from random import Random
from functools import partial
import numpy as np
import pandas as pd


def to_uint8(string):
    return np.frombuffer(string.encode("ascii"), dtype=np.uint8)


# function to convert only one piece of sequence to np.array
def make_one_hot_encode(alphabet: str = "ACGT", dtype=np.float32) -> np.ndarray:
    """
    One-hot encode for a sequence.
    A -> [1,0,0,0]
    C -> [0,1,0,0]
    G -> [0,0,1,0]
    T -> [0,0,0,1]
    N -> [0,0,0,0]
    """

    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    return hash_table


HASH_TABLE = make_one_hot_encode()


def one_hot_encode(sequence: str):
    return HASH_TABLE[to_uint8(sequence)]


def pad_one_hot(one_hot: np.ndarray, to: int = 1024):
    """
    Padded to 1024 because certain neural network layers have not been implemented
    for lengths that are not a power of 2 on all types of devices (e.g. Apple MPS).
    """
    assert one_hot.ndim == 2 and one_hot.shape[1] == 4, one_hot.shape
    assert len(one_hot) <= to, len(one_hot)
    arr_len = len(one_hot)
    pad = (to - arr_len) / 2
    return np.pad(one_hot, [(int(pad), int(np.ceil(pad))), (0, 0)], mode="constant")


def unpad_one_hot(one_hot: np.ndarray):
    # Drop all rows that are all zeros (zero padding)
    return one_hot[one_hot.sum(axis=1) != 0]


def tensor_reverse_complement(x):
    # Flip along the length dimension (dim=2).
    x_rev = torch.flip(x, dims=[2])

    # Swap A <-> T and C <-> G channels.
    # Channel order is (A=0, C=1, G=2, T=3)
    channel_map = [3, 2, 1, 0]  # T, G, C, A
    x_revcomp = x_rev[:, channel_map, :]
    return x_revcomp


def gen_shifts(
    seq_len: int,
    random_obj: Random,
    max_len: int = 1024,
    num_shifts: int = 2,
):
    p_right = (max_len - seq_len) // 2
    p_left = max_len - seq_len - p_right
    assert p_left + p_right + seq_len == max_len
    out = set()
    while len(out) < num_shifts:
        num = random_obj.randint(-p_left, p_right)
        if num == 0 or num in out:
            continue  # this dataset was already generated
        yield num
        out.add(num)


def shift_padded_onehot(arr: np.ndarray, shift: int):
    # since we just change the padding from one side to the other,
    # slicing and concatenating is enough. Zeros are preserver on both sides
    # NB Negative shift works as expected.
    return np.concatenate((arr[shift:, :], arr[:shift, :]), axis=0)


def make_augment_col(
    enrichment: pd.Series,
    min_augment: int = 0,
    max_augment: int = 24,
    num_bins: int = 100,
):
    """
    This was an experiment. It did not seem to improve spearman/pearson correlation.

    Make a column with the number of times each sequence should be augmented.
    The less samples for an enrichment bin, the more it should be augmented to keep it
    representative during the training.

    `max_augment` is the maximum number of times any sequence can be augmented. It can
    go up to 24 because the sequences are <=1000 base pairs long and we are padding all
    sequences to 1024 for technical reasons, which is convenient for the augmentation.
    """
    counts, bins = np.histogram(enrichment, bins=num_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # +1 because the augmentation is a + {N multiples of the original count},
    # the original data is always included.
    counts_max = (min_augment + 1) * counts.max()
    counts = counts.clip(1, None)  # avoid division by 0
    # the intension here is to generate a uniform distribution of across the bins, but
    # we must clip to the max_augment since we have limited augmentation possibilities
    # In the end the distribution will be similar to a normal distribution but truncated
    # at the top.
    augment = (counts_max / counts).clip(min_augment + 1, max_augment + 1) - 1
    return np.int64(np.interp(enrichment, bin_centers, augment).round())


def augment_data(df_input: pd.DataFrame, random_state: int = 913):
    """
    This was an experiment to augment the data by shifting the sequences within the
    padded edges.
    HOWEVER! Augmenting data with differently padded sequences is irrelevant because the
    convolution layers slide over the DNA sequence!
    """
    random_obj = Random(random_state)
    seq_lens = df_input.Seq.str.len()
    dfs = [df_input]  # preserve a copy of the original data
    for num_shifts, df in df_input.groupby("augment"):
        if num_shifts == 0:
            continue
        if not isinstance(num_shifts, int):
            raise ValueError(f"num_shifts must be an integer, got {num_shifts}")
        gen = partial(
            gen_shifts, random_obj=random_obj, num_shifts=num_shifts, max_len=1024
        )
        shifts_per_seq = [tuple(gen(seq_len)) for seq_len in seq_lens]
        for i in range(num_shifts):
            df = df.copy()
            df["SeqEnc"] = [
                shift_padded_onehot(arr, shift=shifts[i])
                for arr, shifts in zip(df.SeqEnc, shifts_per_seq)
            ]
            dfs.append(df)
    df = pd.concat(dfs)
    return df.sample(frac=1, random_state=random_state)
