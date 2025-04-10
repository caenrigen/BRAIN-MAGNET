from random import Random
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


def pad_arr(snippet: np.ndarray, arr_pre: int = 1024):
    assert len(snippet) <= arr_pre, len(snippet)
    arr_len = len(snippet)
    pad = (arr_pre - arr_len) / 2
    return np.pad(snippet, [(int(pad), math.ceil(pad)), (0, 0)], mode="constant")


def tensor_reverse_complement(x):
    # Flip along the length dimension (dim=3).
    x_rev = torch.flip(x, dims=[3])

    # Swap A <-> T and C <-> G channels.
    # Channel order is (A=0, C=1, G=2, T=3)
    channel_map = [3, 2, 1, 0]  # T, G, C, A
    x_revcomp = x_rev[:, channel_map, :, :]
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


def augment_data(df_train: pd.DataFrame, num_shits: int = 2, random_state: int = 913):
    """"""
    random_obj = Random(random_state)
    seq_lens = df_train.Seq.str.len()
    shifts_per_seq = [
        tuple(
            gen_shifts(
                seq_len, random_obj=random_obj, num_shifts=num_shits, max_len=1024
            )
        )
        for seq_len in seq_lens
    ]
    dfs = [df_train]
    for i in range(num_shits):
        df = df_train.copy()
        df["SeqEnc"] = [
            shift_padded_onehot(arr, shift=shifts[i])
            for arr, shifts in zip(df_train.SeqEnc, shifts_per_seq)
        ]
        dfs.append(df)
    df = pd.concat(dfs)
    return df.sample(frac=1, random_state=random_state)
