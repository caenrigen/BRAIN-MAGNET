from pathlib import Path
from typing import Optional, Sequence, Union, Literal
from functools import partial
import time

import pandas as pd
import torch
import numpy as np
from numpy.random import default_rng
from sklearn.metrics import mean_squared_error
from scipy import stats
from tensorboard.backend.event_processing import event_accumulator


def empty_cache(device: torch.device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def make_version(fold: Optional[int] = None, version: Optional[str] = None):
    """
    Generate a random version string for this training run, it is used to name the
    folder where the results of the training are saved.
    """
    rng_version = default_rng(int(time.time()))
    if version is None:
        version = rng_version.random(1).tobytes()[:4].hex()
    if fold is not None:
        return f"{version}_f{fold}"
    else:
        return version


def to_uint8(seq: str):
    return np.frombuffer(seq.encode("ascii"), dtype=np.uint8)


# function to convert only one piece of sequence to np.array
def make_one_hot_encode(alphabet: str = "ACGT", dtype=np.uint8) -> np.ndarray:
    """
    One-hot encode for a sequence.
    A -> [1,0,0,0]
    C -> [0,1,0,0]
    G -> [0,0,1,0]
    T -> [0,0,0,1]
    N -> [0,0,0,0] (or any other character)
    """
    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    return hash_table


HASH_TABLE = make_one_hot_encode()


def one_hot_encode(
    sequence: str, pad_to: Optional[int] = None, transpose: bool = False
):
    one_hot = HASH_TABLE[to_uint8(sequence)]
    if pad_to is not None:
        one_hot = pad_one_hot(one_hot, to=pad_to)
    if transpose:
        one_hot = one_hot.T
    return one_hot


def pad_one_hot(one_hot: np.ndarray, to: int):
    """
    You might want to pad to 1024 because certain neural network layers have not been
    implemented for lengths that are not powers of 2 on all types of devices (e.g.
    Apple MPS). For the simplified model we used, it is not necessary.
    """
    assert one_hot.ndim == 2 and one_hot.shape[1] == 4, one_hot.shape
    assert len(one_hot) <= to, len(one_hot)
    if len(one_hot) == to:
        return one_hot
    arr_len = len(one_hot)
    pad = (to - arr_len) / 2
    return np.pad(one_hot, [(int(pad), int(np.ceil(pad))), (0, 0)], mode="constant")


def unpad_one_hot(one_hot: np.ndarray, acgt_axis: Literal[0, 1] = 1):
    """Drop all zeros at the start and/or end of the sequence."""
    assert one_hot.ndim == 2, one_hot.shape
    assert acgt_axis in (0, 1), acgt_axis
    # +1 because np.diff "shifts" towards the start of the array
    idxs = one_hot.sum(axis=acgt_axis).nonzero()[0]
    if idxs.size == 0:
        return one_hot  # no padding, nothing to unpad
    if acgt_axis == 1:
        return one_hot[idxs[0] : idxs[-1] + 1]
    else:
        return one_hot[:, idxs[0] : idxs[-1] + 1]


def one_hot_reverse_complement(one_hot: Union[np.ndarray, torch.Tensor]):
    """
    Reverse complement a one-hot encoded sequence or a batch of one-hot encoded
    sequences.

    ! Reminder: If the `one_hot` input is padded, the padding will be flipped too.
    """
    # We need to flip along the bp_num dimension to obtain the "reverse".
    # To obtain the complement we have to swap the channels:
    # A <-> T and C <-> G
    # Since the channel are encoded as (A=0, C=1, G=2, T=3), flipping this dimension
    # has the desired effect too.
    assert any(s == 4 for s in one_hot.shape[-2:])
    assert one_hot.ndim in (2, 3), one_hot.shape
    # Flipping return a view of the array, cheap operation.
    if isinstance(one_hot, np.ndarray):
        return np.flip(one_hot, (-2, -1))
    elif isinstance(one_hot, torch.Tensor):
        return torch.flip(one_hot, (-2, -1))
    else:
        raise NotImplementedError(f"Unsupported {type(one_hot) = !r}")


def sequences_str_to_1hot(
    sequences: Sequence[str],
    pad_to: int = 1000,
    transpose: bool = True,
):
    """Transpose to match how pytorch organizes data: (batch_size, channels=4, num_bp)"""
    func = partial(one_hot_encode, pad_to=pad_to, transpose=transpose)
    return np.stack(list(map(func, sequences)), axis=0)


def read_tensorboard_log(
    fp: Path,
    scalars: Sequence[str] = ("loss/train", "loss/val", "pearson/train", "pearson/val"),
):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    assert fp.exists(), fp
    size_guidance = {event_accumulator.SCALARS: 0}
    ea = event_accumulator.EventAccumulator(str(fp), size_guidance=size_guidance)
    ea.Reload()  # load events from file
    assert all(s in ea.Tags()["scalars"] for s in scalars)
    dfs = []
    for s in scalars:
        df = pd.DataFrame(ea.Scalars(s))
        df["scalar"] = s
        dfs.append(df)
    df = pd.concat(dfs)
    df["scalar"] = df.scalar.str.replace("/", "_")
    df.set_index(["scalar", "step"], inplace=True)
    df = df.unstack(level="scalar").value  # there is also .wall_time
    df.reset_index(level="step", inplace=True)
    df.set_index("step", drop=True, inplace=True)
    df.sort_index(inplace=True, ascending=True)
    df.columns.name = None
    return df


def model_stats(targets: np.ndarray, preds: np.ndarray):
    mse = mean_squared_error(targets, preds)
    pearson = float(stats.pearsonr(targets, preds).statistic)  # type: ignore
    spearman = float(stats.spearmanr(targets, preds).statistic)  # type: ignore
    return mse, pearson, spearman


def run_tests():
    """Run tests for some of the functions in this module"""
    seq = "ACCAGCT"
    assert len(seq) % 2 == 1  # odd length since it is more tricky to handle

    seq_revcomp = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    one_hot = one_hot_encode(seq, pad_to=10)
    assert one_hot.shape == (10, 4)

    batch = sequences_str_to_1hot([seq], pad_to=10, transpose=False)
    assert batch.shape == (1, 10, 4)
    assert np.all(batch[0] == one_hot)

    batch = sequences_str_to_1hot([seq], pad_to=10, transpose=True)
    assert batch.shape == (1, 4, 10)
    assert np.all(batch[0].T == one_hot)

    expected_1hot_list = [
        [0, 0, 0, 0],  # padding
        [1, 0, 0, 0],  # A
        [0, 1, 0, 0],  # C
        [0, 1, 0, 0],  # C
        [1, 0, 0, 0],  # A
        [0, 0, 1, 0],  # G
        [0, 1, 0, 0],  # C
        [0, 0, 0, 1],  # T
        [0, 0, 0, 0],  # padding
        [0, 0, 0, 0],  # padding
    ]

    expected_1hot = np.array(expected_1hot_list)
    assert np.all(one_hot == expected_1hot)

    unpadded = unpad_one_hot(one_hot)
    assert np.all(unpadded == expected_1hot[1:-2])
    assert np.all(unpad_one_hot(one_hot[1:-2]) == expected_1hot[1:-2])
    assert np.all(unpad_one_hot(one_hot[:-2]) == expected_1hot[1:-2])
    assert np.all(unpad_one_hot(one_hot[1:]) == expected_1hot[1:-2])

    reverse_complement = one_hot_reverse_complement(one_hot)
    assert expected_1hot.shape == reverse_complement.shape
    assert not np.all(reverse_complement == expected_1hot[::-1])

    recovered = one_hot_reverse_complement(reverse_complement)
    assert np.all(recovered == one_hot)

    revcomp_1hot = one_hot_encode(seq_revcomp, pad_to=10)

    # Expected to be different because the of the padding of a odd-length sequence
    assert not np.all(reverse_complement == revcomp_1hot)
    assert np.all(reverse_complement[1:] == revcomp_1hot[:-1])

    batch = sequences_str_to_1hot([seq], pad_to=10, transpose=False)
    batch_revcomp = one_hot_reverse_complement(batch)
    assert np.all(reverse_complement == batch_revcomp[0])

    batch = sequences_str_to_1hot([seq], pad_to=10, transpose=True)
    batch_revcomp = one_hot_reverse_complement(batch)
    assert np.all(reverse_complement == batch_revcomp[0].T)

    one_hot_torch = torch.from_numpy(one_hot)
    revcomp_torch = one_hot_reverse_complement(one_hot_torch)
    assert np.all(reverse_complement == revcomp_torch.numpy())

    batch_torch = torch.from_numpy(batch)
    batch_revcomp_torch = one_hot_reverse_complement(batch_torch)
    assert np.all(reverse_complement == batch_revcomp_torch.numpy()[0].T)


if __name__ == "__main__":
    run_tests()
