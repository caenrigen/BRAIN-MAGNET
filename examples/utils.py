from pathlib import Path
from typing import Optional, Sequence
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
    N -> [0,0,0,0]
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


def pad_one_hot(one_hot: np.ndarray, to: int = 1000):
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


def unpad_one_hot(one_hot: np.ndarray):
    # Drop all rows that are all zeros (zero padding)
    return one_hot[one_hot.sum(axis=1, dtype=np.bool)]


def one_hot_reverse_complement(one_hot, is_transposed: bool = False):
    # Swap A <-> T and C <-> G channels. Channel order is (A=0, C=1, G=2, T=3)
    channel_map = [3, 2, 1, 0]  # T, G, C, A
    if is_transposed:
        assert one_hot.shape[0] == 4, one_hot.shape
        return np.flip(one_hot, axis=0)[channel_map, :]
    else:
        assert one_hot.shape[1] == 4, one_hot.shape
        return np.flip(one_hot, axis=1)[:, channel_map]


def tensor_reverse_complement(x: torch.Tensor):
    assert x.ndim == 3, x.shape
    assert x.shape[1] == 4, x.shape
    # Flip along the length dimension (dim=2).
    x_rev = torch.flip(x, dims=[2])
    # Swap A <-> T and C <-> G channels. Channel order is (A=0, C=1, G=2, T=3)
    channel_map = [3, 2, 1, 0]  # T, G, C, A
    x_revcomp = x_rev[:, channel_map, :]
    return x_revcomp


def sequences_str_to_1hot(
    sequences: Sequence[str],
    pad_to: int = 1000,
    transpose: bool = True,
):
    """Transpose to match how pytorch organizes data: (batch_size, channels=4, num_bp)"""
    func = partial(one_hot_encode, pad_to=pad_to, transpose=transpose)
    seqs_1hot = list(map(func, sequences))
    # Return the seqs_1hot so that we can use it for reverse complement
    return np.stack(seqs_1hot, axis=0)


def sequences_1hot_to_reverse_complement(
    sequences_1hot: Sequence[np.ndarray],
    is_transposed: bool = True,
):
    """Reverse complement numpy one-hot encoded sequences"""
    func = partial(one_hot_reverse_complement, is_transposed=is_transposed)
    return np.stack(list(map(func, sequences_1hot)), axis=0)


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
