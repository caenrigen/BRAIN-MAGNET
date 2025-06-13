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
    return np.frombuffer(seq.encode(), dtype=np.uint8)


DNA_ALPHABET = "ACGT"


def make_one_hot_encoded_alphabet(
    alphabet: str = DNA_ALPHABET, dtype=np.uint8
) -> np.ndarray:
    """
    One-hot encode for a sequence.
    A -> [1,0,0,0]
    C -> [0,1,0,0]
    G -> [0,0,1,0]
    T -> [0,0,0,1]
    N -> [0,0,0,0] (any other character)
    """
    one_hot_alphabet = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    one_hot_alphabet[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    return one_hot_alphabet


ONE_HOT_ENCODED_ALPHABET = make_one_hot_encoded_alphabet()


def one_hot_encode(
    sequence: str,
    pad_to: Optional[int] = None,
    transpose: bool = False,
    validate: bool = True,
):
    if validate:
        assert sequence.isupper(), f"{sequence.isupper() = !r}, {sequence = !r}"
    one_hot = ONE_HOT_ENCODED_ALPHABET[to_uint8(sequence)]
    if pad_to is not None:
        one_hot = pad_one_hot(one_hot, to=pad_to)
    if transpose:
        one_hot = one_hot.T
    return one_hot


DNA_ALPHABET_UINT8 = to_uint8(DNA_ALPHABET)


def one_hot_decode(one_hot: np.ndarray, validate: bool = True):
    """Decode a one-hot encoded sequence."""
    if validate:
        assert one_hot.ndim == 2, one_hot.shape
        assert one_hot.shape[1] == len(DNA_ALPHABET_UINT8), one_hot.shape
        assert np.all(one_hot.sum(axis=1) == 1), "all rows must sum to 1, maybe padded?"
    return DNA_ALPHABET_UINT8[np.argmax(one_hot, axis=-1)].tobytes().decode()


def pad_one_hot(one_hot: np.ndarray, to: int, alphabet_len: int = len(DNA_ALPHABET)):
    """
    Pad a one-hot encoded sequence to a given length `to`.
    Works for a single one-hot encoded sequence or a batch of one-hot encoded
    sequences.
    Expects shapes (seqs_len, alphabet_len) or (batch_size, seqs_len, alphabet_len).
    """
    assert one_hot.ndim >= 2 and one_hot.shape[-1] == alphabet_len, one_hot.shape
    assert one_hot.shape[-2] <= to, one_hot.shape
    if one_hot.shape[-2] == to:
        return one_hot
    arr_len = one_hot.shape[-2]
    pad = (to - arr_len) / 2
    pad_width = [(int(pad), int(np.ceil(pad))), (0, 0)]
    if one_hot.ndim > 2:
        pad_width = (one_hot.ndim - 2) * [(0, 0)] + pad_width
    return np.pad(one_hot, pad_width, mode="constant")


def unpad_one_hot_idxs(
    one_hot: np.ndarray,
    acgt_axis: Literal[0, 1] = 1,
    alphabet_len: int = len(DNA_ALPHABET),
):
    assert one_hot.ndim == 2, f"{one_hot.ndim = !r}, {one_hot.shape = !r}"
    assert acgt_axis in (0, 1), f"{acgt_axis = !r} not in (0, 1)"
    assert one_hot.shape[acgt_axis] == alphabet_len, (
        f"{one_hot.shape[acgt_axis] = !r} != {alphabet_len = !r}"
    )
    # +1 because np.diff "shifts" towards the start of the array
    idxs = one_hot.sum(axis=acgt_axis).nonzero()[0]
    if idxs.size != 0:
        return np.array([idxs[0], idxs[-1]])  # return only the indices we care about
    return idxs


def unpad_one_hot(
    one_hot: np.ndarray,
    acgt_axis: Literal[0, 1] = 1,
    idxs: Optional[np.ndarray] = None,
    alphabet_len: int = len(DNA_ALPHABET),
):
    """Drop all zeros at the start and/or end of the sequence."""
    if idxs is None:
        idxs = unpad_one_hot_idxs(one_hot, acgt_axis, alphabet_len)
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
    pad_to: Optional[int] = 1000,
    # transpose=True shapes it as a tensor batch (batch_size, channels=4, num_bp)
    transpose: bool = True,
):
    """
    Convert string sequences to a one-hot encoded batch of numpy arrays.
    The batch is ready to be converted to a tensor is `transpose=True`.
    Set `pad_to=None` to not pad the sequences (sequences must have the same length).
    """
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

    seq_revcomp = seq.translate(str.maketrans(DNA_ALPHABET, DNA_ALPHABET[::-1]))[::-1]

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
