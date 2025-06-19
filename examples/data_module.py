import logging
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import train_test_split
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

import utils as ut

logger = logging.getLogger(__name__)

# For reference
DATASET_COLS = (
    "Chr",  # Chromosome
    "Start",  # Start position in the chromosome
    "End",  # End position in the chromosome
    "Seq_len",  # Length of the sequence
    "GC_counts",  # Number of "G"s and "C"s in the sequence
    "N_counts",  # Number of "N"s in the sequence
    "NSC_log2_enrichment",  # Log2 enrichment of NSC
    "ESC_log2_enrichment",  # Log2 enrichment of ESC
    "Seq",  # Sequence
)
# for convenience, because we often don't need the sequences (which is heavy on memory)
DATASET_COLS_NO_SEQ = DATASET_COLS[:-1]


def bins(s, n: int = 8):
    return pd.cut(s, bins=n, labels=False)


def bins_log2(s, n: int = 8):
    # Can't do less bins AND have enough elements in each bin
    # There are several "outliers" in the NCREs activity,
    # take another log2 to compact the data further for assigning bins
    return bins(np.log2(s + 1), n=n)


def interleave_with_rev_comp(data: np.ndarray, data_rev: np.ndarray):
    return np.stack([data, data_rev]).T.flatten()


def interleave_undo(data: np.ndarray):
    return data.reshape(len(data) // 2, 2).T


class MemMapDataset(Dataset):
    def __init__(
        self,
        fp_npy_1hot_seqs: Path,
        targets: np.ndarray,
        rev_comp: bool = False,
    ):
        self.targets = targets
        self.seqs_1hot: np.ndarray = np.load(fp_npy_1hot_seqs, mmap_mode="r")
        self.rev_comp = rev_comp

        if self.rev_comp:
            self.seqs_1hot = ut.one_hot_reverse_complement(self.seqs_1hot)  # type: ignore

        if len(self.seqs_1hot) != len(self.targets):
            raise ValueError(f"{len(self.seqs_1hot)} != {len(self.targets)}")

        if self.seqs_1hot.ndim != 3:
            raise ValueError(f"{self.seqs_1hot.ndim} must be 3")
        if self.seqs_1hot.shape[1] != 4:
            raise ValueError(f"{self.seqs_1hot.shape[1]} must be 4")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: Union[int, slice, np.ndarray, List[int]]):
        return self.seqs_1hot[idx], self.targets[idx]


def collate(batch: List[Tuple[np.ndarray, np.ndarray]]):
    """
    Collate a list of tuples (numpy_array, target_scalar) into a single tuple
    (torch.Tensor, torch.Tensor).

    The input has the shape (batch_size, 4, L) and the targets have
    the shape (batch_size,).

    The input comes from indexing/slicing the MemMapDataset (i.e. the return
    of MemMapDataset.__getitem__).
    """
    seqs, targets = zip(*batch)
    # Stack all NumPy arrays along new batch dimension
    seqs_batch = np.stack(seqs)

    expected_shape = (len(batch), 4, len(batch[0][0][0]))
    assert seqs_batch.shape == expected_shape, (seqs_batch.shape, expected_shape)

    # Convert to torch.Tensor (float32)
    seqs_tensor = torch.tensor(
        seqs_batch.astype(np.float32), dtype=torch.float32
    ).unsqueeze(2)
    # Targets: small list (compared to seqs_batch), so direct tensor conversion
    targets_tensor = torch.tensor(
        [t.astype(np.float32) for t in targets], dtype=torch.float32
    )

    # add one dimension in targets: [batch] -> [batch, 1]
    return seqs_tensor, targets_tensor.unsqueeze(1)


class DataModule:
    def __init__(
        self,
        fp_dataset: Union[Path, str],
        augment_w_rev_comp: bool,
        targets_col: str,
        x_col: str = "Seq",
        fp_npy_1hot_seqs: Optional[Union[Path, str]] = None,
        random_state: int = 20240413,
        frac_test: float = 0.00,
        frac_val: float = 0.10,
        validate_split: bool = True,
        bins_func: Callable = bins_log2,
        padding: int = 1000,
        # DataLoader kwargs:
        batch_size: int = 128,
        **dataloader_kwargs,
    ):
        super().__init__()

        self.fp_dataset = Path(fp_dataset)
        if not self.fp_dataset.is_file():
            raise FileNotFoundError(self.fp_dataset)

        if fp_npy_1hot_seqs:
            self.fp_npy_1hot_seqs = Path(fp_npy_1hot_seqs)
            if not self.fp_npy_1hot_seqs.is_file():
                raise FileNotFoundError(self.fp_npy_1hot_seqs)
        else:
            self.fp_npy_1hot_seqs: Optional[Path] = None
            self.fp_npy_1hot_seqs = self.fp_dataset.parent / "seqs.npy"

        self.augment_w_rev_comp = augment_w_rev_comp
        self.x_col = x_col
        self.targets_col = targets_col
        self.random_state = random_state
        self.frac_test = frac_test
        self.frac_val = frac_val
        self.bins_func = bins_func
        self.padding = padding
        self.validate_split = validate_split

        self.targets: Optional[np.ndarray] = None
        self.indices_train: Optional[np.ndarray] = None
        self.indices_test: Optional[np.ndarray] = None
        self.indices_val: Optional[np.ndarray] = None

        # A convenience wrapper for instantiating DataLoader
        self.DataLoader = partial(
            DataLoader,
            # we already shuffle the indices when splitting data.
            # For the full dataset, we want to stay consistent with the original order.
            shuffle=False,
            collate_fn=collate,
            batch_size=batch_size,
            **dataloader_kwargs,
        )

    def concat_rev(self, indices):
        if self.augment_w_rev_comp:
            indices = np.asarray(indices, dtype=np.int64)
            # Interleave forward and reverse complement indices, instead of simply
            # concatenating them. Might help to learn faster.
            # return np.concatenate([indices, indices + len(self.targets)])
            assert self.targets is not None
            return interleave_with_rev_comp(indices, indices + len(self.targets))
        return indices

    def prepare_data(self):
        """
        For "one time actions" before any training experiments.
        This is where you typically download the dataset.
        ! Do not make state assignments here (e.g. self.indices_train = ..., etc.),
        ! use `setup()` instead
        """

        assert self.fp_npy_1hot_seqs is not None

        if self.fp_npy_1hot_seqs.exists():
            return

        seqs_str = pd.read_csv(self.fp_dataset, usecols=[self.x_col])[self.x_col]
        if self.fp_npy_1hot_seqs.exists():
            logger.info(
                f"Using precomputed one-hot encoded sequences: {self.fp_npy_1hot_seqs}"
            )
            return

        seqs_1hot = ut.sequences_str_to_1hot(
            seqs_str,  # type: ignore
            pad_to=self.padding,
            transpose=True,
        )
        if not self.fp_npy_1hot_seqs.exists():
            logger.info(f"Saving one-hot encoded sequences: {self.fp_npy_1hot_seqs}")
            np.save(self.fp_npy_1hot_seqs, seqs_1hot)

    def setup(self, stage: Optional[str] = None):
        """It is ok to make state assignments in this method"""

        df = pd.read_csv(self.fp_dataset, usecols=[self.targets_col])
        self.targets = df[self.targets_col].to_numpy()

        assert self.fp_npy_1hot_seqs is not None
        self.dataset_fwd = MemMapDataset(
            fp_npy_1hot_seqs=self.fp_npy_1hot_seqs,
            targets=self.targets,
        )
        if self.augment_w_rev_comp:
            self.dataset_rev = MemMapDataset(
                fp_npy_1hot_seqs=self.fp_npy_1hot_seqs,
                targets=self.targets,
                rev_comp=True,
            )
            self.dataset = ConcatDataset([self.dataset_fwd, self.dataset_rev])
        else:
            self.dataset_rev = None
            self.dataset = self.dataset_fwd

        indices_train_val = np.arange(len(self.targets))
        self.indices_full = self.concat_rev(indices_train_val)
        if self.frac_test:
            # Put aside a fraction of the data for a final testing
            indices_train_val, self.indices_test = train_test_split(
                indices_train_val,
                test_size=self.frac_test,
                random_state=self.random_state,
                stratify=self.bins_func(self.targets),
                shuffle=True,  # stratify requires shuffle=True
            )
            self.indices_test = self.concat_rev(self.indices_test)

        indices_train, indices_val = train_test_split(
            indices_train_val,
            test_size=self.frac_val,
            random_state=self.random_state,
            stratify=self.bins_func(self.targets[indices_train_val]),
            shuffle=True,  # stratify requires shuffle=True
        )
        self.indices_train = self.concat_rev(indices_train)
        self.indices_val = self.concat_rev(indices_val)

        if self.validate_split:
            self.validate_data_split()

    def validate_data_split(self):
        assert self.indices_train is not None
        assert self.indices_val is not None

        set_train = set(self.indices_train)
        set_val = set(self.indices_val)
        # The train and val sets must be disjoint
        assert set_train & set_val == set(), f"{set_train} != {set_val}"

        num_samples = len(self.indices_train) + len(self.indices_val)
        unique_samples = set_train | set_val

        if self.frac_test:
            assert self.indices_test is not None
            set_test = set(self.indices_test)
            assert set_test & set_train == set()
            assert set_test & set_val == set()
            num_samples += len(self.indices_test)
            unique_samples |= set_test

        assert self.targets is not None
        len_targets = len(self.targets)
        if self.augment_w_rev_comp:
            len_targets *= 2

        # The data split must cover the whole dataset
        assert num_samples == len_targets, f"{num_samples} != {len_targets}"
        assert len(unique_samples) == len_targets, (
            f"{len(unique_samples)} != {len_targets}"
        )

    def train_dataloader(self):
        assert self.indices_train is not None
        return self.DataLoader(self.dataset, sampler=self.indices_train)

    def val_dataloader(self):
        assert self.indices_val is not None
        return self.DataLoader(self.dataset, sampler=self.indices_val)

    def test_dataloader(self):
        assert self.indices_test is not None
        return self.DataLoader(self.dataset, sampler=self.indices_test)

    def full_dataloader(self):
        assert self.indices_full is not None
        return self.DataLoader(self.dataset, sampler=self.indices_full)
