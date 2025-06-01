import logging
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import train_test_split
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union, Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

import utils as ut

logger = logging.getLogger(__name__)


def bins(s, n: int = 8):
    return pd.cut(s, bins=n, labels=False)


def bins_log2(s, n: int = 8):
    # Can't do less bins AND have enough elements in each bin
    # There are several "outliers" in the NCREs activity,
    # take another log2 to compact the data further for assigning bins
    return bins(np.log2(s + 1), n=n)


def interleave_with_rev_comp(data: np.ndarray, data_rev: np.ndarray):
    return np.stack([data, data_rev]).T.flatten()


class MemMapDataset(Dataset):
    def __init__(self, fp_npy_1hot_seqs: Path, targets: np.ndarray):
        self.targets = targets
        self.seqs_1hot: np.ndarray = np.load(fp_npy_1hot_seqs, mmap_mode="r")

        if len(self.seqs_1hot) != len(self.targets):
            raise ValueError(f"{len(self.seqs_1hot) = } != {len(self.targets) = }")

        if len(self.seqs_1hot.shape) != 3:
            raise ValueError(f"{self.seqs_1hot.shape = } must be 3")
        if self.seqs_1hot.shape[1] != 4:
            raise ValueError(f"{self.seqs_1hot.shape[1] = } must be 4")

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
    seqs_tensor = torch.from_numpy(seqs_batch).float()
    # Targets: small list (compared to seqs_batch), so direct tensor conversion
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    # add one dimension in targets: [batch] -> [batch, 1]
    return seqs_tensor, targets_tensor.unsqueeze(1)


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        fp_dataset: Union[Path, str],
        x_col: str = "Seq",
        targets_col: str = "ESC_log2_enrichment",
        fp_npy_1hot_seqs: Optional[Union[Path, str]] = None,
        fp_npy_1hot_seqs_rev_comp: Optional[Union[Path, str]] = None,
        random_state: int = 20240413,
        folds: Optional[int] = None,  # Number of folds for cross-validation
        fold: int = 0,  # Current fold index (0 to folds-1)
        frac_test: float = 0.10,
        frac_val: float = 0.10,
        validate_split: bool = True,
        bins_func: Callable = bins_log2,
        padding: int = 1000,
        # DataLoader kwargs:
        batch_size: int = 128,
        # On macOS with MPS (Apple Silicon) using multiprocessing did not give any speed up.
        # With CUDA maybe it does, thought out MemMapDataset is already efficient.
        num_workers: int = 0,
        # Only relevant if num_workers > 0
        # Ref: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        # multiprocessing_context="fork" won't consume a lot of RAM,
        # but might not work (well) on all OSes/versions.
        multiprocessing_context=None,
        persistent_workers: Optional[bool] = None,
        **dataloader_kwargs,
    ):
        super().__init__()
        if folds is not None and fold >= folds:
            raise ValueError(f"{fold = } must be less than {folds = }")
        if frac_val and folds:
            # To ensure there is no confusion about how the folds work
            raise ValueError(
                "frac_val does not apply for folds. "
                "Set frac_val=0 if you are using folds (or set folds=None)."
            )

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

        if fp_npy_1hot_seqs_rev_comp:
            self.fp_npy_1hot_seqs_rev_comp = Path(fp_npy_1hot_seqs_rev_comp)
            if not self.fp_npy_1hot_seqs_rev_comp.is_file():
                raise FileNotFoundError(self.fp_npy_1hot_seqs_rev_comp)
        else:
            self.fp_npy_1hot_seqs_rev_comp = self.fp_dataset.parent / "seqs_rev.npy"

        self.x_col = x_col
        self.targets_col = targets_col
        self.random_state = random_state
        self.folds = folds or None  # 0 is equivalent to no folds
        self.fold = fold
        self.frac_test = frac_test
        self.frac_val = frac_val
        self.bins_func = bins_func
        self.padding = padding
        self.validate_split = validate_split

        self.targets: Optional[np.ndarray] = None
        self.indices_train: Optional[np.ndarray] = None
        self.indices_test: Optional[np.ndarray] = None
        self.indices_val: Optional[np.ndarray] = None

        if persistent_workers is None:
            persistent_workers = num_workers > 0

        # A convenience wrapper for instantiating DataLoader
        self.DataLoader = partial(
            DataLoader,
            shuffle=False,  # we already shuffle the indices when splitting data
            collate_fn=collate,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
            persistent_workers=persistent_workers,
            batch_size=batch_size,
            **dataloader_kwargs,
        )

    def concat_rev(self, indices):
        if self.dataset_rev:
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
        assert self.fp_npy_1hot_seqs_rev_comp is not None

        if self.fp_npy_1hot_seqs.exists() and self.fp_npy_1hot_seqs_rev_comp.exists():
            return

        seqs_str = pd.read_csv(self.fp_dataset, usecols=[self.x_col])[self.x_col]
        if self.fp_npy_1hot_seqs.exists() and self.fp_npy_1hot_seqs_rev_comp.exists():
            logger.info(
                f"Using precomputed one-hot encoded sequences: "
                f"{self.fp_npy_1hot_seqs} and {self.fp_npy_1hot_seqs_rev_comp}"
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

        if not self.fp_npy_1hot_seqs_rev_comp.exists():
            seqs_1hot_rev_comp = ut.one_hot_reverse_complement(seqs_1hot)
            logger.info(
                f"Saving reverse-complement one-hot encoded sequences: "
                f"{self.fp_npy_1hot_seqs_rev_comp}"
            )
            np.save(self.fp_npy_1hot_seqs_rev_comp, seqs_1hot_rev_comp)

    def setup(self, stage: Literal["fit", "test", None] = None):
        """It is safe to make state assignments here"""

        df = pd.read_csv(self.fp_dataset, usecols=[self.targets_col])
        self.targets = df[self.targets_col].to_numpy()

        assert self.fp_npy_1hot_seqs is not None
        self.dataset_fwd = MemMapDataset(
            fp_npy_1hot_seqs=self.fp_npy_1hot_seqs,
            targets=self.targets,
        )
        if self.fp_npy_1hot_seqs_rev_comp:
            self.dataset_rev = MemMapDataset(
                fp_npy_1hot_seqs=self.fp_npy_1hot_seqs_rev_comp,
                targets=self.targets,
            )
            self.dataset = ConcatDataset([self.dataset_fwd, self.dataset_rev])
        else:
            self.dataset_rev = None
            self.dataset = self.dataset_fwd

        indices_train_val = np.arange(len(self.targets))
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

        if stage == "test":
            return  # the rest of the code is not needed for test stage

        # Make a train/val split, k-folds for cross-validation or simple split
        if self.folds is not None:
            kf = StratifiedKFold(
                n_splits=self.folds,
                shuffle=True,
                random_state=self.random_state,
            )
            targets_train_val = self.targets[indices_train_val]
            # X is dummy, it is not relevant for stratified splitting
            split = kf.split(X=targets_train_val, y=self.bins_func(targets_train_val))
            for i, (train_idxs, val_idxs) in enumerate(split):
                if i != self.fold:
                    continue  # only run for the current fold, continue otherwise
                assert len(val_idxs) < len(train_idxs)
                self.indices_train = self.concat_rev(indices_train_val[train_idxs])
                self.indices_val = self.concat_rev(indices_train_val[val_idxs])
        else:
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
        assert set_train & set_val == set(), f"{set_train = } != {set_val = }"

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
        if self.dataset_rev:
            len_targets *= 2

        # The data split must cover the whole dataset
        assert num_samples == len_targets, f"{num_samples = } != {len_targets = }"
        assert len(unique_samples) == len_targets, (
            f"{len(unique_samples) = } != {len_targets = }"
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

    def predict_dataloader(self):
        return self.test_dataloader()


def extract_fold(fp: Path):
    parts = fp.parent.parent.name.split("_f")
    if len(parts) == 2:
        version, fold = parts
    else:
        version, fold = parts[0], None
    return version, fold


def extract_epoch(fp: Path):
    # name = "epoch000.ckpt"
    return int(fp.name.split(".")[0].split("=")[-1])


def list_checkpoints(
    dp_train: Path, task: str, version: str = "", read_tb: bool = True
):
    dp = dp_train / task
    dps_folds = list(fp for fp in dp.glob(f"{version}*") if fp.is_dir())
    rows = []
    for dp in dps_folds:
        dp = dp / "checkpoints"
        rows += list(dp.glob("*.ckpt"))

    s_fps = pd.Series(rows)
    s_fps.name = "fp"
    df = s_fps.to_frame()

    df["version"], df["fold"] = zip(*df.fp.map(extract_fold))  # type: ignore
    df["fold"] = df["fold"].astype(int)
    df["epoch"] = df.fp.map(extract_epoch)
    df.sort_values(["fold", "epoch"], inplace=True)

    if not read_tb:
        return df

    dfs = []
    for _, df_v in df.groupby("version"):
        for _, df_f in df_v.groupby("fold"):
            fp_tb = list(df_f.fp.iloc[0].parent.parent.glob("events.out.tfevents.*"))[0]
            df_tb = ut.read_tensorboard_log(fp_tb)  # comes sorted by step
            for col in df_tb.columns:
                df_f[col] = df_tb[col].values
            dfs.append(df_f)

    df = pd.concat(dfs)
    return df


def pick_checkpoint(
    df: pd.DataFrame, col_train: str = "loss_train", col_val: str = "loss_val", ax=None
):
    df = df.copy()
    df.set_index("epoch", inplace=True)
    df.sort_index(inplace=True)
    if ax:
        cols = [col_train, col_val]
        min_, max_ = min(df[cols].min()), max(df[cols].max())
        df.plot(y=cols, marker=".", ax=ax)

    m = df[col_train] < df[col_val]
    # Choose the lowest validation loss
    epoch = df[m].sort_values(by="loss_val").index.values[0]

    if ax:
        ax.vlines(epoch, min_, max_, color="red", linestyle="--")

    return int(epoch)
