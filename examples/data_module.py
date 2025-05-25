import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import seaborn as sns

import utils as ut


def bins(s, n: int = 8):
    return pd.cut(s, bins=n, labels=False)


def bins_log2(s, n: int = 8):
    # Can't do less bins AND have enough elements in each bin
    # There are several "outliers" in the NCREs activity,
    # take another log2 to compact the data further for assigning bins
    return bins(np.log2(s + 1), n=n)


def make_tensor_dataset(df: pd.DataFrame, x_col: str, y_col: str, device: torch.device):
    x = np.stack(df[x_col].values)  # type: ignore
    # [batch, seq_len, 4] -> [batch, 4, seq_len]
    # Convolutional layers expect tensors with shape [batch, channels, length]
    tensor_x = torch.tensor(x, device=device, dtype=torch.float32).permute(0, 2, 1)
    # add one dimension in targets: [batch] -> [batch, 1]
    tensor_y = torch.tensor(
        df[y_col].values, device=device, dtype=torch.float32
    ).unsqueeze(1)
    return TensorDataset(tensor_x, tensor_y)


# These seem to be experimental errors, these 10 sequences are predicted by all models
# to have high activity.
# m_outliers = (targets < 0.2) & (preds > 2)
OUTLIER_INDICES = [
    7885,
    19948,
    20312,
    33019,
    46116,
    53222,
    75207,
    111863,
    120174,
    128136,
]


class MemMapDataset(Dataset):
    def __init__(
        self,
        fp_npy_1hot_seqs: Path,
        targets: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ):
        self.seqs_1hot = np.load(fp_npy_1hot_seqs, mmap_mode="r")
        self.targets = targets

        if len(self.seqs_1hot) != len(self.targets):
            raise ValueError(f"{len(self.seqs_1hot) = } != {len(self.targets) = }")

        if len(self.seqs_1hot.shape) != 3:
            raise ValueError(f"{self.seqs_1hot.shape = } must be 3")
        if self.seqs_1hot.shape[1] != 4:
            raise ValueError(f"{self.seqs_1hot.shape[1] = } must be 4")

        if indices is None:
            if len(self.seqs_1hot) != len(self.targets):
                raise ValueError(f"{len(self.seqs_1hot) = } != {len(self.targets) = }")
            self.indices = np.arange(len(targets), dtype=np.int64)
        else:
            if len(indices) > len(targets) or len(indices) > len(self.seqs_1hot):
                raise ValueError(f"{len(indices) = } > {len(targets) = }")
            if max(indices) >= len(targets):
                raise ValueError(f"{max(indices) = } >= {len(targets) = }")
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(
        self, idx: Union[int, slice, np.ndarray, List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        real_idx = self.indices[idx]
        return self.seqs_1hot[real_idx], self.targets[real_idx]


def collate(batch: List[Tuple[np.ndarray, np.ndarray]], device: torch.device):
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
    seqs_tensor = torch.from_numpy(seqs_batch).float().to(device)
    # Targets: small list (compared to seqs_batch), so direct tensor conversion
    targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)

    # add one dimension in targets: [batch] -> [batch, 1]
    return seqs_tensor, targets_tensor.unsqueeze(1)


def make_dl(
    df: pd.DataFrame,
    y_col: str,
    device: torch.device,
    batch_size: int = 256,
    shuffle: bool = False,
):
    return DataLoader(
        make_tensor_dataset(df=df, x_col="SeqEnc", y_col=y_col, device=device),
        batch_size=batch_size,
        shuffle=shuffle,
    )


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        fp_npy_1hot_seqs: Path,
        targets: np.ndarray,
        device: torch.device,
        batch_size: int = 256,
        random_state: int = 913,
        n_folds: Optional[int] = None,  # Number of folds for cross-validation
        fold: int = 0,  # Current fold index (0 to n_folds-1)
        frac_for_test: float = 0.10,
        frac_for_val: float = 0.10,
        bins_func: Callable = bins_log2,
        num_workers: int = 4,
    ):
        super().__init__()
        self.fp_npy_1hot_seqs = fp_npy_1hot_seqs
        self.targets = targets
        self.indices_train = self.indices_test = self.indices_val = None
        self.ds_train = self.ds_val = self.ds_test = None
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_folds = n_folds
        self.fold = fold
        self.frac_for_test = frac_for_test
        self.frac_for_val = frac_for_val
        self.bins_func = bins_func
        self.device = device
        self.num_workers = num_workers
        if n_folds is not None and fold >= n_folds:
            raise ValueError(f"{fold = } must be less than {n_folds = }")

        self.make_dataset = partial(
            MemMapDataset,
            fp_npy_1hot_seqs=self.fp_npy_1hot_seqs,
            targets=self.targets,
        )
        self.make_dataloader = partial(
            DataLoader,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=partial(collate, device=self.device),
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers),
        )

    def prepare_data(self):
        if self.frac_for_test:
            # Put aside a fraction of the data for a final testing
            indices_train_val, self.indices_test = train_test_split(
                np.arange(len(self.targets)),
                test_size=self.frac_for_test,
                random_state=self.random_state,
                stratify=self.bins_func(self.targets),
                shuffle=True,  # stratify requires shuffle=True
            )
        else:
            indices_train_val = np.arange(len(self.targets))

        # Make a train/val split, simple or k-folds cross-validated

        if self.n_folds is not None:
            kf = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            targets_train_val = self.targets[indices_train_val]
            # X is dummy, it is not relevant for stratified splitting
            split = kf.split(X=targets_train_val, y=self.bins_func(targets_train_val))
            for i, (train_indices, val_indices) in enumerate(split):
                if i == self.fold:
                    assert len(val_indices) < len(train_indices)
                    # Get the real indices
                    self.indices_train = indices_train_val[train_indices]
                    self.indices_val = indices_train_val[val_indices]
                    break
        else:
            self.indices_train, self.indices_val = train_test_split(
                indices_train_val,
                test_size=self.frac_for_val,
                random_state=self.random_state,
                stratify=self.bins_func(self.targets[indices_train_val]),
                shuffle=True,  # stratify requires shuffle=True
            )

        self.validate_data_split()

    def validate_data_split(self):
        assert self.indices_train is not None
        assert self.indices_val is not None

        # The train and val sets must be disjoint
        set_all = set(np.arange(len(self.targets)))
        set_train = set(self.indices_train)
        set_val = set(self.indices_val)
        assert set_train & set_val == set(), f"{set_train = } != {set_val = }"

        t = len(self.indices_train) + len(self.indices_val)
        s = set_train | set_val

        if self.frac_for_test:
            assert self.indices_test is not None
            set_test = set(self.indices_test)
            assert set_test & set_train == set()
            assert set_test & set_val == set()
            t += len(self.indices_test)
            s |= set_test
        # The data split must must cover the whole dataset
        assert t == len(self.targets), f"{t = } != {len(self.targets) = }"
        assert s == set_all

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            assert self.indices_train is not None
            assert self.indices_val is not None
            self.ds_train = self.make_dataset(indices=self.indices_train)
            self.ds_val = self.make_dataset(indices=self.indices_val)
        elif stage == "test":
            assert self.indices_test is not None
            self.ds_test = self.make_dataset(indices=self.indices_test)
        else:
            raise NotImplementedError(f"{stage = }")

    def train_dataloader(self):
        assert self.ds_train is not None
        return self.make_dataloader(self.ds_train)

    def val_dataloader(self):
        assert self.ds_val is not None
        return self.make_dataloader(self.ds_val)

    def test_dataloader(self):
        assert self.ds_test is not None
        return self.make_dataloader(self.ds_test)


def list_fold_checkpoints(dp_train: Path, version: str, task: str):
    fps = []
    dp = dp_train / f"starr_{task}" / version
    for dp in dp.glob(r"fold_*"):
        dp /= "epoch_checkpoints"
        fps += list(dp.glob(rf"{task}*.pt"))
    return fps


def checkpoint_fps_to_df(fps):
    data = []
    for fp in fps:
        fn = fp.name
        fold = int(fp.parent.parent.name.split("fold_")[1].split("_")[0])
        # Extract val_loss and epoch from filename (format:
        # {task}_ep{epoch}_vloss{val_loss}_tloss{train_loss}.pt)
        epoch_str = fn.split("ep")[1].split("_")[0]
        epoch = int(epoch_str)
        val_loss_str = fn.split("vloss")[1].split("_")[0]
        val_loss = float(val_loss_str) / 1000  # Convert from integer representation
        train_loss_str = fn.split("tloss")[1].split(".")[0]
        train_loss = float(train_loss_str) / 1000  # Convert from integer representation
        data.append(
            {
                "fp": fp,
                "fn": fn,
                "fold": fold,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "epoch": epoch,
            }
        )
    df = pd.DataFrame(data)
    df.sort_values(by=["fold", "epoch"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def pick_checkpoint(df, fold: int, tolerance: float = 0.10, ax=None):
    df = df[df.fold == fold]
    df.set_index("epoch", inplace=True)
    df.sort_index(inplace=True)
    min_, max_ = df.mse.min(), df.mse.max()
    if ax:
        sns.lineplot(data=df, x="epoch", y="mse", hue="set_name", marker="o", ax=ax)

    df = df.pivot(columns="set_name", values="mse")
    df["diff_train_val"] = (df.train - df.val).abs()
    df["stop"] = (df.train < df.val) & (df.diff_train_val <= tolerance * df.val)
    df = df[df.stop]
    df.sort_values(by=["val", "diff_train_val"], inplace=True)
    epoch = df.index.values[0]
    if ax:
        ax.vlines(epoch, min_, max_, color="red", linestyle="--")
    return int(epoch)


def pick_best_checkpoint(dp_train: Path, version: str, task: str, fold: int):
    fp = dp_train / f"starr_{task}" / version / "stats.pkl.bz2"
    df_models = pd.read_pickle(fp)
    epoch = pick_checkpoint(df_models, fold=fold)
    dp_checkpoints = (
        dp_train / f"starr_{task}" / version / f"fold_{fold}" / "epoch_checkpoints"
    )
    fp_model_checkpoint = list(dp_checkpoints.glob(f"{task}_ep{epoch:02d}*.pt"))[0]
    return fp_model_checkpoint
