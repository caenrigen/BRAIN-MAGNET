import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Union
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


def load_enrichment_data(
    fp: Path,
    y_col: str = "NSC_log2_enrichment",
    drop_indices: Optional[List[int]] = None,
    pad_to: int = 1000,
):
    usecols = [
        # "Chr",
        # "Start",
        # "End",
        # "NSC_log2_enrichment",
        # "ESC_log2_enrichment",
        y_col,
        "Seq",
        # "SeqRevComp",
    ]
    df = pd.read_csv(fp, usecols=usecols)
    df["SeqLen"] = df.Seq.str.len()
    df["SeqEnc"] = df.Seq.map(ut.one_hot_encode).map(partial(ut.pad_one_hot, to=pad_to))
    if drop_indices:
        df.drop(drop_indices, inplace=True)
    return df


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


class DMSTARR(L.LightningDataModule):
    def __init__(
        self,
        df_enrichment: pd.DataFrame,
        device: torch.device,
        y_col: str = "NSC_log2_enrichment",
        sample: Optional[int] = None,
        batch_size: int = 256,
        random_state: int = 913,
        n_folds: Optional[int] = None,  # Number of folds for cross-validation
        fold_idx: int = 0,  # Current fold index (0 to n_folds-1)
        augment: Optional[Union[Callable, int]] = None,
        frac_for_test: float = 0.1,
        frac_for_val: float = 0.1,
        bins_func: Callable = bins_log2,
    ):
        super().__init__()
        self.df_enrichment = df_enrichment
        self.y_col = y_col
        self.sample = sample  # for quick code tests with small data sample
        self.df = self.df_train = self.df_val = self.df_test = None
        self.ds_train = self.ds_val = self.ds_test = None
        self.batch_size = batch_size
        self.random_state = random_state
        if n_folds is not None and fold_idx >= n_folds:
            raise ValueError(f"{fold_idx = } must be less than {n_folds = }")
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.augment = augment
        self.frac_for_test = frac_for_test
        self.frac_for_val = frac_for_val
        assert y_col in df_enrichment.columns
        self.bins_func = bins_func
        self.device = device

    def prepare_data(self):
        if self.augment:
            if isinstance(self.augment, int):
                self.df_enrichment["augment"] = self.augment
            else:
                self.df_enrichment["augment"] = self.augment(
                    self.df_enrichment[self.y_col]
                )

        if self.sample:
            _, self.df = train_test_split(
                self.df_enrichment,
                test_size=self.sample,
                random_state=self.random_state,
                stratify=self.bins_func(self.df_enrichment[self.y_col]),
            )
        else:
            self.df = self.df_enrichment

        if self.frac_for_test:
            self.df, self.df_test = train_test_split(
                self.df,
                test_size=self.frac_for_test,
                random_state=self.random_state,
                stratify=self.bins_func(self.df[self.y_col]),
            )
        if self.augment and self.df_test:
            self.df_test = ut.augment_data(self.df_test, random_state=self.random_state)

        if self.n_folds is not None:
            kf = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )
            for i, (train_idxs, val_idxs) in enumerate(
                kf.split(self.df, self.bins_func(self.df[self.y_col]))
            ):
                if i == self.fold_idx:
                    assert len(val_idxs) < len(train_idxs)
                    self.df_val = self.df.iloc[val_idxs]
                    self.df_train = self.df.iloc[train_idxs]
                    break
        else:
            self.df_train, self.df_val = train_test_split(
                self.df,
                test_size=self.frac_for_val,
                random_state=self.random_state,
                stratify=self.bins_func(self.df[self.y_col]),
            )

        if self.augment:
            self.df_train = ut.augment_data(
                self.df_train, random_state=self.random_state
            )
            if callable(self.augment):
                # Augment the validation set as well, otherwise the validation loss is
                # not comparable to the training loss when the augment changes the
                # targets distribution.
                self.df_val = ut.augment_data(
                    self.df_val, random_state=self.random_state
                )

    def setup(self, stage: Optional[str] = None):
        func = partial(
            make_tensor_dataset, x_col="SeqEnc", y_col=self.y_col, device=self.device
        )
        if stage == "fit":
            self.ds_train = func(df=self.df_train)
            self.ds_val = func(df=self.df_val)
        elif stage == "test":
            self.ds_test = func(df=self.df_test)
        else:
            raise NotImplementedError(f"{stage = }")

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False)


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
