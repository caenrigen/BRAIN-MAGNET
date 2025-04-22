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


def bins(s, n: int = 8):
    return pd.cut(s, bins=n, labels=False)


def bins_log2(s, n: int = 8):
    # Can't do less bins AND have enough elements in each bin
    # There are several "outliers" in the NCREs activity,
    # take another log2 to compact the data further for assigning bins
    return bins(np.log2(s + 1), n=n)


def make_tensor_dataset(df: pd.DataFrame, x_col: str, y_col: str):
    x = np.stack(df[x_col].values)
    # convert input: [batch, seq_len, 4] -> [batch, 4, 1, seq_len]
    tensor_x = torch.Tensor(x).permute(0, 2, 1).unsqueeze(2)
    # add one dimension in targets: [batch] -> [batch, 1]
    tensor_y = torch.Tensor(df[y_col].values).unsqueeze(1)
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
    df["SeqEnc"] = df.Seq.map(one_hot_encode).map(pad_arr)
    if drop_indices:
        df = df.drop(drop_indices)
    return df


def make_dl(df: pd.DataFrame, y_col: str, batch_size: int = 256, shuffle: bool = False):
    return DataLoader(
        make_tensor_dataset(df=df, x_col="SeqEnc", y_col=y_col),
        batch_size=batch_size,
        shuffle=shuffle,
    )


class DMSTARR(L.LightningDataModule):
    def __init__(
        self,
        df_enrichment: pd.DataFrame,
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
            self.df_test = augment_data(self.df_test, random_state=self.random_state)

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
            self.df_train = augment_data(self.df_train, random_state=self.random_state)
            if callable(self.augment):
                # Augment the validation set as well, otherwise the validation loss is
                # not comparable to the training loss when the augment changes the
                # targets distribution.
                self.df_val = augment_data(self.df_val, random_state=self.random_state)

    def setup(self, stage: Optional[str] = None):
        func = partial(make_tensor_dataset, x_col="SeqEnc", y_col=self.y_col)
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
