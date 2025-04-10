import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from functools import partial
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def bins(s):
    # Can't do less bins AND have enough elements in each bin
    # There are several "outliers" in the NCREs activity,
    # take another log2 to compact the data further for assigning bins
    return pd.cut(np.log2(s + 1), bins=8, labels=False)


def make_tensor_dataset(df: pd.DataFrame, x_col: str, y_col: str):
    x = np.stack(df[x_col].values)
    # convert input: [batch, seq_len, 4] -> [batch, 4, 1, seq_len]
    tensor_x = torch.Tensor(x).permute(0, 2, 1).unsqueeze(2)
    # add one dimension in targets: [batch] -> [batch, 1]
    tensor_y = torch.Tensor(df[y_col].values).unsqueeze(1)
    return TensorDataset(tensor_x, tensor_y)


class DMSTARR(L.LightningDataModule):
    def __init__(
        self,
        fp: Path,
        y_col: str = "NSC_log2_enrichment",
        sample: Optional[int] = None,
        batch_size: int = 256,
        random_state: int = 913,
        n_folds: Optional[int] = None,  # Number of folds for cross-validation
        fold_idx: int = 0,  # Current fold index (0 to n_folds-1)
        augment: int = 0,
    ):
        super().__init__()
        self.fp = fp
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

    def prepare_data(self):
        usecols = [
            # "Chr",
            # "Start",
            # "End",
            # "NSC_log2_enrichment",
            # "ESC_log2_enrichment",
            self.y_col,
            "Seq",
            # "SeqRevComp",
        ]
        df = pd.read_csv(self.fp, usecols=usecols)
        df["SeqEnc"] = df.Seq.map(one_hot_encode).map(pad_arr)

        if self.sample:
            _, df = train_test_split(
                df,
                test_size=self.sample,
                random_state=self.random_state,
                stratify=bins(df[self.y_col]),
            )

        self.df = df

        if self.n_folds is None:
            df, self.df_test = train_test_split(
                df,
                test_size=0.10,
                random_state=self.random_state,
                stratify=bins(df[self.y_col]),
            )
        else:
            kf = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )
            for i, (df_idxs, test_idxs) in enumerate(
                kf.split(df, bins(df[self.y_col]))
            ):
                if i == self.fold_idx:
                    assert len(test_idxs) < len(df_idxs)
                    self.df_test = df.iloc[test_idxs]
                    df = df.iloc[df_idxs]
                    break

        self.df_train, self.df_val = train_test_split(
            df,
            test_size=0.10,
            random_state=self.random_state,
            stratify=bins(df[self.y_col]),
        )
        if self.augment:
            self.df_train = augment_data(
                df_train=self.df_train,
                num_shits=self.augment,
                random_state=self.random_state,
            )

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
