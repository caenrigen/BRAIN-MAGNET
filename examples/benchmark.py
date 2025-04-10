# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: g
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import math
import time
import random
from tqdm.auto import tqdm
from pathlib import Path
from typing import Optional
from functools import partial

# %%
import lightning as L
import torch
from torch import nn
import gc
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from random import randbytes

# %%
random_state = 913
# random.seed(random_state)

dbm = Path("/Volumes/Famafia/brain-magnet/")
dbmrd = dbm / "rd_APP_data"
dbmt = dbm / "train"

# %%
print(torch.cuda.is_available(), torch.backends.mps.is_available())
# device = torch.device("cuda")
# device = torch.device("cpu")
device = torch.device("mps")  # might have priblems for macOS <14.0
device

# %%
# Evaluate the python files within the notebook namespace
# %run -i auxiliar.py
# %run -i cnn_starr.py
# %run -i data_module.py
n_folds = 5


# %%
def list_fold_checkpoints(version: str, task: str):
    fps = []
    dp = dbmt / f"starr_{task}" / version
    for dp in dp.glob(r"fold_*"):
        dp /= "threshold_checkpoints"
        fps += list(dp.glob(rf"{task}*.pt"))
    return fps


def checkpoint_fps_to_df(fps):
    data = []
    for fp in fps:
        fn = fp.name
        fold = int(fp.parent.parent.name.split("fold_")[1].split("_")[0])
        # Extract val_loss and epoch from filename (format:
        # {task}_vloss{val_loss}_ep{epoch}.pt)
        val_loss_str = fn.split("vloss")[1].split("_")[0]
        epoch_str = fn.split("ep")[1].split(".")[0]
        val_loss = float(val_loss_str) / 1000  # Convert from integer representation
        epoch = int(epoch_str)
        data.append(
            {"fp": fp, "fn": fn, "fold": fold, "val_loss": val_loss, "epoch": epoch}
        )
    df = pd.DataFrame(data)
    df.sort_values(by=["fold", "epoch"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# %%
def load_model(fp: Path, device=device):
    model = CNNSTARR(revcomp=True, log_vars_prefix="")
    model.to(device)
    model.load_state_dict(torch.load(fp))
    model.to(device)
    return model


# %%
from sklearn.metrics import mean_squared_error
from scipy import stats


def eval_model(model: CNNSTARR, dataloader: DataLoader):
    preds = []
    targets = []

    model.eval()
    with torch.no_grad():
        for _batch, data in enumerate(dataloader):
            inputs, targets_ = data
            inputs = inputs.to(device)
            targets_ = targets_.to(device)
            outputs = model(inputs)

            targets.append(targets_)
            preds.append(outputs)

    targets = torch.cat(targets, dim=0).squeeze().cpu().numpy()
    preds = torch.cat(preds, dim=0).squeeze().cpu().numpy()
    return targets, preds


def model_stats(targets, preds):
    mse = mean_squared_error(targets, preds)
    pearon = float(stats.pearsonr(targets, preds).statistic)
    spearman = float(stats.spearmanr(targets, preds).statistic)
    return mse, pearon, spearman



# %%
task = "ESC"
for task, version in tqdm([("ESC", "f219f565"), ("NSC", "d0fe3e2e")]):
    fps = list_fold_checkpoints(version, task=task)
    dfc = checkpoint_fps_to_df(fps)

    df_enrichment = load_enrichment_data(
        fp=dbmrd / "Enhancer_activity_w_seq.csv.gz", y_col=f"{task}_log2_enrichment"
    )
    rows = []
    for fold, df in tqdm(dfc.groupby("fold"), total=dfc.fold.nunique()):
        for epoch in tqdm(df.epoch, leave=False):
            fp = df[df.epoch == epoch].fp.iloc[0]
            model = load_model(fp)
            data_loader = DMSTARR(
                df_enrichment=df_enrichment,
                sample=None,
                y_col=f"{task}_log2_enrichment",
                n_folds=n_folds,
                fold_idx=fold,
                augment=0,
            )
            data_loader.prepare_data()
            for name in ["train", "val", "test"]:
                df_for_loader = getattr(data_loader, f"df_{name}")
                targets, preds = eval_model(model, data_loader.make_dl(df_for_loader))
                mse, pearon, spearman = model_stats(targets, preds)
                rows.append(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "set_name": name,
                        "mse": mse,
                        "pearon": pearon,
                        "spearman": spearman,
                        "targets": targets,
                        "preds": preds,
                    }
                )
    df_stats = pd.DataFrame(rows)
    df_stats.to_pickle(dbmt / f"starr_{task}" / version / "stats.pkl.bz2", index=False)

# %%

# %%
