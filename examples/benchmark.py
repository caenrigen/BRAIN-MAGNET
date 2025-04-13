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
#     display_name: Python 3.13 (RMBP+SSHFS)
#     language: python
#     name: ssh_mbp_ext
# ---

# %%
# cd ./sshpyk_code/examples/

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
def load_model(fp: Path, device=device, forward_mode="main"):
    model = CNNSTARR(forward_mode=forward_mode, log_vars_prefix="")
    model.to(device)
    model.load_state_dict(torch.load(fp))
    model.to(device)
    return model


# %%
from sklearn.metrics import mean_squared_error
from scipy import stats


def eval_model(model: CNNSTARR, dataloader: DataLoader):
    inputs = []
    preds = []
    targets = []

    model.eval()
    with torch.no_grad():
        for _batch, data in enumerate(dataloader):
            inputs_, targets_ = data
            inputs_ = inputs_.to(device)
            targets_ = targets_.to(device)
            outputs = model(inputs_)

            inputs.append(inputs_)
            targets.append(targets_)
            preds.append(outputs)

    inputs = torch.cat(inputs, dim=0).squeeze().cpu().numpy()
    targets = torch.cat(targets, dim=0).squeeze().cpu().numpy()
    preds = torch.cat(preds, dim=0).squeeze().cpu().numpy()
    return targets, preds, inputs


def model_stats(targets, preds):
    mse = mean_squared_error(targets, preds)
    pearon = float(stats.pearsonr(targets, preds).statistic)
    spearman = float(stats.spearmanr(targets, preds).statistic)
    return mse, pearon, spearman



# %% [markdown]
# # Eval models

# %%
tasks = [
    ("ESC", "f219f565"),
    ("NSC", "d0fe3e2e"),
]


# %%
for task, version in tqdm(tasks, desc="tasks"):
    fps = list_fold_checkpoints(version, task=task)
    dfc = checkpoint_fps_to_df(fps)

    df_enrichment = load_enrichment_data(
        fp=dbmrd / "Enhancer_activity_w_seq.csv.gz", y_col=f"{task}_log2_enrichment"
    )
    rows = []
    for fold, df in tqdm(dfc.groupby("fold"), total=dfc.fold.nunique(), desc="folds"):
        for epoch in tqdm(df.epoch, leave=False, desc="epoch"):
            fp = df[df.epoch == epoch].fp.iloc[0]
            model = load_model(fp, forward_mode="both")
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
                dl = data_loader.make_dl(df_for_loader)
                for forward_mode in ["main", "revcomp", "both"]:
                    model.forward_mode = forward_mode
                    targets, preds, inputs = eval_model(model, dl)
                    mse, pearon, spearman = model_stats(targets, preds)
                    rows.append(
                        {
                            "fold": fold,
                            "epoch": epoch,
                            "set_name": name,
                            "mse": mse,
                            "pearon": pearon,
                            "spearman": spearman,
                            "inputs": inputs,
                            "targets": targets,
                            "preds": preds,
                            "forward_mode": forward_mode,
                        }
                    )
    df_stats = pd.DataFrame(rows)
    df_stats.to_pickle(dbmt / f"starr_{task}" / version / "stats.pkl.bz2")

# %% [markdown]
# # Benchmark results

# %%
task, version = tasks[0]
print(task, version)
fp = dbmt / f"starr_{task}" / version / "stats.pkl.bz2"
print(fp, fp.exists())

# %%
df_stats = pd.read_pickle(fp)
df_stats.head()

# %% [markdown]
# ## Inspect differcent in prediction for reversed complement

# %%
df = df_stats
df = df[df.fold == 0]
df = df[df.set_name == "train"]
df = df[df.epoch == 1]
df.head()
df.set_index("epoch", inplace=True)
df.sort_index(inplace=True)

preds_main = df[df.forward_mode == "main"].preds.values[0]
preds_revcomp = df[df.forward_mode == "revcomp"].preds.values[0]
preds_both = df[df.forward_mode == "both"].preds.values[0]

df


# %%
df = pd.DataFrame(
    {
        "main": pd.Series(preds_main),
        "revcomp": pd.Series(preds_revcomp),
        "both": pd.Series(preds_both),
    }
)
df["main"] = (df.main - df.both) / df.both * 100
df["revcomp"] = (df.revcomp - df.both) / df.both * 100
del df["both"]
sum(df.main > 10) / len(df.main)

# %%
plt.figure(figsize=(10, 6))
sns.histplot(data=df, bins=100)


# %% [markdown]
# ## Criteria for stopping the training and picking checkpoint

# %%
def pick_checkpoint(df, fold: int, tolerance: float = 0.05, plot: bool = True):
    df = df[df.fold == fold]
    df = df[df.forward_mode == "both"]
    df.set_index("epoch", inplace=True)
    df.sort_index(inplace=True)
    min_, max_ = df.mse.min(), df.mse.max()
    if plot:
        sns.lineplot(data=df, x="epoch", y="mse", hue="set_name", marker="o")

    df = df.pivot(columns="set_name", values="mse")
    df["diff_train_val"] = (df.train - df.val).abs()
    df["stop"] = (df.train < df.val) & (df.diff_train_val <= tolerance * df.val)
    df = df[df.stop]
    df.sort_values(by=["val", "diff_train_val"], inplace=True)
    epoch = df.index.values[0]
    if plot:
        plt.vlines(epoch, min_, max_, color="red", linestyle="--")
        plt.show()
    return epoch


epoch_per_fold = {}
for fold in range(n_folds):
    epoch = pick_checkpoint(df_stats, fold, plot=False)
    epoch_per_fold[fold] = epoch
epoch_per_fold

# %%
dfs = []
for fold in range(n_folds):  # TBD
    df = df_stats
    df = df[df.set_name == "test"]
    df = df[df.forward_mode == "both"]
    df = df[df.fold == fold]
    df = df[df.epoch == epoch_per_fold[fold]]
    dfs.append(df)

df = pd.concat(dfs)
df

# %%
[len(a) for a in df.preds.to_list()]

# %%
np.array(df.preds.to_list()).mean(axis=1)

# %%
df = df_stats
df = df[df.forward_mode == "both"]
df = df[df.fold == 0]
df = df[df.epoch == 2]
df

# %%
