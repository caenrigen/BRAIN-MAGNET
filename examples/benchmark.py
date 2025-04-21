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
# cd ./sshpyk_code/examples/

# %%
random_state = 913
# random.seed(random_state)

# dbm = Path("/Volumes/Famafia/brain-magnet/")
dbm = Path("../../sshpyk_data/")
dbmrd = dbm / "rd_APP_data"
dbmt = dbm / "train"
dbm.is_dir()

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

# %% [markdown]
# # Eval models

# %%
tasks = [
    ("ESC", "f952c4c5"),
    # ("NSC", ""),
]

# %%
for task, version in tqdm(tasks, desc="tasks"):
    fps = list_fold_checkpoints(dp_train=dbmt, version=version, task=task)
    dfc = checkpoint_fps_to_df(fps)
    y_col = f"{task}_log2_enrichment"
    df_enrichment = load_enrichment_data(
        fp=dbmrd / "Enhancer_activity_w_seq.csv.gz",
        y_col=y_col,
        log2log2norm=True,
    )
    y_col = f"{task}_log2log2norm_enrichment"
    rows = []
    for fold, df in tqdm(dfc.groupby("fold"), total=dfc.fold.nunique(), desc="folds"):
        for epoch in tqdm(df.epoch, leave=False, desc="epoch"):
            fp = df[df.epoch == epoch].fp.iloc[0]
            model = load_model(fp, forward_mode="both", device=device)
            data_loader = DMSTARR(
                df_enrichment=df_enrichment,
                sample=None,
                y_col=y_col,
                n_folds=n_folds,
                fold_idx=fold,
                augment=0,
                frac_for_test=0,
                frac_for_val=0.05,
                bins_func=bins,
            )
            data_loader.prepare_data()
            for name in ["train", "val"]:
                df_for_loader = getattr(data_loader, f"df_{name}")
                dl = make_dl(df_for_loader, y_col=y_col)
                targets, preds = eval_model(model, dl, device=device)
                mse, pearson, spearman = model_stats(targets, preds)
                rows.append(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "set_name": name,
                        "mse": mse,
                        "pearson": pearson,
                        "spearman": spearman,
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
df_stats

# %% [markdown]
# ## Criteria for stopping the training and picking checkpoint

# %%
epoch_per_fold = {}
n_folds = 5
for fold in range(n_folds):
    epoch = pick_checkpoint(df_stats, fold, plot=True)
    epoch_per_fold[fold] = epoch
epoch_per_fold

# %% [markdown]
# ## Check if there are significant differences between folds

# %%
dfs = []
for fold in range(n_folds):  # TBD
    df = df_stats
    df = df[df.set_name == "val"]
    df = df[df.fold == fold]
    df = df[df.epoch == epoch_per_fold[fold]]
    dfs.append(df)

df = pd.concat(dfs)
df

# %% [markdown]
# It is not worth to use an ensable of models!
#
# The deviations between folds is pretty small:

# %%
print(f"{round(df.mse.std() / df.mse.mean() * 100, 2)}%")

# %% [markdown]
# # Check performance per category

# %%
fps = list_fold_checkpoints(dp_train=dbmt, version=version, task=task)
dfc = checkpoint_fps_to_df(fps)
df = dfc
fold = 0
df = df[df.fold == fold]
# display(df)
epoch = 2
fp = df[df.epoch == epoch].fp.iloc[0]
print(fp)

# %%
model = load_model(fp, forward_mode="both", device=device)

# %%
y_col = f"{task}_log2_enrichment"
df_enrichment = load_enrichment_data(
    fp=dbmrd / "Enhancer_activity_w_seq.csv.gz", y_col=y_col, log2log2norm=True
)
display(df_enrichment.head())
y_col = f"{task}_log2log2norm_enrichment"
dl = make_dl(df_enrichment, y_col=y_col, batch_size=256, shuffle=False)

# %%
targets, preds = eval_model(model, dl, device=device)

# %%
mse, pearson, spearman = model_stats(targets, preds)
mse, pearson, spearman

# %%
sns.histplot(targets, bins=100, log_scale=False)
preds_fixed = preds.copy()
preds_fixed /= preds_fixed.std()
preds_fixed -= preds_fixed.mean()
sns.histplot(preds_fixed, bins=100, log_scale=False)
# sns.histplot(preds, bins=100, log_scale=False)
plt.show()
targets.mean(), targets.std(), preds_fixed.mean(), preds_fixed.std()

# %%
y_col = "ESC_log2_enrichment"
std, mean = df_enrichment[y_col].std(), df_enrichment[y_col].mean()
targets_shifted_back = 2 ** (targets * std + mean) - 1
preds_shifted_back = 2 ** (preds * std + mean) - 1
preds_fixed_shifted_back = 2 ** (preds_fixed * std + mean) - 1
print(model_stats(targets_shifted_back, preds_shifted_back))
print(model_stats(targets_shifted_back, preds_fixed_shifted_back))

# %%
targets_shifted_back.min()


# %%
sns.histplot(targets_shifted_back, bins=100, log_scale=False, stat="percent")
sns.histplot(preds_fixed_shifted_back, bins=100, log_scale=False, stat="percent")
sns.histplot(preds_shifted_back, bins=100, log_scale=False, stat="percent")

# %%
df_enrichment[y_col].describe()

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn


def density_scatter(x, y, ax, fig, sort=True, bins=100, **kwargs):
    """Scatter plot colored by 2d histogram"""
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
    )

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)

    # norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    # cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    # cbar.ax.set_ylabel("Density")

    return ax



# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
density_scatter(
    x=targets_shifted_back, y=preds_shifted_back, ax=ax1, fig=fig, cmap="plasma_r"
)
# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
density_scatter(
    x=targets_shifted_back,
    y=preds_fixed_shifted_back,
    ax=ax2,
    fig=fig,
    cmap="plasma_r",
)
ax1.set_ylim(-1, 15)
ax2.set_ylim(-1, 15)
ax1.set_aspect("equal")
ax2.set_aspect("equal")


# %%
df = df_enrichment
df["cat_enh"] = bins(df[y_col], n=10)
display(df.head())
rows = []
for cat, df_ in tqdm(df.groupby("cat_enh"), total=df.cat_enh.nunique()):
    df_.sort_values(by=y_col, inplace=True)
    dl = make_dl(df_, y_col=y_col, batch_size=256, shuffle=False)
    targets, preds = eval_model(model, dl, device=device)
    offset = 10
    mse_rel = 1000 * np.mean((preds - targets) ** 2 / (targets + offset) ** 2)
    mse, pearson, spearman = model_stats(targets, preds)
    rows.append(
        {
            "cat": cat,
            "mse": mse,
            "mse_rel": mse_rel,
            "pearson": pearson,
            "spearman": spearman,
        }
    )
df_cat_stats = pd.DataFrame(rows)

# %%
df_cat_stats

# %%
df = df_cat_stats.copy()
df.set_index("cat", inplace=True)
df.mse_rel.plot(marker="o")
# df.mse_rel_log.plot(marker="o")

# %%
sns.histplot(df_enrichment, x=y_col, bins=100, log_scale=False)
plt.show()
# sns.histplot(df_enrichment, x=y_col, bins=100, log_scale=True)

# %%
