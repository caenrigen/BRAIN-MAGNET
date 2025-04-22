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
task, version = ("ESC", "82c66f85")

# %%
y_col = f"{task}_log2_enrichment"
df_enrichment = load_enrichment_data(
    fp=dbmrd / "Enhancer_activity_w_seq.csv.gz",
    y_col=y_col,
    drop_indices=OUTLIER_INDICES,
)

# %%
fps = list_fold_checkpoints(dp_train=dbmt, version=version, task=task)
dfc = checkpoint_fps_to_df(fps)
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
fig, ax = plt.subplots(1, n_folds, figsize=(15, 3), sharex=True, sharey=True)
if n_folds == 1:
    ax = [ax]
for fold in range(n_folds):
    epoch = pick_checkpoint(df_stats, fold, ax=ax[fold])
    epoch_per_fold[fold] = epoch
fig.tight_layout()

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
df_best = df.copy()
df

# %% [markdown]
# It does not seem worth to use an ensable of models.
# The deviations between folds is pretty small:

# %%
print(f"{round(df.mse.std() / df.mse.mean() * 100, 2)}%")

# %% [markdown]
# # Check ensemble performance

# %%
display(df_enrichment.head())
dl = make_dl(df_enrichment, y_col=y_col, batch_size=256, shuffle=False)

# %%
fps = list_fold_checkpoints(dp_train=dbmt, version=version, task=task)
dfc = checkpoint_fps_to_df(fps)
preds_all = []
for fold, epoch in tqdm(zip(df_best.fold, df_best.epoch), total=len(df_best)):
    df = dfc
    df = df[df.fold == fold]
    fp = df[df.epoch == epoch].fp.iloc[0]
    print(fp)
    model = load_model(fp, forward_mode="both", device=device)
    targets, preds = eval_model(model, dl, device=device)
    preds_all.append(preds)

# %%
preds_ensemble = np.mean(preds_all, axis=0)
preds_ensemble

# %% [markdown]
# It improves but that is expected since averaging over all kfolds allows to learn from **all** data.

# %%
model_stats(targets, preds_ensemble)

# %% [markdown]
# # Check performance per category

# %%
fps = list_fold_checkpoints(dp_train=dbmt, version=version, task=task)
dfc = checkpoint_fps_to_df(fps)
df = dfc
fold = 0
df = df[df.fold == fold]
epoch = 1
fp = df[df.epoch == epoch].fp.iloc[0]
print(fp)

# %%
model = load_model(fp, forward_mode="both", device=device)

# %%
# y_col = f"{task}_log2_enrichment"
# df_enrichment = load_enrichment_data(
#     fp=dbmrd / "Enhancer_activity_w_seq.csv.gz", y_col=y_col
# )

# %%
targets, preds = eval_model(model, dl, device=device)

# %%
model_stats(targets, preds)

# %%
m_outliers = (targets < 0.2) & (preds > 2)
model_stats(targets[~m_outliers], preds[~m_outliers])

# %%
preds_fixed = (preds - preds.mean()) / preds.std() * targets.std() + targets.mean()
model_stats(targets, preds_fixed)

# %%
sns.histplot(targets, log_scale=False, stat="probability")
sns.histplot(preds_fixed, log_scale=False, stat="probability")
sns.histplot(preds, log_scale=False, stat="probability")
plt.show()

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5), sharex=True, sharey=True)
density_scatter(x=targets, y=preds, ax=ax1, fig=fig, cmap="plasma_r")
# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
density_scatter(x=targets, y=preds_fixed, ax=ax2, fig=fig, cmap="plasma_r")
for ax in [ax1, ax2]:
    ax.set_ylim(-0.5, 6)
    ax.set_xlim(-0.5, 6)
    ax.set_aspect("equal")
# ax1.scatter(targets[m_outliers], preds[m_outliers])
fig.tight_layout()


# %%
df = df_enrichment
df["cat_enh"] = bins_log2(df[y_col], n=10)
display(df.head())
rows = []
for cat, df_ in tqdm(df.groupby("cat_enh"), total=df.cat_enh.nunique()):
    df_.sort_values(by=y_col, inplace=True)
    dl = make_dl(df_, y_col=y_col, batch_size=256, shuffle=False)
    targets, preds = eval_model(model, dl, device=device)
    preds_fixed = (preds - preds.mean()) / preds.std() * targets.std() + targets.mean()

    offset = 10  # To avoid small values exploding the fraction
    mae_rel = np.mean(np.abs(preds - targets) / (targets + offset))
    mae_rel_fixed = np.mean(np.abs(preds_fixed - targets) / (targets + offset))

    mse, pearson, spearman = model_stats(targets, preds)
    mse_fixed, pearson_fixed, spearman_fixed = model_stats(targets, preds_fixed)
    rows.append(
        {
            "cat": cat,
            "mse": mse,
            "mse_fixed": mse_fixed,
            "mae_rel": mae_rel,
            "mae_rel_fixed": mae_rel_fixed,
            "pearson": pearson,
            "pearson_fixed": pearson_fixed,
            "spearman": spearman,
            "spearman_fixed": spearman_fixed,
        }
    )
df_cat_stats = pd.DataFrame(rows)

# %%
df_cat_stats

# %%
df = df_cat_stats.copy()
df.set_index("cat", inplace=True)
(df.mae_rel * 100).plot(marker="o")
(df.mae_rel_fixed * 100).plot(marker="o")

# %%
