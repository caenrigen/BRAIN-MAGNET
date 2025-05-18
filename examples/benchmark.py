# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
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
from random import randbytes

# %%
# # cd ./sshpyk_code/examples/

# %%
random_state = 913
# random.seed(random_state)

dbm = Path("/Volumes/Famafia/brain-magnet/")
# dbm = Path("../../sshpyk_data/")
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
from importlib import reload

import utils as ut
import cnn_starr as cnn
import data_module as dm
import plot_utils as put

reload(ut)
reload(put)
reload(cnn)
reload(dm)

n_folds = 5

# %% [markdown]
# # Eval models
#

# %%
task, version = ("ESC", "cc0e922b")

# %%
y_col = f"{task}_log2_enrichment"
df_enrichment = dm.load_enrichment_data(
    fp=dbmrd / "Enhancer_activity_w_seq.csv.gz",
    y_col=y_col,
)

# %%
fps = dm.list_fold_checkpoints(dp_train=dbmt, version=version, task=task)
dfc = dm.checkpoint_fps_to_df(fps)
rows = []
for fold, df in tqdm(dfc.groupby("fold"), total=dfc.fold.nunique(), desc="folds"):
    assert isinstance(fold, int), fold
    for epoch in tqdm(df.epoch, leave=False, desc="epoch"):
        fp = df[df.epoch == epoch].fp.iloc[0]
        model = cnn.load_model(fp, forward_mode="both", device=device)
        data_loader = dm.DMSTARR(
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
            dl = dm.make_dl(df_for_loader, y_col=y_col)
            targets, preds = cnn.eval_model(model, dl, device=device)
            mse, pearson, spearman = cnn.model_stats(targets, preds)
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
    # Save it on each fold so that we can have a look at it quicker while the rest of
    # the folds it being evaluated.
    df_stats = pd.DataFrame(rows)
    df_stats.to_pickle(dbmt / f"starr_{task}" / version / "stats.pkl.bz2")

# %% [markdown]
# # Benchmark results
#

# %%
print(task, version)
fp = dbmt / f"starr_{task}" / version / "stats.pkl.bz2"
print(fp, fp.exists())

# %%
df_stats = pd.read_pickle(fp)
df_stats

# %% [markdown]
# ## Criteria for stopping the training and picking checkpoint
#

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
#

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
#

# %%
max_diff = df.mse.max() - df.mse.min()
print(f"{round(max_diff / df.mse.mean() * 100, 2)}%")

# %%
fps = list_fold_checkpoints(dp_train=dbmt, version=version, task=task)
dfc = checkpoint_fps_to_df(fps)
# Does not matter much which one we pick
idx = 0
fold_best, epoch_best = df_best.fold.iloc[idx], df_best.epoch.iloc[idx]
fp_best = dfc[(dfc.epoch == epoch_best) & (dfc.fold == fold_best)].fp.iloc[0]
fp_best

# %% [markdown]
# # Forward vs reverse complement
#

# %%
df_enr = df_enrichment.drop(OUTLIER_INDICES)
df = df_enr
dl = make_dl(df, y_col=y_col, batch_size=256, shuffle=False)

for forward_mode in tqdm(["main", "revcomp"]):
    model = load_model(fp_best, forward_mode=forward_mode, device=device)
    targets, preds = eval_model(model, dl, device=device)
    df[f"preds_{forward_mode}"] = preds

# %% [markdown]
# There can be "outliers" when giving the model only the "main" or the "revcomp" sequence.
#
# It seems it is better to average the two values.
#

# %%
fig, ax = plt.subplots(1, 1)
x = df.preds_main.values
y = df.preds_revcomp.values
density_scatter(x, y, ax=ax, fig=fig, cmap="magma")
model_stats(x, y)


# %% [markdown]
# # Scratch
#

# %%
t = df_sample.SeqEnc.iloc[0]
t

# %%
x = np.stack([t])
# convert input: [batch, seq_len, 4] -> [batch, 4, 1, seq_len]
tensor_x = torch.Tensor(x).permute(0, 2, 1).unsqueeze(2)
seq_len = sum(tensor_x[:, i, :, :] for i in range(4)).sum()
seq_len

# %% [markdown]
# # Residuals vs SeqLen
#

# %%
df = df_enr
df["SeqLen"] = df.Seq.map(len)

# %%
df.SeqLen.hist()

# %%
df = df_enr
df = df[(df.SeqLen != 500) & (df.SeqLen != 1000)]
x = df.SeqLen.values
y = (df.ESC_log2_enrichment - df.preds_scaled_both).values
fig, ax = plt.subplots(1, 1)
density_scatter(x, y, fig=fig, ax=ax)

# %% [markdown]
# # GC content
#

# %%
df_enrichment["SeqLen"] = df_enrichment.Seq.map(len)
func = df_enrichment.Seq.str.count
df_enrichment["GC"] = (func("G") + func("C")) / df_enrichment.SeqLen
df_enrichment.head()

# %%
dl = make_dl(df_enrichment, y_col=y_col, batch_size=256, shuffle=False)

# %%
fps = list_fold_checkpoints(dp_train=dbmt, version=version, task=task)
dfc = checkpoint_fps_to_df(fps)
for fold, epoch in tqdm(zip(df_best.fold, df_best.epoch), total=len(df_best)):
    df = dfc
    df = df[df.fold == fold]
    fp = df[df.epoch == epoch].fp.iloc[0]
    print(fp)
    model = load_model(fp, forward_mode="both", device=device)
    targets, preds = eval_model(model, dl, device=device)
    preds_scaled = (preds - preds.mean()) / preds.std() * targets.std() + targets.mean()
    df_enrichment[f"preds_{fold}_{epoch}"] = preds
    df_enrichment[f"preds_scaled_{fold}_{epoch}"] = preds_scaled

# %%
df_enrichment.head()

# %% [markdown]
# Little GC-content does seem to affect the results a bit.
#
# Lower GC content seems to correlate with higher relative residuals.
#
# But this might be an indication that the model actually learnt the patterns and not the
# noise of the data.
#

# %%
df = df_enrichment
preds = df["preds_scaled_0_5"].values
targets = df[y_col].values
gc_content = df.GC.values
# plt.scatter(targets, preds)
residuals = (preds - targets) / (targets + 10)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
density_scatter(x=gc_content, y=residuals, ax=ax, fig=fig, cmap="plasma")

# %% [markdown]
# # Check performance per category
#

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
