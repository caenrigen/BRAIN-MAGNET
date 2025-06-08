# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: g
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch
from itertools import combinations

# %%
# %load_ext autoreload
# %autoreload explicit
# %aimport utils, cnn_starr, data_module, plot_utils, notebook_helpers

import utils as ut
import data_module as dm
import plot_utils as put
import notebook_helpers as nh

# %%
dir_data = Path("./data")
dir_train = dir_data / "train"
assert dir_data.is_dir() and dir_train.is_dir()

fp_dataset = dir_data / "Enhancer_activity_with_str_sequences.csv.gz"

device = torch.device("mps")

# %% [markdown]
# # Pick best checkpoints
#

# %%
# k-folds models trained on 90% of the data, 10% for testing
task, version = ("ESC", "8a6b9616")

# %%
df_ckpts = dm.list_checkpoints(dp_train=dir_train, task=task, version=version)
df_ckpts.tail()

# %%
best_checkpoints, fig, axs = nh.pick_best_checkpoints(df_ckpts, plot=True)
best_checkpoints

# %% [markdown]
# # Evaluate k-folds on the test data
#

# %%
eval_results = []
for fold, fp in best_checkpoints.items():
    eval_result = nh.evaluate_model(
        fp_checkpoint=fp,
        fp_dataset=fp_dataset,
        device=device,
        dataloader="test_dataloader",
        augment_w_rev_comp=True,
    )
    eval_result["fold"] = fold
    eval_results.append(eval_result)

# %%
rows = [
    {k: v for k, v in eval_result.items() if isinstance(v, (int, float))}
    for eval_result in eval_results
]
df_stats = pd.DataFrame(rows).set_index("fold").round(3)
df_stats

# %%
s = {
    "mean": df_stats.mean(),
    "std": df_stats.std(),
    "std %": round(100 * (df_stats.std() / df_stats.mean()), 2),
    "max_diff %": round(100 * (df_stats.max() - df_stats.min()) / df_stats.mean(), 2),
}
pd.DataFrame(s).round(3)

# %% [markdown]
# ## Verify consistency between k-folds
#

# %%
rows = []

for i, j in combinations(range(len(eval_results)), 2):
    preds_i, preds_j = eval_results[i]["preds"], eval_results[j]["preds"]
    _, pearson, spearman = ut.model_stats(preds_i, preds_j)
    rows.append({"fold_a": i, "fold_b": j, "pearson": pearson, "spearman": spearman})
df_corr = pd.DataFrame(rows).set_index(["fold_a", "fold_b"]).unstack().round(2)

display(df_corr.pearson)
display(df_corr.spearman)

# %% [markdown]
# The models trained on different data folds seem very consistent between each other.
# All combinations are consistent within ~1% pearson/spearman.
#

# %% [markdown]
# # Predictions on full dataset
#

# %%
# Final model trained on 90% of the data, 10% for validation
task, version = ("ESC", "762acb33")
df_ckpts = dm.list_checkpoints(dp_train=dir_train, task=task, version=version)
best_checkpoints, fig, axs = nh.pick_best_checkpoints(df_ckpts, plot=True)
best_checkpoints[0]

# %%
fp = best_checkpoints[0]
eval_result = nh.evaluate_model(
    fp_checkpoint=fp,
    fp_dataset=fp_dataset,
    device=device,
    dataloader="full_dataloader",
    augment_w_rev_comp=True,
)

# %%
preds = eval_result["preds"]
targets = eval_result["targets"]

# Undo interleaving of forward and reverse complement indices
preds_fw, preds_rc = dm.interleave_undo(preds)
preds_av = (preds_fw + preds_rc) / 2

targets_fw, targets_rc = dm.interleave_undo(targets)
assert np.all(targets_fw == targets_rc)

preds, preds_fw, preds_rc

# %% [markdown]
# ## Forward vs reverse strands
#

# %%
_ = nh.plot_corr(
    x=preds_fw,
    y=preds_rc,
    title="Predictions forward vs reverse",
    min_=min(preds_fw.min(), preds_rc.min()),
    max_=max(preds_fw.max(), preds_rc.max()),
)

# %% [markdown]
# The prediction on the forward strand are fairly consitent with the predictions on the reverse strand.
#

# %% [markdown]
# ## Targets vs Predictions
#

# %%
combs = [
    (targets, preds, "Targets vs Predictions (both)"),
    (targets_fw, preds_av, "Targets vs Pred. mean(forward, reverse)"),
    (targets_fw, preds_fw, "Targets vs Pred. forward"),
    (targets_fw, preds_rc, "Targets vs Pred. reverse"),
]

fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
for (x, y, title), ax in zip(combs, axs.flatten()):
    nh.plot_corr(x=x, y=y, title=title, ax=ax, min_=targets.min(), max_=targets.max())
fig.tight_layout()

# %% [markdown]
# ## Distribution of predictions vs targets
#

# %%
round((targets_fw.mean() - preds_av.mean()) / targets_fw.mean(), 4)

# %%
num_bins = 200
_, bins = np.histogram(targets_fw, bins=num_bins)
sns.histplot(targets_fw, bins=bins, stat="probability")
fig, ax = plt.gcf(), plt.gca()
plt.vlines(targets_fw.mean(), 0, ax.get_ylim()[1], color="C0", linestyle="--")
sns.histplot(preds_av, bins=bins, stat="probability")
plt.vlines(preds_av.mean(), 0, ax.get_ylim()[1], color="C1", linestyle="--")
# ax.set_xlim(0, 4)
fig.set_figwidth(15)


# %% [markdown]
# The centers of the distributions are very close which is a good sign.
#
# The std of the predicitons is smaller. That is expected from a statistical model.
#

# %% [markdown]
# ## Residuals vs Sequence Length
#

# %%
usecols = list(set(dm.DATASET_COLS) - {"Seq"})  # type: ignore
df_enrichment = pd.read_csv(fp_dataset, usecols=usecols)
df_enrichment.head()

# %%
# Verify that the targets are the same
assert np.allclose(df_enrichment.ESC_log2_enrichment.to_numpy(), targets_fw)

# %%
df_enrichment.Seq_len.hist()
_ = plt.gca().set_xlabel("Sequence length")
_ = plt.gca().set_ylabel("Sequence count")

# %%
df = df_enrichment
x = df.Seq_len.to_numpy()
y = targets_fw - preds_av  # residuals
m = x != 1000
for y, x in ((y, x), (y[m], x[m])):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    _ = put.density_scatter(x, y, ax=ax1)
    sns.histplot(y=y, bins=100, ax=ax2)
    fig.tight_layout()

# %% [markdown]
# The residuals increase a bit for shorter sequences, but does not seem extreme.
#

# %% [markdown]
# ## GC content
#

# %%
df = df_enrichment
x = (df.GC_counts / df.Seq_len).to_numpy()
y = targets_fw - preds_av  # residuals
fig, ax = plt.subplots(1, 1)
_ = put.density_scatter(x, y, ax=ax)
ax.set_xlabel("GC content")
ax.set_ylabel("Residuals")
fig.tight_layout()


# %% [markdown]
# Lower GC content is correlated with higher residuals.
#
# This might be an indication that the model actually learnt the patterns and not the
# noise of the data.
#
