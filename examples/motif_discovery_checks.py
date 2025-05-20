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
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# %%
from importlib import reload
import utils as ut
import cnn_starr as cnn
import data_module as dm
import plot_utils as put
import motif_discovery as md
from tqdm.auto import tqdm

_ = reload(ut)
_ = reload(cnn)
_ = reload(dm)
_ = reload(put)
_ = reload(md)

# %%
random_state = 913
dbmc = Path("/Users/victor/Documents/ProjectsDev/genomics/BRAIN-MAGNET")
dbm = Path("/Volumes/Famafia/brain-magnet")
dbmce = str(dbmc / "examples")
dbmrd = dbm / "rd_APP_data"
dbmt = dbm / "train"
os.chdir(dbmce)

# %%
print(torch.cuda.is_available(), torch.backends.mps.is_available())
device = torch.device("mps")  # cpu/mps/cuda
device

# %% [raw]
# task = "ESC"
# df = dm.load_enrichment_data(
#     fp=dbmrd / "Enhancer_activity_w_seq.csv.gz",
#     y_col=f"{task}_log2_enrichment",
# )
# q = df.ESC_log2_enrichment.sort_values(ascending=True).quantile(0.95)
# df_sample = df[df.ESC_log2_enrichment > q]
# print(df_sample.shape)
# fp = dbmrd / "Enhancer_activity_w_seq_top_5_percent.csv.gz"
# df_sample.to_csv(fp, index=False)
# df_sample

# %%
task = "ESC"
y_col = f"{task}_log2_enrichment"
df_sample = dm.load_enrichment_data(
    fp=dbmrd / "Enhancer_activity_w_seq_top_ESC_5_percent.csv.gz",
    y_col=y_col,
)
df_sample.sort_values(by=y_col, ascending=False, inplace=True)
df_sample_mini = df_sample[:10].copy()

# %%
dataset = dm.make_tensor_dataset(
    df=df_sample_mini,
    x_col="SeqEnc",
    y_col=y_col,
    device=device,
)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
dataloader


# %%
def load_model(
    version: str,
    fold: int,
    device: torch.device = device,
):
    fp_model_checkpoint = dm.pick_best_checkpoint(
        dbmt, version=version, task=task, fold=fold
    )
    return cnn.load_model(
        fp=fp_model_checkpoint,
        device=device,
        forward_mode="main",
    )


# %% [markdown]
# # Impact of averaging with rev. compl. and the number of shuffles
#

# %%
version, fold = "cc0e922b", 0
model_trained = load_model(version, fold)

# %%
reload(md)

# Two distinct seeds to use for calculating a Pearson correlation
seed_a = 123
seed_b = 456
seeds = [seed_a, seed_b]

# Test some amounts of shuffled sequences to use as reference for
# calculating hypothetical contribution scores
num_shufs_list = [3, 10, 30, 50, 100, 200, 300, 500]

pearson = {False: {}, True: {}}
for avg_w_revcomp in [False, True]:
    for num_shufs in tqdm(num_shufs_list, desc=f"avg_w_revcomp={avg_w_revcomp}"):
        res = {}
        for seed in seeds:
            inputs, shap_vals = md.calc_contrib_scores(
                dataloader,
                model_trained=model_trained,
                device=device,
                random_state=seed,
                num_shufs=num_shufs,
                avg_w_revcomp=avg_w_revcomp,
            )
            res[seed] = shap_vals

        # Take the average using along the sequences of our sample dataset
        pearson[avg_w_revcomp][num_shufs] = np.mean(
            [
                stats.pearsonr(
                    res[seed_a][i].flatten(), res[seed_b][i].flatten()
                ).statistic
                for i in range(inputs.shape[0])
            ]
        )

pearson

# %%
fig, ax = plt.subplots(1, 1)
for avg_w_revcomp in [False, True]:
    ax.plot(
        list(pearson[avg_w_revcomp]),
        list(pearson[avg_w_revcomp].values()),
        "o-",
        label=f"avg_w_revcomp={avg_w_revcomp}",
    )
ax.set_xlabel("Num. shuffled sequences used as reference for hyp. contrib. scores")
ax.set_ylabel("Pearson correlation for 2 distinct RNG seeds")
ax.set_xscale("log")
_ = ax.legend()

# %% [markdown]
# This shows that:
#
# 1. There is a significant variation in the calculated contribution scores depending on which shuffled sequences are used as reference. In other words for relatively little amount of shuffles, the chosen seed matters.
# 2. As expected, increasing the number of shuffled reference sequences leads to more consistent results decreasing the dependence on the seed used for generating the pseudo-random shuffles.
# 3. Averaging the contribution scores obtained from inputting both the "forward" and reverse complement strands greatly enhances the robustness.
#
# If **not using** reverse complement averaging:
#
# - 100 shuffles is a solid amount for achieving very high correlation (~97%).
# - 30 shuffles is fairly good for quicker computation without big compromise (~90% correlation).
# - 10 shuffles seems a bit low with "only" ~80% correlation.
#
# If **using** reverse complement averaging (requires x2 GPU computation and <= x2 CPU computation):
#
# - 100 shuffles is likely unnecessarily large amount (~99% correlation).
# - 30 shuffles is excellent (~96% correlation).
# - 10 shuffles might be already enough (~89% correlation).
#

# %% [markdown]
# # Impact of the training/validation data split
#
# Here we check the correlation of the contributions scores obtained from evaluating the same input DNA sequences (and same references shuffles, i.e. same random seed) on 5 models trained on a different train/validation data splits (k folds).
# The correlation is calculated pairwise between each model pair combinations.
#

# %%
num_shufs = 30
version = "cc0e922b"
folds = 5  # we trained 5 kfolds

pearson = {}
res = {}
for fold in tqdm(range(folds)):
    inputs, shap_vals = md.calc_contrib_scores(
        dataloader,
        model_trained=load_model(version, fold),
        device=device,
        random_state=random_state,
        num_shufs=num_shufs,
        avg_w_revcomp=True,
    )
    res[fold] = shap_vals


for fold_i in range(folds):
    for fold_j in range(fold_i + 1, folds):
        if fold_i == fold_j:
            continue
        pearson[(fold_i, fold_j)] = np.mean(
            [
                stats.pearsonr(
                    res[fold_i][i].flatten(), res[fold_j][i].flatten()
                ).statistic
                for i in range(inputs.shape[0])
            ]
        )

pearson

# %% [markdown] vscode={"languageId": "raw"}
# Since the correlation between the contributions derived from different models is NOT very consistent among the best models from the different k-folds, it might be worth averaging the contribution scores from several models to obtain a more robust output akin to "wisdom of the crowd".
#

# %%
reload(md)
dataset = dm.make_tensor_dataset(
    df=df_sample_mini[:1],
    x_col="SeqEnc",
    y_col=y_col,
    device=device,
)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

res = []
for fold in range(folds):
    (input_seq_T, *_), (shap_val, *_) = md.calc_contrib_scores(
        dataloader,
        model_trained=load_model(version, fold),
        device=device,
        random_state=random_state,
        num_shufs=num_shufs,
    )
    res.append(shap_val)

shap_val_avg = np.array(res).mean(axis=0)
shap_val_avg.shape

# %%
reload(put)
start, stop = None, None
hypothetical = False

# last fold from the loop above
put.plot_weights(input_seq_T, shap_val, start, stop, hypothetical)
put.plot_weights(input_seq_T, shap_val_avg, start, stop, hypothetical)

# %%
