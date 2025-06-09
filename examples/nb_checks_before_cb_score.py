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
# Reload all python modules before executing each cell
# %load_ext autoreload
# %autoreload all

# %%
from itertools import combinations
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch
from tqdm.auto import tqdm

# %%
import cnn_starr as cnn
import data_module as dm
import plot_utils as put
import motif_discovery as md
import notebook_helpers as nh

# %%
dir_data = Path("./data")
assert dir_data.is_dir()
dir_train = dir_data / "train"
dir_train.mkdir(exist_ok=True)

fp_dataset = dir_data / "Enhancer_activity_with_str_sequences.csv.gz"

device = torch.device("mps")

# %%
task, version = "ESC", "762acb33"
fp_checkpoint = dir_train / f"{task}/{version}/checkpoints/epoch=014.ckpt"
assert fp_checkpoint.exists()

# %%
df_enrichment = pd.read_csv(fp_dataset, usecols=dm.DATASET_COLS_NO_SEQ)
df_enrichment.head()


# %%
def load_model_and_full_dataloader(
    fp_checkpoint: Path, fp_dataset: Path, indices: np.ndarray
):
    model = cnn.ModelModule.load_from_checkpoint(fp_checkpoint)
    datamodule = dm.DataModule(
        fp_dataset=fp_dataset,
        targets_col=f"{task}_log2_enrichment",
        augment_w_rev_comp=False,
        batch_size=256,
    )
    datamodule.setup()
    # `datamodule.DataLoader` is based on a partial function, we only need to pass
    # the indices to the sampler
    dataloader = datamodule.DataLoader(datamodule.dataset, sampler=indices)
    return model, dataloader


# %%
sample_size = 10  # small sample for some fast estimates, shap is pretty slow
targets = df_enrichment[f"{task}_log2_enrichment"]
top_targets = targets.sort_values(ascending=False)[:sample_size]
idxs_sample = top_targets.index.to_numpy()
top_targets

# %% [markdown]
# # Impact of averaging with rev. compl. and the number of shuffles
#

# %%
# Two distinct seeds to use for calculating a Pearson correlation
seed_a = 123
seed_b = 456
seeds = [seed_a, seed_b]

# Test some amounts of shuffled sequences to use as reference for
# calculating hypothetical contribution scores
num_shufs_list = [5, 10, 15, 20, 30, 50, 100, 200]

model_trained, dataloader = load_model_and_full_dataloader(
    fp_checkpoint=fp_checkpoint,
    fp_dataset=fp_dataset,
    indices=idxs_sample,
)
pearson = {False: {}, True: {}}
for avg_w_revcomp in [False, True]:
    for num_shufs in tqdm(num_shufs_list, desc=f"avg_w_revcomp={avg_w_revcomp}"):
        res = {}
        for seed in seeds:
            inputs, shap_vals = md.calc_contrib_scores_concatenated(
                dataloader,
                model_trained=model_trained,
                device=device,
                random_state=seed,
                num_shufs=num_shufs,
                avg_w_revcomp=avg_w_revcomp,
                tqdm_bar=False,
            )
            res[seed] = shap_vals

        # Take the average over the sequences of our sample dataset
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
# 3. Averaging the contribution scores obtained from inputting both the "forward" and the reverse complement strands greatly enhances the robustness.
#
# If **not** averaging with the reverse complement:
#
# - 100 shuffles is a solid amount for achieving very high correlation (~98%).
# - 30 shuffles is fairly good for quicker computation without big compromise (~90-95% correlation).
# - 10 shuffles seems a bit low with "only" ~80-85% correlation.
#
# If averaging with reverse complement (requires x2 GPU computation and <= x2 CPU computation):
#
# - 100 shuffles is likely unnecessarily large amount (~99% correlation).
# - 30 shuffles is excellent (~96-97% correlation).
# - 10 shuffles might be already enough (~90% correlation).
#

# %% [markdown]
# # Impact of the training/validation data split
#
# Here we check the correlation of the contributions scores obtained from evaluating the same input DNA sequences (and same references shuffles, i.e. same random seed) on 5 models trained on a different train/validation data splits (k folds).
# The correlation is calculated pairwise between each model pair combinations.
#

# %%
task, version = ("ESC", "8a6b9616")
df_ckpts = dm.list_checkpoints(dp_train=dir_train, task=task, version=version)
best_checkpoints, *_ = nh.pick_best_checkpoints(df_ckpts, plot=False)
best_checkpoints

# %%
random_state = 20240413
num_shufs = 30
version = "cc0e922b"

pearson = {}
res = {}
for fold, fp in tqdm(best_checkpoints.items()):
    inputs, shap_vals = md.calc_contrib_scores_concatenated(
        dataloader,
        model_trained=cnn.ModelModule.load_from_checkpoint(fp),
        device=device,
        random_state=random_state,
        num_shufs=num_shufs,
        avg_w_revcomp=True,
        tqdm_bar=False,
    )
    res[fold] = shap_vals


for fa, fb in combinations(range(len(best_checkpoints)), 2):
    pearson[(fa, fb)] = np.mean(
        [
            stats.pearsonr(res[fa][i].flatten(), res[fb][i].flatten()).statistic
            for i in range(inputs.shape[0])
        ]
    )

pearson

# %% [markdown] vscode={"languageId": "raw"}
# Since the correlation between the contributions derived from different models is NOT very consistent among the best models from the different k-folds, it might be worth averaging the contribution scores from several models to obtain a more robust output akin to "wisdom of the crowd".
#

# %%
res = []
for fold, fp in tqdm(best_checkpoints.items()):
    (input_seq_T, *_), (shap_val, *_) = md.calc_contrib_scores_concatenated(
        dataloader,
        model_trained=cnn.ModelModule.load_from_checkpoint(fp),
        device=device,
        random_state=random_state,
        num_shufs=num_shufs,
        avg_w_revcomp=True,
        tqdm_bar=False,
    )
    res.append(shap_val)

shap_val_avg = np.array(res).mean(axis=0)
shap_val_avg.shape

# %%
start, stop = None, None
hypothetical = False

# Plot for a single fold (last fold from the loop above)
put.plot_weights(input_seq_T, shap_val, start, stop, hypothetical)
# Plot for the average of all folds
put.plot_weights(input_seq_T, shap_val_avg, start, stop, hypothetical)


# %% [markdown]
# There are some significant differences that can be observed visually between using a single model or averaging across all 5 models from the 5-folds training.
#
