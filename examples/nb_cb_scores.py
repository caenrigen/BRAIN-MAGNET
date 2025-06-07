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
import cnn_starr as cnn
import data_module as dm
import motif_discovery as md
import notebook_helpers as nh
from tqdm.auto import tqdm

# %%
from typing import Optional
from pathlib import Path
import numpy as np
import torch

# %%
dir_data = Path("/Volumes/Famafia/brain-magnet/")
assert dir_data.is_dir()
dir_train = dir_data / "train"
dir_train.mkdir(exist_ok=True)
dir_cb_score = dir_data / "cb_score"
dir_cb_score.mkdir(exist_ok=True)

fp_dataset = dir_data / "Enhancer_activity_with_str_sequences.csv.gz"

device = torch.device("mps")

# %% [markdown]
# # Choose models and dataloader
#

# %%
task, version = ("ESC", "8a6b9616")
df_ckpts = dm.list_checkpoints(dp_train=dir_train, task=task, version=version)
best_checkpoints, *_ = nh.pick_best_checkpoints(df_ckpts, plot=False)

# %%
# Set to an integer to use a small sample of the dataset, the first `sample` sequences
sample: Optional[int] = None

datamodule = dm.DataModule(
    fp_dataset=fp_dataset,
    targets_col=f"{task}_log2_enrichment",
    # we use calc_contrib_scores(..., avg_w_revcomp=True) instead
    augment_w_rev_comp=False,
    batch_size=256,  # bigger batches consume more RAM but did not seem faster
)
datamodule.setup()

if sample:
    indices = np.arange(sample)
    dataloader = datamodule.DataLoader(dataset=datamodule.dataset, sampler=indices)
else:
    dataloader = datamodule.full_dataloader()

len(dataloader)  # number of batches

# %% [markdown]
# # Calculate contribution score
#

# %% [markdown]
# ⚠️ Runnning `calc_contrib_scores` on the full dataset can take ~2 hours.
#

# %%
random_state = 20240413
# 10 works well enough (specialy if averaging with the reverse complement),
# 30 if you want to be safe extra safe
# Calculating the contribution scores for the full ~150k sequences takes:
# num_shufs=10 --> ~1h
# num_shufs=30 --> ~1.5h (~50% longer)
num_shufs = 10
num_folds = len(best_checkpoints)

fp_out_inputs = dir_cb_score / f"{task}_{version}_input_seqs_{num_shufs}shufs.npy"
fp_out_shap_av = dir_cb_score / f"{task}_{version}_shap_vals_{num_shufs}shufs.npy"

for fold, fp in tqdm(best_checkpoints.items()):
    sum_inplace = fold > 0
    div_after_sum = num_folds if fold == num_folds - 1 else None
    gen = md.calc_contrib_scores(
        dataloader=dataloader,
        model_trained=cnn.BrainMagnetCNN.load_from_checkpoint(fp),
        fp_out_shap=fp_out_shap_av,
        # Only save inputs once
        fp_out_inputs=fp_out_inputs if fold == 0 else None,
        overwrite=fold > 0,
        # after writing the first fold, sum inplace
        sum_inplace=sum_inplace,
        # on last fold devide by the number of folds to obtain the average
        div_after_sum=div_after_sum,
        device=device,
        random_state=random_state,
        num_shufs=num_shufs,
        # avg_w_revcomp=True requires ~2x computation but it is likley to produce less
        # noisy contributions scores.
        avg_w_revcomp=False,
        tqdm_bar=True,
    )
    for inputs, shap_vals in gen:
        # we are writing to disk, so we don't need to keep the data in memory
        del inputs, shap_vals
