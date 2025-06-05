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
import utils as ut
import cnn_starr as cnn
import data_module as dm
import plot_utils as put
import motif_discovery as md
import notebook_helpers as nh
from tqdm.auto import tqdm

# %%
import numpy as np
from pathlib import Path
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

# %%
task, version = ("ESC", "8a6b9616")
df_ckpts = dm.list_checkpoints(dp_train=dir_train, task=task, version=version)
best_checkpoints, *_ = nh.pick_best_checkpoints(df_ckpts, plot=False)

# %%
# Set to 0 to use the entire dataset
# Set to some small fraction to use a small random sample of the dataset
sample_frac = 0.0

datamodule = dm.DataModule(
    fp_dataset=fp_dataset,
    targets_col=f"{task}_log2_enrichment",
    augment_w_rev_comp=True,
    batch_size=256,
    frac_test=sample_frac,
)
datamodule.setup()

if sample_frac:
    dataloader = datamodule.test_dataloader()
else:
    dataloader = datamodule.full_dataloader()

# %% [markdown]
# # Calculate contribution score
#

# %% [markdown]
# ⚠️ **WARNING**
#
# Runnning `calc_contrib_scores` on the full dataset can take 3+ hours and 30GB of RAM!
#

# %%
random_state = 20240413
num_shufs = 30

for fold, fp in tqdm(best_checkpoints.items()):
    inputs, shap_vals = md.calc_contrib_scores(
        dataloader,
        model_trained=cnn.BrainMagnetCNN.load_from_checkpoint(fp),
        device=device,
        random_state=random_state,
        num_shufs=num_shufs,
        avg_w_revcomp=True,
        tqdm_bar=True,
    )
    fp = dir_cb_score / f"{task}_{version}_{fold}_shap_vals.npz"
    np.savez_compressed(fp, shap_vals)

# These we only need to save once
fp = dir_cb_score / f"{task}_{version}_input_seqs.npz"
np.savez_compressed(fp, inputs)

# %% [raw] vscode={"languageId": "raw"}
# modisco motifs -w 1000 -s ESC_cc0e922b_0_top_5_percent_input_seqs.npz -a ESC_cc0e922b_0_top_5_percent_shap_vals.npz -n 2000 -o modisco_results.h5
# modisco report -i modisco_results.h5 -o report/ -s report/
