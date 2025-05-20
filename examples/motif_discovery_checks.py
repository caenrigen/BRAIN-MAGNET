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

# %%
task = "ESC"
df_sample = dm.load_enrichment_data(
    fp=dbmrd / "Enhancer_activity_w_seq_sample.csv.gz",
    y_col=f"{task}_log2_enrichment",
)
df_sample_mini = df_sample[:10].copy()

# %%
dataset = dm.make_tensor_dataset(
    df=df_sample_mini,
    x_col="SeqEnc",
    y_col=f"{task}_log2_enrichment",
    device=device,
)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
dataloader

# %%
version, fold = "cc0e922b", 4
fp = dbmt / f"starr_{task}" / version / "stats.pkl.bz2"
df_models = pd.read_pickle(fp)
epoch = dm.pick_checkpoint(df_models, fold=fold)
dp_checkpoints = dbmt / f"starr_{task}" / version / f"fold_{fold}" / "epoch_checkpoints"
fp_model_checkpoint = list(dp_checkpoints.glob(f"{task}_ep{epoch:02d}*.pt"))[0]
model_trained = cnn.load_model(
    fp=fp_model_checkpoint,
    device=device,
    forward_mode="main",
)


# %%
def unpad_shap_vals(input_seq, shap_val):
    m = input_seq.sum(axis=0, dtype=np.bool)
    m = np.broadcast_to(m, shap_val.shape)
    return shap_val[m]


# %%
pearson = {}

seq_idx = 0
# Two distinct seeds to use for calculating a Pearson correlation
seed_a = 123
seed_b = 456
seeds = [seed_a, seed_b]

# Test some amounts of shuffled sequences to use as reference for
# calculating hypothetical contribution scores
num_shufs_list = [10, 30, 50, 100, 200, 300, 500]

for num_shufs in tqdm(num_shufs_list):
    res = {}
    for seed in seeds:
        inputs, shap_vals = md.calc_contrib_scores(
            dataloader,
            model_trained=model_trained,
            device=device,
            random_state=seed,
            num_shufs=num_shufs,
        )
        shaps = [
            unpad_shap_vals(inputs[i], shap_vals[i]) for i in range(inputs.shape[0])
        ]
        res[seed] = shaps

    # Take the average using along the sequences of our sample dataset
    pearson[num_shufs] = np.mean(
        [
            stats.pearsonr(res[seed_a][i], res[seed_b][i]).statistic
            for i in range(inputs.shape[0])
        ]
    )

pearson

# %% [markdown]
# This shows that there is a significant variation in the calcualted contribution scores, depending on which shuffled sequences used as reference.
#
# As expected, increasing the number of reference shuffled sequences leads to more consistent results decreasing the dependence on the seed used for generating the pseudo-random shuffles.
#
# - 100 shuffles is a solid amount for achieving very high correlation (~97%).
# - 30 shuffles is fairly good for quicker coputation without big compromise (~91%).
# - 10 shuffles seems on the low end with only ~76% correlation.
#

# %%
fig, ax = plt.subplots(1, 1)
ax.plot(list(pearson), list(pearson.values()), "o-")
ax.set_xlabel("Num. shuffled sequences used as reference for hyp. contrib. scores")
ax.set_ylabel("Pearson correlation")
