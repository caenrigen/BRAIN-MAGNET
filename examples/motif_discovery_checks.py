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
from importlib import reload
import utils as ut
import cnn_starr as cnn
import data_module as dm
import plot_utils as put
import motif_discovery as md

_ = reload(ut)
_ = reload(cnn)
_ = reload(dm)
_ = reload(put)
_ = reload(md)

# %%
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from deeplift.visualization import viz_sequence

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
df_sample = df_sample[:10].copy()

# %%
dataset = dm.make_tensor_dataset(
    df=df_sample,
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
reload(md)
inputs, shap_vals = md.calc_contrib_scores(
    dataloader, model_trained=model_trained, device=device
)

# %%
seq_idx = 0
put.plot_weights(inputs[seq_idx], shap_vals[seq_idx])
