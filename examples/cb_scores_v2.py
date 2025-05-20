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
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import math
import time
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# %%
random_state = 913
# random.seed(random_state)

dbmc = Path("/Users/victor/Documents/ProjectsDev/genomics/BRAIN-MAGNET")
dbm = Path("/Volumes/Famafia/brain-magnet")

# dbmc = Path("/Users/victor/sshpyk_code")
# dbm = Path("/Users/victor/sshpyk_data/")

dbmce = str(dbmc / "examples")
dbmrd = dbm / "rd_APP_data"
dbmt = dbm / "train"
os.chdir(dbmce)

# %%
print(torch.cuda.is_available(), torch.backends.mps.is_available())
# device = torch.device("cuda")
# device = torch.device("cpu")
device = torch.device("mps")
device

# %%
task = "ESC"
y_col = f"{task}_log2_enrichment"
df_sample = dm.load_enrichment_data(
    fp=dbmrd / "Enhancer_activity_w_seq_top_ESC_5_percent.csv.gz",
    y_col=y_col,
)
df_sample_mini = df_sample.sort_values(by=y_col, ascending=False)[:10].copy()
df_sample_mini

# %% [markdown]
# # Calculate contribution score
#

# %%
dataset = dm.make_tensor_dataset(
    df=df_sample_mini, x_col="SeqEnc", y_col=f"{task}_log2_enrichment", device=device
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataloader

# %%
version, fold = "cc0e922b", 0
fp_model_checkpoint = dm.pick_best_checkpoint(
    dp_train=dbmt, version=version, task=task, fold=fold
)
model_trained = cnn.load_model(
    fp=fp_model_checkpoint,
    device=device,
    forward_mode="main",
)
fp_model_checkpoint

# %%
reload(md)
inputs, shap_vals = md.calc_contrib_scores(
    dataloader, model_trained=model_trained, device=device
)

# %%
seq_idx = 0
put.plot_weights(inputs[seq_idx], shap_vals[seq_idx])
