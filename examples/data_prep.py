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
# %load_ext autoreload
# %autoreload explicit

# %%
# %aimport utils, data_module

import utils as ut
import data_module as dm

# %%
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from functools import partial

# %%
dbm = Path("/Volumes/Famafia/brain-magnet/")
dbmrd = dbm / "rd_APP_data"
dbmt = dbm / "train"
assert dbm.is_dir()

# %%
print(torch.cuda.is_available(), torch.backends.mps.is_available())
# device = torch.device("cuda")
# device = torch.device("cpu")
device = torch.device("mps")  # might have priblems for macOS <14.0
device

# %%
fp = dbmrd / "Enhancer_activity_with_ACTG_sequences.csv.gz"
df_enrichment = pd.read_csv(fp)

# %%
n_samples = 5000
df = df_enrichment[:n_samples].copy()

# %%
# Transpose to match how pytorch organizes data: (batch_size, channels=4, num_bp)
func = partial(ut.one_hot_encode, pad_to=1000, transpose=True)
arr = np.stack(df.Seq.map(func).values, axis=0)  # type: ignore
arr.shape

# %%
fp = dbmrd / f"seqs_shape{arr.shape}.npy".replace(" ", "")
np.save(fp, arr)
fp_size = round(fp.stat().st_size / 2**20)  # size in MB
fp_size, fp

# %%
arr_memmap = np.load(fp, mmap_mode="r")
arr_memmap.shape

# %%
from torch.utils.data import Subset

