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
# %aimport utils

import utils as ut

# %%
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial

# %%
dbm = Path("/Volumes/Famafia/brain-magnet/")
dbmrd = dbm / "rd_APP_data"
dbmt = dbm / "train"
assert dbm.is_dir()

# %%
fp = dbmrd / "Enhancer_activity_with_ACTG_sequences.csv.gz"
df_enrichment = pd.read_csv(fp, usecols=["Seq"])
df_enrichment.head()

# %%
# n_samples = 5000
# df = df_enrichment[:n_samples].copy()
df = df_enrichment

# %%
# Transpose to match how pytorch organizes data: (batch_size, channels=4, num_bp)
func = partial(ut.one_hot_encode, pad_to=1000, transpose=True)
seqs_1hot = df.Seq.map(func)  # type: ignore
arr = np.stack(seqs_1hot.values, axis=0)
arr.shape

# %%
fp = dbmrd / f"seqs{arr.shape}.npy".replace(" ", "")
np.save(fp, arr)
round(fp.stat().st_size / 2**20), fp  # size in MB

# %%
func = partial(ut.one_hot_reverse_complement, is_transposed=True)
arr_rev = np.stack(seqs_1hot.map(func).values, axis=0)
arr_rev.shape

# %%
fp = dbmrd / f"seqs_rev_comp{arr.shape}.npy".replace(" ", "")
np.save(fp, arr)
round(fp.stat().st_size / 2**20), fp

# %%
# Test loading
arr_memmap = np.load(fp, mmap_mode="r")
arr_memmap.shape
