# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext jupyter_black
# %load_ext autotime


# %%
import numpy as np
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import math
import time
import random
from tqdm.auto import tqdm
from pathlib import Path
from typing import Optional
from functools import partial

# %%
import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# %%
random_state = 913
random.seed(random_state)

dbm = Path("/Volumes/Famafia/brain-magnet/")
dbmrd = dbm / "rd_APP_data"
dbmt = dbm / "train"

# %%
print(torch.cuda.is_available(), torch.backends.mps.is_available())
# device = torch.device("cuda")
# device = torch.device("cpu")
device = torch.device("mps")  # might have priblems for macOS <14.0
device

# %%
# Evaluate the python files within the notebook namespace
# %run -i auxiliar.py
# %run -i cnn_starr.py
# %run -i data_module.py

task = "ESC"

model = CNNSTARR(lr=0.005, weight_decay=0, revcomp=True, log_vars_prefix=task)
model.to(device)

fp = dbmrd / "Enhancer_activity_w_seq.csv.gz"
data_loader = DMSTARR(fp=fp, sample=None, y_col=f"{task}_log2_enrichment")

early_stop = EarlyStopping(
    monitor=f"{task}_loss_val",
    min_delta=0.001,
    patience=20,
    verbose=True,
    mode="min",
)

logger = TensorBoardLogger(dbmt, name=f"starr_{task}")
trainer = L.Trainer(
    max_epochs=100,
    logger=logger,
    callbacks=[early_stop],
)

trainer.fit(model, data_loader)  # run training
