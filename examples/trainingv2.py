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
# Test tqdm progress bar
import time
from tqdm.auto import tqdm

for _ in tqdm(range(10)):
    time.sleep(0.1)

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
from tqdm.auto import tqdm
from pathlib import Path
from typing import Optional
from functools import partial
import gc

import lightning as L
import torch
from torch import nn
import gc
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.loggers import TensorBoardLogger
from random import randbytes

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

# %% [raw] vscode={"languageId": "raw"}
# # Logs
# - 347bc336: lr=0.001, weight_decay=1e-6, dropout=0.1
# - 288fa45b: lr=0.0005, weight_decay=1e-6, dropout=0.1
# - 3dc19952: lr=0.0001, weight_decay=1e-6, dropout=0.1
# - 04b9c60f: lr=0.0001, weight_decay=1e-7, dropout=0.1
# - 86aa4f1d: lr=0.001, weight_decay=5e-6, dropout=0.1
# - 1fcdfb89: lr=0.005, weight_decay=1e-5, dropout=0.1
# - ca9be902: lr=0.01, weight_decay=8e-6, dropout=0.1
# - e7dcf63e: lr=0.01, weight_decay=5e-6, dropout=0.1
# - 91f17bcc: lr=0.01, weight_decay=2e-6, dropout=0.1
# - c92d9749: lr=0.01, weight_decay=1e-6, dropout=0.1, 0.05, 64->32 head
# - 2ae21203: lr=0.01, weight_decay=1e-6, dropout=0.1, 64->32
# - a3604d58: lr=0.01, weight_decay=1e-6, dropout=0.1, 32->16 all
# - ebd0e997: lr=0.01, weight_decay=1e-6, dropout=0.1, 16 all except nn.Conv2d(4, 32, kernel_size=(1, 13), ...)
# - af2f90f7: lr=0.01, weight_decay=1e-6, dropout=0.1, 16 all except nn.Conv2d(4, 32, kernel_size=(1, 15), ...), 11(?), 9(?)
# - 674707d4: lr=0.01, weight_decay=1e-6, dropout=0.1, 16 all except nn.Conv2d(4, 32, kernel_size=(1, 13), ...), 9, 7
# - a8099c41: lr=0.01, weight_decay=1e-6, dropout=0.1, 16 all except nn.Conv2d(4, 32, kernel_size=(1, 11/9/7), ...)
# - 6adb5711: lr=0.01, weight_decay=1e-6, dropout=0.1, 16 all except nn.Conv2d(4, 32, kernel_size=(1, 15/13/11), ...)
# - a5ffba4d: lr=0.01, weight_decay=1e-6, dropout=0.1, 16 all except nn.Conv2d(4, 32, kernel_size=(1, 17/15/13), ...)
# - cd8eaa74: lr=0.01, weight_decay=1e-6, dropout=0.1, 16 all except nn.Conv2d(4, 32, kernel_size=(1, 13/11/9), ...)
# ---
# - 2b336432: lr=0.01, weight_decay=1e-6, dropout=0.1, 16 all except nn.Conv2d(4, 32, kernel_size=(1, 15/13/11), ...)
# - f219f565: 2b336432 but with augment=4
# - a8e7dd9b: f219f565 but with augment=8
# ---
# - d2dd90b5: f219f565 but no backbone+head
# ---
# - 4cdb5f99: fixed the data module to keep test set separate
# ---
# - 44c93be6/: no test, 5% val
# ---
# - f952c4c5: no test, 5% val, log2log2norm targets
# ---
# - c8d9a9e7: ???
# - 82c66f85: no test, 5% val, augment=6, final(?)
# - 26a38237: no test, 5% val, augment=6, 16/16, 15/13/11 experiment
# - f4ceccfc: no test, 5% val, augment=6, 16/16, 13/11/9 experiment
# - 6fb254fd: no test, 5% val, augment=6, 16/16, 17/15/13 experiment
# - b854ab5f: no test, 5% val, augment=4, 16/16, 15/13/11, final
# - 050ccf4e: no test, 5% val, augment=10, 16/16, 15/13/11
# - f9bd95fa: no test, 5% val, augment=4, all 16, 15/13/11, AvgPool2d
# - 02fe8ebd: no test, 5% val, augment=4, all 16, 15/13/11, swap conv2d->conv1d
# - 5a41adbe: no test, 5% val, augment=None, all 16, 15/13/11, swap conv2d->conv1d
# - cc0e922b: no test, 5% val, augment=None, all 16, 15/13/11, Conv1D, fixed loss logging

# %%
import utils as ut
import cnn_starr as cnn
import data_module as dm

# %%
task = "ESC"
# task = "NSC"
df_enrichment = dm.load_enrichment_data(
    fp=dbmrd / "Enhancer_activity_w_seq.csv.gz", y_col=f"{task}_log2_enrichment"
)
df_enrichment.head()

# %%
gc.collect()
torch.mps.empty_cache()

# %%
version = randbytes(4).hex()
print(f"{version = }")
sample = None

n_folds = 5
folds = range(n_folds)
# folds = [0]

max_epochs = 10 * 5

frac_for_test = 0
frac_for_val = 0.05
# augment = 4
augment = None

y_col = f"{task}_log2_enrichment"


def train(df_enrichment, task, max_epochs, n_folds, fold_idx=0):
    model = cnn.CNNSTARR(
        lr=0.01,
        weight_decay=1e-6,
        log_vars_prefix=task,
    )
    model.to(device)

    data_loader = dm.DMSTARR(
        df_enrichment=df_enrichment,
        sample=sample,
        y_col=y_col,
        n_folds=n_folds,
        fold_idx=fold_idx,
        augment=augment,
        frac_for_test=frac_for_test,
        frac_for_val=frac_for_val,
        device=device,
    )

    logger = TensorBoardLogger(
        dbmt,
        name=f"starr_{task}",
        version=version,
        sub_dir=f"fold_{fold_idx}",
    )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        # callbacks=[early_stop],
        callbacks=[cnn.EpochCheckpoint(task=task)],
    )

    try:
        trainer.fit(model, data_loader)  # run training
    except (KeyboardInterrupt, NameError):
        print("Training interrupted by user")
        return False

    # Free up memory
    model.cpu()
    del model, data_loader, logger, trainer
    gc.collect()
    torch.mps.empty_cache()

    return True


if n_folds:
    for fold_idx in tqdm(folds, desc="Folds"):
        # if fold_idx < 4:
        #     continue
        res = train(df_enrichment, task, max_epochs, n_folds, fold_idx=fold_idx)
        if not res:
            break
else:
    res = train(df_enrichment, task, max_epochs, n_folds)

# %%
version

# %%
