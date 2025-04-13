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
#     display_name: Python 3.13 (RMBP)
#     language: python
#     name: ssh_mbp_no_sshfs
# ---

# %%
# cd sshpyk_code/examples

# %%
import time
from tqdm.auto import tqdm

for _ in tqdm(range(10)):
    time.sleep(0.2)

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
import gc
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from random import randbytes

# %%
random_state = 913
# random.seed(random_state)

dbmc = Path("/Users/victor/sshpyk_code")
dbm = Path("/Users/victor/sshpyk_data/")
dbmrd = dbm / "rd_APP_data"
dbmt = dbm / "train"

# %%
print(torch.cuda.is_available(), torch.backends.mps.is_available())
# device = torch.device("cuda")
# device = torch.device("cpu")
device = torch.device("mps")  # might have priblems for macOS <14.0
device

# %% [raw] vscode={"languageId": "raw"}
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
# - : fixed the data module to keep test set separate

# %%
# Evaluate the python files within the notebook namespace
# %run -i auxiliar.py
# %run -i cnn_starr.py
# %run -i data_module.py

task = "ESC"
threshold = 0.14
# task = "NSC"
# threshold = 0.17

# %%
# %run -i auxiliar.py
# %run -i cnn_starr.py
# %run -i data_module.py
df_enrichment = load_enrichment_data(
    fp=dbmrd / "Enhancer_activity_w_seq.csv.gz", y_col=f"{task}_log2_enrichment"
)

# %%
version = randbytes(4).hex()
# version = "d2dd90b5"
n_folds = 5

for fold_idx in tqdm(range(n_folds), desc="Folds"):
    # if fold_idx < 4:
    #     continue
    model = CNNSTARR(lr=0.01, weight_decay=1e-6, log_vars_prefix=task)
    model.to(device)

    fp = dbmrd / "Enhancer_activity_w_seq.csv.gz"
    data_loader = DMSTARR(
        df_enrichment=df_enrichment,
        sample=None,
        y_col=f"{task}_log2_enrichment",
        n_folds=n_folds,
        fold_idx=fold_idx,
        augment=4,
    )

    logger = TensorBoardLogger(
        dbmt,
        name=f"starr_{task}",
        version=version,
        sub_dir=f"fold_{fold_idx}",
    )
    trainer = L.Trainer(
        max_epochs=5,
        logger=logger,
        # callbacks=[early_stop],
        callbacks=[ThresholdCheckpoint(threshold=threshold, task=task)],
    )

    try:
        trainer.fit(model, data_loader)  # run training
    except (KeyboardInterrupt, NameError):
        print("Training interrupted by user")
        break

    # test_result = trainer.test(model, data_loader)[0]
    # results.append(test_result)

    del model, data_loader, logger, trainer

# %%
