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
from pathlib import Path
from functools import partial
import logging
from typing import Literal

import torch
from tqdm.auto import tqdm

# logging.basicConfig(level=logging.INFO)

# %%
# %load_ext autoreload
# %autoreload explicit
# %aimport utils, cnn_starr, data_module, notebook_helpers

import utils as ut
import cnn_starr as cnn
import data_module as dm
import notebook_helpers as nh

# %%
print(torch.cuda.is_available(), torch.backends.mps.is_available())
device = torch.device("mps")  # mps/cuda/cpu
device

# %%
dir_data = Path("/Volumes/Famafia/brain-magnet")
dir_train = dir_data / "train"
fp_dataset = dir_data / "Enhancer_activity_with_str_sequences.csv.gz"

task: Literal["ESC", "NSC"] = "ESC"

# The training should result in exactly the same models using the same seed,
# same data loading and processing order, same model, same hyperparameters, same
# software packages and same hardware.
# Nontheless, be aware that even if you keep everything the same but the hardware
# is different you might get slightly different results. On the same machine results
# should be exactly the same.
random_state = 20240413  # for reproducibility

# We train for a fixed number of epochs and post select the best model(s)
max_epochs = 30

batch_size = 256
learning_rate = 0.01

# Train 5 models, each one trained on 4/5=80% of the data and validated on 1/5=20% of
# the data. Each time the data used for validation is different.
folds = 5

folds_list = range(folds) if folds else []
frac_val = 0.00  # only relevant if not using folds

# Fraction of the initial dataset to set aside for testing.
# 💡 Tip: You can increase it a lot to e.g. 0.90 for a quick training round.
frac_test = 0.10

train = partial(
    nh.train,
    save_dir_tensorboard=dir_train,
    fp_dataset=fp_dataset,
    batch_size=batch_size,
    task=task,
    learning_rate=learning_rate,
    max_epochs=max_epochs,
    frac_test=frac_test,
    frac_val=frac_val,
    folds=folds,
    random_state=random_state,
    device=device,
)

if folds:
    version = ut.make_version()
    for fold in tqdm(folds_list, desc="Folds"):
        completed = train(fold=fold, version=version)
        if not completed:
            break
else:
    res = train(fold=0)

# %%
