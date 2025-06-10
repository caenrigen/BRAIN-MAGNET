# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: g
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload all

# %%
from pathlib import Path
from functools import partial
from typing import Literal

import torch
from tqdm.auto import tqdm

# %%
# local modules
import utils as ut
import notebook_helpers as nh


# %%
print(torch.cuda.is_available(), torch.backends.mps.is_available())
device = torch.device("mps")  # mps/cuda/cpu
device

# %% [markdown]
# To visualize the training progress in more detail run in a terminal:
#
# ```bash
# tensorboard --logdir ./data/train
# ```
#
# which should output something like:
#
# ```bash
# Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
# TensorBoard 2.19.0 at http://localhost:6006/ (Press CTRL+C to quit)
# ```
#
# Open in the browser the URL printed by the command above, e.g. http://localhost:6006/.
#
# Explore the tabs for different visualizations:
#
# - http://localhost:6006/#scalars plots all the logged metrics
# - http://localhost:6006/#custom_scalars plots both training and validation loss on the same figure
# - http://localhost:6006/#hparams allows to compare between training runs (usually with different hyperparameters)
#

# %%
dir_data = Path("./data")
dir_train = dir_data / "train"  # Tensorboard logs and model checkpoints
fp_dataset = dir_data / "Enhancer_activity_with_str_sequences.csv.gz"
assert fp_dataset.exists()

task: Literal["ESC", "NSC"] = "ESC"

# The training should result in exactly the same weights using the same seed,
# same data loading and processing order, same hyperparameters, same
# software packages and same hardware.
# Nonetheless, be aware that even if you keep everything the same but the hardware
# is different you might get slightly different results. On the same machine results
# should be exactly the same.
random_state = 20240413  # for reproducibility

# We train for a fixed number of epochs and post select the best model
max_epochs = 75

batch_size = 256
learning_rate = 0.01

# Train 5 models, each one trained on 4/5=80% of the data and validated on 1/5=20% of
# the data. Each time the data used for validation is different.
folds = None

folds_list = range(folds) if folds else []
frac_val = 0.10  # only relevant if not using folds

# Fraction of the initial dataset to set aside for testing.
# 💡 Tip: You can increase it a lot to e.g. 0.90 for a quick training round.
frac_test = 0.00

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
        # break  # if we want to run a single fold only
else:
    res = train(fold=0)
