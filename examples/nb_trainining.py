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
import lightning as L
from tqdm.auto import tqdm

# %%
# local modules
import utils as ut
import notebook_helpers as nh
import data_module as dm
import explainn as enn


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
dir_train = dir_data / "train_explainn"  # Tensorboard logs and model checkpoints
dir_train.mkdir(exist_ok=True)
fp_dataset = dir_data / "Enhancer_activity_with_str_sequences.csv.gz"
assert fp_dataset.exists()

augment_w_rev_comp = True

task: Literal["ESC", "NSC"] = "ESC"

# The training should result in exactly the same weights using the same seed,
# same data loading and processing order, same hyperparameters, same
# software packages and same hardware.
# Nonetheless, be aware that even if you keep everything the same but the hardware
# is different you might get slightly different results. On the same machine results
# should be exactly the same.
seed_split = 413  # for reproducibility

# Some NVIDIA GPUs implement certain operations in a non-deterministic way.
# So results might still be slightly different for the same seed.
seed_train = 416

# We train for a fixed number of epochs and post select the best model
max_epochs = 15

batch_size = 256
learning_rate = 0.001
weight_decay = 1e-4  # tiny weight decay to avoid huge weights (regularization)

# Train 4 models, each one trained on 3/4=75% of the data and validated on 1/4=25% of
# the data. Each time the data used for validation is different.
folds = 5
one_fold_only = False

folds_list = range(folds) if folds else []
frac_val = 0.00  # only relevant if not using folds

# Fraction of the initial dataset to set aside for testing.
# 💡 Tip: You can increase it a lot to e.g. 0.90 for a quick training round.
frac_test = 0.00

# Fraction of the training set to eval on after each epoch, used as an extra sanity
# check. Monitors the performance of the model on the training set.
frac_train_sample = 0.20

explainn_hyper_params = dict(
    num_cnns=20,
    filter_size=11,
    # num_fc=2,
    pool_size=7,
    pool_stride=7,
    num_classes=1,
    channels_mid=100,  # only relevant if num_fc >= 2
    input_length=1000,
)
# make_model = partial(enn.ExplaiNN, **explainn_hyper_params)
make_model = partial(enn.make_explainn, **explainn_hyper_params)

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
    frac_train_sample=frac_train_sample,
    folds=folds,
    # groups_func=partial(dm.bp_dist_groups, threshold=100_000),
    groups_func=lambda df: df.Cluster_id,
    seed_train=seed_train,
    seed_split=seed_split,
    device=device,
    weight_decay=weight_decay,
    augment_w_rev_comp=augment_w_rev_comp,
    make_model=make_model,
    **explainn_hyper_params,
)

if folds:
    version = ut.make_version()
    for fold in tqdm(folds_list, desc="Folds"):
        completed = train(fold=fold, version=version)
        if not completed:
            break
        if one_fold_only:
            break
else:
    res = train(fold=0)

# %%
