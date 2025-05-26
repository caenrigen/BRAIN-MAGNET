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
import gc
from random import randbytes
from tqdm.auto import tqdm
import pandas as pd

import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

# %%
random_state = 913
dbm = Path("/Volumes/Famafia/brain-magnet")
dbmrd = dbm / "rd_APP_data"
dbmt = dbm / "train"

# %%
print(torch.cuda.is_available(), torch.backends.mps.is_available())
device = torch.device("mps")  # mps/cuda/cpu
device

# %%
# %load_ext autoreload
# %autoreload explicit
# %aimport utils, cnn_starr, data_module

import cnn_starr as cnn
import data_module as dm

# %%
task = "ESC"
y_col = f"{task}_log2_enrichment"
fp = dbmrd / "Enhancer_activity_with_ACTG_sequences.csv.gz"
targets = pd.read_csv(fp, usecols=[y_col])[y_col].to_numpy()
targets.shape

# %%
# fp_npy_1hot_seqs = dbmrd / "seqs(5000,4,1000).npy"
# targets = targets[:5000]

fp_npy_1hot_seqs = dbmrd / "seqs(148114,4,1000).npy"
assert fp_npy_1hot_seqs.exists()
fp_npy_1hot_seqs_rev_comp = dbmrd / "seqs_rev_comp(148114,4,1000).npy"
assert fp_npy_1hot_seqs_rev_comp.exists()


# %%
gc.collect()
torch.mps.empty_cache()

# %%
version = randbytes(4).hex()
print(f"{version = }")

batch_size = 256
learning_rate = 0.01
weight_decay = 1e-6

n_folds = 5
folds = range(n_folds) if n_folds else []
frac_for_val = 0.0  # only relevant if not using n_folds

max_epochs = 50

# Fraction of the initial dataset to set aside for testing.
# 💡 Tip: You can increase it a lot to e.g. 0.90 for a quick test training.
frac_for_test = 0.05


def train(version: str, fold: int, batch_size: int):
    model = cnn.BrainMagnetCNN(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        # Don't change this for training, reverse complement is handled by the data
        # module as augmentation data.
        forward_mode="forward",
        # The rest are hyperparameters for logging purposes.
        task=task,
        batch_size=batch_size,
        frac_for_test=frac_for_test,
        frac_for_val=frac_for_val,
        n_folds=n_folds,
        fold=fold,
        num_samples=len(targets),
        max_epochs=max_epochs,
    )
    model.to(device)

    data_loader = dm.DataModule(
        fp_npy_1hot_seqs=fp_npy_1hot_seqs,
        fp_npy_1hot_seqs_rev_comp=fp_npy_1hot_seqs_rev_comp,
        targets=targets,
        n_folds=n_folds or None,
        fold=fold,
        frac_for_test=frac_for_test,
        frac_for_val=frac_for_val,
        # DataLoader kwargs:
        batch_size=batch_size,
        # These might give some speed up if cuda is available
        # pin_memory=True,
        # pin_memory_device="cuda",
    )

    logger = TensorBoardLogger(
        dbmt,
        name=task,
        version=version,
        sub_dir=f"fold_{fold}",
        default_hp_metric=True,
    )
    trainer = L.Trainer(
        accelerator="mps",
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[cnn.EpochCheckpoint()],
    )

    try:
        trainer.fit(model, datamodule=data_loader)  # run training
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
    for fold in tqdm(folds, desc="Folds"):
        # if fold < 4:
        #     continue
        res = train(fold=fold, version=version, batch_size=batch_size)
        if not res:
            break
else:
    res = train(fold=0, version=version, batch_size=batch_size)

# %%
