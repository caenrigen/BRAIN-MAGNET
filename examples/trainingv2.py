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
from numpy.random import default_rng
from tqdm.auto import tqdm
import pandas as pd
import time

import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# %%
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
fp_npy_1hot_seqs = dbmrd / "seqs(148114,4,1000).npy"
assert fp_npy_1hot_seqs.exists()
fp_npy_1hot_seqs_rev_comp = dbmrd / "seqs_rev_comp(148114,4,1000).npy"
assert fp_npy_1hot_seqs_rev_comp.exists()


# %%
# Generate a random version string for this trianing run, it is used to name the
# folder where the results of the training are saved.
# Here we actually want it to be always random.
rng_version = default_rng(int(time.time()))
version = rng_version.random(1).tobytes()[:4].hex()
print(f"{version = }")

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
folds = None

folds_list = range(folds) if folds else []
frac_val = 0.10  # only relevant if not using folds

# Fraction of the initial dataset to set aside for testing.
# 💡 Tip: You can increase it a lot to e.g. 0.90 for a quick training round.
frac_test = 0.90


def train(version: str, fold: int, batch_size: int):
    # We did not use workers, but we keep it here for future reference and reminder.
    L.seed_everything(random_state, workers=True)  # for reproducibility

    model = cnn.BrainMagnetCNN(
        learning_rate=learning_rate,
        # Don't change this for training, reverse complement is handled by the data
        # module as augmentation data.
        forward_mode="forward",
        # The rest are hyperparameters for logging purposes.
        task=task,
        batch_size=batch_size,
        frac_test=frac_test,
        frac_val=frac_val,
        folds=folds,
        fold=fold,
        samples=len(targets),
        max_ep=max_epochs,
    )

    data_loader = dm.DataModule(
        fp_npy_1hot_seqs=fp_npy_1hot_seqs,
        fp_npy_1hot_seqs_rev_comp=fp_npy_1hot_seqs_rev_comp,
        random_state=random_state,
        targets=targets,
        folds=folds or None,
        fold=fold,
        frac_test=frac_test,
        frac_val=frac_val,
        # DataLoader kwargs:
        batch_size=batch_size,
        # These might give some speed up if cuda is available
        # pin_memory=True,
        # pin_memory_device="cuda",
    )

    logger = TensorBoardLogger(
        save_dir=dbmt,
        name=task,
        version=f"{version}_f{fold}" if folds else version,
        # avoid inserting a dummy metric with an initial value
        default_hp_metric=False,
    )
    checkpoints_callback = ModelCheckpoint(
        filename="{epoch:03d}",
        every_n_epochs=1,
        save_top_k=-1,
        # Set it to True if you intend to, e.g., be able to resume training from a
        # checkpoint and need things like optimizer state, etc. to be saved.
        save_weights_only=False,
    )
    trainer = L.Trainer(
        accelerator=device.type,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[cnn.LogMetrics(), checkpoints_callback],
        deterministic=True,  # for reproducibility
        enable_checkpointing=True,
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


# Free up memory
gc.collect()
if device.type == "mps":
    torch.mps.empty_cache()

if folds:
    for fold in tqdm(folds_list, desc="Folds"):
        res = train(fold=fold, version=version, batch_size=batch_size)
        if not res:
            break
else:
    res = train(fold=0, version=version, batch_size=batch_size)

# %%
