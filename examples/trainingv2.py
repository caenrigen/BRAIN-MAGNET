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
task = "ESC"
y_col = f"{task}_log2_enrichment"
fp = dbmrd / "Enhancer_activity_with_ACTG_sequences.csv.gz"
targets = pd.read_csv(fp, usecols=[y_col])[y_col].to_numpy()
targets.shape

# %%
# fp_npy_1hot_seqs = dbmrd / "seqs_shape(5000,4,1000).npy"
# targets = targets[:5000]

fp_npy_1hot_seqs = dbmrd / "seqs_shape(148114,4,1000).npy"
assert fp_npy_1hot_seqs.exists()

# %%
gc.collect()
torch.mps.empty_cache()

# %% [raw] vscode={"languageId": "raw"}
# Old data loader: ~4 min, 4.5-5GB
# New data loader: ~4 min, 1.8GB
# 512 batch size: ~2 min

# %%
version = randbytes(4).hex()
print(f"{version = }")
sample = None

n_folds = None
folds = range(n_folds) if n_folds else []
# folds = [0]

max_epochs = 10

frac_for_test = 0
frac_for_val = 0.05


def train(fold: int = 0):
    model = cnn.CNNSTARR(
        lr=0.01,  # learning rate
        weight_decay=1e-6,
        log_vars_prefix=task,
    )
    model.to(device)

    # On macOS with MPS (Apple Silicon) using multiprocessing did not give any speed up.
    # For CUDA it might very well be worth it.
    num_workers = 0

    # Ref: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # "fork" won't consume a lot of RAM, but might not work (well) on all OSes/versions.
    # Set it first to None if you run into any issues.
    multiprocessing_context = "fork" if num_workers else None

    data_loader = dm.DataModule(
        fp_npy_1hot_seqs=str(fp_npy_1hot_seqs),
        targets=targets,
        n_folds=n_folds,
        fold=fold,
        frac_for_test=frac_for_test,
        frac_for_val=frac_for_val,
        # DataLoader kwargs:
        batch_size=512,
        num_workers=num_workers,
        persistent_workers=bool(num_workers),
        multiprocessing_context=multiprocessing_context,
        # These might give some speed up if cuda is available
        # pin_memory=True,
        # pin_memory_device="cuda",
    )

    logger = TensorBoardLogger(
        dbmt,
        name=f"starr_{task}",
        version=version,
        sub_dir=f"fold_{fold}",
    )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[cnn.EpochCheckpoint(task=task)],
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
        res = train(fold=fold)
        if not res:
            break
else:
    fold = 0
    res = train(fold=fold)

# %%
