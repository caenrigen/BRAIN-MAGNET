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
#     display_name: py37
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload all

# %%
from pathlib import Path
from numpy.random import default_rng
import time

import torch
from torch import nn
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

# %%
# local modules
import utils as ut
import data_module as dm
# import notebook_helpers as nh

# %%
device = torch.device("cpu")
device


# %%
class ModelModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 128, (1, 11), padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Conv2d(128, 256, (1, 9), padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Conv2d(256, 512, (1, 7), padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Flatten(),
            nn.Linear(512 // (2**3) * 1000, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def forward(self, x):
        return self.model(x)


loss_fn = nn.MSELoss()
model = ModelModule()
model

# %%
dir_data = Path("./data")
dir_train = dir_data / "train"  # Tensorboard logs and model checkpoints
fp_dataset = dir_data / "Enhancer_activity_with_str_sequences.csv.gz"
assert fp_dataset.exists()

dir_train_2d = dir_train / "2d_py37"
dir_train_2d.mkdir(parents=True, exist_ok=True)

# %%
random_state = 20240413 + 1
datamodule = dm.DataModule(
    fp_dataset=fp_dataset,
    augment_w_rev_comp=True,
    targets_col="ESC_log2_enrichment",
    random_state=random_state,
    batch_size=128,
    frac_val=0.10,
    frac_test=0.00,
)
datamodule.prepare_data()
datamodule.setup()
dataloader_train = datamodule.train_dataloader()
dataloader_val = datamodule.val_dataloader()

# %%
rng_version = default_rng(int(time.time()))
version = rng_version.random(1).tobytes()[:4].hex()
print(version)
dir_train_round = dir_train_2d / version

train_losses_avg = []
val_losses_avg = []

val_losses_avg_min = 100
epochs = 10
it_epoch = tqdm(range(epochs), desc="Epochs")
for epoch in it_epoch:
    train_losses = []
    val_losses = []
    model.train()
    for inputs, targets in tqdm(dataloader_train, desc="tb", leave=False):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        train_losses.append(loss.detach().numpy())

        model.optimizer.zero_grad()  # reset gradients
        loss.backward()  # compute gradients
        model.optimizer.step()  # update model weights

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader_val, desc="vb", leave=False):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_losses.append(loss.detach().numpy())

    train_losses_avg.append(np.mean(train_losses))
    val_losses_avg.append(np.mean(val_losses))

    it_epoch.set_postfix(
        train_loss=train_losses_avg[-1],
        val_loss=val_losses_avg[-1],
    )

    min_val_loss = min(val_losses_avg)
    if min_val_loss < val_losses_avg_min:
        val_losses_avg_min = min_val_loss
        print(
            f"New best model at epoch {epoch}: "
            + f"{val_losses_avg_min} (tloss: {train_losses_avg[-1]})"
        )

    vl = int(val_losses_avg[-1] * 10**4)
    tl = int(train_losses_avg[-1] * 10**4)
    fp = dir_train_round / f"epoch={epoch:03d}_vl={vl:04d}_tl={tl:04d}.ckpt"
    dir_train_round.mkdir(parents=False, exist_ok=True)
    torch.save(model.state_dict(), fp)

# %%
plt.plot(train_losses_avg, "o-", label="train")
plt.plot(val_losses_avg, "o-", label="val")
plt.legend()

# %%
