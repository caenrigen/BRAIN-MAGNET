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

# %% [raw]
# %load_ext autoreload
# %autoreload all

# %%
from typing import Optional, Union, List
from pathlib import Path
import numpy as np
import torch
from torch import nn

# %%
# local modules
import data_module as dm
import deep_explainer as de
# import notebook_helpers as nh

# %%
dir_data = Path("./data")
assert dir_data.is_dir()
dir_train = dir_data / "train"
dir_train.mkdir(exist_ok=True)
dir_cb_score = dir_data / "cb_score"
dir_cb_score.mkdir(exist_ok=True)

dir_train_2d = dir_train / "2d_py37"
dir_train_2d.mkdir(parents=True, exist_ok=True)

fp_dataset = dir_data / "Enhancer_activity_with_str_sequences.csv.gz"

device = torch.device("cpu")

# %% [markdown]
# # Choose models and dataloader
#

# %%
sample_indices = [
    0,
    71187,
    72893,
    96960,
    98223,
    103904,
    105091,
    106890,
    114846,
    126371,
    26636,
    31597,
    36769,
    69558,
    89821,
]

# %%
# Set to an integer to use a small sample of the dataset, the first `sample` sequences
sample: Optional[Union[int, List[int]]] = sample_indices
task = "ESC"
random_state = 20240413 + 1
datamodule = dm.DataModule(
    fp_dataset=fp_dataset,
    targets_col=f"{task}_log2_enrichment",
    # we use calc_contrib_scores(..., avg_w_revcomp=True) instead
    augment_w_rev_comp=False,
    batch_size=128,  # bigger batches consume more RAM but did not seem faster
    random_state=random_state,
    frac_test=0.00,
    frac_val=0.10,
)
datamodule.setup()

if sample:
    if isinstance(sample, int):
        indices = np.arange(sample)
    else:
        indices = sample
    dataloader = datamodule.DataLoader(dataset=datamodule.dataset, sampler=indices)
else:
    dataloader = datamodule.full_dataloader()

len(dataloader)  # number of batches


# %% [markdown]
# # Calculate contribution score
#

# %%
class ModelModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # self.model = model or make_model()
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

version = "86847720"
# fp_model = dir_train_2d / version / "epoch=002_vl=1298_tl=1440.ckpt"
# fp_model = dir_train_2d / version / "epoch=003_vl=1374_tl=1428.ckpt"
fp_model = dir_train_2d / version / "epoch=004_vl=1420_tl=1410.ckpt"

version = "de05c7a2"
fp_model = dir_train_2d / version / "epoch=001_vl=1475_tl=1443.ckpt"
model.load_state_dict(torch.load(fp_model))

# %%
avg_w_revcomp = False
num_shufs = 100

suffix = "_sample"
fp_out_inputs = dir_cb_score / f"{task}_{version}_seqs_{num_shufs}shufs{suffix}.npy"
fp_out_shap_av = (
    dir_cb_score / f"{task}_{version}_shap_vals_{num_shufs}shufs{suffix}.npy"
)

gen = de.calc_contrib_scores(
    dataloader=dataloader,
    model_trained=model,
    fp_out_shap=fp_out_shap_av,
    fp_out_inputs=fp_out_inputs,
    overwrite=True,  # ! Reminder: this will overwrite the file if it exists
    sum_inplace=False,
    div_after_sum=False,
    device=device,
    random_state=random_state,
    num_shufs=num_shufs,
    avg_w_revcomp=avg_w_revcomp,
    tqdm_bar=False,
)
for inputs, shap_vals in gen:
    # we are writing to disk, so we don't need to keep the data in memory
    del inputs, shap_vals

# %%
