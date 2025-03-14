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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext jupyter_black
# %load_ext autotime

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
import gc

random_state = 913
random.seed(random_state)

dbm = Path("/Volumes/Famafia/brain-magnet/")
dbmrd = dbm / "rd_APP_data"
dbmt = dbm / "train"

# %%
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import (
    Sequential,
    Conv2d,
    BatchNorm2d,
    BatchNorm1d,
    ReLU,
    MaxPool2d,
    Flatten,
    Linear,
    Dropout,
    MSELoss,
    AdaptiveAvgPool2d,
)
from torchsummary import summary
from torchmetrics import SpearmanCorrCoef, PearsonCorrCoef
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %%
torch.cuda.is_available(), torch.backends.mps.is_available()

# %%
# device = torch.device("cuda")
# device = torch.device("cpu")
device = torch.device("mps")  # might have priblems for macOS <14.0
device


# %% [markdown]
# ### Utilities to encode data


# %%
def to_uint8(string):
    return np.frombuffer(string.encode("ascii"), dtype=np.uint8)


# function to convert only one piece of sequence to np.array
def make_one_hot_encode(alphabet: str = "ACGT", dtype=np.float32) -> np.ndarray:
    """
    One-hot encode for a sequence.
    A -> [1,0,0,0]
    C -> [0,1,0,0]
    G -> [0,0,1,0]
    T -> [0,0,0,1]
    N -> [0,0,0,0]
    """

    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    return hash_table


HASH_TABLE = make_one_hot_encode()


def one_hot_encode(sequence: str):
    return HASH_TABLE[to_uint8(sequence)]


def pad_arr(snippet: np.ndarray, arr_pre: int = 1024):
    assert len(snippet) <= arr_pre, len(snippet)
    arr_len = len(snippet)
    pad = (arr_pre - arr_len) / 2
    return np.pad(snippet, [(int(pad), math.ceil(pad)), (0, 0)], mode="constant")


seq = "ACGTATCnotdna"
to_uint8(seq), one_hot_encode(seq)


# %% [markdown]
# ### Convert data to tensors


# %%
def create_dataloader(x: pd.Series, y: pd.Series, batch_size: int):
    x = np.stack(x.values)

    # convert input: [batch, seq_len, 4] -> [batch, 4, 1, seq_len]
    tensor_x = torch.Tensor(x).permute(0, 2, 1).unsqueeze(2)

    # add one dimension in targets: [batch] -> [batch, 1]
    tensor_y = torch.Tensor(y.values).unsqueeze(1)

    tensor_data = TensorDataset(tensor_x, tensor_y)
    return DataLoader(tensor_data, batch_size)


def reverse_complement(x):
    """
    x: A tensor of shape (batch, 4, 1, seq_len)
       channel 0 = A, channel 1 = C, channel 2 = G, channel 3 = T.
    Returns the reverse-complement version of x (same shape).
    """
    # 1) Flip along the length dimension (dim=3).
    x_rev = torch.flip(x, dims=[3])

    # 2) Swap A <-> T and C <-> G channels.
    #    If your channel order is (A=0, C=1, G=2, T=3), you can do:
    channel_map = [3, 2, 1, 0]  # T, G, C, A
    x_revcomp = x_rev[:, channel_map, :, :]
    return x_revcomp


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Define EarlyStopping class


# %%
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=20,
        verbose=False,
        delta=0,
        trace_func=print,
        task="NSC",
        path=dbmt,
    ):
        """
        patience (int): How long to wait after last time validation loss improved.
        verbose (bool): If True, prints a message for each validation loss improvement.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        path (str): Path for the checkpoint to be saved to.
        trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.task = task
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, task)
        elif score < self.best_score + self.delta:
            self.counter += 1
            msg = f"EarlyStopping {self.counter = } out of {self.patience = }"
            self.trace_func(msg)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, task)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, task):
        """Saves model when validation loss decrease."""
        fp = dbmt / f"checkpoint_{task}.pth"
        if self.verbose:
            msg = f"Validation loss decreased ({self.val_loss_min:.5f} --> {val_loss:.5f}). Saving model: {fp}"
            self.trace_func(msg)
        torch.save(model.state_dict(), fp)
        self.val_loss_min = val_loss


# %% [markdown]
# ### Build deep learning model


# %%
class CNN_STARR(nn.Module):
    def __init__(self, revcomp: bool = False):
        super().__init__()
        self.revcomp = revcomp
        self.backbone = Sequential(
            Conv2d(4, 128, kernel_size=(1, 11), padding="same"),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d((1, 2), (1, 2)),
            Conv2d(128, 256, kernel_size=(1, 9), padding="same"),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d((1, 2), (1, 2)),
            Conv2d(256, 512, kernel_size=(1, 7), padding="same"),
            BatchNorm2d(512),
            ReLU(),
            MaxPool2d((1, 2), (1, 2)),
            AdaptiveAvgPool2d((1, 8)),
            # Flatten(),
        )
        self.head = Sequential(
            Linear(512 * 8, 1024),
            # Linear(65536, 1024),
            BatchNorm1d(1024),
            ReLU(),
            Dropout(0.4),
            Linear(1024, 1024),
            BatchNorm1d(1024),
            ReLU(),
            Dropout(0.4),
            Linear(1024, 1),
        )

    def forward_backbone_only(self, x):
        z = self.backbone(x)  # shape [batch, 512, 1, 1]
        z = z.view(z.size(0), -1)  # flatten to [batch, 512]
        return z

    def forward(self, x_input):
        embed_fwd = self.forward_backbone_only(x_input)
        if self.revcomp:
            embed_rc = self.forward_backbone_only(reverse_complement(x_input))
            embed_merged = embed_fwd + embed_rc
        else:
            embed_merged = embed_fwd
        out = self.head(embed_merged)
        return out


cnn_starr = CNN_STARR(revcomp=True)
summary(cnn_starr, input_size=(4, 1, 1024), batch_size=128)
cnn_starr.to(device)
sum(p.numel() for p in cnn_starr.parameters())


# %% [markdown]
# ### Specify Loss function and Optimizer

# %%
# loss function
loss_fn = MSELoss()
loss_fn.to(device)

# optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(cnn_starr.parameters(), lr=learning_rate)


# %% [markdown]
# ### Train the model using EarlyStopping


# %%
def train_model(
    model,
    loss_fn,
    optimizer,
    train_dataloader,
    valid_dataloader,
    batch_size=128,
    patience=20,
    epochs=100,
    task="NSC",
    device=device,
):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping_object
    early_stopping = EarlyStopping(patience, verbose=True)

    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        #################
        # Train the model
        #################

        ### index: 0 is NSC, index: 1 is ESC ###
        model.train()
        for batch, data in enumerate(train_dataloader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            # record training loss
            train_losses.append(loss.item())

            # optimize model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ####################
        # Validate the model
        ####################
        model.eval()  # pre model for evaluation
        with torch.no_grad():
            for batch, data in enumerate(valid_dataloader):
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                valid_losses.append(loss.item())

        # print training/validation statisctics
        # calculte average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        end_time = time.time()
        dt = end_time - start_time

        msg = f"[{epoch + 1:2d}/{epochs}| {dt=:4.2f}s], train_loss: {train_loss:.5f}, valid_loss {valid_loss:.5f}"
        print(msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decrease
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    fp = dbmt / f"checkpoint_{task}.pth"
    model.load_state_dict(torch.load(fp))

    return model, avg_train_losses, avg_valid_losses


# %% [markdown]
# ### Load data

# %%
# Generated with get_seqs_from_hg38.R based on Enhancer_activity.txt
usecols = [
    "Chr",
    "Start",
    "End",
    "NSC_log2_enrichment",
    # "ESC_log2_enrichment",
    "Seq",
    # "SeqRevComp",
]
dfe = pd.read_csv(dbmrd / "Enhancer_activity_w_seq.csv.gz", usecols=usecols)

# %% [markdown]
# #### DNA sequences: encode and pad

# %%
for c in ("Seq",):
    dfe[f"{c}Enc"] = dfe[c].map(one_hot_encode).map(pad_arr)


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# #### DNA shape data

# %% [raw]
# data = np.load(dbmrd / "shapes.npz")
# list(data.keys())

# %% [raw]
# seqs = np.stack(
#     list(np.nan_to_num(data[c]).astype(np.float32) for c in data.keys()),
#     axis=2,
# )
# # pad to 1024 (12, 1000, 12)
# seqs = np.pad(seqs, ((0, 0), (12, 12), (0, 0)), mode="constant", constant_values=0.0)
# dfe["SeqShapes"] = list(seqs)
# print(seqs.shape)
# del seqs, data

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# #### Coupled DNA encoded sequence & DNA shape

# %% [raw]
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# seqs = np.stack(
#     list(
#         scaler.fit_transform(np.nan_to_num(data[c]).astype(np.float32))
#         for c in data.keys()
#     ),
#     axis=2,
# )
# # pad to 1024 (12, 1000, 12)
# seqs = np.pad(seqs, ((0, 0), (12, 12), (0, 0)), mode="constant", constant_values=0.0)
# seqs.shape

# %% [raw]
# seq_enc = np.stack(list(dfe.SeqEnc))
# seq_enc.shape

# %% [raw]
# dfe["SeqAndShapes"] = list(np.concatenate((seq_enc, seqs), axis=2))
# del dfe["Seq"]
# del dfe["SeqEnc"]
# del seqs, seq_enc, data

# %% [markdown]
# ### Split data for training

# %%
def bins(s):
    # Can't do less bins AND have enough elements in each bin
    # There are several "outliers" in the NCREs activity,
    # take another log2 to compact the data further for assigning bins
    return pd.cut(np.log2(s + 1), bins=8, labels=False)


# Global sample for quick tests
_, df = train_test_split(
    dfe,
    test_size=0.20,
    random_state=random_state,
    stratify=bins(dfe.NSC_log2_enrichment),
)

# df = dfe

# Split
df_train, df = train_test_split(
    df,
    test_size=0.10,
    random_state=random_state,
    stratify=bins(df.NSC_log2_enrichment),
)

df_val, df_test = train_test_split(
    df,
    test_size=0.50,
    random_state=random_state,
    stratify=bins(df.NSC_log2_enrichment),
)

del df

# %% [markdown]
# ### Run the pipline

# %%
# Tests on macOS M1 Pro
# batch_size = 32  # 720s+/epoch
# batch_size = 128  # 500s/epoch
batch_size = 256  # 454s/epoch, uses more RAM
# batch_size = 512  # 455s/epoch, just uses more RAM

epochs = 100
patience = 20

task = "NSC"

# col_x = "SeqEnc"
# col_x = "SeqShapes"
# col_x = "SeqAndShapes"
col_y = f"{task}_log2_enrichment"

# create data
train_dataloader = create_dataloader(
    x=df_train[col_x],
    y=df_train[col_y],
    batch_size=batch_size,
)
valid_dataloader = create_dataloader(
    x=df_val[col_x],
    y=df_val[col_y],
    batch_size=batch_size,
)

# train model
cnn_starr, train_loss, valid_loss = train_model(
    model=cnn_starr,
    loss_fn=loss_fn,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    batch_size=batch_size,
    patience=patience,
    epochs=epochs,
    task=task,
)

# %% [markdown]
# ### Function to visualiz the loss and the Early Stopping Checkpoint


# %%
def loss_fig(train_loss, valid_loss, task):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label="Validation Loss")

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle="--", color="r", label="Early Stopping Checkpoint")

    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.ylim(0, 1)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ### Functions to evaluate model performance
# mean squared error (MSE), Person (PCC) and Spearman (SCC) correlation coefficients


# %%
def summmary_statistic(set_name, dataloader, main_model, task):
    # initialize loss
    test_loss = 0.0
    whole_preds = []
    whole_preds = []
    whole_targets = []
    whole_targets = []

    main_model.eval()  # pre model for evaluation
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = main_model(inputs)

            # return the whole dataset of targets and preds
            whole_targets.append(targets)
            whole_preds.append(outputs)

    whole_targets = torch.cat(whole_targets, dim=0)
    whole_targets = whole_targets.squeeze().cpu().detach().numpy()

    whole_preds = torch.cat(whole_preds, dim=0)
    whole_preds = whole_preds.cpu().detach().numpy()

    MSE = mean_squared_error(whole_targets, whole_preds.squeeze(1))
    pearon = stats.pearsonr(whole_targets, whole_preds.squeeze(1))
    spearman = stats.spearmanr(whole_targets, whole_preds.squeeze(1))

    # summary data for NSC

    print("---------Summary on {} data---------".format(set_name))
    print("the MSE of {}: {:.5f}".format(task, MSE))
    print("the pearson r of {}: {:.5f}".format(task, pearon[0]))
    print("the spearman r of {}: {:.5f}".format(task, spearman[0]))

    # plot correlation
    df = pd.DataFrame({"targets": whole_targets, "preds": whole_preds.squeeze(1)})
    return df


# %%
# plot Train and Valid loss
loss_fig(train_loss, valid_loss, task)

# %% [raw]
# test_dataloader = create_dataset(df_test, batch_size, task)
#
# # summary statistics
# # cnn_starr.load_state_dict(torch.load("/data/scratch/rdeng/enhancer_project/model/checkpoint_NSC_212697.2D.pth".format(task), map_location=torch.device('cpu')))
# summmary_statistic("Train", train_dataloader, cnn_starr, task)
# summmary_statistic("Valid", valid_dataloader, cnn_starr, task)
# summmary_statistic("Test", test_dataloader, cnn_starr, task)
