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

random_state = 913
random.seed(random_state)

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
)
from torchsummary import summary
from torchmetrics import SpearmanCorrCoef, PearsonCorrCoef
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %%
torch.cuda.is_available(), torch.backends.mps.is_available()

# %%
# device = torch.device("cuda")
device = torch.device("cpu")
# device = torch.device("mps") # for macOS >=14.0
device


# %% [raw]
# # import sys
# # sys.path.append("/data/scratch/rdeng/enhancer_project/ipython_notebooks/")
# from helper import IOHelper, SequenceHelper, utils
# # np.set_printoptions(threshold=sys.maxsize)

# %% [markdown]
# ### Funtions to load data

# %%
def to_uint8(string):
    return np.frombuffer(string.encode("ascii"), dtype=np.uint8)


# function to convert only one piece of sequence to np.array
def make_one_hot_encode(alphabet: str = "ACGT", dtype=np.int8) -> np.ndarray:
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


def pad_post_one_hot(snippet: np.ndarray, arr_pre: int = 1000):
    assert len(snippet) <= arr_pre, len(snippet)
    arr_len = len(snippet)
    pad = (arr_pre - arr_len) / 2
    return np.pad(snippet, [(int(pad), math.ceil(pad)), (0, 0)], mode="constant")


seq = "ACGTATCnotdna"
to_uint8(seq), one_hot_encode(seq)

# %% [markdown]
# ### Load sequences data

# %%
from pathlib import Path

dbm = Path("/Volumes/Famafia/brain-magnet/")
dbmrd = dbm / "rd_APP_data"
dbmt = dbm / "train"

# %%
# Generated with get_seqs_from_hg38.R based on Enhancer_activity.txt
dfe = pd.read_csv(dbmrd / "Enhancer_activity_w_seq_aug.txt.gz")
dfe


# %% [markdown]
# ### Convert data to tensors

# %%
# function to generate all sequences to np.array (one-hot encoding matrix)
def generate_sequence_matrix(sequences):
    """
    After the function of one_hot_encode,
    convert the whole dataset to one_hot encoding matrix
    """
    sequence_matrix = []
    for seq in sequences:
        snippet = one_hot_encode(seq)
        sequence_matrix.append(pad_post_one_hot(snippet))
    return np.array(sequence_matrix)


def prepare_tensor(X, Y):
    """
    convert np.array to tensor
    """
    # convert input: [batch, 1000, 4] -> [batch, 4, 1, 1000]
    tensor_x = torch.Tensor(X).permute(0, 2, 1).unsqueeze(2)
    # add one dimension in targets: [batch] -> [batch, 1]
    tensor_y = torch.Tensor(Y).unsqueeze(1)
    return TensorDataset(tensor_x, tensor_y)


def create_dataset(df: pd.DataFrame, batch_size: int, task: str):
    """
    Load sequences and enhancer activity,
    convert to tensor type as the input of model
    """
    assert task in ("NSC", "ESC"), task
    Y = df[f"{task}_log2_enrichment"].values

    # Convert each sequence to one-hot encoding
    seq_matrix = generate_sequence_matrix(df.Seq.values)

    # Replace NaN with zero and infinity with large finite numbers
    X = np.nan_to_num(seq_matrix)
    X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    tensor_data = prepare_tensor(X_reshaped, Y)

    return DataLoader(tensor_data, batch_size)


# %% [markdown]
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
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
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
    """
    CNN model: 4 convolution layers followed with two linear layers.
    forward two heads.
    """

    def __init__(self):
        super(CNN_STARR, self).__init__()
        self.model = Sequential(
            Conv2d(4, 128, (1, 11), padding="same"),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d((1, 2), (1, 2)),
            Conv2d(128, 256, (1, 9), padding="same"),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d((1, 2), (1, 2)),
            Conv2d(256, 512, (1, 7), padding="same"),
            BatchNorm2d(512),
            ReLU(),
            MaxPool2d((1, 2), (1, 2)),
            #             Conv1d(512, 1024, 3, padding=1),
            #             BatchNorm1d(1024),
            #             ReLU(),
            #             MaxPool1d(2),
            Flatten(),
            Linear(64000, 1024),
            BatchNorm1d(1024),
            ReLU(),
            Dropout(0.4),
            Linear(1024, 1024),
            BatchNorm1d(1024),
            ReLU(),
            Dropout(0.4),
            Linear(1024, 1),
        )

    def forward(self, x):
        return self.model(x)


cnn_starr = CNN_STARR()
cnn_starr.to(device)

print(cnn_starr)
summary(cnn_starr, input_size=(4, 1, 1000), batch_size=128)


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
    start_time = time.time()
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
# ### Evaluating the model
# #### Model performace: mean squared error (MSE), Person (PCC) and Spearman (SCC) correlation coefficients


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


# %% [markdown]
# ### Split data from training

# %%
# Small sample for testing code
df = dfe.sample(n=4_000, random_state=random_state)

df_train, df_tmp = train_test_split(df, test_size=0.10, random_state=random_state)
df_val, df_test = train_test_split(df_tmp, test_size=0.50, random_state=random_state)
del df_tmp

# %%
df_train

# %% [markdown]
# ### Run the pipline

# %%
batch_size = 128
epochs = 100
patience = 10

task = "NSC"

# create data
train_dataloader = create_dataset(df_train, batch_size, task)
valid_dataloader = create_dataset(df_val, batch_size, task)
test_dataloader = create_dataset(df_test, batch_size, task)

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

# %%
# plot Train and Valid loss
loss_fig(train_loss, valid_loss, task)

# summary statistics
# cnn_starr.load_state_dict(torch.load("/data/scratch/rdeng/enhancer_project/model/checkpoint_NSC_212697.2D.pth".format(task), map_location=torch.device('cpu')))
summmary_statistic("Train", train_dataloader, cnn_starr, task)
summmary_statistic("Valid", valid_dataloader, cnn_starr, task)
summmary_statistic("Test", test_dataloader, cnn_starr, task)


# %%
def combine_df(cell, set):
    file_seq = str(
        "/data/scratch/rdeng/enhancer_project/data/train_set/Sequences_" + set + ".fa"
    )
    input_fasta = IOHelper.get_fastas_from_file(file_seq, uppercase=True)

    targets = np.load(
        "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/preds_targets/targets_"
        + cell
        + "_"
        + set
        + ".npy"
    )
    preds = np.load(
        "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/preds_targets/preds_"
        + cell
        + "_"
        + set
        + ".npy"
    )
    DF_targets = pd.DataFrame(targets)
    DF_preds = pd.DataFrame(preds)

    DF_merged = pd.concat([input_fasta, DF_targets, DF_preds], axis=1)
    DF_merged.columns = ["location", "sequence", "targets", "preds"]

    DF_merged = DF_merged[~DF_merged.location.str.contains("Reversed")]

    return DF_merged


# %%
test_nsc = combine_df("ESC", "Test")
valid_nsc = combine_df("ESC", "Valid")
train_nsc = combine_df("ESC", "Train")
whole_nsc = pd.concat([test_nsc, valid_nsc, train_nsc], axis=0)

whole_nsc = whole_nsc[["location", "targets", "preds"]]
whole_nsc.to_csv(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/preds_targets/whole_ESC.csv",
    index=False,
)
# whole_nsc[['location','start']] = whole_nsc['location'].str.split(':',expand=True)
# whole_nsc[['start','end']] = whole_nsc['start'].str.split('-',expand=True)


# whole_nsc = whole_nsc.sort_values(by=['location', 'start'], ascending=[True, True])
# whole_nsc = whole_nsc[['location', 'start', 'end', 'targets', 'preds']]
