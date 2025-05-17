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
#     display_name: g
#     language: python
#     name: python3
# ---

# %%
# Evaluate the python files within the notebook namespace
# %run -i auxiliar.py
# %run -i cnn_starr.py
# %run -i data_module.py

# %%
import os
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
from typing import Optional
from functools import partial
import gc

import lightning as L
import torch
from torch import nn
import gc
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.loggers import TensorBoardLogger
from random import randbytes
from numpy.random import RandomState

# %%
random_state = 913
# random.seed(random_state)

dbmc = Path("/Users/victor/Documents/ProjectsDev/genomics/BRAIN-MAGNET")
dbm = Path("/Volumes/Famafia/brain-magnet")

# dbmc = Path("/Users/victor/sshpyk_code")
# dbm = Path("/Users/victor/sshpyk_data/")

dbmce = str(dbmc / "examples")
dbmrd = dbm / "rd_APP_data"
dbmt = dbm / "train"
os.chdir(dbmce)

# %%
print(torch.cuda.is_available(), torch.backends.mps.is_available())
# device = torch.device("cuda")
# device = torch.device("cpu")
device = torch.device("mps")
device

# %%
task = "ESC"
# task = "NSC"
df_enrichment = load_enrichment_data(
    fp=dbmrd / "Enhancer_activity_w_seq.csv.gz",
    y_col=f"{task}_log2_enrichment",
)
df_enrichment.head()

# %%
df_enrichment["SeqLen"] = df_enrichment.Seq.str.len()

# %%
df_sample = df_enrichment.loc[OUTLIER_INDICES].copy()
df_sample_1000 = df_sample[df_sample.SeqLen == 1000].copy()
df_sample

# %% [markdown]
# # Scratch pad
#

# %%
df_enrichment.shape

# %%
(df_enrichment.Seq.str.len() < 980).sum()

# %%
seq = df_sample.SeqEnc.iloc[-1]

rng = RandomState(random_state)
_ = ds.dinuc_shuffle(seq, num_shufs=30, rng=rng)


# %%
# WIP
def make_shuffled_seqs(
    input_seq_tensor: torch.Tensor,
    num_shufs: int = 30,
    rng: RandomState = RandomState(random_state),
    device: torch.device = device,
):
    print(input_seq_tensor.shape)
    raise RuntimeError("stop here")
    one_hot = input_seq_tensor.squeeze().permute(1, 0).cpu().numpy()
    shuffled_seqs = ds.dinuc_shuffle(
        unpad_one_hot(one_hot), num_shufs=num_shufs, rng=rng
    )
    shuffled_seqs_padded = np.asarray(
        [pad_one_hot(seq) for seq in shuffled_seqs], dtype=np.float32
    )
    return torch.tensor(shuffled_seqs_padded).permute(0, 2, 1).to(device)



# %% [markdown]
# # Imports
#

# %%
# https://github.com/kundajelab/shap/commit/29d2ffab405619340419fc848de6b53e2ef0f00c
# My fork fixes an issue for data that is on GPU/MPS
# https://github.com/caenrigen/shap/commit/0db4abbc916688f1d937ca1f62003e4a149ba0df
import shap

# https://github.com/kundajelab/deeplift/commit/0201a218965a263b9dd353099feacbb6f6db0051
import deeplift

from importlib import reload

reload(shap)
reload(shap.explainers)
reload(shap.explainers.deep)
reload(shap.explainers.deep.deep_pytorch)


import deeplift.dinuc_shuffle as ds
import dinuc_shuffle_v0_6_11_0 as ds0611

# %% [markdown]
# # Calculate contribution score, GradientShap with gradient correction
#

# %%
dataset = make_tensor_dataset(
    df=df_sample_1000, x_col="SeqEnc", y_col=f"{task}_log2_enrichment"
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataloader

# %%
from typing import List


# this function performs a dinucleotide shuffle of a one-hot encoded sequence
# It expects the supplied input in the format (length x 4)
def onehot_dinuc_shuffle(s):
    s = np.squeeze(s)
    assert len(s.shape) == 2
    assert s.shape[1] == 4
    # The argmax identifies the index that == 1 in the one-hot encoded sequence
    argmax_vals = "".join([str(x) for x in np.argmax(s, axis=-1)])
    # shuffled_argmax_vals = [
    #     int(x)
    #     for x in ds0611.traverse_edges(
    #         argmax_vals, ds0611.shuffle_edges(ds0611.prepare_edges(argmax_vals))
    #     )
    # ]
    shuffled_argmax_vals = [int(x) for x in ds0611.dinuc_shuffle(argmax_vals)]
    to_return = np.zeros_like(s).astype(np.float32)
    to_return[list(range(len(s))), shuffled_argmax_vals] = 1
    return to_return


def tensor_to_onehot(t):
    return t.detach().squeeze().transpose(1, 0).cpu().numpy()


def onehot_to_tensor_shape(one_hot: np.ndarray):
    return one_hot.transpose(1, 0)[:, None, :]


# This generates 100 dinuc-shuffled references per sequence
# In my (Avanti Shrikumar) experience,
# 100 references per sequence is on the high side; around 10 work well in practice
# I am using 100 references here just for demonstration purposes.
# Note that when an input of None is supplied, the function returns a tensor
# that has the same dimensions as actual input batches
def shuffle_several_times(inp, num_shufs: int = 100, num_bp: int = 1000):
    # I am assuming len(inp) == 1 because this function is designed for models with one
    # input mode (i.e. just sequence as the input mode)
    assert (inp is None) or len(inp) == 1
    if inp is None:
        return torch.tensor(np.zeros((1, 4, 1, num_bp), dtype=np.float32)).to(device)
    else:
        tensor_1hot = inp[0]
        # Some reshaping/transposing needs to be performed before calling
        # onehot_dinuc_shuffle becuase the input to the DeepSEA model
        # is in the format (4 x 1 x length) for each sequence, whereas
        # onehot_dinuc_shuffle expects (length x 4)
        it = (
            onehot_to_tensor_shape(onehot_dinuc_shuffle(tensor_to_onehot(tensor_1hot)))
            for _ in range(num_shufs)
        )
        to_return = torch.tensor(np.array(list(it), dtype=np.float32)).to(device)
        return to_return


# This combine_mult_and_diffref function can be used to generate hypothetical
# importance scores for one-hot encoded sequence.
# Hypothetical scores can be thought of as quick estimates of what the
# contribution *would have been* if a different base were present. Hypothetical
# scores are used as input to the importance score clustering algorithm
# TF-MoDISco (https://github.com/kundajelab/tfmodisco)
# Hypothetical importance scores are discussed more in this pull request:
# https://github.com/kundajelab/deeplift/pull/36
def combine_mult_and_diffref(
    mult: List[np.ndarray],  # shape of the (only) element: (num_shufs, 4, 1, N)
    orig_inp: List[np.ndarray],  # shape of the (only) element: (4, 1, N)
    bg_data: List[np.ndarray],  # shape of the (only) element: (num_shufs, 4, 1, N)
):
    assert len(mult) == len(orig_inp) == len(bg_data) == 1
    to_return = []

    # Perform some reshaping/transposing because the code was designed
    # for inputs that are in the format (length x 4), whereas the DeepSEA
    # model has inputs in the format (4 x 1 x length)
    # List[(num_shufs, 4, 1, N)] -> List[(num_shufs, N, 4)]
    mult = [x.squeeze().transpose(0, 2, 1) for x in mult]
    # List[(num_shufs, 4, 1, N)] -> List[(num_shufs, N, 4)]
    bg_data = [x.squeeze().transpose(0, 2, 1) for x in bg_data]
    # List[(4, 1, N)] -> List[(N, 4)]
    orig_inp = [x.squeeze().transpose(1, 0) for x in orig_inp]

    for l_idx in range(len(mult)):
        # At each position in the input sequence, we iterate over the one-hot encoding
        # possibilities (eg: for genomic sequence, this is ACGT i.e.
        # 1000, 0100, 0010 and 0001) and compute the hypothetical
        # difference-from-reference in each case. We then multiply the hypothetical
        # differences-from-reference with the multipliers to get the hypothetical contributions.
        # For each of the one-hot encoding possibilities,
        # the hypothetical contributions are then summed across the ACGT axis to estimate
        # the total hypothetical contribution of each position. This per-position hypothetical
        # contribution is then assigned ("projected") onto whichever base was present in the
        # hypothetical sequence.
        # The reason this is a fast estimate of what the importance scores *would* look
        # like if different bases were present in the underlying sequence is that
        # the multipliers are computed once using the original sequence, and are not
        # computed again for each hypothetical sequence.

        len_one_hot, num_bp = 4, orig_inp[l_idx].shape[0]

        assert len(orig_inp[l_idx].shape) == 2, orig_inp[l_idx].shape
        assert orig_inp[l_idx].shape[-1] == len_one_hot, orig_inp[l_idx].shape

        # We don't need zeros, these will be overwritten
        projected_hyp_contribs = np.empty_like(bg_data[l_idx], dtype=np.float32)
        hyp_contribs = np.empty_like(bg_data[l_idx], dtype=np.float32)

        ident = np.eye(len_one_hot, dtype=np.float32)
        # Iterate over 4 hypothetical sequences, each made of the same base,
        # e.g. for idx_col_1hot == 0: "AAAA....AAAA" (but one hot encoded of course)
        for idx_col_1hot in range(len_one_hot):
            # ##########################################################################
            # These two lines allocate extra memory
            # // hyp_seq_1hot = np.zeros_like(orig_inp[l_idx], dtype=np.float32)
            # // hyp_seq_1hot[:, idx_col_1hot] = 1.0
            # This trick avoids memory allocation
            hyp_seq_1hot = np.broadcast_to(ident[idx_col_1hot], (num_bp, 4))
            # ##########################################################################

            # `hyp_seq_1hot[None, :, :]` shapes it such that it can match the
            # shape of `bg_data[l_idx]` that has the extra dimension of num_shufs.
            # It is only a view of the underlying memory, so it is efficient.
            np.subtract(hyp_seq_1hot[None, :, :], bg_data[l_idx], out=hyp_contribs)
            np.multiply(hyp_contribs, mult[l_idx], out=hyp_contribs)

            # Sum on the one-hot axis, save directly to `projected_hyp_contribs`.
            # The sum is to get the total hypothetical contribution (at that bp)
            hyp_contribs.sum(axis=-1, out=projected_hyp_contribs[:, :, idx_col_1hot])

        # Average on the num_shufs axis to arrive to the final hypothetical
        # contribution scores (at each bp).
        p_h_cbs_mean = onehot_to_tensor_shape(projected_hyp_contribs.mean(axis=0))
        to_return.append(torch.tensor(p_h_cbs_mean).to(device))
    return to_return


# calculate contribution score, and save it as npy for TF-modisco
def contri_score(dataloader, model_trained: CNNSTARR):
    whole_inputs = []
    whole_shap_explanations = []

    for batch, data in enumerate(dataloader):
        inputs, _targets = data
        inputs = inputs.to(device)
        # targets = targets.to(device) # not needed for shap

        # calculate shap
        e = shap.DeepExplainer(
            model=model_trained,
            data=shuffle_several_times,
            combine_mult_and_diffref=combine_mult_and_diffref,
        )

        shap_explanations = e.shap_values(inputs)

        # # ? what is this? commend in the wrong place? outdated?
        # process gradients with gradient correction (Majdandzic et al. 2022)
        inputs = inputs.detach().cpu().numpy()

        whole_inputs.append(inputs)
        whole_shap_explanations.append(shap_explanations)

    whole_inputs = np.concatenate(whole_inputs, axis=0)
    # remove additional dimention for TFmodisco, (Batch, 4, 1, 1000)->(Batch, 4, 1000)
    whole_inputs = whole_inputs.squeeze()

    whole_shap_explanations = np.concatenate(whole_shap_explanations, axis=0)
    # remove additional dimention for TFmodisco, (Batch, 4, 1, 1000)->(Batch, 4, 1000)
    whole_shap_explanations = whole_shap_explanations.squeeze()

    # np.save(
    #     "contri_score/shap_explanations_" + task + ".npy",
    #     whole_shap_explanations,
    # )
    # np.save(
    #     "contri_score/inp_" + task + ".npy",
    #     whole_inputs,
    # )
    return whole_inputs, whole_shap_explanations


# %%
version = "f9bd95fa"
fp = dbmt / f"starr_{task}" / version / "stats.pkl.bz2"
df_models = pd.read_pickle(fp)
fig, ax = plt.subplots(1, 1)
fold = 0
epoch = pick_checkpoint(df_models, fold=fold, ax=ax)
fold, epoch


# %%
dp_checkpoints = dbmt / f"starr_{task}" / version / f"fold_{fold}" / "epoch_checkpoints"
fp_model_checkpoint = list(dp_checkpoints.glob(f"{task}_ep{epoch:02d}*.pt"))[0]
fp_model_checkpoint


# %%
model_trained = load_model(
    fp=fp_model_checkpoint,
    device=device,
    forward_mode="main",
)
model_trained

# %%
inputs, shap_explanations = contri_score(dataloader, model_trained=model_trained)

# %%
inputs.shape, shap_explanations.shape


# %%
from deeplift.visualization import viz_sequence

for input_seq, hyp_imp_scores in zip(inputs, shap_explanations):
    start, end = 750, 1000
    segment = input_seq[:, start:end]
    hyp_imp_scores_segment = hyp_imp_scores[:, start:end]
    viz_sequence.plot_weights(hyp_imp_scores_segment, subticks_frequency=20)
    # * The actual importance scores can be computed using an element-wise product of
    # * the hypothetical importance scores and the actual importance scores
    viz_sequence.plot_weights(hyp_imp_scores_segment * segment, subticks_frequency=20)
    break


# %% [markdown]
# ## Percentile contribution score
#

# %%
# load the contribution scores and input sequences
NSC_contri = np.load(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/Whole/shap_explanations_NSC.npy"
)
NSC_inp = np.load(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/Whole/inp_NSC.npy"
)
NSC_actural = np.multiply(NSC_contri, NSC_inp)


# %%
# Read fasta files of Test dataset
file_seq = str("/data/scratch/rdeng/enhancer_project/data/Enhancer.fa")
input_fasta = IOHelper.get_fastas_from_file(file_seq, uppercase=True)
# Don't include augment fasta: Reversed
input_fasta = input_fasta[~input_fasta["location"].str.contains("Reversed")]


# %%
# Your genome coordinates
# coordinates = input_fasta.location[:20] # time-consuming, test 20 enhancers!!!
coordinates = input_fasta.location


# Function to extract start and end coordinates from a string
def extract_coordinates(coord_string):
    chr_, coordinates_str = coord_string.split(":")
    start, end = map(int, coordinates_str.split("-"))
    return chr_, start, end


# Create a DataFrame with the original coordinates
df = pd.DataFrame({"Enhancer": coordinates})

# Extract chromosome, start, and end coordinates
df[["Chr", "Start", "End"]] = df["Enhancer"].apply(
    lambda x: pd.Series(extract_coordinates(x))
)

# Add a new column with consecutive numbers for each range starting from start+1
df["Pos"] = df.apply(lambda row: list(range(row["Start"] + 1, row["End"] + 1)), axis=1)
df["Length"] = df["End"] - df["Start"]

###################
# Your numpy data
# numpy_data = NSC_actural[:20,::] # time-consuming, test 20 enhancers!!!
numpy_data = NSC_actural

# Lengths
# lengths = df.Length[:20] # time-consuming, test 20 enhancers!!!
lengths = df.Length

result = []

for i, length in enumerate(lengths):
    arr_pre = 1000
    arr_len = length
    arr_int = int((arr_pre - arr_len) / 2)
    arr_end = arr_int + length

    # Extract subarray based on arr_int and arr_ceil
    subarray = numpy_data[i, :, arr_int:arr_end]
    max_num = np.abs(np.sum(subarray, axis=0))

    result.append(max_num)
###################


df["Contri_score"] = result

# Explode the list of consecutive numbers into separate rows
df = df.explode(["Pos", "Contri_score"]).reset_index(drop=True)

# Reorder columns
df = df[["Chr", "Pos", "Enhancer", "Length", "Contri_score"]]
df

# Calculate percentiles for Contri_score
# Rank Contri_score with handling ties by using the 'min' method
df["Rank"] = df["Contri_score"].rank(method="min")

# Calculate percentiles
df["Percentile"] = df["Rank"] / len(df) * 100


df = df.sort_values(by=["Contri_score"])
df

# %%
df.to_csv("Whole_contri_percentile.txt", index=False, sep="\t")


# %%
# load SNP info


# %% [markdown]
# # Viz sequences
#

# %%
from deeplift.visualization import viz_sequence

# %% [markdown]
# ### visualize sequence with motifs of NSC
#

# %%
import numpy as np

NSC_top10 = 1.652084635
ESC_top10 = 1.820433242

# load preds and targets data
preds = np.load(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/preds_targets/preds_NSC_High.npy"
)

targets = np.load(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/preds_targets/targets_NSC_High.npy"
)

# Find indices of elements greater than NSC_top10 in both arrays
precise_idx = np.where(np.logical_and(preds > NSC_top10, targets > NSC_top10))[0]
# load contribution score from Test data
contri_scores = np.load(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/shap_explanations_NSC.npy"
)
inps = np.load(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/inp_NSC.npy"
)

# %%
import h5py

filename = "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/10000/out_NSC.h5"
f = h5py.File(filename, "r")

# in total three layers of the h5
print(list(f["pos_patterns"]))
print(list(f["pos_patterns"]["pattern_0"]))
print(list(f["pos_patterns"]["pattern_0"]["seqlets"]))

motif_idx = f["pos_patterns"]["pattern_1"]["seqlets"]["example_idx"][()]
selected_idx = np.intersect1d(precise_idx, motif_idx)
print("The selected idx for TP53 is {}".format(selected_idx))

motif_idx = f["pos_patterns"]["pattern_6"]["seqlets"]["example_idx"][()]
selected_idx = np.intersect1d(precise_idx, motif_idx)
print("The selected idx for YY2 is {}".format(selected_idx))

# %%
# example of YY2 (MA0748.2), 470:481
# example of TP53 (MA0106.3), 579:600
# the hypothetical_contribs
mod_viz = contri_scores[13776]
viz_sequence.plot_weights(mod_viz[:, :], subticks_frequency=20)

# %%
# the actual contribs_scores
mod_viz = np.multiply(contri_scores[13776], inps[13776])
viz_sequence.plot_weights(
    mod_viz[:, :], subticks_frequency=100, highlight={"red": ([470, 481], [579, 596])}
)

# %%
# chr8:123479279-123480279
print(preds[13776])
print(targets[13776])

# %% [markdown]
# ### visualize sequence with motifs of ESC
#

# %%
import numpy as np

NSC_top10 = 1.652084635
ESC_top10 = 1.820433242

# load preds and targets data
preds = np.load(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/preds_targets/preds_ESC_High.npy"
)

targets = np.load(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/preds_targets/targets_ESC_High.npy"
)

# Find indices of elements greater than NSC_top10 in both arrays
precise_idx = np.where(np.logical_and(preds > ESC_top10, targets > ESC_top10))[0]
# load contribution score from Test data
contri_scores = np.load(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/shap_explanations_ESC.npy"
)
inps = np.load(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/inp_ESC.npy"
)

# %%
import h5py

filename = "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/10000/out_ESC.h5"
f = h5py.File(filename, "r")

motif_idx = f["neg_patterns"]["pattern_0"]["seqlets"]["example_idx"][()]
selected_idx1 = np.intersect1d(precise_idx, motif_idx)
print("The selected idx for ZKSCAN5 is {}".format(selected_idx1))

motif_idx = f["neg_patterns"]["pattern_2"]["seqlets"]["example_idx"][()]
selected_idx2 = np.intersect1d(precise_idx, motif_idx)
print("The selected idx for Pou5f1::Sox2 is {}".format(selected_idx2))

motif_idx = f["neg_patterns"]["pattern_7"]["seqlets"]["example_idx"][()]
selected_idx3 = np.intersect1d(precise_idx, motif_idx)
print("The selected idx for Pou5f1::Sox2 is {}".format(selected_idx3))
np.intersect1d(selected_idx1, selected_idx3)

# I visualized 6511

# %%
# chr2:121490069-121490672
print(preds[6511])
print(targets[6511])

# %%
High_table_NSC = pd.read_csv(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/High_table_NSC.txt",
    sep="\t",
)
print(High_table_NSC.iloc[13776, 0])
print(len(High_table_NSC.iloc[13776, 1]))

High_table_ESC = pd.read_csv(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/High_table_ESC.txt",
    sep="\t",
)
print(High_table_ESC.iloc[6511, 0])
print(len(High_table_ESC.iloc[6511, 1]))


# %% [markdown]
# ### example_idx to extract the genomic coordinates for closest gene expression and epigenome data
#

# %%
filename = "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/10000/out_NSC.h5"
f = h5py.File(filename, "r")

patterns = f["pos_patterns"]
pattern_names = list(patterns.keys())

data = []
for pattern_name in pattern_names:
    seqlets = patterns[pattern_name]["seqlets"]
    example_idx = seqlets["example_idx"][()]
    for idx in example_idx:
        data.append((pattern_name, idx))

df = pd.DataFrame(data, columns=["pattern_name", "example_idx"])
High_table_NSC = pd.read_csv(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/High_table_NSC.txt",
    sep="\t",
)


# %%
import h5py
import pandas as pd

filename = "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/10000/out_NSC.h5"
f = h5py.File(filename, "r")

data = []

for pattern_type in ["pos_patterns", "neg_patterns"]:
    patterns = f[pattern_type]
    pattern_names = list(patterns.keys())
    for pattern_name in pattern_names:
        seqlets = patterns[pattern_name]["seqlets"]
        example_idx = seqlets["example_idx"][()]
        for idx in example_idx:
            data.append((pattern_type, pattern_name, idx))

df = pd.DataFrame(data, columns=["pattern_type", "pattern_name", "example_idx"])
High_table_NSC = pd.read_csv(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/High_table_NSC.txt",
    sep="\t",
)

# merge example_idx with index of genomic coordinates
merged_df = pd.merge(df, High_table_NSC, left_on="example_idx", right_index=True)
merged_df[["pattern_type", "pattern_name", "location"]].to_csv(
    f"/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/patterns_coordinates_NSC.txt",
    index=False,
    header=True,
    sep="\t",
)


# %% [markdown]
# ### Trial trim_zero after getting the contribution scores
#

# %%
contri_scores = np.load(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/shap_explanations_NSC.npy"
)
inps = np.load(
    "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/inp_NSC.npy"
)


# %%
def padded_idx(inps):
    """
    Return the index of padded zero of DNA sequences
    inps:  onehot coded matrix: [num, 4, length]
    """
    inp_len = inps.shape[2]
    first_nonzero = []
    for num in range(inps.shape[0]):
        for i in range(inps[num].shape[1]):
            if inps[num][:, : i + 1].sum() != 0:
                break
        first_nonzero.append(i)

    last_nonzero = []
    for num in range(inps.shape[0]):
        if inps[num][:, inp_len - 1 :].sum() != 0:
            last_nonzero.append(inp_len)
        else:
            for i in range(inps[num].shape[1]):
                if inps[num][:, -i - 1 : -1].sum() != 0:
                    break
            last_nonzero.append(inp_len - i)
    return zip(first_nonzero, last_nonzero)


def unpadded(idx_nonzero, inps):
    """
    Return unpadded arrary based on the padded_idx function
    idx_zero: the resulr from padded_index, zip(first_nonzero, last_nonzero)
    inps:  onehot coded matrix: [num, 4, length]

    """
    non_padded = []
    for num, (first_nonzero, last_nonzero) in enumerate(idx_nonzero):
        # first_nonzero, last_nonzero = 0, 1000 # IMPORTANT: this won't trim zeros
        num_unpadded = inps[num, :, first_nonzero:last_nonzero]
        # convert [4,length] -> [length, 4] for the TFmodisco
        num_unpadded = num_unpadded.transpose(1, 0)
        non_padded.append(num_unpadded.astype("float32"))
    return non_padded


inps_unpad = unpadded(padded_idx(inps), inps)
hypoth_unpad = unpadded(padded_idx(inps), contri_scores)

# create an empty list to store the result
contri_unpad = []
# iterate through each array in the lists and multiply the elements
for i in range(len(hypoth_unpad)):
    temp = np.multiply(hypoth_unpad[i], inps_unpad[i])
    contri_unpad.append(temp)

# from collections import OrderedDict

# hypoth_unpad = OrderedDict([('task0', hypoth_unpad)])
# contri_unpad = OrderedDict([('task0', contri_unpad)])

# %%
import h5py
import numpy as np

# %matplotlib inline
import modisco

# Uncomment to refresh modules for when tweaking code during development:
from importlib import reload

reload(modisco.util)
reload(modisco.pattern_filterer)
reload(modisco.aggregator)
reload(modisco.core)
reload(modisco.seqlet_embedding.advanced_gapped_kmer)
reload(modisco.affinitymat.transformers)
reload(modisco.affinitymat.core)
reload(modisco.affinitymat)
reload(modisco.cluster.core)
reload(modisco.cluster)
reload(modisco.tfmodisco_workflow.seqlets_to_patterns)
reload(modisco.tfmodisco_workflow)
reload(modisco)

null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=5000)
tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
    # Slight modifications from the default settings
    sliding_window_size=15,
    flank_size=5,
    target_seqlet_fdr=0.15,
    seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
        # Note: as of version 0.5.6.0, it's possible to use the results of a motif discovery
        # software like MEME to improve the TF-MoDISco clustering. To use the meme-based
        # initialization, you would specify the initclusterer_factory as shown in the
        # commented-out code below:
        initclusterer_factory=modisco.clusterinit.memeinit.MemeInitClustererFactory(
            meme_command="meme",
            base_outdir="/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/High/Note_Modisco/",
            max_num_seqlets_to_use=10000,
            nmotifs=10,
            n_jobs=1,
        ),
        trim_to_window_size=21,
        initial_flank_to_add=10,
        final_flank_to_add=10,
        final_min_cluster_size=60,
        # use_pynnd=True can be used for faster nn comp at coarse grained step
        # (it will use pynndescent), but note that pynndescent may crash
        # use_pynnd=True,
        n_cores=10,
    ),
)(
    task_names=["task0"],  # , "task1", "task2"],
    contrib_scores=contri_unpad,
    hypothetical_contribs=hypoth_unpad,
    one_hot=inps_unpad,
    null_per_pos_scores=null_per_pos_scores,
)
