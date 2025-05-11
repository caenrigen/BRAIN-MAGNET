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

df_sample = df_enrichment.sample(100)
df_sample.head()

# %% [markdown]
# # Calculate contribution score, GradientShap with gradient correction
#

# %%
### IMPORTANT MESSAGE!!! In order to run the code on the GPU, I changed the "/trinity/home/rdeng/enhancer/lib/python3.7/site-packages/torch/_tensor.py"
### return self.numpy() -> return self.cpu().detach().numpy()
### return self.numpy().astype(dtype, copy=False) -> return self.cpu().detach().numpy().astype(dtype, copy=False)

# %%
# https://github.com/kundajelab/shap/commit/29d2ffab405619340419fc848de6b53e2ef0f00c
import shap

# https://github.com/kundajelab/deeplift/commit/0201a218965a263b9dd353099feacbb6f6db0051
import deeplift

# from importlib import reload
# reload(shap.explainers.deep)
# reload(shap.explainers.deep.deep_pytorch)
# reload(shap.explainers)
# reload(shap)

import deeplift.dinuc_shuffle as ds

# %%
shap.DeepExplainer


# %%
def standard_combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = [
        (mult[l] * (orig_inp[l] - bg_data[l])).mean(0) for l in range(len(orig_inp))
    ]
    return to_return


# %%
seq = df_sample.Seq.iloc[0]

rng = RandomState(random_state)
ds.dinuc_shuffle(seq, num_shufs=None, rng=rng)

# %%
deeplift.dinuc_shuffle


# %%
# This generates 100 dinuc-shuffled references per sequence
# In my (Avanti Shrikumar) experience,
# 100 references per sequence is on the high side; around 10 work well in practice
# I am using 100 references here just for demonstration purposes.
# Note that when an input of None is supplied, the function returns a tensor
# that has the same dimensions as actual input batches
def shuffle_several_times(inp):
    # I am assuming len(inp) == 1 because this function is designed for models with one
    # input mode (i.e. just sequence as the input mode)
    assert (inp is None) or len(inp) == 1
    if inp is None:
        return torch.tensor(np.zeros((1, 4, 1, 1000)).astype("float32")).to(device)
    else:
        # Some reshaping/transposing needs to be performed before calling
        # onehot_dinuc_shuffle becuase the input to the DeepSEA model
        # is in the format (4 x 1 x length) for each sequence, whereas
        # onehot_dinuc_shuffle expects (length x 4)
        to_return = torch.tensor(
            np.array(
                [
                    onehot_dinuc_shuffle(
                        inp[0].detach().cpu().numpy().squeeze().transpose(1, 0)
                    ).transpose((1, 0))[:, None, :]
                    for i in range(100)
                ]
            ).astype("float32")
        ).to(device)
        return to_return


# This combine_mult_and_diffref function can be used to generate hypothetical
# importance scores for one-hot encoded sequence.
# Hypothetical scores can be thought of as quick estimates of what the
# contribution *would have been* if a different base were present. Hypothetical
# scores are used as input to the importance score clustering algorithm
# TF-MoDISco (https://github.com/kundajelab/tfmodisco)
# Hypothetical importance scores are discussed more in this pull request:
#  https://github.com/kundajelab/deeplift/pull/36
def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []
    # Perform some reshaping/transposing because the code was designed
    # for inputs that are in the format (length x 4), whereas the DeepSEA
    # model has inputs in the format (4 x 1 x length)
    mult = [x.squeeze().transpose((0, 2, 1)) for x in mult]
    orig_inp = [x.squeeze().transpose((1, 0)) for x in orig_inp]
    bg_data = [x.squeeze().transpose((0, 2, 1)) for x in bg_data]
    for l in range(len(mult)):
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
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape) == 2, orig_inp[l].shape
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:, i] = 1.0
            hypothetical_difference_from_reference = (
                hypothetical_input[None, :, :] - bg_data[l]
            )
            hypothetical_contribs = hypothetical_difference_from_reference * mult[l]
            projected_hypothetical_contribs[:, :, i] = np.sum(
                hypothetical_contribs, axis=-1
            )
        to_return.append(
            torch.tensor(
                np.mean(projected_hypothetical_contribs, axis=0).transpose((1, 0))[
                    :, None, :
                ]
            ).to(device)
        )
    return to_return


def load_model(task):
    if task == "NSC":
        # cnn_starr.load_state_dict(torch.load("/data/scratch/rdeng/enhancer_project/model/checkpoint_NSC_212697.2D.pth", map_location=torch.device('cpu')))
        cnn_starr.load_state_dict(
            torch.load(
                "/data/scratch/rdeng/enhancer_project/model/checkpoint_NSC_212697.2D.pth"
            )
        )
    elif task == "ESC":
        #         cnn_starr.load_state_dict(torch.load("/data/scratch/rdeng/enhancer_project/model/checkpoint_ESC_212696.2D.pth", map_location=torch.device('cpu')))
        cnn_starr.load_state_dict(
            torch.load(
                "/data/scratch/rdeng/enhancer_project/model/checkpoint_ESC_212696.2D.pth"
            )
        )
    else:
        print("Please provide correct cell type")
    return cnn_starr


# calculate contribution score, and save it as npy for TF-modisco
def contri_score(dataloader, task):
    whole_inputs = []
    whole_shap_explanations = []

    for batch, data in enumerate(dataloader):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        # calculate shap
        e = shap.DeepExplainer(
            (cnn_starr),
            shuffle_several_times,
            combine_mult_and_diffref=combine_mult_and_diffref,
        )

        shap_explanations = e.shap_values(inputs)

        # shap_explanations = shap_explanations.cpu().detach().numpy()

        # process gradients with gradient correction (Majdandzic et al. 2022)
        inputs = inputs.cpu().detach().numpy()

        whole_inputs.append(inputs)
        whole_shap_explanations.append(shap_explanations)

    whole_inputs = np.concatenate(whole_inputs, axis=0)
    whole_inputs = whole_inputs.squeeze()  # remove additional dimention for TFmodisco, (Batch, 4, 1, 1000)->(Batch, 4, 1000)
    whole_shap_explanations = np.concatenate(whole_shap_explanations, axis=0)
    whole_shap_explanations = whole_shap_explanations.squeeze()  # remove additional dimention for TFmodisco, (Batch, 4, 1, 1000)->(Batch, 4, 1000)

    np.save(
        "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/shap_explanations_"
        + task
        + ".npy",
        whole_shap_explanations,
    )
    np.save(
        "/data/scratch/rdeng/enhancer_project/ipython_notebooks/2D/contri_score/inp_"
        + task
        + ".npy",
        whole_inputs,
    )


# %%
# calculate contribution score

# NSC
task = "NSC"
test_dataloader_NSC = create_dataset(5, task=task)
load_model(task)
contri_score(test_dataloader_NSC, task=task)
test = contri_score(test_dataloader_NSC, task=task)

# ESC
# task = "ESC"
# test_dataloader_ESC = create_dataset(5, task = task)
# load_model(task)
# contri_score(test_dataloader_ESC, task = task)

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
        #         first_nonzero, last_nonzero = 0, 1000 # IMPORTANT: this won't trim zeros
        num_unpadded = inps[num, :, first_nonzero:last_nonzero]
        num_unpadded = num_unpadded.transpose(
            1, 0
        )  # convert [4,length] -> [length, 4] for the TFmodisco
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
