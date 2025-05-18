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
from importlib import reload
import utils as ut
import cnn_starr as cnn
import data_module as dm
import motif_discovery as md

_ = reload(ut)
_ = reload(cnn)
_ = reload(dm)
_ = reload(md)

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
from pathlib import Path
import torch

from deeplift.visualization import viz_sequence

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
df_enrichment = dm.load_enrichment_data(
    fp=dbmrd / "Enhancer_activity_w_seq.csv.gz",
    y_col=f"{task}_log2_enrichment",
)
df_enrichment.head()

# %%
df_enrichment["SeqLen"] = df_enrichment.Seq.str.len()

# %%
df_sample = df_enrichment.loc[dm.OUTLIER_INDICES].sort_values(
    by="SeqLen", ascending=False
)
df_sample_1000 = df_sample[df_sample.SeqLen == 1000]
df_sample

# %% [markdown]
# # Calculate contribution score
#

# %%
dataset = dm.make_tensor_dataset(
    df=df_sample_1000, x_col="SeqEnc", y_col=f"{task}_log2_enrichment"
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataloader

# %%
# version, fold = "f9bd95fa", 0  # Conv2D
version, fold = "cc0e922b", 0  # Conv1D
fp = dbmt / f"starr_{task}" / version / "stats.pkl.bz2"
df_models = pd.read_pickle(fp)
fig, ax = plt.subplots(1, 1)
epoch = dm.pick_checkpoint(df_models, fold=fold, ax=ax)
fold, epoch

# %%
dp_checkpoints = dbmt / f"starr_{task}" / version / f"fold_{fold}" / "epoch_checkpoints"
fp_model_checkpoint = list(dp_checkpoints.glob(f"{task}_ep{epoch:02d}*.pt"))[0]
fp_model_checkpoint


# %%
model_trained = cnn.load_model(
    fp=fp_model_checkpoint,
    device=device,
    forward_mode="main",
)

# %%
reload(md)
inputs, shap_vals = md.calc_contrib_scores(
    dataloader, model_trained=model_trained, device=device
)

# %%
# dp = dbm / "cb_tmp" / "shap_vals_conv2d.npz"
dp = dbm / "cb_tmp" / "shap_vals_conv1d.npz"
np.savez_compressed(dp, inputs=inputs, shap_vals=shap_vals)

# %%
inputs.shape, shap_vals.shape


# %%
def plot_weights(inputs, shap_vals, start: int, end: int):
    segment = inputs[:, start:end]
    hyp_imp_scores_segment = shap_vals[:, start:end]
    # viz_sequence.plot_weights(hyp_imp_scores_segment, subticks_frequency=20)
    # * The actual importance scores can be computed using an element-wise product of
    # * the hypothetical importance scores and the actual importance scores
    viz_sequence.plot_weights(hyp_imp_scores_segment * segment, subticks_frequency=20)



# %%
dp = dbm / "cb_tmp"
fps = [dp / "shap_vals_conv1d.npz"]
for fp in fps:
    loaded = np.load(fp)
    inputs, shap_vals = loaded["inputs"], loaded["shap_vals"]
    seq_idx = 0
    plot_weights(inputs[seq_idx], shap_vals[seq_idx], 12, 1012)

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
