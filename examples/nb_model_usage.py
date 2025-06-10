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

# %% [markdown]
# This notebook shows how to use a trained model for predictions.
#

# %%
# %load_ext autoreload
# %autoreload all

# %%
from pathlib import Path
import torch

# %%
# local modules
import utils as ut
import cnn_starr as cnn

# %%
dir_data = Path("./data")
dir_train = dir_data / "train"
assert dir_data.is_dir() and dir_train.is_dir()

fp_dataset = dir_data / "Enhancer_activity_with_str_sequences.csv.gz"

device = torch.device("mps")  # cpu/mps/cuda/etc

# %% [markdown]
# # Load a model
#

# %%
fp_trained_model = dir_train / "ESC/762acb33/checkpoints/epoch=014.ckpt"
assert fp_trained_model.exists()

# %%
trained_model = cnn.ModelModule.load_from_checkpoint(fp_trained_model)

# Choose the sequence forwarding mode.
# "mean": pass through the model both the forward and reverse complement sequences and take the average
# "forward": pass through the model only the forward sequence
# "reverse_complement": pass through the model only the reverse complement sequence
trained_model.forward_mode = "mean"  # mean/forward/reverse_complement

# Disable gradient calculation, we are only using the model for predictions
trained_model.eval()
trained_model.freeze()

trained_model

# %% [markdown]
# # Prepare sequences
#

# %%
# gh38, chr3, Start=128680750, End=128681335
seq_str_example = "TCACACCCAGTCCCAGGAAGCGAGCCCCACGTCGCCCCAGCTCTGCATTCACAGCCGCCAAGCGGTCCCGGAAGCCGAATGCCGGATAGGTCAAAGACAGCGCCGCTGCTCCGGCACTGCCGCCAGAGGGCGCGGAGTCGCCGTGTTGGCCTGCACCTCCCTTGTCACGTCAGCCGAACGGCTACGGCAATGGAGGATTTTGCCGTATGCGACGACAGGATGACTGCTTCCCCTGAAGCTTCACCAAACACGGAGAGGCTTTCCTGTCCCTGGGAGACGCGGTGATGTGACGCAGCGGTTGCAGCGAAGGTTTCTTGTAAAAGATGGCCGCTGCGGAGTTGCTAGGTGTCTCCTGGCAGGCGGCGGCCGCACCACAAGATGGCGGCCCGCCCCAGGGTCCCACACGCGGGGCTGCGGAGGCGGCGGCCCCGAGGGTCCCCGCCCCTTGCTGGCCCTGTCCCCGCGTGCGGGGCTGCGACCTGCCTTTGTGTGCGGGTGGAAGTGCGGGCCGCCTTGGGGCCGAGTTGCAACTTCGCGACACTCCACTGGAGTGCGATGAGACGATGAGATGTTACCTCCTAACAC"

# %%
# emulate a few sequences for demonstration purposes
seqs_str = [seq_str_example, seq_str_example]
seqs_for_tensor = ut.sequences_str_to_1hot(seqs_str, pad_to=None, transpose=True)
seqs_for_tensor


# %%
batch = torch.from_numpy(seqs_for_tensor).float().to(device)
batch

# %% [markdown]
# # Run predictions
#

# %%
trained_model(batch)

# %%
