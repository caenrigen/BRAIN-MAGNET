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
#     display_name: g
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload all

# %%
import os
from pathlib import Path
import numpy as np
import pickle

# %%
import modiscolite

# %%
dir_data = Path("./data")
assert dir_data.is_dir()
dir_cb_score = dir_data / "cb_score"
dir_cb_score.mkdir(exist_ok=True)

# %%
# https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_non-redundant_pfms_meme.txt
fp_jaspar = dir_data / "JASPAR2024_CORE_non-redundant_pfms_meme.txt"
assert fp_jaspar.is_file()

# %% [markdown]
# # Input sequences and the SHAP values scores
#
# SHAP values, attribution scores or hypothetical contribution scores refer to more or less the same thing.
#

# %%
fp_sequences = dir_data / "seqs.npy"
fp_shaps = dir_cb_score / "ESC_cdb72439_shap_vals_30shufs.npy"

sequences = np.load(fp_sequences, mmap_mode="r")
hypothetical_contribs = np.load(fp_shaps, mmap_mode="r")
print(sequences.dtype, hypothetical_contribs.dtype)
print(sequences.shape, hypothetical_contribs.shape)
sequences = sequences.transpose(0, 2, 1)
hypothetical_contribs = hypothetical_contribs.transpose(0, 2, 1)
print(sequences.shape, hypothetical_contribs.shape)

# %% [markdown]
# # Discover motifs via TFMoDISco
#

# %%
# This controls how many seqlets are considered initially for clustering.
# Here we use a large number since we have a big dataset.
max_seqlets = 50000

assert hypothetical_contribs.dtype == np.float32, hypothetical_contribs.dtype
pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
    hypothetical_contribs=hypothetical_contribs,
    one_hot=sequences,
    max_seqlets_per_metacluster=max_seqlets,
    verbose=True,
    # The rest are the default values of the `modisco` command line tool
    sliding_window_size=20,
    flank_size=5,
    trim_to_window_size=30,
    initial_flank_to_add=10,
    final_flank_to_add=0,
    target_seqlet_fdr=0.05,
    n_leiden_runs=2,
)

# %%
# Above we used the full 1000bp sequences,
# instead of modisco's default value of only the central 400bp.
window = 1000
fp_patterns_h5 = fp_shaps.parent / f"{fp_shaps.stem}_n{max_seqlets}.h5"
modiscolite.io.save_hdf5(
    filename=fp_patterns_h5,
    pos_patterns=pos_patterns,
    neg_patterns=neg_patterns,
    window_size=window,
)

# %%
# In case we want to do something with the patterns later, dump to disk in a format
# that can be directly recovered in Python with the `pickle` module.
fp_pos_patterns = fp_patterns_h5.parent / f"{fp_patterns_h5.stem}_pos_patterns.pkl"
with open(fp_pos_patterns, "wb") as f:
    pickle.dump(pos_patterns, f)
fp_neg_patterns = fp_patterns_h5.parent / f"{fp_patterns_h5.stem}_neg_patterns.pkl"
with open(fp_neg_patterns, "wb") as f:
    pickle.dump(neg_patterns, f)

# %% [markdown]
# # Match motifs against JASPAR database and visualizing matches
#

# %%
# `modiscolite.report.report_motifs` uses `tomtom` for matching motifs against a
# MEME database.
tomtom_exe = Path("/opt/homebrew/anaconda3/envs/tomtom/bin/tomtom")
assert tomtom_exe.is_file()
# Due to restrictive dependency version requirements, `tomtom` was installed in a
# separate conda environment.
# Inject the parent directory of the `tomtom` executable into the PATH so that it
# can be found by `shutil.which` used in `modiscolite.report`.
os.environ["PATH"] = f"{os.environ['PATH']}:{tomtom_exe.parent}"

# %% [markdown]
# Running the next cell can take 15-30min+ to generate a report.
#
# This will generate inside the `output_dir` an HTML `motifs.html` file.
# It can be opening the in a browser, e.g., Firefox.
#
# At the time of the writing the HTML report has some visualization "bugs". E.g., not all matched motifs have their logos displayed.
#

# %%
top_n_matches = 10

fp_patterns_h5 = fp_shaps.parent / f"{fp_shaps.stem}_n{max_seqlets}.h5"
dir_report = fp_patterns_h5.parent / f"{fp_patterns_h5.stem}_report"

modiscolite.report.report_motifs(
    modisco_h5py=fp_patterns_h5,
    output_dir=dir_report,
    img_path_suffix="./",
    meme_motif_db=fp_jaspar,
    is_writing_tomtom_matrix=False,
    top_n_matches=top_n_matches,
)
