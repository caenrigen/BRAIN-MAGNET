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
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# %%
import data_module as dm
import plot_utils as put


# %%
dir_data = Path("./data")
fp_dataset = dir_data / "Enhancer_activity_with_str_sequences.csv.gz"

# %%
df_enrichment = pd.read_csv(fp_dataset, usecols=dm.DATASET_COLS_NO_SEQ)
df_enrichment["Chr_id"] = df_enrichment.Chr.str.replace("chr", "")
df_enrichment["GC_content"] = df_enrichment.GC_counts / df_enrichment.Seq_len
df_enrichment

# %% [markdown]
# A few sequences in our dataset have `N`s in the human genome reference from which the sequences have been extracted.
#

# %%
df_enrichment[df_enrichment.N_counts > 0]

# %% [markdown]
# # Sequences distribution across chromosomes
#

# %%
_ = df_enrichment.Chr_id.value_counts().plot(kind="bar")
ax = plt.gca()
ax.set_xlabel("Chromosome")
_ = ax.set_ylabel("# Sequences")

# %% [markdown]
# # Sequences length
#

# %%
_ = df_enrichment.Seq_len.hist()
_ = plt.gca().set_xlabel("Sequence Length")
_ = plt.gca().set_ylabel("# Sequences")

# %% [markdown]
# # GC content
#

# %%
df_enrichment.GC_content.hist()
_ = plt.gca().set_xlabel("GC Content")
_ = plt.gca().set_ylabel("# Sequences")

# %% [markdown]
# # Enhancer activity
#

# %%
fig, axs = plt.subplots(1, 2, figsize=(2 * 4 * 1.61, 4), sharey=True)
for log_scale, ax in zip([False, True], axs):
    for col in ["ESC_log2_enrichment", "NSC_log2_enrichment"]:
        sns.histplot(
            df_enrichment,
            x=col,
            ax=ax,
            log_scale=log_scale,
            label=col,
        )
    ax.set_ylabel("# Sequences")
    if log_scale:
        ax.set_xlim(0.1, 10)
    else:
        ax.set_xlim(0, 4)
    ax.legend()
fig.tight_layout()


# %%
x = df_enrichment.NSC_log2_enrichment
y = df_enrichment.ESC_log2_enrichment
ax = put.density_scatter(x=x, y=y)
ax.set_xlabel("NSC Log2 Enrichment")
ax.set_ylabel("ESC Log2 Enrichment")
ax.set_title("Comparative ESC vs NSC")
ax.set_aspect("equal")
_ = ax.set_ylim(ax.get_ylim()[0], ax.get_xlim()[1])

# %% [markdown]
# # Export sequences to Fasta
#

# %% [markdown]
# Export to fast so that we can run SpanSeq on the sequences.
# References:
#
# - https://doi.org/10.1093/nargab/lqae106
# - https://github.com/genomicepidemiology/SpanSeq
#

# %%
df_enrichment = pd.read_csv(fp_dataset, usecols=dm.DATASET_COLS)

fp_fasta = fp_dataset.parent / "Enhancers_sequences.fa"

with open(fp_fasta, "w") as f:
    for r in df_enrichment.itertuples():
        line = f">{r.Index:06d}\n{r.Seq}\n"
        f.write(line)
fp_fasta  # ~110 MB

# %% [markdown]
# We pass the output file to the `spanseq` command:
#
# ```bash
# spanseq split -c 0.25 -i Enhancers_sequences.fa -s nucleotides -o ./spanseq_output -k 17 -m 15 -n 10 -f merged_table -d mash -l 1000 -a hobohm_split -hd 0.90 -H -b 5 -CP "/opt/homebrew/anaconda3/envs/spanseq_ccphylo/bin/ccphylo"
# ```
#
# 💡 **Tip for macOS on Apple Silicon (CPU M1, M1 Pro, etc.)**
#
# `-CP "/opt/homebrew/anaconda3/envs/spanseq_ccphylo/bin/ccphylo"` was used to point to a `ccphylo` installed in a conda environment with:
#
# ```bash
# conda create -n spanseq_ccphylo
# conda activate spanseq_ccphylo
# conda config --env --set subdir osx-64
# conda install ccphylo
# conda deactivate # deactivate&activate to be sure the `ccphylo` command is found
# conda activate spanseq_ccphylo
# which ccphylo # show path to the `ccphylo` executable
# ```
#
# This was needed because `ccphylo` is not available for the `osx-arm64` CPU architecture (at the time of the writing).
#
# Similarly, the `fasta3` conda package is not available for `osx-arm64`, but that is required only if you invoke the (slow) `spanseq -d identity ...` command which requires `ggsearch36` command (provided by `fasta3`). Because we used `-d mash`, we skipped installing `fasta3` package altogether (e.g. remove it from `spanseqenv.yml` before creating the `spanseq` conda env).
#
# You don't need any of this if you installed `ccphylo` in the same `spanseq` environment (or via some other means and it is available in your shell when invoking `spanseq ...` command).
#
