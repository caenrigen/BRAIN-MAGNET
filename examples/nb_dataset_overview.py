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
# # Similarity clusters to prevent leakage
#

# %%
fp_clusters = dir_data / "mmseq2res_cluster.tsv"
df_clusters = pd.read_table(fp_clusters, names=["cluster_id", "seq_id"])
s = df_clusters.cluster_id.value_counts()

s[:30].plot(kind="bar")

s[s > 1].sum()

# %% [markdown]
# ## Add cluster id to dataset file
#
# We will use the cluster ids to create groups that must remain together when spliting the data into training/validation sets.
#

# %%
df_enrichment = pd.read_csv(fp_dataset, usecols=dm.DATASET_COLS)

df_clusters.set_index("seq_id", inplace=True)
df_clusters.sort_index(inplace=True)

df_enrichment["Cluster_id"] = df_clusters.cluster_id.to_numpy()
df_enrichment.head()

# %%
df_enrichment.to_csv(fp_dataset, index=False)

# %%
df_enrichment = pd.read_csv(fp_dataset, usecols=dm.DATASET_COLS)
df_enrichment.head()

# %%
