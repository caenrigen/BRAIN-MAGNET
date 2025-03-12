# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
import pandas as pd

# %%
from pathlib import Path

# %%
dbm = Path("/Volumes/Famafia/brain-magnet/")
dbmst = dbm / "Supplementary Tables"

# %%
fp = dbmst / "Supplementary Table 5 - Scaffold, Categories, Targets, HPO.xlsx"
ef = pd.ExcelFile(fp)

# %%
ef.sheet_names

# %%
df_nsc = pd.read_excel(ef, sheet_name='NSC')
df_esc = pd.read_excel(ef, sheet_name='ESC')

# %%
df_nsc

# %%
df_nsc.NCREs.unique().shape

# %% [raw]
# chr1_KI270709v1_random:3184-4184

# %%
df = df_nsc.set_index("NCREs")
df[df.index.isin(["chr1_KI270709v1_random:3184-4184"])]

# %%
df = df_esc.set_index("NCREs")
df[df.index.isin(["chr1_KI270709v1_random:3184-4184"])]

# %%
df_esc.NCREs.unique().shape

# %%
pd.Index(df_nsc.NCREs.unique())

# %%
df_esc.NCREs

# %%
to_comp = {c: cc for c, cc in zip("ACGT", "TGCA")}
to_comp

# %%
import random

def generate_random_dna_sequence(length):
    nucleotides = ['A', 'T', 'C', 'G']
    return ''.join(random.choice(nucleotides) for _ in range(length))


# %%
seq = generate_random_dna_sequence(10)
seq_comp = "".join(to_comp[c] for c in seq)
seq, seq_comp

# %%
seq == seq[::-1], seq == seq_comp, seq[::-1] == seq_comp

# %% [raw]
# GAGATGAGTT
# CTCTACTCAA

# %%
