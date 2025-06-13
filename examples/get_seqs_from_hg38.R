# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:percent
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% [markdown]
# This R Script was used to extract the DNA sequences from hg38
#

# %%
library(BSgenome.Hsapiens.UCSC.hg38)
library(GenomicRanges)
library(Biostrings)
library(reticulate)
library(data.table)  # for fast file reading/writing

# %%
setwd("./data")

# %%
enhancers <- fread("Enhancer_activity.txt", sep = "\t")

# %%
# Create a GRanges object from the enhancer coordinates
gr_enh <- GRanges(
  seqnames = enhancers$Chr,
  ranges = IRanges(start = enhancers$Start, end = enhancers$End - 1)
)
gr_enh

# %%
seqs <- getSeq(BSgenome.Hsapiens.UCSC.hg38, gr_enh)
seqs

# %%
enhancers[, Seq := as.character(seqs)]
fwrite(enhancers, file = "Enhancer_activity_w_seq.csv.gz", sep = ",", compress = "gzip")
