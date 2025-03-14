# ---
# jupyter:
#   jupytext:
#     formats: R:percent
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %%
library(BSgenome.Hsapiens.UCSC.hg38)
library(GenomicRanges)
library(Biostrings)
library(DNAshapeR)
library(reticulate)
library(data.table)  # for fast file reading/writing
np <- import("numpy")

# %%
setwd("/Volumes/Famafia/brain-magnet/rd_APP_data")

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
seqs_revcomp <- reverseComplement(seqs)
seqs_revcomp

# %%
enhancers[, Seq := as.character(seqs)]
enhancers[, SeqRevComp := as.character(seqs_revcomp)]
fwrite(enhancers, file = "Enhancer_activity_w_seq.csv.gz", sep = ",", compress = "gzip")

# %% [raw]
# # Alternative export format
# enhancers[, Seq := as.character(seqs)]
# enhancers[, RevComp := 0]
# enhancers_revcomp <- copy(enhancers)
# enhancers_revcomp[, Seq := as.character(seqs_revcomp)]
# enhancers_revcomp[, RevComp := 1]
# enhancers_aug <- rbind(enhancers, enhancers_revcomp)
# fwrite(enhancers_aug, file = "Enhancer_activity_w_seq_aug.txt.gz", sep = ",", compress = "gzip")

# %% [markdown]
# # Calculate DNA shape metrics

# %%
# seqs must all have the same length
get_dna_shapes <- function (seqs, target_num_cols=1000) {
    tmp_fasta <- "tmp_enhancers.fa"
    writeXStringSet(seqs, tmp_fasta)
    preds <- getShape(tmp_fasta, shapeType=c("MGW","ProT","HelT","Roll"))
    
    num_rows <- nrow(preds$MGW)
    num_cols <- ncol(preds$MGW)
    
    na_column <- rep(NA, num_rows)
    # HelT & Roll have -1 columns, add one more
    preds$HelT <- cbind(preds$HelT, na_column)
    preds$Roll <- cbind(preds$Roll, na_column)

    pad = (target_num_cols - num_cols) / 2
    prepend_cols <- matrix(NA, nrow = num_rows, ncol = floor(pad))
    append_cols <- matrix(NA, nrow = num_rows, ncol = ceiling(pad))

    preds$MGW <- cbind(prepend_cols, preds$MGW, append_cols)
    preds$ProT <- cbind(prepend_cols, preds$ProT, append_cols)
    preds$HelT <- cbind(prepend_cols, preds$HelT, append_cols)
    preds$Roll <- cbind(prepend_cols, preds$Roll, append_cols)

    colnames(preds$MGW) <- NULL
    colnames(preds$ProT) <- NULL
    colnames(preds$HelT) <- NULL
    colnames(preds$Roll) <- NULL

    return(preds)
}

# %%
get_all_dna_shapes <- function(seqs, target_num_cols = 1000) {
  # Total number of sequences
  N <- length(seqs)
  
  # Pre-allocate big matrices with N rows and target_num_cols columns
  big_MGW  <- matrix(NA, nrow = N, ncol = target_num_cols)
  big_ProT <- matrix(NA, nrow = N, ncol = target_num_cols)
  big_HelT <- matrix(NA, nrow = N, ncol = target_num_cols)
  big_Roll <- matrix(NA, nrow = N, ncol = target_num_cols)
  
  # Group sequences by their width
  seqs_by_width <- split(seqs, width(seqs))
  
  # Iterate over each group of sequences with the same width
  for (w in names(seqs_by_width)) {
    # Extract the group of sequences
    group_seqs <- seqs_by_width[[w]]
    
    # Identify the original indices for these sequences in 'seqs'
    current_indices <- which(width(seqs) == as.numeric(w))
    
    # Get DNA shape predictions for the current group
    shapes <- get_dna_shapes(group_seqs, target_num_cols = target_num_cols)
    
    # Fill in the pre-allocated matrices at the correct row positions
    big_MGW[current_indices, ]  <- shapes$MGW
    big_ProT[current_indices, ] <- shapes$ProT
    big_HelT[current_indices, ] <- shapes$HelT
    big_Roll[current_indices, ] <- shapes$Roll
  }
  
  # Return a list containing the four big matrices
  return(list(MGW = big_MGW, ProT = big_ProT, HelT = big_HelT, Roll = big_Roll))
}

# %%
# Process only the first 10 sequences (or all if there are fewer than 10)
# dna_set_subset <- head(seqs, 10)
dna_set_subset <- head(seqs_revcomp, 10)
dna_set_subset

# %%
# Silence stdout and stderr
sink(tempfile())
suppressMessages(
    preds <- get_all_dna_shapes(seqs_revcomp)
)
sink()

# %%
np$savez_compressed(
    # "shapes.npz",
    "shapes_revcomp.npz",
    mgw=preds$MGW,
    prot=preds$ProT,
    helt=preds$HelT,
    roll=preds$Roll,
)

# %%
