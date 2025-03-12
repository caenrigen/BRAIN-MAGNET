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
library(data.table)  # for fast file reading/writing

# %%
setwd("/Volumes/Famafia/brain-magnet/rd_APP_data")

# %%
enhancers <- fread("Enhancer_activity.txt", sep = "\t")
enhancers

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
# Add the sequences as a new column to the data table (converting DNAStringSet to character)
enhancers[, Seq := as.character(seqs)]
enhancers[, RevComp := 0]
nchar(enhancers$Seq[1])

# %%
enhancers_revcomp <- copy(enhancers)
enhancers_revcomp[, Seq := as.character(seqs_revcomp)]
enhancers_revcomp[, RevComp := 1]
nchar(enhancers_revcomp$Seq[1])

# %%
enhancers_aug <- rbind(enhancers, enhancers_revcomp)

# %%
fwrite(enhancers_aug, file = "Enhancer_activity_w_seq_aug.txt.gz", sep = ",", compress = "gzip")

# %%
