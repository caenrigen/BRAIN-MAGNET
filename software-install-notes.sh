# Installing genomics software packages on macOS (Apple Silicon M1)

# Make sure you are using only conda-forge (highest priority) and bioconda
conda config --show channels

conda env create -n g -f bioconda.yaml
pip install jupyterlab jupytext

# In R
install.packages("BiocManager")
BiocManager::install(version = "3.20")
BiocManager::valid()
# Update packages if needed
# BiocManager::install(c(...), update = TRUE, ask = FALSE, force = TRUE)

# Just in case some R packages are not following the guidelines of passing downstream
# the same flags that were used to build R
Sys.setenv(MACOSX_DEPLOYMENT_TARGET = "11.0")

BiocManager::install(c(
"Biostrings",
"PRROC",
"GenomicRanges"
))

# Required by phastCons100way.UCSC.hg38
# Update should be available from Bioconductor shortly
BiocManager::install('grimbough/rhdf5filters', ref = 'devel', force = TRUE)
BiocManager::install(c("phastCons100way.UCSC.hg38"))
BiocManager::install(c("BSgenome.Hsapiens.UCSC.hg38"))

# NB BiocManager::valid() reports some old packages, I left them be
# Old packages: 'abind', 'bit64', 'caret', 'clock', 'cpp11', 'curl', 'gower', 'lme4', 'MASS', 'ps', 'textshaping', 'xml2'
