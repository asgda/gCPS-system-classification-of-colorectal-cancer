# Step 23. ESTIMATE stromal, immune, ESTIMATE and tumor-purity scores from the Swedish tumor TPM matrix
suppressMessages(library(estimate))

root <- Sys.getenv("GCPS_ROOT", getwd())
out  <- Sys.getenv("GCPS_OUT", file.path(root, "results"))
tpm  <- Sys.getenv("GCPS_TPM", file.path(Sys.getenv("GCPS_DATA"), "CRC.SW.mRNA.symbol.TPM.txt"))
dir_out <- file.path(out, "23_estimate"); dir.create(dir_out, recursive = TRUE, showWarnings = FALSE)

x <- read.delim(tpm, check.names = FALSE, row.names = 1)
x <- x[, grepl("\\.T2?$", colnames(x))]

# filterCommonGenes reads a tab-delimited matrix (genes in rows, tumors in columns) and writes a GCT file
txt <- file.path(dir_out, "tumors_tpm.txt")
write.table(x, txt, sep = "\t", quote = FALSE, col.names = NA)

common <- file.path(dir_out, "tumors_tpm_common_genes.gct")
filterCommonGenes(input.f = txt, output.f = common, id = "GeneSymbol")
estimateScore(input.ds = common, output.ds = file.path(dir_out, "estimate_scores.gct"))   # default platform
file.remove(txt)
