# Step 22. GSVA Hallmark and KEGG pathway scores for the Swedish tumors (Figure 6F-G, Supplementary Table S7h-i)
suppressMessages({ library(GSVA); library(msigdbr); library(stringr) })

root <- Sys.getenv("GCPS_ROOT", getwd())
out  <- Sys.getenv("GCPS_OUT", file.path(root, "results"))
tpm  <- Sys.getenv("GCPS_TPM", file.path(Sys.getenv("GCPS_DATA"), "CRC.SW.mRNA.symbol.TPM.txt"))
dir_out <- file.path(out, "22_gsva"); dir.create(dir_out, recursive = TRUE, showWarnings = FALSE)
set.seed(as.integer(Sys.getenv("GCPS_SEED", "42")))

x <- read.delim(tpm, check.names = FALSE, row.names = 1)
x <- as.matrix(x[, grepl("\\.T2?$", colnames(x))])
x[x < 0] <- 0
x <- log2(x + 1)
z <- t(scale(t(x))); z[is.nan(z)] <- 0          # gene-wise z-score across the Swedish tumors
cat(nrow(z), "genes,", ncol(z), "tumors\n")

collections <- list(
  HALLMARK = msigdbr(species = "Homo sapiens", collection = "H"),
  KEGG     = msigdbr(species = "Homo sapiens", collection = "C2", subcollection = "CP:KEGG_LEGACY"))

labels <- read.csv(file.path(out, "05_subtypes", "gcps_labels.csv"))
labels$sid <- str_match(labels$sample_id, "((?:UM|U)[0-9]+)")[, 2]
sid <- str_match(colnames(z), "((?:UM|U)[0-9]+)")[, 2]

for (name in names(collections)) {
  sets <- split(collections[[name]]$gene_symbol, collections[[name]]$gs_name)
  scores <- gsva(gsvaParam(z, sets), verbose = FALSE)
  write.csv(data.frame(sample_id = colnames(scores), t(scores), check.names = FALSE),
            file.path(dir_out, paste0(name, "_gsva_scores.csv")), row.names = FALSE)

  d <- data.frame(t(scores), check.names = FALSE)
  d$gCPS <- labels$gCPS[match(sid, labels$sid)]
  d <- d[!is.na(d$gCPS), ]
  features <- setdiff(colnames(d), "gCPS")
  stats <- do.call(rbind, lapply(features, function(f) {
    means <- tapply(d[[f]], d$gCPS, mean)
    data.frame(pathway = f, t(means), kruskal_P = kruskal.test(d[[f]] ~ factor(d$gCPS))$p.value)
  }))
  stats$P_adj <- p.adjust(stats$kruskal_P, "BH")
  write.csv(stats[order(stats$P_adj), ], file.path(dir_out, paste0(name, "_subtype_means.csv")), row.names = FALSE)
  cat(name, ":", sum(stats$P_adj < 0.05), "of", nrow(stats), "pathways differ across subtypes (P-adj < 0.05)\n")
}
