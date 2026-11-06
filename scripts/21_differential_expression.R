# Step 21. limma-voom differential expression, each subtype against the mean of the other three
# (Figure 6A, Supplementary Figure S3A-D, Supplementary Table S7a-b)
suppressMessages({ library(limma); library(edgeR); library(stringr) })

root  <- Sys.getenv("GCPS_ROOT", getwd())
out   <- Sys.getenv("GCPS_OUT", file.path(root, "results"))
rna   <- Sys.getenv("GCPS_COUNTS", file.path(Sys.getenv("GCPS_DATA"), "CRC.SW.mRNA.symbol.count.txt"))
dir_out <- file.path(out, "21_deg"); dir.create(dir_out, recursive = TRUE, showWarnings = FALSE)
set.seed(as.integer(Sys.getenv("GCPS_SEED", "42")))

counts <- read.delim(rna, check.names = FALSE, row.names = 1)
counts <- counts[, grepl("\\.T", colnames(counts))]
id <- function(x) str_match(x, "((?:UM|U)[0-9]+)")[, 2]

labels <- read.csv(file.path(out, "05_subtypes", "gcps_labels.csv")); labels$sid <- id(labels$sample_id)
tumor_id <- id(colnames(counts))
keep <- tumor_id %in% labels$sid
counts <- counts[, keep]
group <- factor(labels$gCPS[match(tumor_id[keep], labels$sid)])
cat(sprintf("%d tumors: %s\n", ncol(counts), paste(levels(group), table(group), collapse = ", ")))

y <- DGEList(as.matrix(counts))
y <- y[filterByExpr(y, group = group), , keep.lib.sizes = FALSE]
y <- calcNormFactors(y, method = "TMM")
cat(nrow(y), "genes after filtering\n")

design <- model.matrix(~ 0 + group); colnames(design) <- levels(group)
fit <- lmFit(voom(y, design), design)
contrasts <- sapply(levels(group), function(s) { w <- setNames(rep(-1 / 3, nlevels(group)), levels(group)); w[s] <- 1; w })
fit <- eBayes(contrasts.fit(fit, contrasts))

summary_rows <- list()
for (s in levels(group)) {
  tab <- topTable(fit, coef = s, number = Inf); tab$gene <- rownames(tab)
  write.csv(tab, file.path(dir_out, paste0("limma_", s, "_vs_rest.csv")), row.names = FALSE)
  up <- sum(tab$adj.P.Val < 0.05 & tab$logFC > 0.5); down <- sum(tab$adj.P.Val < 0.05 & tab$logFC < -0.5)
  summary_rows[[s]] <- data.frame(subtype = s, up = up, down = down, total = up + down)
}
degs <- do.call(rbind, summary_rows)
write.csv(degs, file.path(dir_out, "deg_counts.csv"), row.names = FALSE)
print(degs)
