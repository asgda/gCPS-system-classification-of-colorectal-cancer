# Step 26. Predicted drug response (GDSC2, oncoPredict) for the Swedish tumors
suppressMessages({ library(oncoPredict); library(data.table); library(dplyr) })

root <- Sys.getenv("GCPS_ROOT", getwd())
out  <- Sys.getenv("GCPS_OUT", file.path(root, "results"))
inp  <- Sys.getenv("GCPS_ONCOPREDICT", file.path(Sys.getenv("GCPS_DATA"), "oncopredict_inputs"))
dir_out <- file.path(out, "26_oncopredict"); dir.create(dir_out, recursive = TRUE, showWarnings = FALSE)
setwd(dir_out)
set.seed(as.integer(Sys.getenv("GCPS_SEED", "42")))

# Test expression: Swedish tumors, log2(TPM + 1), gene-wise z-score
tpm <- data.frame(fread(file.path(inp, "CRC.SW.mRNA.symbol.TPM.txt"), sep = "\t"), row.names = 1, check.names = FALSE)
tumor <- log2(tpm[, grepl("\\.T.*$", colnames(tpm))] + 1)
tumor <- as.data.frame(t(scale(t(tumor), center = TRUE, scale = TRUE)))
tumor <- tumor %>% mutate(across(where(is.numeric), ~ ifelse(is.nan(.), 0, .)))
cat("Swedish tumors:", ncol(tumor), "| genes:", nrow(tumor), "\n")

# Training data: GDSC2 ln(IC50), back-transformed because calcPhenotype transforms the phenotype again
train_response <- exp(readRDS(file.path(inp, "GDSC2_Res.rds")))
train_expr <- readRDS(file.path(inp, "GDSC2_Expr_short.rds"))
train_expr <- t(scale(t(train_expr), center = TRUE, scale = TRUE))

calcPhenotype(trainingExprData = train_expr, trainingPtype = train_response, testExprData = as.matrix(tumor),
              batchCorrect = "standardize", powerTransformPhenotype = TRUE, removeLowVaryingGenes = 0.2,
              minNumSamples = 10, selection = 1, printOutput = TRUE, pcr = FALSE,
              removeLowVaringGenesFrom = "rawData", report_pc = FALSE, cc = FALSE, percent = 80, rsq = FALSE)

predictions <- data.frame(fread("calcPhenotype_Output/DrugPredictions.csv", sep = ",", header = TRUE),
                          row.names = 1, check.names = FALSE)
fwrite(cbind(sample = rownames(predictions), log(predictions)), "DrugPredictions_lnIC50.csv")
cat("predictions:", dim(predictions), "\n")
