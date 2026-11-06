# Verification against the manuscript

Each step was run on the Nunes et al. cohort and its output compared with the value reported in the manuscript.
All 31 steps were then run in sequence through `run_all.sh`: the Python, ShatterSeek, TIDE steps on the
analysis server, and the limma-voom, GSVA, ESTIMATE and oncoPredict steps (21-23, 26) on a workstation with the R packages listed in
`versions/R_packages.txt`. Step 28 was run in `--evaluate` mode on the saved ResNet model.

| Step | Quantity | Manuscript | Script output |
|---|---|---|---|
| 01-04 | Binary matrix | 1,062 x 371 (210 SNV, 130 CNV, 31 SV) | 1,063 x 371; identical to the matrix used for the paper (the extra row, U3201, is dropped in step 05 for lack of survival data) |
| 05 | Subtype sizes | 426 / 274 / 268 / 94 | 426 / 274 / 268 / 94; labels identical to the frozen partition |
| 05 | Cophenetic correlation (null) | 0.962 (0.255) | 0.962 (0.255) |
| 05 | Dispersion; Hamming silhouette (null) | 0.727; 0.184 (0.015) | 0.727; 0.184 (0.015) |
| 05 | Calinski-Harabasz; Davies-Bouldin | 133.2; 2.717 | 133.2; 2.717 |
| 05 | Resampling ARI (5th percentile); minimum Jaccard | 0.944 (0.928); 0.820 | 0.944 (0.928); 0.820 |
| 05 | Log-rank P, k = 4 | 0.0246 | 0.0246 |
| 06 | WGD, gCPS1 / gCPS4 | 74.4% / 4.3% | 74.4% / 4.3% |
| 06 | Loss delta against ploidy, all tumors | 0.035 | 0.035 |
| 06 | Within WGD-negative tumors: loss against ploidy; LOH | 0.563; 0.484 | 0.563; 0.484 |
| 08 | gCPS4 vs gCPS1, adjusted OS | HR 1.58 (1.18-2.14), P 0.003, P-adj 0.015 | HR 1.58 (1.18-2.14), P 0.0025, P-adj 0.0151 |
| 08 | gCPS4 vs gCPS1, adjusted RFS | HR 1.83 (1.11-3.02), P 0.018, P-adj 0.107 | HR 1.83 (1.11-3.02), P 0.018, P-adj 0.107 |
| 08 | Log-rank P, OS / RFS | 0.0246 / 0.0263 | 0.0246 / 0.0263 |
| 09 | Table 1 (11 confounder models) | HR 1.54-1.78 | HR 1.54-1.78; every row agrees to two decimals |
| 09 | Stage II-III MSS, gCPS4 vs gCPS1 | HR 1.58 (1.11-2.27), P 0.012 | HR 1.58 (1.11-2.27), P 0.012 |
| 10 | Enriched features (P-adj < 0.05) | gCPS1 115 CNV; gCPS3 203 SNV | gCPS1 115 CNV; gCPS3 203 SNV |
| 11 | Heatmap features | 48 (16 per modality) | the same 48 |
| 12 | Etiology P-adj: MMR, colibactin, HRD | 4.2e-91, 9.0e-7, 0.012 | 4.2e-91, 9.0e-7, 0.012 |
| 13 | gCPS4 vs other subtypes, dual hit | 22.3% vs 8.2%, OR 3.24, P-adj 1.22e-4 | 22.3% vs 8.2%, OR 3.24, P-adj 1.2e-4 |
| 14 | Median TMB in gCPS3 | 92.4 mut/Mb | 92.4 mut/Mb |
| 14 | SV junction medians, gCPS1-4 | 72 / 50 / 44.5 / 84 | 72 / 50 / 44.5 / 84 |
| 15 | Inversion delta; chi-square P; Cramer's V | 0.23; 1.5e-6; 0.117 | 0.23; 1.5e-6; 0.117 |
| 18 | High-confidence chromothripsis, gCPS4 vs gCPS1 | not enriched | 4.3% vs 4.7%, P-adj 1.0 |
| 19 | Scar score, gCPS1 vs gCPS4 (median); WGD-negative delta | 35 vs 29; 0.36 | 35 vs 29; 0.357 |
| 20 | Within gCPS4: dual hit; scar split | HR 1.35, P 0.33; HR 0.82, P 0.45 | HR 1.35, P 0.33; HR 0.82, P 0.45 |
| 21 | DEGs, gCPS1-4 | 689 / 718 / 3,082 / 143 | 689 / 718 / 3,082 / 143 |
| 22 | GSVA Hallmark | stored scores | median correlation 1.0 |
| 23 | ESTIMATE | stored scores | correlation 0.996-0.9995 with the stored scores |
| 24 | TIDE | stored scores | correlation 1.0 |
| 25 | EMT / stemness / proliferation P-adj | 3.6e-9 / 7.3e-4 / 2.9e-9 | 3.6e-9 / 7.3e-4 / 2.9e-9 |
| 25 | CIBERSORT features with P-adj < 0.05 | 14 of 22 | 14 of 22 |
| 26, 30 | Compounds differing across subtypes | 134 of 198 | 134 of 198 |
| 30 | Talazoparib, gCPS3 delta | 0.23 | 0.226 |
| 27 | Test F1 range; AUROC range | 0.843-0.966; 0.953-0.998 | 0.843-0.966; 0.953-0.998 (six of seven classifiers match Table S8b exactly; XGBoost see below) |
| 28 | ResNet test accuracy, F1, AUROC | 0.962, 0.948, 0.996 | 0.962, 0.948, 0.996 with `--evaluate` on the saved model; 0.944, 0.921, 0.991 when retrained by the script |
| 29 | Complete input ARI (95% CI 0.939-0.974); SNV+CNV ARI | 0.958; 0.938 | 0.958; 0.938 |
| 29 | CNV alone: ARI, gCPS4 recall and precision; SV alone ARI | 0.439, 92.6%, 43.5%; -0.004 | 0.439, 92.6%, 43.5%; -0.004 |
| 29 | Complete input OS / RFS log-rank P | 0.035 / 0.031 | 0.035 / 0.031 |
| 31 | ARI with CMS / CRPS / iCMS | 0.159 / 0.186 / 0.216 | 0.159 / 0.186 / 0.216 |
| 31 | Likelihood-ratio P, gCPS added to CRPS / expression added to gCPS (iCMS) | 0.0086; 1.8e-4 | 0.0086; 1.8e-4 |

## Known differences

- **ESTIMATE (step 23).** Re-running the `estimate` package on the TPM matrix gives scores correlated at r = 0.996-0.9995 with the file
  used for the manuscript, but not identical values (the original input matrix differed). Conclusions are unchanged.
- **ResNet (step 28).** The script uses a fixed seed (42) and deterministic TensorFlow operations; two independent retrains gave identical
  results (test accuracy 0.944, macro-F1 0.921, AUROC 0.991). These are slightly below the saved model reported in the manuscript
  (0.962, 0.948, 0.996), which was trained before determinism was enforced; `--evaluate` reproduces the manuscript values from that model.
  Both sets lie inside the range of the other classifiers.
- **XGBoost (step 27).** The script gives test accuracy 0.962, macro-F1 0.940, AUROC 0.996, against 0.958, 0.936 and 0.996 in Table S8b. The result does not depend on the thread count (1-28 threads tested), so the earlier table was most likely produced with a different XGBoost build; the conclusions are unchanged.
- **limma-voom (step 21).** The original script was not archived; it was rebuilt from the Methods (filterByExpr, TMM, voom, one subtype against the mean of the other three, eBayes) and reproduces the reported DEG counts and log fold changes (maximum difference 1e-14).
