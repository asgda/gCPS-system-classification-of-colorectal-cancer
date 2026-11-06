# Software versions

Versions were checked against the Methods ("Statistics and software") of the manuscript and against the environments on
the analysis server (cgntlab4-gpu) and the workstation used for the final runs. Rows marked **extra** are not named in the
manuscript text but were used by the scripts.

## Python environments

| Environment | Python | Steps | Packages |
|---|---|---|---|
| `discovery` (manuscript) | 3.13.5 | 05 | scikit-learn 1.8.0, NumPy 2.4.4, pandas 2.3.3, SciPy 1.17.1, umap-learn 0.5.12, joblib 1.5.3 (**extra**), matplotlib 3.10.9 (**extra**), openpyxl 3.1.5 (**extra**) |
| `analysis` (manuscript) | 3.10.14 | 01-04, 06, 07, 09-16, 18-20, 25, 27, 29-31 | scikit-learn 1.7.2, imbalanced-learn 0.14.2, XGBoost 3.2.0, SHAP 0.48.0, SciPy 1.15.2, statsmodels 0.14.6, lifelines 0.30.0, NumPy 1.26.4, pandas 2.3.3, matplotlib 3.10.9 (**extra**), joblib 1.6.0 (**extra**), openpyxl 3.1.5 (**extra**), seaborn 0.13.2 (**extra**), umap-learn 0.5.12 (**extra**), numba 0.68.1 and llvmlite 0.50.0 (**extra**, UMAP and SHAP dependencies) |
| `resnet` (manuscript) | 3.11.16 | 28 | TensorFlow 2.21.0, Keras 3.15.1, scikit-learn 1.7.2, imbalanced-learn 0.14.2 (**extra**), NumPy 1.26.4, pandas 2.3.3 (**extra**), SciPy 1.15.2 (**extra**) |
| `tide` (manuscript) | 3.9 / 3.10 | 24 | tidepy 1.3.8, NumPy 1.26.4 (**extra**), pandas 2.1.4 (**extra**), setuptools 65.5.0 (**extra**, pkg_resources) |

Step 05 gives identical labels in both Python environments (checked: 1,062 of 1,062 tumors, ARI 1.0; model-selection
statistics agree to within 3 x 10^-4). The `discovery` environment is the one reported in the manuscript.

## R

| Environment | R | Steps | Packages |
|---|---|---|---|
| `survival` (manuscript) | 4.6.0 | 08, 16-17 | survival 3.8.6, ShatterSeek 1.1, survminer 0.5.2 (**extra**), multcomp 1.4.30 (**extra**), broom 1.0.13 (**extra**), dplyr 1.2.1, tidyr 1.3.2, readxl 1.5.0, stringr 1.6.0, ggplot2 4.0.3, svglite 2.2.2, data.table 1.18.4 (**extra**) |
| `expression` (manuscript: R 4.6.1) | 4.6.2 on the final run | 21, 26 | limma 3.68.3, edgeR 4.10.1, oncoPredict 1.2, data.table 1.18.6.1 (**extra**), stringr 1.6.0 (**extra**), dplyr 1.2.1 (**extra**) |
| `gsva` (manuscript: R 4.5.0) | 4.6.2 on the final run | 22 | GSVA 2.6.2 (manuscript: 2.2.1), msigdbr 26.1.0 (**extra**; Hallmark "H" and KEGG_LEGACY collections), stringr 1.6.0 (**extra**) |
| `estimate` (manuscript: R 4.5.0) | 4.6.2 on the final run | 23 | estimate 1.0.13 (installed from R-Forge) |

GSVA 2.6.2 reproduced the Hallmark scores originally computed with GSVA 2.2.1 (median correlation across pathways 1.0).

## Reference data and external tools

| Item | Version or release | Used for |
|---|---|---|
| GRCh38 / hg38 | UCSC cytoBand and Ensembl protein-coding gene BED | gene and arm coordinates (steps 02, 19) |
| MSigDB (via msigdbr 26.1.0) | Hallmark (H), C2 CP:KEGG_LEGACY | GSVA (step 22) |
| GDSC2 | GDSC2_Res.rds and GDSC2_Expr_short.rds | oncoPredict training data (step 26) |
| COSMIC signatures | v3 refit, from Nunes et al. supplementary data | mutational signatures (step 12) |
| Variant calls | TNhaplotyper, TNhaplotyper2, TNsnv, TNscope, VEP-annotated by Nunes et al.; FACETS segments; BRASS structural variants | steps 01-03 (read, not re-run) |

## Files

`requirements_discovery.txt`, `requirements_analysis.txt`, `requirements_resnet.txt` and `requirements_tidepy.txt`
pin the Python packages above. `R_packages.txt` lists the R packages and installation sources.

## Random seeds

All stochastic steps use the master seed 42 (`GCPS_SEED`).
