# Input data

All inputs come from Nunes et al. (Nature 2024) and are not redistributed here.

| File or folder | Source | Used by |
|---|---|---|
| `short_variants/`, `vep_output/*.vep.vcf.gz` | European Variation Archive, PRJEB61514 (VEP-annotated TNscope/TNhaplotyper calls) | step 01 |
| `cnv/*.vcf.gz` | European Variation Archive, PRJEB61514 (FACETS segments) | steps 02, 06, 16, 19 |
| `sv/*.vcf.gz` | European Variation Archive, PRJEB61514 (BRASS) | steps 03, 16 |
| `Supplementary_Table_01.xlsx` | Nunes et al., clinical and sample table | steps 06-09, 14, 20, 25, 29, 31 |
| `SBS_signatures.xlsx` | Nunes et al., SBS/DBS/ID signature activities | step 12 |
| `Supplementary_Table_29.xlsx` | Nunes et al., CIBERSORT fractions | step 25 |
| `CRC.SW.mRNA.symbol.TPM.txt`, `CRC.SW.mRNA.symbol.count.txt` | ArrayExpress E-MTAB-12862 | steps 21-26 |
| `clinical_data.tsv` | overall-survival table derived from the supplementary data (sample ID, OS_MONTHS, OS_STATUS) | step 05 |
| `GDSC2_Res.rds`, `GDSC2_Expr_short.rds` | GDSC2 training data for oncoPredict | step 26 |
| `cytoBand.txt.gz`, `genes_protein_coding_hg38.bed` | UCSC hg38 cytoband; protein-coding genes from GENCODE v44 | steps 02, 19 |

`resources/crc_panel_genes.txt` is the 346-gene CRC panel (82 IntOGen drivers and five MSigDB CRC gene sets) used for the
feature universe.
