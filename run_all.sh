#!/usr/bin/env bash
# Runs the full gCPS pipeline, from the Nunes et al. VCFs to every table and figure panel.
# Usage: bash run_all.sh [first_step] [last_step]
# Steps listed in GCPS_SKIP (space separated, e.g. "28") are skipped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source config/paths.env
mkdir -p "$GCPS_OUT/logs"

FIRST=${1:-1}
LAST=${2:-32}

run() {   # run <step number> <command...> <script> [args]
  local n=$1; shift
  (( 10#$n < FIRST || 10#$n > LAST )) && return 0
  [[ " ${GCPS_SKIP:-} " == *" $n "* ]] && return 0
  local script
  for a in "$@"; do [[ $a == *.py || $a == *.R ]] && { script=$a; break; }; done
  echo "[$(date +%H:%M:%S)] step $n: $script"
  (cd scripts && "$@") > "$GCPS_OUT/logs/${script%.*}.log" 2>&1
}

PY=$GCPS_PYTHON
RS=$GCPS_RSCRIPT

run 01 $PY 01_build_snv_features.py
run 02 $PY 02_build_cnv_features.py
run 03 $PY 03_build_sv_features.py
run 04 $PY 04_merge_modalities.py
run 05 $PY 05_discover_gcps_subtypes.py
run 06 $PY 06_facets_ploidy_wgd.py
run 07 $PY 07_baseline_characteristics.py
run 08 $RS 08_survival_analysis.R
run 09 $PY 09_survival_sensitivity.py
run 10 $PY 10_feature_enrichment.py
run 11 $PY 11_feature_heatmap.py
run 12 $PY 12_mutational_signatures.py
run 13 $PY 13_pathway_analysis.py
run 14 $PY 14_genomic_burden.py
run 15 $PY 15_sv_mechanism.py
run 16 $PY 16_prepare_shatterseek_inputs.py
run 17 $RS 17_run_shatterseek.R
run 18 $PY 18_chromothripsis_by_subtype.py
run 19 $PY 19_hrd_scar_score.py
run 20 $PY 20_within_gcps4_survival.py
run 21 $RS 21_differential_expression.R
run 22 $RS 22_gsva_pathways.R
run 23 $RS 23_estimate_scores.R
run 24 env TIDEPY="$GCPS_TIDEPY" $PY 24_tide_scores.py
run 25 $PY 25_immune_signature_tests.py
run 26 $RS 26_oncopredict.R
run 27 $PY 27_supervised_classifiers.py
run 28 $GCPS_PYTHON_TF 28_resnet1d_classifier.py ${GCPS_RESNET_MODEL:+--evaluate "$GCPS_RESNET_MODEL"}
run 29 $PY 29_modality_dropout.py
run 30 $PY 30_drug_sensitivity.py
run 31 $PY 31_expression_system_concordance.py
run 32 $PY 32_classifier_roc_panels.py

echo "done: results in $GCPS_OUT"
