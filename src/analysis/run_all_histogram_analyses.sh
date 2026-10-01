#!/usr/bin/env bash
# Run per-grid-point error histograms AND per-IC RMSE histograms for all
# 7 publication models sequentially (one GPU at a time).
#
# Usage:
#   bash src/analysis/run_all_histogram_analyses.sh [cuda:0|cuda:1|cpu]
#
# Outputs per model:
#   results/<MODEL_NAME>/error_histograms.npz
#   results/<MODEL_NAME>/rmse_histograms.npz
#   results/figures/fig_error_histograms_<MODEL_NAME>.{png,pdf}
#   results/figures/fig_rmse_histograms_<MODEL_NAME>.{png,pdf}

set -e
cd "$(dirname "$0")/../.."   # project root

DEVICE="${1:-cuda:1}"
EXTRA_DIR="data/2021_2025"
PY="conda run -n BoB_Surf_2 python"
ERR_SCRIPT="src/analysis/compute_error_histograms.py"
RMSE_SCRIPT="src/analysis/compute_rmse_histograms.py"
COMMON="--extra_data_dir $EXTRA_DIR --device $DEVICE --pdf"

run_model() {
    local TYPE=$1 CONFIG=$2 MODEL=$3
    local NAME
    NAME=$(basename "$MODEL" .pth)
    echo ""
    echo "=========================================="
    echo "  $NAME  ($TYPE)"
    echo "=========================================="

    echo "--- Error histograms ---"
    $PY $ERR_SCRIPT \
        --model_type "$TYPE" --config_file "$CONFIG" \
        --model_path "$MODEL" $COMMON

    echo "--- RMSE histograms ---"
    $PY $RMSE_SCRIPT \
        --model_type "$TYPE" --config_file "$CONFIG" \
        --model_path "$MODEL" $COMMON
}

# ---- AFNO ----
run_model afno afno_bob_surf_e13.yaml results/models/AFNO_BoB_Surf_E13.pth
run_model afno afno_bob_surf_e14.yaml results/models/AFNO_BoB_Surf_E14.pth

# ---- FNO ----
run_model fno  fno_bob_surf_e03.yaml  results/models/FNO_BoB_Surf_E03.pth
run_model fno  fno_bob_surf_e04.yaml  results/models/FNO_BoB_Surf_E04.pth

# ---- TFNO ----
run_model tfno tfno_bob_surf_e03.yaml results/models/TFNO_BoB_Surf_E03.pth
run_model tfno tfno_bob_surf_e04.yaml results/models/TFNO_BoB_Surf_E04.pth

# ---- UNO ----
run_model uno  uno_bob_surf_e04.yaml  results/models/UNO_BoB_Surf_E04.pth

echo ""
echo "=========================================="
echo "All histogram analyses complete."
ls -lh results/figures/fig_error_histograms_*.png \
        results/figures/fig_rmse_histograms_*.png 2>/dev/null
echo "=========================================="
