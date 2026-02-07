#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Configuration (edit here)
# -------------------------
EXP_NUM=14
# If cl_aldi, start stage = 1 and end stage = 4, if cl_cardinality, start stage = 1 and end stage = 18
START_STAGE=1
END_STAGE=4
EPOCHS=1
BATCH_SIZE=24
DATASET_PATH="./aggregated_final/stage_1_and_gpt.csv"
# 1 for cardinality-based CL, 0 for ALDI-based CL
CL_METHOD=0
# ORDER=(18 2 3 1 12 7 4 15 17 11 6 8 16 14 5 9 10 13)
ORDER=(4 1 3 2)
SEED=42

# Toggle steps
SKIP_DATA_PREP=false   # set to true to skip
SKIP_STAGE_0=false     # set to true to skip
# -------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python}"

# Compose optional flags
OPTS=()
if [[ "${SKIP_DATA_PREP}" == "true" ]]; then OPTS+=("--skip-data-prep"); fi
if [[ "${SKIP_STAGE_0}" == "true" ]]; then OPTS+=("--skip-stage-0"); fi

ORDER_CSV="$(IFS=,; echo "${ORDER[*]}")"
exec "$PYTHON_CMD" "$SCRIPT_DIR/main.py"       --exp-num "$EXP_NUM"       --start-stage "$START_STAGE"       --end-stage "$END_STAGE"       --epochs "$EPOCHS"       --batch-size "$BATCH_SIZE"       --dataset_path "$DATASET_PATH"       --order "$ORDER_CSV" --seed "$SEED" --cl_method "$CL_METHOD"    ${OPTS[@]+"${OPTS[@]}"}