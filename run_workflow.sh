#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Configuration (edit here)
# -------------------------
EXP_NUM=9
START_STAGE=1
END_STAGE=16
EPOCHS=1
BATCH_SIZE=24
STAGE_0_EPOCHS=1
DATASET_PATH="gpt_data_v2.csv"

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

exec "$PYTHON_CMD" "$SCRIPT_DIR/main.py"       --exp-num "$EXP_NUM"       --start-stage "$START_STAGE"       --end-stage "$END_STAGE"       --epochs "$EPOCHS"       --batch-size "$BATCH_SIZE"       --stage-0-epochs "$STAGE_0_EPOCHS"       --dataset_path "$DATASET_PATH"       "${OPTS[@]}"
