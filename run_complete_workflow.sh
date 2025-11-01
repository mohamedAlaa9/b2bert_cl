#!/bin/bash

################################################################################
# Simplified Curriculum Learning Training Script
# No environment checks - runs training directly
################################################################################

set -e  # Exit on error

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default configuration
EXP_NUM=3
START_STAGE=1
END_STAGE=16
STAGE_0_EPOCHS=1
CURRICULUM_EPOCHS=1
BATCH_SIZE=24
SKIP_DATA_PREP=false
SKIP_STAGE_0=false
PYTHON_CMD="python"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-num) EXP_NUM="$2"; shift 2 ;;
        --start-stage) START_STAGE="$2"; shift 2 ;;
        --end-stage) END_STAGE="$2"; shift 2 ;;
        --stage-0-epochs) STAGE_0_EPOCHS="$2"; shift 2 ;;
        --epochs) CURRICULUM_EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --skip-data-prep) SKIP_DATA_PREP=true; shift ;;
        --skip-stage-0) SKIP_STAGE_0=true; shift ;;
        --python) PYTHON_CMD="$2"; shift 2 ;;
        --help|-h)
            echo "Simplified training script - no environment checks"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --exp-num NUM          Experiment number (default: 28)"
            echo "  --start-stage NUM      Starting stage (default: 1)"
            echo "  --end-stage NUM        Ending stage (default: 15)"
            echo "  --stage-0-epochs NUM   Epochs for stage 0 (default: 2)"
            echo "  --epochs NUM           Epochs per stage (default: 2)"
            echo "  --batch-size NUM       Batch size (default: 24)"
            echo "  --skip-data-prep       Skip data preparation"
            echo "  --skip-stage-0         Skip stage 0 training"
            echo "  --python CMD           Python command (default: python)"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Print configuration
echo ""
log_info "=========================================="
log_info "Experiment: $EXP_NUM"
log_info "Stage 0 Epochs: $STAGE_0_EPOCHS"
log_info "Stages $START_STAGE-$END_STAGE: $CURRICULUM_EPOCHS epochs each"
log_info "Batch Size: $BATCH_SIZE"
log_info "=========================================="
echo ""

TOTAL_START=$(date +%s)

################################################################################
# STEP 1: Data Preparation
################################################################################

if [ "$SKIP_DATA_PREP" = false ]; then
    log_info "STEP 1: Preparing Curriculum Data"
    
    cat <<EOF | $PYTHON_CMD
import sys
from prepare_data import prepare_curriculum_data

try:
    stage_paths = prepare_curriculum_data(
        dataset_path="gpt_data_v1.csv",
        output_dir="./CL_stages",
        computed_filter=True
    )
    print(f"Created {len(stage_paths)} curriculum stages")
    sys.exit(0)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        log_success "Data preparation completed"
    else
        log_error "Data preparation failed"
        exit 1
    fi
    echo ""
else
    log_info "STEP 1: Data Preparation [SKIPPED]"
    echo ""
fi

# ################################################################################
# # STEP 2: Train Stage 0
# ################################################################################

# if [ "$SKIP_STAGE_0" = false ]; then
#     log_info "STEP 2: Training Stage 0"
    
#     STEP_START=$(date +%s)
    
#     cat <<EOF | $PYTHON_CMD
# import sys
# from config import ExperimentConfig
# from main_training import train_single_stage

# try:
#     exp_config = ExperimentConfig(
#         exp_num=$EXP_NUM,
#         stage=0,
#         model_name="UBC-NLP/MARBERT",
#         threshold=0.3,
#         batch_size=$BATCH_SIZE,
#         epochs=$STAGE_0_EPOCHS,
#         use_previous_stage_model=False
#     )
#     train_single_stage(exp_config)
#     sys.exit(0)
# except Exception as e:
#     print(f"ERROR: {e}", file=sys.stderr)
#     import traceback
#     traceback.print_exc()
#     sys.exit(1)
# EOF
    
#     if [ $? -eq 0 ]; then
#         STEP_END=$(date +%s)
#         DURATION=$((STEP_END - STEP_START))
#         MINUTES=$((DURATION / 60))
#         log_success "Stage 0 completed in ${MINUTES} minutes"
#     else
#         log_error "Stage 0 training failed"
#         exit 1
#     fi
#     echo ""
# else
#     log_info "STEP 2: Training Stage 0 [SKIPPED]"
#     echo ""
# fi

################################################################################
# STEP 3: Train Curriculum Sequence
################################################################################

log_info "STEP 3: Training Curriculum Sequence (Stages $START_STAGE-$END_STAGE)"

STEP_START=$(date +%s)

cat <<EOF | $PYTHON_CMD
import sys
from main_training import train_curriculum_sequence

try:
    train_curriculum_sequence(
        exp_num=$EXP_NUM,
        start_stage=$START_STAGE,
        end_stage=$END_STAGE,
        epochs=$CURRICULUM_EPOCHS,
        batch_size=$BATCH_SIZE
    )
    sys.exit(0)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    STEP_END=$(date +%s)
    DURATION=$((STEP_END - STEP_START))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    log_success "Curriculum sequence completed in ${HOURS}h ${MINUTES}m"
else
    log_error "Curriculum training failed"
    exit 1
fi

################################################################################
# Summary
################################################################################

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
log_success "=========================================="
log_success "TRAINING COMPLETED!"
log_success "=========================================="
log_success "Experiment: $EXP_NUM"
log_success "Total Time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
log_success "Models saved to: ./exp_${EXP_NUM}/"
log_success "=========================================="
echo ""

exit 0