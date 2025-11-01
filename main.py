import argparse
import sys
import time
import traceback

# Color codes (same as shell script semantics)
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

def log_info(msg: str) -> None:
    print(f"{BLUE}[INFO]{NC} {msg}")

def log_success(msg: str) -> None:
    print(f"{GREEN}[SUCCESS]{NC} {msg}")

def log_error(msg: str) -> None:
    print(f"{RED}[ERROR]{NC} {msg}", file=sys.stderr)

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Simplified training script - Python port of embedded code"
    )
    # Defaults mirror the variables at the top of the shell script
    p.add_argument("--exp-num", type=int, default=3, help="Experiment number (default: 3)")
    p.add_argument("--start-stage", type=int, default=1, help="Starting stage (default: 1)")
    p.add_argument("--end-stage", type=int, default=16, help="Ending stage (default: 16)")
    p.add_argument("--stage-0-epochs", type=int, default=1, help="Epochs for stage 0 (default: 1)")
    p.add_argument("--epochs", type=int, default=1, help="Epochs per stage (default: 1)")
    p.add_argument("--batch-size", type=int, default=24, help="Batch size (default: 24)")
    p.add_argument("--skip-data-prep", action="store_true", help="Skip data preparation step")
    p.add_argument("--skip-stage-0", action="store_true", help="Skip Stage 0 training")
    p.add_argument("--dataset_path", type=str, default="gpt_data_v1.csv", help="Path to the input dataset")
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)

    print()
    log_info("==========================================")
    log_info(f"Experiment: {args.exp_num}")
    log_info(f"Stage 0 Epochs: {args.stage_0_epochs}")
    log_info(f"Stages {args.start_stage}-{args.end_stage}: {args.epochs} epochs each")
    log_info(f"Batch Size: {args.batch_size}")
    log_info("==========================================")
    print()

    total_start = time.time()

    # STEP 1: Data Preparation
    if not args.skip_data_prep:
        log_info("STEP 1: Preparing Curriculum Data")
        try:
            # Matches the embedded Python from the shell here-doc
            from prepare_data import prepare_curriculum_data  # noqa: E402

            stage_paths = prepare_curriculum_data(
                dataset_path=args.dataset_path,
                output_dir="./CL_stages",
                computed_filter=True,
            )
            print(f"Created {len(stage_paths)} curriculum stages")
            log_success("Data preparation completed")
            print()
        except Exception as e:
            log_error(f"Data preparation failed: {e}")
            traceback.print_exc()
            return 1
    else:
        log_info("STEP 1: Data Preparation [SKIPPED]")
        print()

    # STEP 2: Train Curriculum Sequence
    log_info(f"STEP 3: Training Curriculum Sequence (Stages {args.start_stage}-{args.end_stage})")
    step_start = time.time()
    try:
        from train import train_curriculum_sequence  # noqa: E402
        train_curriculum_sequence(
            exp_num=args.exp_num,
            start_stage=args.start_stage,
            end_stage=args.end_stage,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        duration = int(time.time() - step_start)
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        log_success(f"Curriculum sequence completed in {hours}h {minutes}m")
    except Exception as e:
        log_error(f"Curriculum training failed: {e}")
        traceback.print_exc()
        return 1

    # Summary
    total_duration = int(time.time() - total_start)
    total_hours = total_duration // 3600
    total_minutes = (total_duration % 3600) // 60

    print()
    log_success("==========================================")
    log_success("TRAINING COMPLETED!")
    log_success("==========================================")
    log_success(f"Experiment: {args.exp_num}")
    log_success(f"Total Time: {total_hours}h {total_minutes}m")
    log_success(f"Models saved to: ./exp_{args.exp_num}/")
    log_success("==========================================")
    print()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
