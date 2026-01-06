"""
Three-Stage Training Orchestrator
==================================

This script runs all three training stages sequentially:
1. Stage 1: Pure Format SFT (until 90%+ format accuracy)
2. Stage 2: Light GRPO (strategy learning with format masking)
3. Stage 3: Full GRPO (polish with all penalties)

Usage:
    python run_all_stages.py [--stage N] [--skip-data-gen]

Options:
    --stage N         Run only stage N (1, 2, or 3)
    --skip-data-gen   Skip synthetic data generation (use existing data)

Examples:
    python run_all_stages.py                # Run all stages
    python run_all_stages.py --stage 2      # Run only stage 2
    python run_all_stages.py --skip-data-gen  # Run all, skip data gen
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime


def log(msg):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def run_stage(script_name, stage_name):
    """Run a training stage script"""
    log(f"Starting {stage_name}...")
    log(f"Running: python {script_name}")

    try:
        result = subprocess.run(
            ["python", script_name],
            check=True,
            capture_output=False,
            text=True
        )
        log(f"✅ {stage_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        log(f"❌ {stage_name} failed with error code {e.returncode}")
        log(f"Error: {e}")
        return False


def check_stage_output(stage_num):
    """Check if a stage has already been completed"""
    output_dir = f"stage{stage_num}_output/final_model"
    metadata_file = f"{output_dir}/stage{stage_num}_metadata.json"

    if Path(output_dir).exists() and Path(metadata_file).exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        log(f"Stage {stage_num} already completed on {metadata.get('training_date', 'unknown')}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Run three-stage Wordle training pipeline")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Run only specific stage")
    parser.add_argument("--skip-data-gen", action="store_true", help="Skip synthetic data generation")
    parser.add_argument("--force", action="store_true", help="Force re-run even if stage already completed")

    args = parser.parse_args()

    log("=" * 80)
    log("THREE-STAGE WORDLE TRAINING PIPELINE")
    log("=" * 80)
    log("")
    log("Pipeline stages:")
    log("  Stage 1: Pure Format SFT (teach output format to 90%+ accuracy)")
    log("  Stage 2: Light GRPO (strategy learning with format masking)")
    log("  Stage 3: Full GRPO (polish with all penalties)")
    log("")

    # Determine which stages to run
    if args.stage:
        stages_to_run = [args.stage]
        log(f"Running only Stage {args.stage}")
    else:
        stages_to_run = [1, 2, 3]
        log("Running all stages sequentially")

    log("")

    # Stage 0: Generate synthetic data (if needed)
    if 1 in stages_to_run and not args.skip_data_gen:
        data_file = Path("sft_synthetic_data.jsonl")
        if data_file.exists():
            log("Synthetic data already exists, skipping generation")
        else:
            log("=" * 80)
            log("STAGE 0: Generating Synthetic SFT Data")
            log("=" * 80)
            if not run_stage("generate_sft_data.py", "Data Generation"):
                log("❌ Data generation failed. Aborting.")
                sys.exit(1)
            log("")

    # Stage 1: Format SFT
    if 1 in stages_to_run:
        if not args.force and check_stage_output(1):
            log("Stage 1 already completed. Use --force to re-run.")
        else:
            log("=" * 80)
            log("STAGE 1: Pure Format SFT")
            log("=" * 80)
            if not run_stage("stage1_format_sft.py", "Stage 1"):
                log("❌ Stage 1 failed. Aborting.")
                sys.exit(1)
        log("")

    # Stage 2: Light GRPO
    if 2 in stages_to_run:
        # Check if Stage 1 is complete (prerequisite)
        if not check_stage_output(1):
            log("❌ Stage 1 must be completed before running Stage 2")
            sys.exit(1)

        if not args.force and check_stage_output(2):
            log("Stage 2 already completed. Use --force to re-run.")
        else:
            log("=" * 80)
            log("STAGE 2: Light GRPO with Format Masking")
            log("=" * 80)
            if not run_stage("stage2_light_grpo.py", "Stage 2"):
                log("❌ Stage 2 failed. Aborting.")
                sys.exit(1)
        log("")

    # Stage 3: Full GRPO
    if 3 in stages_to_run:
        # Check if Stage 2 is complete (prerequisite)
        if not check_stage_output(2):
            log("❌ Stage 2 must be completed before running Stage 3")
            sys.exit(1)

        if not args.force and check_stage_output(3):
            log("Stage 3 already completed. Use --force to re-run.")
        else:
            log("=" * 80)
            log("STAGE 3: Full GRPO Polish")
            log("=" * 80)
            if not run_stage("stage3_full_grpo.py", "Stage 3"):
                log("❌ Stage 3 failed. Aborting.")
                sys.exit(1)
        log("")

    # Summary
    log("=" * 80)
    log("🎉 TRAINING PIPELINE COMPLETE!")
    log("=" * 80)
    log("")
    log("Outputs:")

    if 1 in stages_to_run and check_stage_output(1):
        log("  Stage 1: stage1_output/final_model")

    if 2 in stages_to_run and check_stage_output(2):
        log("  Stage 2: stage2_output/final_model")

    if 3 in stages_to_run and check_stage_output(3):
        log("  Stage 3: stage3_output/final_model")
        log("           stage3_output/best_model (recommended for deployment)")

    log("")
    log("Next steps:")
    log("  1. Evaluate the final model: python ../test_model_comparison.py")
    log("  2. Compare all stages: python compare_stages.py")
    log("  3. Deploy best model from stage3_output/best_model")
    log("")


if __name__ == "__main__":
    main()