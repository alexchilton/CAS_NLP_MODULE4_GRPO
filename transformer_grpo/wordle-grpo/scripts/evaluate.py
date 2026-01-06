#!/usr/bin/env python3
"""
Evaluation script for Wordle GRPO model.

This script evaluates a trained model by playing Wordle games and
measuring performance metrics including win rate, average guesses,
and reward scores.

Usage:
    # Evaluate a checkpoint
    python scripts/evaluate.py --checkpoint checkpoints/checkpoint_epoch_10

    # Evaluate with specific number of games
    python scripts/evaluate.py --checkpoint checkpoints/checkpoint_epoch_10 --num-games 100

    # Evaluate baseline (untrained model)
    python scripts/evaluate.py --config configs/dev_config.yaml --baseline --num-games 50

    # Compare checkpoint to baseline
    python scripts/evaluate.py --checkpoint checkpoints/checkpoint_epoch_10 --compare-baseline
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config, print_config_summary
from utils.device import get_device, print_device_info
from utils.logging import setup_logger
from model.setup import load_model_and_tokenizer, load_checkpoint
from data.dataset import load_word_list_csv
from evaluation.evaluator import evaluate_model, compare_checkpoints, generate_report
from training.reward_functions import CombinedReward


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Wordle GRPO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a trained checkpoint
  python scripts/evaluate.py --checkpoint checkpoints/checkpoint_epoch_10 --num-games 100

  # Evaluate baseline (untrained model)
  python scripts/evaluate.py --config configs/dev_config.yaml --baseline --num-games 50

  # Compare checkpoint to baseline
  python scripts/evaluate.py --checkpoint checkpoints/checkpoint_epoch_10 --compare-baseline

  # Evaluate multiple checkpoints
  python scripts/evaluate.py --compare checkpoints/checkpoint_epoch_5 checkpoints/checkpoint_epoch_10

  # Save results to specific directory
  python scripts/evaluate.py --checkpoint checkpoints/checkpoint_epoch_10 --output-dir eval_results
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dev_config.yaml",
        help="Path to configuration YAML file (default: configs/dev_config.yaml)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint directory to evaluate"
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate baseline (untrained) model instead of checkpoint"
    )

    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare checkpoint performance to baseline model"
    )

    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="Compare multiple checkpoints (provide space-separated paths)"
    )

    parser.add_argument(
        "--num-games",
        type=int,
        default=None,
        help="Number of games to evaluate (overrides config)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )

    parser.add_argument(
        "--word-list",
        type=str,
        default=None,
        help="Path to word list CSV file (if not using default)"
    )

    return parser.parse_args()


def load_word_list(args, config):
    """Load word list for evaluation."""
    # Try to load from args first
    if args.word_list:
        word_list_path = Path(args.word_list)
    else:
        # Try to get from config
        word_list_path = None
        if hasattr(config, 'data') and hasattr(config.data, 'word_list_path'):
            word_list_path = Path(config.data.word_list_path)

    # Default word list (you may need to update this path)
    if word_list_path is None or not word_list_path.exists():
        # Use a default set of common Wordle words
        print("⚠️  Word list path not found, using default word set")
        return ["CRANE", "TRAIN", "BRAIN", "GRAIN", "DRAIN", "SLATE", "CRATE", "TRACE"]

    try:
        word_list = load_word_list_csv(word_list_path)
        print(f"✓ Loaded {len(word_list)} words from {word_list_path}")
        return word_list
    except Exception as e:
        print(f"⚠️  Error loading word list: {e}")
        print("Using default word set")
        return ["CRANE", "TRAIN", "BRAIN", "GRAIN", "DRAIN", "SLATE", "CRATE", "TRACE"]


def main():
    """Main evaluation function."""
    args = parse_args()

    # Validate arguments
    if not args.baseline and args.checkpoint is None and args.compare is None:
        print("❌ Error: Must provide either --checkpoint, --baseline, or --compare")
        print("Use --help for usage examples")
        sys.exit(1)

    # Print header
    print("\n" + "=" * 60)
    print(" " * 15 + "WORDLE GRPO EVALUATION")
    print("=" * 60)
    print(f"Config: {args.config}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    if args.baseline:
        print("Mode: BASELINE (untrained model)")
    if args.compare:
        print(f"Comparing {len(args.compare)} checkpoints")
    print("=" * 60 + "\n")

    try:
        # Load configuration
        print("Step 1/6: Loading configuration...")
        config = load_config(args.config)
        print_config_summary(config)

        # Setup logging
        print("\nStep 2/6: Setting up logging...")
        log_level = logging.DEBUG if args.verbose else logging.INFO
        output_dir = Path(args.output_dir) if args.output_dir else Path("evaluation_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(
            name="wordle_eval",
            log_dir=output_dir,
            level=log_level,
        )
        logger.info("=" * 60)
        logger.info("WORDLE GRPO EVALUATION STARTED")
        logger.info("=" * 60)
        logger.info(f"Config file: {args.config}")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Setup device
        print("\nStep 3/6: Detecting device...")
        device = get_device()
        print_device_info()
        logger.info(f"Using device: {device}")

        # Load word list
        print("\nStep 4/6: Loading word list...")
        word_list = load_word_list(args, config)
        logger.info(f"Loaded {len(word_list)} words")

        # Determine number of games
        num_games = args.num_games
        if num_games is None:
            if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'num_eval_games'):
                num_games = config.evaluation.num_eval_games
            else:
                num_games = 100
        logger.info(f"Evaluating with {num_games} games")

        # Initialize reward function (optional, for scoring)
        reward_function = CombinedReward(
            format_weight=1.0,
            feedback_weight=0.5,
            value_weight=0.3,
        )

        # Handle different evaluation modes
        results = {}

        if args.compare:
            # Compare multiple checkpoints
            print("\nStep 5/6: Comparing multiple checkpoints...")
            checkpoint_paths = [Path(cp) for cp in args.compare]
            logger.info(f"Comparing {len(checkpoint_paths)} checkpoints")

            results = compare_checkpoints(
                checkpoint_paths=checkpoint_paths,
                config=config,
                word_list=word_list,
                num_games=num_games,
                output_dir=output_dir,
            )

        else:
            # Single model evaluation
            print("\nStep 5/6: Loading model...")

            if args.baseline:
                # Evaluate baseline (untrained) model
                logger.info("Loading baseline (untrained) model")
                model, tokenizer = load_model_and_tokenizer(config, device)
                model_name = "baseline"
            else:
                # Load checkpoint
                checkpoint_path = Path(args.checkpoint)
                logger.info(f"Loading checkpoint: {checkpoint_path}")
                model, tokenizer = load_checkpoint(checkpoint_path, config, device)
                model_name = checkpoint_path.name

            model.eval()
            logger.info("Model loaded successfully")

            # Run evaluation
            print("\nStep 6/6: Running evaluation...")
            print(f"Playing {num_games} games...")

            metrics = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                word_list=word_list,
                num_games=num_games,
                config=config,
                output_dir=output_dir,
                reward_function=reward_function,
            )

            results[model_name] = metrics

            # If comparing to baseline
            if args.compare_baseline and not args.baseline:
                print("\nComparing to baseline model...")
                logger.info("Evaluating baseline for comparison")

                # Load baseline
                baseline_model, baseline_tokenizer = load_model_and_tokenizer(config, device)
                baseline_model.eval()

                baseline_metrics = evaluate_model(
                    model=baseline_model,
                    tokenizer=baseline_tokenizer,
                    word_list=word_list,
                    num_games=num_games,
                    config=config,
                    output_dir=output_dir / "baseline",
                    reward_function=reward_function,
                )

                results["baseline"] = baseline_metrics

                # Clean up baseline model
                del baseline_model, baseline_tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Print results summary
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60 + "\n")

        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  Total games: {metrics['total_games']}")
            print(f"  Wins: {metrics['wins']}")
            print(f"  Losses: {metrics['losses']}")
            print(f"  Win rate: {metrics['win_rate']:.2%}")

            if metrics.get('avg_guesses_when_won', 0) > 0:
                print(f"  Avg guesses (wins): {metrics['avg_guesses_when_won']:.2f}")
                print(f"  Min guesses: {metrics['min_guesses']}")
                print(f"  Max guesses: {metrics['max_guesses']}")

                # Guess distribution
                if 'guess_distribution' in metrics:
                    print("\n  Guess distribution:")
                    for n in sorted(metrics['guess_distribution'].keys()):
                        count = metrics['guess_distribution'][n]
                        pct = count / metrics['wins'] * 100 if metrics['wins'] > 0 else 0
                        print(f"    {n} guesses: {count} ({pct:.1f}%)")

        # Generate markdown report
        print(f"\nGenerating report...")
        report_path = output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        generate_report(results, report_path)
        logger.info(f"Report generated: {report_path}")

        # Final summary
        print("\n" + "=" * 60)
        print("✓ EVALUATION COMPLETE")
        print("=" * 60)
        print(f"\nResults saved to: {output_dir}")
        print(f"Report: {report_path}")

        # Best model
        if len(results) > 1:
            best_model = max(results.items(), key=lambda x: x[1]['win_rate'])
            print(f"\nBest model: {best_model[0]} (win rate: {best_model[1]['win_rate']:.2%})")

            # Compare to baseline if available
            if "baseline" in results and len(results) > 1:
                baseline_wr = results["baseline"]["win_rate"]
                print(f"\nBaseline win rate: {baseline_wr:.2%}")
                for name, metrics in results.items():
                    if name != "baseline":
                        improvement = metrics["win_rate"] - baseline_wr
                        print(f"{name} improvement: {improvement:+.2%}")

        print("=" * 60 + "\n")

        logger.info("Evaluation completed successfully")

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ EVALUATION FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("=" * 60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
