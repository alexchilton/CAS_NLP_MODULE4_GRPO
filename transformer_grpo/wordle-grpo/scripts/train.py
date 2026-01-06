#!/usr/bin/env python3
"""
Training script for Wordle GRPO model.

This script orchestrates the complete training process including:
- Configuration loading
- Model setup with LoRA
- Dataset loading
- GRPO training
- Checkpoint saving
- Evaluation

Usage:
    # Development training on Mac
    python scripts/train.py --config configs/dev_config.yaml

    # Production training on GPU
    python scripts/train.py --config configs/prod_config.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/prod_config.yaml --resume checkpoints/checkpoint_epoch_5

    # Dry run (test without training)
    python scripts/train.py --config configs/dev_config.yaml --dry-run
"""

import argparse
import logging
import sys
import signal
from pathlib import Path
from datetime import datetime

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config, print_config_summary, save_config
from utils.device import get_device, print_device_info
from utils.logging import setup_logger, log_metrics
from model.setup import load_model_and_tokenizer, prepare_model_for_training, print_model_info, save_model
from data.dataset import get_dataloader
from training.grpo_trainer import WordleGRPOTrainer
from training.reward_functions import CombinedReward

# Global variables for signal handling
trainer = None
current_epoch = 0


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully by saving checkpoint."""
    global trainer, current_epoch
    print("\n\n" + "=" * 60)
    print("⚠️  Training interrupted by user (Ctrl+C)")
    print("=" * 60)

    if trainer is not None:
        print("Saving emergency checkpoint...")
        try:
            checkpoint_path = trainer.save_checkpoint(
                epoch=current_epoch,
                metrics={"interrupted": True}
            )
            print(f"✓ Emergency checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"✗ Failed to save checkpoint: {e}")

    print("=" * 60)
    print("Exiting...")
    sys.exit(0)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Wordle GRPO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with dev config (Mac, small model, CPU/MPS)
  python scripts/train.py --config configs/dev_config.yaml

  # Train with prod config (GPU, larger model, quantization)
  python scripts/train.py --config configs/prod_config.yaml

  # Resume training from checkpoint
  python scripts/train.py --config configs/prod_config.yaml --resume checkpoints/checkpoint_epoch_5

  # Dry run (test setup without training)
  python scripts/train.py --config configs/dev_config.yaml --dry-run

  # Verbose logging
  python scripts/train.py --config configs/dev_config.yaml --verbose
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dev_config.yaml",
        help="Path to configuration YAML file (default: configs/dev_config.yaml)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run setup and validation without actual training"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    global trainer, current_epoch

    # Parse arguments
    args = parse_args()

    # Print header
    print("\n" + "=" * 60)
    print(" " * 15 + "WORDLE GRPO TRAINING")
    print("=" * 60)
    print(f"Config: {args.config}")
    if args.resume:
        print(f"Resume: {args.resume}")
    if args.dry_run:
        print("Mode: DRY RUN (no training)")
    print("=" * 60 + "\n")

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # 1. Load configuration
        print("Step 1/8: Loading configuration...")
        config = load_config(args.config)

        # Apply command-line overrides
        if args.epochs is not None:
            config.training.epochs = args.epochs
        if args.output_dir is not None:
            config.output.checkpoint_dir = args.output_dir
            config.output.log_dir = args.output_dir

        print_config_summary(config)

        # 2. Setup logging
        print("\nStep 2/8: Setting up logging...")
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logger = setup_logger(
            name="wordle_grpo",
            log_dir=config.output.log_dir,
            level=log_level,
        )
        logger.info("=" * 60)
        logger.info("WORDLE GRPO TRAINING STARTED")
        logger.info("=" * 60)
        logger.info(f"Config file: {args.config}")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Save config to output directory for reproducibility
        config_save_path = Path(config.output.checkpoint_dir) / "config.yaml"
        save_config(config, config_save_path)
        logger.info(f"Configuration saved to: {config_save_path}")

        # 3. Setup device
        print("\nStep 3/8: Detecting device...")
        device = get_device()
        print_device_info()
        logger.info(f"Using device: {device}")

        # Set random seed for reproducibility
        if hasattr(config, 'system') and hasattr(config.system, 'seed'):
            seed = config.system.seed
        else:
            seed = 42
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to: {seed}")

        # 4. Load model and tokenizer
        print("\nStep 4/8: Loading model and tokenizer...")
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            from model.setup import load_checkpoint
            model, tokenizer = load_checkpoint(args.resume, config, device)
            start_epoch = 0  # TODO: Load epoch from checkpoint state
        else:
            model, tokenizer = load_model_and_tokenizer(config, device)
            start_epoch = 0

        # Prepare model for training
        model = prepare_model_for_training(model)
        print_model_info(model, tokenizer)

        # 5. Load dataset
        print("\nStep 5/8: Loading dataset...")
        train_dataloader = get_dataloader(
            dataset_name=config.data.dataset_name,
            split=config.data.train_split,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for main process loading (more stable)
            max_samples=config.training.max_samples if config.training.max_samples > 0 else -1,
        )
        logger.info(f"Dataset loaded: {len(train_dataloader)} batches")

        # 6. Initialize reward function
        print("\nStep 6/8: Initializing reward function...")

        # Get word list path from config
        word_list_path = None
        if hasattr(config, 'data') and hasattr(config.data, 'word_list_path'):
            word_list_path = Path(config.data.word_list_path)
            if not word_list_path.is_absolute():
                # Make relative paths relative to project root
                word_list_path = Path(__file__).parent.parent / word_list_path

        reward_function = CombinedReward(
            format_weight=1.0,
            feedback_weight=0.5,
            value_weight=0.3,
            word_list_path=word_list_path,
        )
        logger.info(f"Reward function initialized (format=1.0, feedback=0.5, value=0.3, word_list={word_list_path})")

        # 7. Initialize trainer
        print("\nStep 7/8: Initializing GRPO trainer...")
        trainer = WordleGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_function=reward_function,
            config=config,
            save_dir=config.output.checkpoint_dir,
        )
        logger.info("GRPO trainer initialized")

        # If dry run, stop here
        if args.dry_run:
            print("\n" + "=" * 60)
            print("✓ DRY RUN COMPLETE")
            print("=" * 60)
            print("All components initialized successfully!")
            print("Ready for training. Remove --dry-run flag to start training.")
            print("=" * 60 + "\n")
            return

        # 8. Training loop
        print("\nStep 8/8: Starting training...")
        print("=" * 60)
        logger.info("Starting training loop")

        num_epochs = config.training.epochs
        logger.info(f"Training for {num_epochs} epochs")

        for epoch in range(start_epoch, num_epochs):
            current_epoch = epoch

            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 60}\n")

            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            # Train for one epoch
            epoch_metrics = trainer.train_epoch(train_dataloader, epoch)

            # Log epoch metrics
            logger.info(f"Epoch {epoch + 1} completed")
            log_metrics(logger, epoch_metrics, prefix=f"Epoch {epoch + 1}")

            # Save checkpoint
            save_interval = getattr(config.output, 'save_every_n_epochs', 1)
            if (epoch + 1) % save_interval == 0:
                print(f"\nSaving checkpoint for epoch {epoch + 1}...")
                checkpoint_path = trainer.save_checkpoint(epoch + 1, epoch_metrics)
                logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Training complete
        print("\n" + "=" * 60)
        print("✓ TRAINING COMPLETE")
        print("=" * 60)
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        # Save final checkpoint
        print("\nSaving final checkpoint...")
        final_checkpoint = trainer.save_checkpoint(num_epochs, {"final": True})
        logger.info(f"Final checkpoint saved: {final_checkpoint}")

        print("\nTraining artifacts:")
        print(f"  Checkpoints: {config.output.checkpoint_dir}")
        print(f"  Logs: {config.output.log_dir}")
        print("\nNext steps:")
        print(f"  1. Evaluate model: python scripts/evaluate.py --checkpoint {final_checkpoint}")
        print(f"  2. Compare checkpoints to see training progress")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        # Signal handler will take care of this
        pass

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TRAINING FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()

        # Try to save emergency checkpoint
        if trainer is not None:
            print("\nAttempting to save emergency checkpoint...")
            try:
                checkpoint_path = trainer.save_checkpoint(
                    epoch=current_epoch,
                    metrics={"error": str(e), "emergency": True}
                )
                print(f"✓ Emergency checkpoint saved: {checkpoint_path}")
            except:
                print("✗ Could not save emergency checkpoint")

        print("=" * 60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
