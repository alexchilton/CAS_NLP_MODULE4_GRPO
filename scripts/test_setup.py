#!/usr/bin/env python3
"""
Comprehensive setup test script.

This script verifies that all components of the Wordle GRPO project
are correctly installed and working. Run this before training to
ensure everything is operational.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_imports():
    """Test that all required packages can be imported."""
    print_section("1. Testing Package Imports")

    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("peft", "PEFT (LoRA)"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
    ]

    failed = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            failed.append(name)

    # Test optional packages
    try:
        import bitsandbytes
        print(f"  ✓ bitsandbytes (optional)")
    except ImportError:
        print(f"  ⚠ bitsandbytes - NOT AVAILABLE (ok for Mac)")

    return failed


def test_device_detection():
    """Test device detection and print device info."""
    print_section("2. Testing Device Detection")

    try:
        from utils.device import get_device, is_cuda_available, is_mps_available, print_device_info
        import torch

        print(f"  PyTorch version: {torch.__version__}")

        device = get_device()
        print(f"  ✓ Detected device: {device}")
        print(f"  CUDA available: {is_cuda_available()}")
        print(f"  MPS available: {is_mps_available()}")

        print("\nDetailed device info:")
        print_device_info()

        return True
    except Exception as e:
        print(f"  ✗ Device detection failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print_section("3. Testing Configuration Loading")

    try:
        from utils.config import load_config, print_config_summary

        # Test loading dev config
        config = load_config("configs/dev_config.yaml")
        print(f"  ✓ Loaded dev_config.yaml")

        # Verify key fields
        assert hasattr(config, 'model'), "Missing 'model' section"
        assert hasattr(config, 'training'), "Missing 'training' section"
        assert hasattr(config, 'data'), "Missing 'data' section"
        print(f"  ✓ Config structure validated")

        # Print summary
        print("\nConfig summary:")
        print(f"  Model: {config.model.name}")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  LoRA rank: {config.model.lora_rank}")

        return config
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_loading(config):
    """Test loading a small model."""
    print_section("4. Testing Model Loading")

    try:
        from model.setup import load_model_and_tokenizer, print_model_info
        import torch

        print(f"  Loading model: {config.model.name}")
        model, tokenizer = load_model_and_tokenizer(config)

        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Tokenizer loaded successfully")

        # Print model info
        print_model_info(model, tokenizer)

        return model, tokenizer
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_dataset_loading(config):
    """Test dataset loading."""
    print_section("5. Testing Dataset Loading")

    try:
        from data.dataset import get_dataloader

        print(f"  Loading dataset: {config.data.dataset_name}")
        print(f"  Split: {config.data.train_split}")

        dataloader = get_dataloader(
            dataset_name=config.data.dataset_name,
            split=config.data.train_split,
            batch_size=1,
            max_samples=5,  # Just load 5 examples for testing
        )

        print(f"  ✓ Dataset loaded successfully")

        # Get first batch
        batch = next(iter(dataloader))
        print(f"  ✓ Retrieved first batch")
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  Number of prompts: {len(batch['prompts'])}")

        if batch['prompts']:
            print(f"\n  Sample prompt:")
            print(f"  {batch['prompts'][0][:200]}...")

        return batch
    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_generation(model, tokenizer):
    """Test text generation."""
    print_section("6. Testing Text Generation")

    try:
        from model.generation import generate_single

        test_prompt = "You are playing Wordle. Make a guess.\n\n<think>"

        print(f"  Test prompt: {test_prompt[:50]}...")
        print(f"  Generating completion...")

        completion = generate_single(
            model=model,
            tokenizer=tokenizer,
            prompt=test_prompt,
            max_new_tokens=50,
            temperature=0.7,
        )

        print(f"  ✓ Generation successful")
        print(f"\n  Generated text:")
        print(f"  {completion}")

        return test_prompt, completion
    except Exception as e:
        print(f"  ✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_reward_computation(prompt, completion):
    """Test reward function computation."""
    print_section("7. Testing Reward Functions")

    try:
        from training.reward_functions import (
            output_format_check,
            uses_previous_feedback,
            guess_value,
            CombinedReward,
        )

        # Create example
        example = {
            "past_guess_history": [],
            "word_list": ["CRANE", "TRAIN", "BRAIN", "GRAIN"],
        }

        print(f"  Testing output_format_check...")
        format_reward = output_format_check(prompt, completion, example)
        print(f"  ✓ Format reward: {format_reward:.3f}")

        print(f"  Testing uses_previous_feedback...")
        feedback_reward = uses_previous_feedback(prompt, completion, example)
        print(f"  ✓ Feedback reward: {feedback_reward:.3f}")

        print(f"  Testing guess_value...")
        value_reward = guess_value(prompt, completion, example)
        print(f"  ✓ Value reward: {value_reward:.3f}")

        print(f"\n  Testing CombinedReward...")
        combined = CombinedReward(
            format_weight=1.0,
            feedback_weight=0.5,
            value_weight=0.3,
        )
        total_reward = combined(prompt, completion, example)
        print(f"  ✓ Combined reward: {total_reward:.3f}")

        return True
    except Exception as e:
        print(f"  ✗ Reward computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wordle_game():
    """Test Wordle game utilities."""
    print_section("8. Testing Wordle Game Logic")

    try:
        from data.wordle_game import (
            validate_guess,
            is_winning_guess,
            format_feedback,
        )

        # Test validation
        secret = "CRANE"
        guess = "TRAIN"

        print(f"  Secret word: {secret}")
        print(f"  Guess: {guess}")

        feedback = validate_guess(secret, guess)
        print(f"  ✓ Feedback: {feedback}")

        formatted = format_feedback(feedback)
        print(f"  ✓ Formatted: {formatted}")

        is_win = is_winning_guess(feedback)
        print(f"  ✓ Is winning guess: {is_win}")

        # Test winning guess
        feedback_win = validate_guess(secret, secret)
        is_win = is_winning_guess(feedback_win)
        print(f"  ✓ Correct guess detected as win: {is_win}")

        return True
    except Exception as e:
        print(f"  ✗ Wordle game logic failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_initialization(model, tokenizer, config):
    """Test GRPO trainer initialization."""
    print_section("9. Testing GRPO Trainer Initialization")

    try:
        from training.grpo_trainer import WordleGRPOTrainer
        from training.reward_functions import CombinedReward

        print(f"  Creating reward function...")
        reward_function = CombinedReward()

        print(f"  Initializing trainer...")
        trainer = WordleGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_function=reward_function,
            config=config,
        )

        print(f"  ✓ Trainer initialized successfully")
        print(f"  Learning rate: {trainer.learning_rate}")
        print(f"  Num generations: {trainer.num_generations}")
        print(f"  Gradient accumulation: {trainer.gradient_accumulation_steps}")

        return True
    except Exception as e:
        print(f"  ✗ Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluator():
    """Test evaluator."""
    print_section("10. Testing Evaluator")

    try:
        from evaluation.evaluator import WordleEvaluator

        print(f"  ✓ Evaluator imports successfully")

        # Don't actually run evaluation, just test initialization
        # would be tested in actual evaluation

        return True
    except Exception as e:
        print(f"  ✗ Evaluator import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print(" " * 15 + "WORDLE GRPO SETUP TEST")
    print("=" * 60)
    print("This script verifies all components are working correctly.")
    print("=" * 60)

    all_passed = True
    model = None
    tokenizer = None
    config = None
    batch = None
    prompt = None
    completion = None

    # 1. Test imports
    failed_imports = test_imports()
    if failed_imports:
        all_passed = False
        print(f"\n⚠️  Missing packages: {', '.join(failed_imports)}")
        print("   Run: pip install -r requirements-dev.txt")
        return False

    # 2. Test device detection
    if not test_device_detection():
        all_passed = False

    # 3. Test config loading
    config = test_config_loading()
    if config is None:
        all_passed = False
        return False

    # 4. Test model loading
    model, tokenizer = test_model_loading(config)
    if model is None or tokenizer is None:
        all_passed = False
        print("\n⚠️  Cannot continue without model. Stopping tests.")
        return False

    # 5. Test dataset loading
    batch = test_dataset_loading(config)
    if batch is None:
        print("\n⚠️  Dataset loading failed but continuing...")
        all_passed = False

    # 6. Test generation
    prompt, completion = test_generation(model, tokenizer)
    if prompt is None or completion is None:
        print("\n⚠️  Generation failed but continuing...")
        all_passed = False

    # 7. Test reward computation
    if prompt and completion:
        if not test_reward_computation(prompt, completion):
            all_passed = False

    # 8. Test Wordle game logic
    if not test_wordle_game():
        all_passed = False

    # 9. Test trainer initialization
    if not test_trainer_initialization(model, tokenizer, config):
        all_passed = False

    # 10. Test evaluator
    if not test_evaluator():
        all_passed = False

    # Summary
    print_section("Test Summary")

    if all_passed:
        print("\n" + "  " + "✅ " * 10)
        print("\n  " + " " * 10 + "ALL SYSTEMS OPERATIONAL!")
        print("\n" + "  " + "✅ " * 10)
        print("\n✓ All components are working correctly!")
        print("✓ Ready for training!")
        print("\nNext steps:")
        print("  1. Review configs/dev_config.yaml for Mac training")
        print("  2. Run: python scripts/train.py --config configs/dev_config.yaml")
        print("  3. Or transfer to laptop and use configs/prod_config.yaml")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        print("\nCommon issues:")
        print("  - Missing packages: pip install -r requirements-dev.txt")
        print("  - Dataset access: Check internet connection")
        print("  - Model loading: May need HuggingFace token for some models")
        sys.exit(1)

    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
