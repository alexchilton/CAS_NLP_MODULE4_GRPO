#!/usr/bin/env python3
"""
Data download script for Wordle GRPO project.

This script downloads all required datasets from HuggingFace and caches them
locally for training and evaluation. It's idempotent and safe to run multiple times.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --data-dir custom_data_dir
    python scripts/download_data.py --no-verify  # Skip verification
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_dataset_library():
    """Check if datasets library is available."""
    try:
        import datasets
        print(f"✓ datasets library available (version {datasets.__version__})")
        return True
    except ImportError:
        print("✗ datasets library not found")
        print("\nPlease install it:")
        print("  pip install datasets")
        return False


def download_hf_dataset(
    dataset_name: str,
    cache_dir: Path,
    force_download: bool = False
) -> Optional[Any]:
    """
    Download a HuggingFace dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'predibase/wordle-grpo')
        cache_dir: Directory to cache the dataset
        force_download: If True, re-download even if cached

    Returns:
        Loaded dataset or None if failed
    """
    try:
        from datasets import load_dataset

        print(f"\nDownloading {dataset_name}...")

        # Check if already cached
        if not force_download:
            try:
                dataset = load_dataset(
                    dataset_name,
                    cache_dir=str(cache_dir),
                    download_mode="reuse_dataset_if_exists"
                )
                print(f"✓ Loaded from cache")
                return dataset
            except:
                pass

        # Download fresh
        dataset = load_dataset(
            dataset_name,
            cache_dir=str(cache_dir),
            download_mode="force_redownload" if force_download else "reuse_dataset_if_exists"
        )

        print(f"✓ Downloaded successfully")
        return dataset

    except Exception as e:
        print(f"✗ Failed to download {dataset_name}: {e}")
        return None


def verify_dataset_structure(dataset, dataset_name: str, expected_splits: list = None) -> bool:
    """
    Verify dataset has expected structure.

    Args:
        dataset: HuggingFace dataset object
        dataset_name: Name for logging
        expected_splits: Expected split names (e.g., ['train', 'test'])

    Returns:
        True if valid, False otherwise
    """
    print(f"\nVerifying {dataset_name}...")

    try:
        # Check splits
        available_splits = list(dataset.keys())
        print(f"  Available splits: {available_splits}")

        if expected_splits:
            for split in expected_splits:
                if split not in available_splits:
                    print(f"  ⚠️  Expected split '{split}' not found")

        # Check each split
        for split_name, split_data in dataset.items():
            num_examples = len(split_data)
            print(f"  {split_name}: {num_examples} examples")

            if num_examples > 0:
                # Show columns
                columns = split_data.column_names
                print(f"    Columns: {columns}")

                # Show first example
                first_example = split_data[0]
                print(f"    Sample keys: {list(first_example.keys())}")

        print(f"✓ {dataset_name} structure verified")
        return True

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def print_dataset_statistics(dataset, dataset_name: str):
    """Print detailed statistics about a dataset."""
    print(f"\n{'=' * 60}")
    print(f"Statistics for {dataset_name}")
    print("=" * 60)

    try:
        total_examples = 0

        for split_name, split_data in dataset.items():
            num_examples = len(split_data)
            total_examples += num_examples

            print(f"\n{split_name.upper()} split:")
            print(f"  Number of examples: {num_examples}")
            print(f"  Columns: {split_data.column_names}")

            # Show data types
            if num_examples > 0:
                first_example = split_data[0]
                print(f"\n  Column types:")
                for key, value in first_example.items():
                    value_type = type(value).__name__
                    print(f"    {key}: {value_type}")

        print(f"\nTotal examples across all splits: {total_examples}")

    except Exception as e:
        print(f"Error computing statistics: {e}")


def show_sample_examples(dataset, dataset_name: str, num_samples: int = 3):
    """Show sample examples from the dataset."""
    print(f"\n{'=' * 60}")
    print(f"Sample Examples from {dataset_name}")
    print("=" * 60)

    try:
        # Get first split
        first_split = next(iter(dataset.values()))

        num_to_show = min(num_samples, len(first_split))

        for i in range(num_to_show):
            print(f"\nExample {i + 1}:")
            print("-" * 60)

            example = first_split[i]

            for key, value in example.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."

                print(f"{key}:")
                print(f"  {value_str}")

    except Exception as e:
        print(f"Error showing examples: {e}")


def download_word_list(data_dir: Path) -> bool:
    """
    Download or verify word list CSV.

    Downloads the official Wordle word list from GitHub.

    Args:
        data_dir: Directory to save word list

    Returns:
        True if successful, False otherwise
    """
    print("\nDownloading word list...")

    word_list_path = data_dir / "wordle_word_list.csv"
    word_list_url = "https://raw.githubusercontent.com/arnavgarg1/arnavgarg1/refs/heads/main/five_letter_words.csv"

    # Check if already exists
    if word_list_path.exists():
        print(f"✓ Word list already exists at {word_list_path}")
        try:
            with open(word_list_path, 'r') as f:
                lines = f.readlines()
                num_words = len([l for l in lines if l.strip() and not l.startswith('#') and l.strip() != 'Word'])
                print(f"  Contains {num_words} words")
            return True
        except Exception as e:
            print(f"⚠️  Could not read word list: {e}")

    # Download from GitHub
    print(f"Downloading from {word_list_url}...")

    try:
        import urllib.request

        # Download the file
        word_list_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(word_list_url, word_list_path)

        # Count words
        with open(word_list_path, 'r') as f:
            lines = f.readlines()
            num_words = len([l for l in lines if l.strip() and not l.startswith('#') and l.strip() != 'Word'])

        print(f"✓ Downloaded {num_words} words to {word_list_path}")
        return True

    except Exception as e:
        print(f"✗ Failed to download word list: {e}")
        print("   You can manually download it from:")
        print(f"   {word_list_url}")
        return False


def save_download_metadata(data_dir: Path, datasets_info: Dict[str, Any]):
    """Save metadata about downloaded datasets."""
    metadata_path = data_dir / "download_metadata.json"

    try:
        import datetime

        metadata = {
            "download_timestamp": datetime.datetime.now().isoformat(),
            "datasets": datasets_info
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Metadata saved to {metadata_path}")

    except Exception as e:
        print(f"⚠️  Could not save metadata: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Wordle GRPO datasets from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python scripts/download_data.py

  # Use custom data directory
  python scripts/download_data.py --data-dir /path/to/data

  # Force re-download
  python scripts/download_data.py --force-download

  # Skip verification step
  python scripts/download_data.py --no-verify
        """
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store downloaded data (default: data/)"
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if data exists"
    )

    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip dataset verification"
    )

    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Don't show sample examples"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of sample examples to show (default: 3)"
    )

    return parser.parse_args()


def main():
    """Main download function."""
    args = parse_args()

    # Print header
    print("\n" + "=" * 60)
    print(" " * 15 + "WORDLE GRPO DATA DOWNLOAD")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    if args.force_download:
        print("Mode: FORCE RE-DOWNLOAD")
    print("=" * 60)

    # Check dependencies
    print_section("1. Checking Dependencies")
    if not check_dataset_library():
        sys.exit(1)

    # Setup data directory
    print_section("2. Setting Up Data Directory")
    data_dir = Path(args.data_dir)
    cache_dir = data_dir / "cache"

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Data directory: {data_dir.absolute()}")
    print(f"✓ Cache directory: {cache_dir.absolute()}")

    # Download datasets
    print_section("3. Downloading HuggingFace Datasets")

    datasets_info = {}
    downloaded_datasets = {}

    # Dataset 1: wordle-grpo (main training dataset)
    print("\n" + "-" * 60)
    print("Dataset 1: predibase/wordle-grpo (Main GRPO Dataset)")
    print("-" * 60)

    grpo_dataset = download_hf_dataset(
        "predibase/wordle-grpo",
        cache_dir,
        args.force_download
    )

    if grpo_dataset:
        downloaded_datasets["wordle-grpo"] = grpo_dataset
        datasets_info["wordle-grpo"] = {
            "name": "predibase/wordle-grpo",
            "status": "downloaded",
            "splits": list(grpo_dataset.keys())
        }
    else:
        datasets_info["wordle-grpo"] = {
            "name": "predibase/wordle-grpo",
            "status": "failed"
        }

    # Dataset 2: wordle-sft (reference/comparison)
    print("\n" + "-" * 60)
    print("Dataset 2: predibase/wordle-sft (Reference SFT Dataset)")
    print("-" * 60)

    sft_dataset = download_hf_dataset(
        "predibase/wordle-sft",
        cache_dir,
        args.force_download
    )

    if sft_dataset:
        downloaded_datasets["wordle-sft"] = sft_dataset
        datasets_info["wordle-sft"] = {
            "name": "predibase/wordle-sft",
            "status": "downloaded",
            "splits": list(sft_dataset.keys())
        }
    else:
        datasets_info["wordle-sft"] = {
            "name": "predibase/wordle-sft",
            "status": "failed"
        }

    # Verify datasets
    if not args.no_verify:
        print_section("4. Verifying Dataset Integrity")

        for name, dataset in downloaded_datasets.items():
            verify_dataset_structure(dataset, name)

    # Download word list
    print_section("5. Checking Word List")
    download_word_list(data_dir)

    # Print statistics
    print_section("6. Dataset Statistics")

    for name, dataset in downloaded_datasets.items():
        print_dataset_statistics(dataset, name)

    # Show sample examples
    if not args.no_samples and downloaded_datasets:
        print_section("7. Sample Examples")

        for name, dataset in downloaded_datasets.items():
            show_sample_examples(dataset, name, args.num_samples)

    # Save metadata
    print_section("8. Saving Metadata")
    save_download_metadata(data_dir, datasets_info)

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    successful = sum(1 for info in datasets_info.values() if info["status"] == "downloaded")
    total = len(datasets_info)

    print(f"\nSuccessfully downloaded: {successful}/{total} datasets")

    if successful == total:
        print("\n✅ All datasets downloaded successfully!")
        print("\nNext steps:")
        print(f"  1. Data is cached in: {cache_dir.absolute()}")
        print("  2. Run setup test: python scripts/test_setup.py")
        print("  3. Start training: python scripts/train.py --config configs/dev_config.yaml")
    else:
        print("\n⚠️  Some datasets failed to download")
        print("Check your internet connection and HuggingFace access")

        failed = [name for name, info in datasets_info.items() if info["status"] == "failed"]
        print(f"\nFailed datasets: {', '.join(failed)}")

    print("=" * 60 + "\n")

    # Return exit code based on success
    sys.exit(0 if successful == total else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
