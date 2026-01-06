#!/usr/bin/env python3
"""
Quick script to view the Wordle GRPO dataset.

Usage:
    python scripts/view_data.py
    python scripts/view_data.py --num-examples 10
    python scripts/view_data.py --show-words
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def view_dataset(num_examples=5, show_words=False):
    """View the dataset examples."""
    from datasets import load_dataset

    data_dir = Path("data/cache")

    print("\n" + "=" * 60)
    print("WORDLE GRPO DATASET")
    print("=" * 60)

    # Load dataset
    grpo_dataset = load_dataset("predibase/wordle-grpo", cache_dir=str(data_dir))
    train_data = grpo_dataset['train']

    print(f"\nTotal examples: {len(train_data)}")
    print(f"Columns: {train_data.column_names}")

    # Show all secret words
    all_secrets = sorted(set(example['secret'] for example in train_data))
    print(f"\nUnique secret words: {len(all_secrets)}")

    if show_words:
        print("\nAll secret words:")
        for i in range(0, len(all_secrets), 10):
            row = all_secrets[i:i+10]
            print("  " + ", ".join(row))

    # Show examples
    print(f"\n{'=' * 60}")
    print(f"SAMPLE EXAMPLES (showing {min(num_examples, len(train_data))})")
    print("=" * 60)

    for i in range(min(num_examples, len(train_data))):
        example = train_data[i]

        print(f"\n{'─' * 60}")
        print(f"Example {i+1}/{len(train_data)}")
        print("─" * 60)

        print(f"Secret word: {example['secret']}")

        # Parse past guesses
        if example['past_guess_history']:
            history = example['past_guess_history']
            # Handle both string and list formats
            if isinstance(history, str):
                history = eval(history) if history else []

            if history:
                print(f"\nPrevious guesses ({len(history)}):")
                for item in history:
                    if isinstance(item, list) and len(item) >= 2:
                        guess, feedback = item[0], item[1]
                        print(f"  {guess} → {feedback}")
                    else:
                        print(f"  {item}")
        else:
            print("\nNo previous guesses (first guess)")

        # Show prompt preview
        prompt_lines = example['prompt'].split('\n')[:10]
        print(f"\nPrompt preview (first 10 lines):")
        for line in prompt_lines:
            if line.strip():
                print(f"  {line[:70]}")

    print(f"\n{'=' * 60}\n")


def view_word_list():
    """View the full Wordle word list."""
    word_list_path = Path("data/wordle_word_list.csv")

    if not word_list_path.exists():
        print("⚠️  Word list not found at data/wordle_word_list.csv")
        print("Download it with:")
        print('  curl -o data/wordle_word_list.csv "https://raw.githubusercontent.com/arnavgarg1/arnavgarg1/refs/heads/main/five_letter_words.csv"')
        return

    print("\n" + "=" * 60)
    print("WORDLE WORD LIST")
    print("=" * 60)

    with open(word_list_path, 'r') as f:
        words = [line.strip() for line in f if line.strip() and line.strip() != 'Word']

    print(f"\nTotal words: {len(words)}")
    print(f"\nFirst 50 words:")
    for i in range(0, min(50, len(words)), 10):
        row = words[i:i+10]
        print("  " + ", ".join(row))

    print(f"\nLast 20 words:")
    for i in range(max(0, len(words)-20), len(words), 10):
        row = words[i:i+10]
        print("  " + ", ".join(row))

    # Statistics
    print(f"\n{'=' * 60}")
    print("STATISTICS")
    print("=" * 60)

    # Letter frequency
    from collections import Counter
    letter_counts = Counter()
    for word in words:
        for letter in word:
            letter_counts[letter.upper()] += 1

    print("\nMost common letters:")
    for letter, count in letter_counts.most_common(10):
        pct = count / sum(letter_counts.values()) * 100
        print(f"  {letter}: {count} ({pct:.1f}%)")

    # Starting letters
    first_letters = Counter(word[0].upper() for word in words)
    print("\nMost common starting letters:")
    for letter, count in first_letters.most_common(5):
        print(f"  {letter}: {count} words")

    print(f"\n{'=' * 60}\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="View Wordle GRPO dataset")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of examples to show (default: 5)"
    )
    parser.add_argument(
        "--show-words",
        action="store_true",
        help="Show all secret words in dataset"
    )
    parser.add_argument(
        "--word-list",
        action="store_true",
        help="Show the full Wordle word list"
    )

    args = parser.parse_args()

    if args.word_list:
        view_word_list()
    else:
        view_dataset(args.num_examples, args.show_words)


if __name__ == "__main__":
    main()
