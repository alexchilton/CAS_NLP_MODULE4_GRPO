

import sys
from pathlib import Path
from datasets import load_dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def view_sft_dataset(num_examples=3):
    """View the SFT dataset examples."""
    data_dir = Path("data/cache")

    print("\n" + "=" * 60)
    print("WORDLE SFT DATASET")
    print("=" * 60)

    # Load dataset
    sft_dataset = load_dataset("predibase/wordle-sft", cache_dir=str(data_dir))
    train_data = sft_dataset['train']

    print(f"\nTotal examples: {len(train_data)}")
    print(f"Columns: {train_data.column_names}")

    # Show examples
    print(f"\n{'=' * 60}")
    print(f"SAMPLE EXAMPLES (showing {min(num_examples, len(train_data))})")
    print("=" * 60)

    for i in range(min(num_examples, len(train_data))):
        example = train_data[i]

        print(f"\n{'─' * 60}")
        print(f"Example {i+1}/{len(train_data)}")
        print("─" * 60)

        print("Prompt:")
        print(example["prompt"])
        print("\nCompletion:")
        print(example["completion"])

    print(f"\n{'=' * 60}\n")

if __name__ == "__main__":
    view_sft_dataset()

