"""
Dataset loader for Wordle GRPO training.

This module provides utilities to load and process the Wordle dataset
from HuggingFace for GRPO training.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from data.prompt_templates import add_few_shot_examples

logger = logging.getLogger(__name__)


class WordleDataset(Dataset):
    """
    PyTorch Dataset wrapper for the Wordle GRPO dataset.

    The dataset contains prompts and game history for training a model
    to play Wordle using Group Relative Policy Optimization (GRPO).
    """

    def __init__(
        self,
        dataset_name: str = "predibase/wordle-grpo",
        split: str = "train",
        word_list_path: Optional[Union[str, Path]] = None,
        max_samples: int = -1,
        use_few_shot: bool = True,
        num_examples: int = 2,
    ):
        """
        Initialize the Wordle dataset.

        Args:
            dataset_name: Name of the HuggingFace dataset.
            split: Dataset split to load (e.g., "train", "train[:100]").
            word_list_path: Optional path to word_list.csv file. If None, will try to load from dataset.
            max_samples: Maximum number of samples to load. -1 means load all.
            use_few_shot: Whether to add few-shot examples to prompts.
            num_examples: Number of few-shot examples to add (if use_few_shot=True).
        """
        self.use_few_shot = use_few_shot
        self.num_examples = num_examples
        logger.info(f"Loading dataset: {dataset_name}, split: {split}")

        # Load dataset from HuggingFace
        self.dataset = load_dataset(dataset_name, split=split)

        # Apply max_samples limit if specified
        if max_samples > 0 and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))
            logger.info(f"Limited dataset to {max_samples} samples")

        logger.info(f"Loaded {len(self.dataset)} samples")

        # Load word list if provided
        self.word_list = None
        if word_list_path is not None:
            self.word_list = self._load_word_list(word_list_path)

        # Validate dataset structure
        self._validate_dataset()

    def _load_word_list(self, word_list_path: Union[str, Path]) -> List[str]:
        """
        Load word list from CSV file.

        Args:
            word_list_path: Path to word_list.csv file.

        Returns:
            List of valid Wordle words.
        """
        word_list_path = Path(word_list_path)

        if not word_list_path.exists():
            logger.warning(f"Word list file not found: {word_list_path}")
            return []

        logger.info(f"Loading word list from: {word_list_path}")

        try:
            df = pd.read_csv(word_list_path)
            # Assuming the CSV has a column with words (adjust column name as needed)
            if "word" in df.columns:
                words = df["word"].tolist()
            elif "words" in df.columns:
                words = df["words"].tolist()
            else:
                # If no named column, assume first column contains words
                words = df.iloc[:, 0].tolist()

            logger.info(f"Loaded {len(words)} words from word list")
            return words
        except Exception as e:
            logger.error(f"Error loading word list: {e}")
            return []

    def _validate_dataset(self) -> None:
        """
        Validate that the dataset has required fields.

        Raises:
            ValueError: If required fields are missing.
        """
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")

        # Check first sample for expected fields
        sample = self.dataset[0]
        logger.info(f"Dataset fields: {list(sample.keys())}")

        # Common fields in Wordle GRPO dataset
        expected_fields = ["prompt"]  # Minimal requirement

        missing_fields = [field for field in expected_fields if field not in sample]

        if missing_fields:
            raise ValueError(f"Missing required fields in dataset: {missing_fields}")

        logger.info("Dataset validation passed")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary containing the sample data.
        """
        sample = self.dataset[idx]

        # Get the original prompt
        prompt = sample.get("prompt", "")

        # Add few-shot examples if enabled
        if self.use_few_shot:
            prompt = add_few_shot_examples(prompt, num_examples=self.num_examples)

        # Create a standardized format
        item = {
            "prompt": prompt,
            "past_guess_history": sample.get("past_guess_history", []),
            "word_list": self.word_list if self.word_list else sample.get("word_list", []),
            "idx": idx,
        }

        # Include any additional fields from the dataset
        for key, value in sample.items():
            if key not in item:
                item[key] = value

        return item

    def get_sample_prompt(self, idx: int) -> str:
        """
        Get the prompt for a specific sample.

        Args:
            idx: Index of the sample.

        Returns:
            The prompt string.
        """
        return self.dataset[idx].get("prompt", "")


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.

    This function handles batching of samples with varying structures.

    Args:
        batch: List of samples from the dataset.

    Returns:
        Batched data as a dictionary.
    """
    # Collect all prompts
    prompts = [item["prompt"] for item in batch]

    # Collect past guess histories (may be lists or None)
    past_guess_histories = [item.get("past_guess_history", []) for item in batch]

    # Collect indices
    indices = [item["idx"] for item in batch]

    # Create batched output
    batched = {
        "prompts": prompts,
        "past_guess_histories": past_guess_histories,
        "indices": indices,
    }

    # Include any other common fields
    if batch:
        for key in batch[0].keys():
            if key not in batched and key not in ["prompt", "past_guess_history", "idx"]:
                batched[key] = [item[key] for item in batch]

    return batched


def get_dataloader(
    dataset_name: str = "predibase/wordle-grpo",
    split: str = "train",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    word_list_path: Optional[Union[str, Path]] = None,
    max_samples: int = -1,
    use_few_shot: bool = True,
    num_examples: int = 2,
) -> DataLoader:
    """
    Create a DataLoader for the Wordle dataset.

    Args:
        dataset_name: Name of the HuggingFace dataset.
        split: Dataset split to load.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes for data loading.
        word_list_path: Optional path to word_list.csv file.
        max_samples: Maximum number of samples to load. -1 means load all.
        use_few_shot: Whether to add few-shot examples to prompts.
        num_examples: Number of few-shot examples to add.

    Returns:
        PyTorch DataLoader instance.
    """
    dataset = WordleDataset(
        dataset_name=dataset_name,
        split=split,
        word_list_path=word_list_path,
        max_samples=max_samples,
        use_few_shot=use_few_shot,
        num_examples=num_examples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        f"Created DataLoader with {len(dataset)} samples, "
        f"batch_size={batch_size}, shuffle={shuffle}"
    )

    return dataloader


def load_word_list_csv(word_list_path: Union[str, Path]) -> List[str]:
    """
    Standalone function to load word list from CSV.

    Args:
        word_list_path: Path to word_list.csv file.

    Returns:
        List of valid Wordle words.
    """
    word_list_path = Path(word_list_path)

    if not word_list_path.exists():
        raise FileNotFoundError(f"Word list file not found: {word_list_path}")

    df = pd.read_csv(word_list_path)

    # Try to find the column with words
    if "word" in df.columns:
        words = df["word"].tolist()
    elif "words" in df.columns:
        words = df["words"].tolist()
    else:
        # Assume first column contains words
        words = df.iloc[:, 0].tolist()

    # Clean and validate words (Wordle words are 5 letters)
    words = [str(word).strip().upper() for word in words if isinstance(word, str)]
    words = [word for word in words if len(word) == 5 and word.isalpha()]

    logger.info(f"Loaded {len(words)} valid 5-letter words")

    return words
