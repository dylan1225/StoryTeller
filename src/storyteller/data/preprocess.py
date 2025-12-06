"""
Preprocess and clean story datasets.

This script:
1. Loads raw text files
2. Cleans and filters text
3. Splits into train/val/test sets
4. Saves processed data

Usage:
    python data/preprocess.py --input_dir data/raw --output_dir data/processed
"""

import argparse
import random
import re
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 consecutive newlines
    text = re.sub(r" {2,}", " ", text)  # Max 1 space
    text = re.sub(r"\t+", " ", text)  # Replace tabs with space

    # Remove special characters that might cause issues
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(""", "'").replace(""", "'")

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def split_into_stories(text: str, min_length: int = 100) -> List[str]:
    """
    Split text into individual stories.

    Args:
        text: Combined text
        min_length: Minimum story length in characters

    Returns:
        List of story strings
    """
    # Split on double newlines (stories are separated this way)
    stories = text.split("\n\n")

    # Filter short stories and clean each one
    cleaned_stories = []
    for story in tqdm(stories, desc="  Cleaning stories", leave=False):
        story = clean_text(story)
        if len(story) >= min_length:
            cleaned_stories.append(story)

    return cleaned_stories


def filter_quality(stories: List[str], max_length: int = 10000) -> List[str]:
    """
    Filter stories based on quality heuristics.

    Args:
        stories: List of story strings
        max_length: Maximum story length in characters

    Returns:
        Filtered list of stories
    """
    filtered = []

    for story in tqdm(stories, desc="  Filtering stories", leave=False):
        # Skip very long stories (might be corrupted or full books)
        if len(story) > max_length:
            continue

        # Skip stories with too many special characters (likely corrupted)
        special_char_ratio = sum(
            1 for c in story if not c.isalnum() and c not in " \n.,!?'\":-;"
        ) / len(story)
        if special_char_ratio > 0.3:
            continue

        # Skip stories that are mostly uppercase (likely metadata or titles)
        if sum(1 for c in story if c.isupper()) / len(story) > 0.5:
            continue

        # Require reasonable word count
        words = story.split()
        if len(words) < 20:  # Too short
            continue

        filtered.append(story)

    return filtered


def train_val_test_split(
    stories: List[str],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split stories into train/val/test sets.

    Args:
        stories: List of story strings
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train, val, test) story lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    if shuffle:
        random.seed(seed)
        stories = stories.copy()
        random.shuffle(stories)

    n = len(stories)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = stories[:train_end]
    val = stories[train_end:val_end]
    test = stories[val_end:]

    return train, val, test


def save_stories(stories: List[str], output_file: Path):
    """Save stories to a text file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for story in tqdm(stories, desc=f"  Saving {output_file.name}", leave=False):
            f.write(story + "\n\n")


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    min_length: int = 100,
    max_length: int = 10000,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
):
    """
    Process all datasets in input directory.

    Args:
        input_dir: Directory containing raw .txt files
        output_dir: Directory to save processed data
        min_length: Minimum story length
        max_length: Maximum story length
        train_ratio: Training set proportion
        val_ratio: Validation set proportion
        test_ratio: Test set proportion
    """
    print("Processing datasets...\n")

    all_stories = []

    # Load and process each dataset
    for txt_file in sorted(input_dir.glob("*.txt")):
        print(f"Processing {txt_file.name}...")

        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into stories
        stories = split_into_stories(text, min_length=min_length)
        print(f"  Found {len(stories):,} stories")

        # Filter by quality
        stories = filter_quality(stories, max_length=max_length)
        print(f"  After filtering: {len(stories):,} stories")

        all_stories.extend(stories)

    print(f"\nTotal stories: {len(all_stories):,}")

    # Split into train/val/test
    print("\nSplitting into train/val/test...")
    train, val, test = train_val_test_split(
        all_stories,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    print(f"  Train: {len(train):,} stories")
    print(f"  Val: {len(val):,} stories")
    print(f"  Test: {len(test):,} stories")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    print("\nSaving processed data...")
    save_stories(train, output_dir / "train.txt")
    save_stories(val, output_dir / "val.txt")
    save_stories(test, output_dir / "test.txt")

    # Calculate statistics
    total_chars = sum(len(s) for s in all_stories)
    total_words = sum(len(s.split()) for s in all_stories)

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total stories: {len(all_stories):,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total words (approx): {total_words:,}")
    print(f"\nData saved to: {output_dir}")
    print("\nNext step:")
    print("  Run tokenizer_training.py to train a custom tokenizer")


def main():
    parser = argparse.ArgumentParser(description="Preprocess story datasets")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw",
        help="Directory containing raw datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=100,
        help="Minimum story length in characters",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=10000,
        help="Maximum story length in characters",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Training set proportion",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation set proportion",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="Test set proportion",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        print("Please run download_stories.py first")
        return

    process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()
