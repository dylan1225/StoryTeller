"""
Download and prepare story datasets.

This script downloads the following datasets:
1. TinyStories (Microsoft) - Simple, clean stories
2. WritingPrompts (Reddit) - Creative fiction from r/WritingPrompts
3. Project Gutenberg - Classic literature (sample)

Usage:
    python data/download_stories.py --output_dir data/raw --datasets all
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def download_tinystories(output_dir: Path):
    """
    Download TinyStories dataset from HuggingFace.

    TinyStories is a dataset of simple, AI-generated stories designed for
    training small language models.
    """
    print("Downloading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    output_file = output_dir / "tinystories.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for example in tqdm(dataset, desc="Processing TinyStories"):
            # Each example has a 'text' field containing the story
            story = example["text"].strip()
            if story:
                f.write(story + "\n\n")  # Double newline between stories

    print(f"✓ TinyStories saved to {output_file}")
    print(f"  Total examples: {len(dataset):,}")


def download_writing_prompts(output_dir: Path, max_examples: int = 100000):
    """
    Download WritingPrompts dataset from HuggingFace.

    WritingPrompts contains creative fiction stories from Reddit's r/WritingPrompts.
    """
    print("Downloading WritingPrompts dataset...")

    try:
        # Try the correct dataset path
        dataset = load_dataset("euclaise/writingprompts", split="train")

        # Limit dataset size for faster training
        if max_examples and len(dataset) > max_examples:
            dataset = dataset.select(range(max_examples))

        output_file = output_dir / "writingprompts.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for example in tqdm(dataset, desc="Processing WritingPrompts"):
                # WritingPrompts has 'prompt' and 'story' fields
                prompt = example.get("prompt", "").strip()
                story = example.get("story", "").strip()

                if story and len(story) > 100:  # Filter very short stories
                    # Optionally include prompt as context
                    # f.write(f"Prompt: {prompt}\n\n")
                    f.write(story + "\n\n")

        print(f"✓ WritingPrompts saved to {output_file}")
        print(f"  Total examples: {min(len(dataset), max_examples):,}")

    except Exception as e:
        print(f"⚠ Could not download WritingPrompts dataset: {e}")
        print("  This is optional and can be skipped.")
        print("  Trying to continue with other datasets...")


def download_gutenberg_sample(output_dir: Path, max_books: int = 100):
    """
    Download a sample of Project Gutenberg books.

    Project Gutenberg provides classic literature in the public domain.
    """
    print("Downloading Project Gutenberg sample...")

    try:
        # Using the pg19 dataset (Project Gutenberg subset)
        dataset = load_dataset("pg19", split="train")

        if max_books and len(dataset) > max_books:
            dataset = dataset.select(range(max_books))

        output_file = output_dir / "gutenberg.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for example in tqdm(dataset, desc="Processing Gutenberg"):
                text = example["text"].strip()
                if text and len(text) > 1000:  # Filter very short texts
                    f.write(text + "\n\n")

        print(f"✓ Gutenberg saved to {output_file}")
        print(f"  Total books: {min(len(dataset), max_books):,}")

    except Exception as e:
        print(f"⚠ Could not download Gutenberg dataset: {e}")
        print("  This is optional and can be skipped.")


def download_bookcorpus_sample(output_dir: Path, max_examples: int = 10000):
    """
    Download a sample of BookCorpus.

    BookCorpus contains modern fiction books.
    Note: The original BookCorpus is no longer available, using alternatives.
    """
    print("Downloading BookCorpus sample...")

    try:
        # Using bookcorpusopen as an alternative
        dataset = load_dataset("bookcorpusopen", split="train")

        if max_examples and len(dataset) > max_examples:
            dataset = dataset.select(range(max_examples))

        output_file = output_dir / "bookcorpus.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for example in tqdm(dataset, desc="Processing BookCorpus"):
                text = example["text"].strip()
                if text and len(text) > 100:
                    f.write(text + "\n\n")

        print(f"✓ BookCorpus saved to {output_file}")
        print(f"  Total examples: {min(len(dataset), max_examples):,}")

    except Exception as e:
        print(f"⚠ Could not download BookCorpus dataset: {e}")
        print("  This is optional and can be skipped.")


def get_dataset_stats(data_dir: Path):
    """Print statistics about downloaded datasets."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    total_size = 0
    total_chars = 0

    for file in sorted(data_dir.glob("*.txt")):
        size = file.stat().st_size / (1024 * 1024)  # MB
        total_size += size

        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            chars = len(content)
            total_chars += chars
            words = len(content.split())

        print(f"\n{file.name}:")
        print(f"  Size: {size:.2f} MB")
        print(f"  Characters: {chars:,}")
        print(f"  Words (approx): {words:,}")

    print(f"\n{'=' * 60}")
    print(f"Total size: {total_size:.2f} MB")
    print(f"Total characters: {total_chars:,}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Download story datasets")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Directory to save downloaded datasets",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "tinystories", "writingprompts", "gutenberg", "bookcorpus"],
        help="Which datasets to download",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples per dataset (for testing)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading datasets to: {output_dir}\n")

    datasets = args.datasets
    if "all" in datasets:
        datasets = ["tinystories", "writingprompts", "gutenberg", "bookcorpus"]

    # Download requested datasets
    success_count = 0
    total_count = len(datasets)

    if "tinystories" in datasets:
        try:
            download_tinystories(output_dir)
            success_count += 1
        except Exception as e:
            print(f"✗ Failed to download TinyStories: {e}")

    if "writingprompts" in datasets:
        max_wp = args.max_examples or 100000
        download_writing_prompts(output_dir, max_examples=max_wp)
        # Check if file was created
        if (output_dir / "writingprompts.txt").exists():
            success_count += 1

    if "gutenberg" in datasets:
        max_gb = args.max_examples or 100
        download_gutenberg_sample(output_dir, max_books=max_gb)
        # Check if file was created
        if (output_dir / "gutenberg.txt").exists():
            success_count += 1

    if "bookcorpus" in datasets:
        max_bc = args.max_examples or 10000
        download_bookcorpus_sample(output_dir, max_examples=max_bc)
        # Check if file was created
        if (output_dir / "bookcorpus.txt").exists():
            success_count += 1

    # Print statistics
    print("\n")
    if output_dir.exists() and any(output_dir.glob("*.txt")):
        get_dataset_stats(output_dir)
    else:
        print("⚠ No datasets were downloaded successfully.")
        return

    print(f"\n✓ Downloaded {success_count}/{total_count} datasets successfully!")
    print("\nNext steps:")
    print(
        "  1. Run: storyteller-preprocess --input_dir data/raw --output_dir data/processed"
    )
    print("  2. Run: storyteller-tokenizer --input_file data/processed/train.txt")


if __name__ == "__main__":
    main()
