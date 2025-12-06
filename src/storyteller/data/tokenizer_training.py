"""
Train a custom BPE tokenizer for story generation.

This script trains a Byte-Pair Encoding (BPE) tokenizer on the processed story data.
BPE is efficient for handling diverse vocabulary while keeping the vocab size reasonable.

Usage:
    python data/tokenizer_training.py --input_file data/processed/train.txt --vocab_size 50000
"""

import argparse
from pathlib import Path
from typing import Optional

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast


def train_bpe_tokenizer(
    input_files: list[str],
    vocab_size: int = 50000,
    min_frequency: int = 2,
    special_tokens: Optional[list[str]] = None,
) -> Tokenizer:
    """
    Train a BPE tokenizer.

    Args:
        input_files: List of text files to train on
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for a token to be included
        special_tokens: List of special tokens (if None, uses defaults)

    Returns:
        Trained tokenizer
    """
    if special_tokens is None:
        special_tokens = [
            "<|endoftext|>",  # End of document
            "<|pad|>",  # Padding token
            "<|unk|>",  # Unknown token
        ]

    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # Pre-tokenizer: split on whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder: convert tokens back to text
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor: add special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    print(f"Training tokenizer on {len(input_files)} file(s)...")
    print(f"Target vocab size: {vocab_size:,}")
    print(f"Special tokens: {special_tokens}")

    # Train the tokenizer
    tokenizer.train(files=input_files, trainer=trainer)

    print(f"✓ Tokenizer trained! Actual vocab size: {tokenizer.get_vocab_size():,}")

    return tokenizer


def test_tokenizer(tokenizer: Tokenizer, test_texts: list[str]):
    """
    Test the tokenizer on sample texts.

    Args:
        tokenizer: Trained tokenizer
        test_texts: List of test strings
    """
    print("\n" + "=" * 60)
    print("Testing Tokenizer")
    print("=" * 60)

    for i, text in enumerate(test_texts, 1):
        encoding = tokenizer.encode(text)
        tokens = encoding.tokens
        ids = encoding.ids
        decoded = tokenizer.decode(ids)

        print(f"\nTest {i}:")
        print(f"  Original: {text[:100]}...")
        print(f"  Tokens ({len(tokens)}): {tokens[:10]}...")
        print(f"  IDs ({len(ids)}): {ids[:10]}...")
        print(f"  Decoded: {decoded[:100]}...")
        print(f"  Compression ratio: {len(text) / len(tokens):.2f} chars/token")


def save_tokenizer(
    tokenizer: Tokenizer,
    output_dir: Path,
    tokenizer_name: str = "storyteller-tokenizer",
):
    """
    Save tokenizer in HuggingFace format.

    Args:
        tokenizer: Trained tokenizer
        output_dir: Directory to save tokenizer
        tokenizer_name: Name for the tokenizer
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wrap in HuggingFace tokenizer for compatibility
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>",  # Use same as EOS for GPT-style models
    )

    # Save
    wrapped_tokenizer.save_pretrained(output_dir / tokenizer_name)

    print(f"\n✓ Tokenizer saved to: {output_dir / tokenizer_name}")

    # Also save the raw tokenizer for reference
    tokenizer.save(str(output_dir / tokenizer_name / "tokenizer.json"))


def load_test_samples(input_file: Path, n_samples: int = 5) -> list[str]:
    """Load some test samples from the dataset."""
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
        stories = [s.strip() for s in text.split("\n\n") if s.strip()]

        # Get diverse samples from different parts of the dataset
        indices = [i * len(stories) // n_samples for i in range(n_samples)]
        for idx in indices:
            if idx < len(stories):
                samples.append(stories[idx][:500])  # First 500 chars of each story

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer for story generation"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/processed/train.txt",
        help="Input text file for training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/tokenizers",
        help="Directory to save the trained tokenizer",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50000,
        help="Target vocabulary size",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for a token",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="storyteller-tokenizer",
        help="Name for the tokenizer",
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)

    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        print("Please run preprocess.py first")
        return

    # Train tokenizer
    tokenizer = train_bpe_tokenizer(
        input_files=[str(input_file)],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    # Test on samples
    test_samples = load_test_samples(input_file)
    if test_samples:
        test_tokenizer(tokenizer, test_samples)

    # Save tokenizer
    save_tokenizer(tokenizer, output_dir, args.tokenizer_name)

    print("\n" + "=" * 60)
    print("Tokenizer Training Complete!")
    print("=" * 60)
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"Saved to: {output_dir / args.tokenizer_name}")
    print("\nNext steps:")
    print("  1. Update model config with correct vocab_size")
    print("  2. Start training the model!")


if __name__ == "__main__":
    main()
