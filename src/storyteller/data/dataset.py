"""
PyTorch dataset for loading and processing story data during training.

This module provides efficient data loading for training language models.
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm


class StoryDataset(Dataset):
    """
    Dataset for loading stories with on-the-fly tokenization.

    This dataset loads stories from a text file and tokenizes them on-the-fly.
    Stories are split into chunks of max_seq_length for efficient training.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_length: int = 2048,
        stride: Optional[int] = None,
        return_full_sequences: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to text file containing stories
            tokenizer: Tokenizer to use
            max_seq_length: Maximum sequence length
            stride: Stride for overlapping chunks (if None, uses max_seq_length // 2)
            return_full_sequences: If True, only return sequences of exactly max_seq_length
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.stride = stride or max_seq_length // 2
        self.return_full_sequences = return_full_sequences

        # Load and tokenize all stories
        self.examples = self._load_and_tokenize()

        print(f"Loaded {len(self.examples)} examples from {self.data_path}")

    def _load_and_tokenize(self) -> List[torch.Tensor]:
        """
        Load stories from file and create training examples.

        Returns:
            List of tokenized sequences
        """
        print(f"Loading data from {self.data_path}...")

        # Read all stories
        with open(self.data_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into stories
        print("Processing stories...")
        stories = [s.strip() for s in text.split("\n\n") if s.strip()]
        print(f"Found {len(stories):,} stories")

        # Process in batches to avoid memory issues
        print("Tokenizing stories in batches...")
        batch_size = 10000  # Process 10k stories at a time
        all_input_ids = []

        for i in tqdm(range(0, len(stories), batch_size), desc="Tokenizing batches"):
            batch_stories = stories[i : i + batch_size]
            batch_text = self.tokenizer.eos_token.join(batch_stories)

            # Tokenize batch
            encoded = self.tokenizer(
                batch_text,
                add_special_tokens=True,
                return_tensors="pt",
            )
            all_input_ids.append(encoded["input_ids"][0])

        # Concatenate all batches
        print("Combining tokenized batches...")
        input_ids = torch.cat(all_input_ids)
        print(f"Generated {len(input_ids):,} tokens")

        # Split into chunks
        print(
            f"Creating training examples with max_seq_length={self.max_seq_length}, stride={self.stride}..."
        )
        examples = []
        num_chunks = (len(input_ids) - self.max_seq_length) // self.stride + 1

        for i in tqdm(
            range(0, len(input_ids) - self.max_seq_length + 1, self.stride),
            desc="Creating chunks",
            total=num_chunks,
        ):
            chunk = input_ids[i : i + self.max_seq_length]

            # Only add if it's the full length (or allow shorter for last chunk)
            if len(chunk) == self.max_seq_length:
                examples.append(chunk)
            elif (
                not self.return_full_sequences and len(chunk) > self.max_seq_length // 2
            ):
                # Pad shorter sequences if not requiring full sequences
                padding = torch.full(
                    (self.max_seq_length - len(chunk),),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                )
                chunk = torch.cat([chunk, padding])
                examples.append(chunk)

        return examples

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.

        Args:
            idx: Example index

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        input_ids = self.examples[idx]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # For language modeling, labels are the same as input_ids
        # (shifted by 1 is handled in the model)
        labels = input_ids.clone()

        # Set padding tokens in labels to -100 (ignored in loss)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class StoryDatasetPreloaded(Dataset):
    """
    Optimized dataset that pre-loads everything into memory.

    Faster than StoryDataset but uses more memory. Good for smaller datasets
    that fit in RAM.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_length: int = 2048,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to text file containing stories
            tokenizer: Tokenizer to use
            max_seq_length: Maximum sequence length
            cache_dir: Directory to cache processed data (optional)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Try to load from cache
        cache_file = None
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{self.data_path.stem}_cache.pt"

        if cache_file and cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            self.examples = torch.load(cache_file)
        else:
            self.examples = self._load_and_tokenize()
            if cache_file:
                print(f"Saving to cache: {cache_file}")
                torch.save(self.examples, cache_file)

        print(f"Loaded {len(self.examples)} examples from {self.data_path}")

    def _load_and_tokenize(self) -> List[Dict[str, torch.Tensor]]:
        """Load and tokenize all data into memory."""
        print(f"Loading data from {self.data_path}...")

        # Same as StoryDataset but pre-creates all dictionaries
        with open(self.data_path, "r", encoding="utf-8") as f:
            text = f.read()

        print("Processing stories...")
        stories = [s.strip() for s in text.split("\n\n") if s.strip()]
        print(f"Found {len(stories):,} stories")

        # Process in batches to avoid memory issues
        print("Tokenizing stories in batches...")
        batch_size = 10000  # Process 10k stories at a time
        all_input_ids = []

        for i in tqdm(range(0, len(stories), batch_size), desc="Tokenizing batches"):
            batch_stories = stories[i : i + batch_size]
            batch_text = self.tokenizer.eos_token.join(batch_stories)

            # Tokenize batch
            encoded = self.tokenizer(
                batch_text,
                add_special_tokens=True,
                return_tensors="pt",
            )
            all_input_ids.append(encoded["input_ids"][0])

        # Concatenate all batches
        print("Combining tokenized batches...")
        input_ids = torch.cat(all_input_ids)
        print(f"Generated {len(input_ids):,} tokens")

        print(
            f"Creating training examples with max_seq_length={self.max_seq_length}..."
        )
        examples = []
        stride = self.max_seq_length
        num_chunks = (len(input_ids) - self.max_seq_length) // stride + 1

        for i in tqdm(
            range(0, len(input_ids) - self.max_seq_length + 1, stride),
            desc="Creating examples",
            total=num_chunks,
        ):
            chunk = input_ids[i : i + self.max_seq_length]

            if len(chunk) == self.max_seq_length:
                attention_mask = torch.ones_like(chunk)
                labels = chunk.clone()

                examples.append(
                    {
                        "input_ids": chunk,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                )

        return examples

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.

        Args:
            idx: Example index

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        return self.examples[idx]


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    sampler: Optional[torch.utils.data.Sampler] = None,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the dataset.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for consistent shapes
    )


if __name__ == "__main__":
    # Example usage
    from transformers import PreTrainedTokenizerFast

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "data/tokenizers/storyteller-tokenizer"
    )

    # Create dataset
    dataset = StoryDataset(
        data_path="data/processed/train.txt",
        tokenizer=tokenizer,
        max_seq_length=512,  # Smaller for testing
    )

    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print("\nSample shape:")
    print(f"  input_ids: {sample['input_ids'].shape}")
    print(f"  attention_mask: {sample['attention_mask'].shape}")
    print(f"  labels: {sample['labels'].shape}")

    # Decode to verify
    decoded = tokenizer.decode(sample["input_ids"])
    print("\nDecoded text (first 200 chars):")
    print(decoded[:200])
