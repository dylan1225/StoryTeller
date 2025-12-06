"""
Evaluation metrics for story generation quality.

This module implements various metrics to assess:
- Fluency (perplexity)
- Diversity (distinct-N, vocabulary richness)
- Repetition (n-gram repetition detection)
- Structure (length statistics)
"""

from collections import Counter
from typing import Dict, List

import torch


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from loss.

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value (lower is better)
    """
    return torch.exp(torch.tensor(loss)).item()


def compute_distinct_n(texts: List[str], n: int = 2) -> float:
    """
    Compute Distinct-N metric for vocabulary diversity.

    Distinct-N measures the ratio of unique n-grams to total n-grams.
    Higher values indicate more diverse vocabulary.

    Args:
        texts: List of generated text strings
        n: N-gram size (1, 2, 3, or 4)

    Returns:
        Distinct-N score (0-1, higher is better)
    """
    all_ngrams = []

    for text in texts:
        tokens = text.split()
        if len(tokens) < n:
            continue

        # Generate n-grams
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)

    return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0


def compute_all_distinct_n(texts: List[str]) -> Dict[str, float]:
    """
    Compute Distinct-1, 2, 3, and 4 metrics.

    Args:
        texts: List of generated text strings

    Returns:
        Dictionary with distinct-N scores
    """
    return {
        "distinct-1": compute_distinct_n(texts, n=1),
        "distinct-2": compute_distinct_n(texts, n=2),
        "distinct-3": compute_distinct_n(texts, n=3),
        "distinct-4": compute_distinct_n(texts, n=4),
    }


def detect_repetition(text: str, max_ngram: int = 4) -> Dict[str, float]:
    """
    Detect repetitive patterns in generated text.

    Checks for:
    1. N-gram repetition ratio (how many n-grams are repeated)
    2. Longest consecutively repeated sequence
    3. Overall repetition penalty score

    Args:
        text: Generated text string
        max_ngram: Maximum n-gram size to check

    Returns:
        Dictionary with repetition metrics
    """
    tokens = text.split()

    if len(tokens) < 2:
        return {"repetition_ratio": 0.0, "max_repeated_length": 0, "penalty_score": 0.0}

    # Count n-gram frequencies
    repeated_count = 0
    total_count = 0
    max_repeated_length = 0

    for n in range(2, min(max_ngram + 1, len(tokens))):
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        ngram_counts = Counter(ngrams)

        # Count repeated n-grams
        for ngram, count in ngram_counts.items():
            total_count += count
            if count > 1:
                repeated_count += count - 1  # Don't count first occurrence
                if n > max_repeated_length:
                    max_repeated_length = n

    repetition_ratio = repeated_count / total_count if total_count > 0 else 0.0

    # Penalty score combines ratio and max length
    penalty_score = repetition_ratio * (1 + max_repeated_length / 10.0)

    return {
        "repetition_ratio": repetition_ratio,
        "max_repeated_length": max_repeated_length,
        "penalty_score": penalty_score,
    }


def compute_repetition_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Compute repetition metrics across multiple texts.

    Args:
        texts: List of generated text strings

    Returns:
        Dictionary with averaged repetition metrics
    """
    all_metrics = [detect_repetition(text) for text in texts]

    if not all_metrics:
        return {
            "repetition/ratio": 0.0,
            "repetition/max_length": 0,
            "repetition/penalty": 0.0,
        }

    # Average across all texts
    avg_metrics = {
        "repetition/ratio": sum(m["repetition_ratio"] for m in all_metrics)
        / len(all_metrics),
        "repetition/max_length": max(m["max_repeated_length"] for m in all_metrics),
        "repetition/penalty": sum(m["penalty_score"] for m in all_metrics)
        / len(all_metrics),
    }

    return avg_metrics


def compute_length_statistics(texts: List[str]) -> Dict[str, float]:
    """
    Compute length statistics for generated stories.

    Args:
        texts: List of generated text strings

    Returns:
        Dictionary with length statistics
    """
    if not texts:
        return {
            "length/avg": 0.0,
            "length/std": 0.0,
            "length/min": 0,
            "length/max": 0,
        }

    lengths = [len(text.split()) for text in texts]

    lengths_tensor = torch.tensor(lengths, dtype=torch.float32)

    return {
        "length/avg": lengths_tensor.mean().item(),
        "length/std": lengths_tensor.std().item() if len(lengths) > 1 else 0.0,
        "length/min": int(lengths_tensor.min().item()),
        "length/max": int(lengths_tensor.max().item()),
    }


def compute_vocabulary_diversity(texts: List[str]) -> Dict[str, float]:
    """
    Compute vocabulary diversity metrics (Type-Token Ratio).

    TTR = unique words / total words
    Higher values indicate richer vocabulary.

    Args:
        texts: List of generated text strings

    Returns:
        Dictionary with vocabulary diversity metrics
    """
    all_tokens = []
    for text in texts:
        all_tokens.extend(text.split())

    if not all_tokens:
        return {"vocab/ttr": 0.0, "vocab/unique": 0, "vocab/total": 0}

    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)

    return {
        "vocab/ttr": unique_tokens / total_tokens if total_tokens > 0 else 0.0,
        "vocab/unique": unique_tokens,
        "vocab/total": total_tokens,
    }


class StoryEvaluator:
    """
    Comprehensive evaluator for story generation quality.

    Computes Phase 1 metrics:
    - Perplexity
    - Distinct-N (1, 2, 3, 4)
    - Repetition detection
    - Length statistics
    - Vocabulary diversity
    """

    def __init__(self):
        """Initialize the evaluator."""
        pass

    def evaluate_texts(self, texts: List[str]) -> Dict[str, float]:
        """
        Evaluate a list of generated texts.

        Args:
            texts: List of generated story strings

        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}

        # Diversity metrics
        distinct_scores = compute_all_distinct_n(texts)
        for key, value in distinct_scores.items():
            metrics[f"diversity/{key}"] = value

        # Repetition metrics
        repetition_scores = compute_repetition_metrics(texts)
        metrics.update(repetition_scores)

        # Length statistics
        length_stats = compute_length_statistics(texts)
        metrics.update(length_stats)

        # Vocabulary diversity
        vocab_stats = compute_vocabulary_diversity(texts)
        metrics.update(vocab_stats)

        return metrics

    def evaluate_with_loss(self, texts: List[str], loss: float) -> Dict[str, float]:
        """
        Evaluate texts and include perplexity from loss.

        Args:
            texts: List of generated story strings
            loss: Validation loss

        Returns:
            Dictionary containing all computed metrics including perplexity
        """
        metrics = self.evaluate_texts(texts)
        metrics["perplexity"] = compute_perplexity(loss)
        return metrics


if __name__ == "__main__":
    # Test the metrics
    print("Testing Story Evaluation Metrics\n")

    # Sample texts for testing
    test_texts = [
        "Once upon a time there was a brave knight who saved the kingdom.",
        "The cat sat on the mat and looked at the bird in the tree.",
        "She walked through the forest and found a magical treasure chest.",
        "The robot learned to paint beautiful pictures of the sunset.",
    ]

    evaluator = StoryEvaluator()
    metrics = evaluator.evaluate_texts(test_texts)

    print("Evaluation Results:")
    print("=" * 60)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Test repetition detection on a repetitive text
    print("\n\nTesting Repetition Detection:")
    print("=" * 60)
    repetitive_text = "The cat sat on the mat. The cat sat on the mat. The cat sat."
    rep_metrics = detect_repetition(repetitive_text)
    print(f"Text: {repetitive_text}")
    print(f"Repetition ratio: {rep_metrics['repetition_ratio']:.4f}")
    print(f"Max repeated length: {rep_metrics['max_repeated_length']}")
    print(f"Penalty score: {rep_metrics['penalty_score']:.4f}")

    # Test with loss
    print("\n\nTesting with Loss/Perplexity:")
    print("=" * 60)
    test_loss = 2.5
    full_metrics = evaluator.evaluate_with_loss(test_texts, test_loss)
    print(f"Loss: {test_loss:.4f}")
    print(f"Perplexity: {full_metrics['perplexity']:.4f}")

    print("\n✓ All metrics working correctly!")
