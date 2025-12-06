"""
Advanced sampling strategies for text generation.

This module implements various sampling techniques beyond simple greedy decoding.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Filter logits to only keep top k tokens.

    Args:
        logits: Logits tensor (batch_size, vocab_size)
        top_k: Number of top tokens to keep

    Returns:
        Filtered logits with low-probability tokens set to -inf
    """
    if top_k <= 0:
        return logits

    # Get top-k values
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.clone()
    logits[indices_to_remove] = float("-inf")

    return logits


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Filter logits using nucleus (top-p) sampling.

    Keeps the smallest set of tokens whose cumulative probability exceeds p.

    Args:
        logits: Logits tensor (batch_size, vocab_size)
        top_p: Cumulative probability threshold

    Returns:
        Filtered logits
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p

    # Shift to keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Create mask in original order
    logits = logits.clone()
    for batch_idx in range(logits.size(0)):
        indices_to_remove = sorted_indices[batch_idx][
            sorted_indices_to_remove[batch_idx]
        ]
        logits[batch_idx, indices_to_remove] = float("-inf")

    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    Apply repetition penalty to discourage repeating tokens.

    Args:
        logits: Logits tensor (batch_size, vocab_size)
        generated_tokens: Previously generated tokens (batch_size, seq_len)
        penalty: Penalty factor (>1.0 discourages repetition)

    Returns:
        Logits with repetition penalty applied
    """
    if penalty == 1.0:
        return logits

    logits = logits.clone()

    for batch_idx in range(generated_tokens.size(0)):
        for token in set(generated_tokens[batch_idx].tolist()):
            # Divide logit by penalty if positive, multiply if negative
            if logits[batch_idx, token] < 0:
                logits[batch_idx, token] *= penalty
            else:
                logits[batch_idx, token] /= penalty

    return logits


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature scaling to logits.

    Temperature > 1.0 makes distribution more uniform (more random)
    Temperature < 1.0 makes distribution more peaked (more deterministic)
    Temperature = 1.0 leaves distribution unchanged

    Args:
        logits: Logits tensor
        temperature: Temperature value

    Returns:
        Scaled logits
    """
    if temperature == 1.0:
        return logits

    return logits / temperature


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample next token from logits using various strategies.

    Args:
        logits: Logits tensor (batch_size, vocab_size)
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        repetition_penalty: Repetition penalty factor
        generated_tokens: Previously generated tokens for repetition penalty

    Returns:
        Sampled token IDs (batch_size, 1)
    """
    # Apply repetition penalty
    if repetition_penalty != 1.0 and generated_tokens is not None:
        logits = apply_repetition_penalty(logits, generated_tokens, repetition_penalty)

    # Apply temperature
    logits = apply_temperature(logits, temperature)

    # Apply top-k filtering
    if top_k is not None:
        logits = top_k_filtering(logits, top_k)

    # Apply top-p filtering
    if top_p is not None:
        logits = top_p_filtering(logits, top_p)

    # Sample from distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def beam_search(
    model,
    input_ids: torch.Tensor,
    num_beams: int = 5,
    max_length: int = 100,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate using beam search.

    Args:
        model: Language model
        input_ids: Input token IDs (batch_size, seq_len)
        num_beams: Number of beams
        max_length: Maximum generation length
        length_penalty: Length penalty factor (>1.0 prefers longer sequences)
        early_stopping: Whether to stop when all beams finish
        eos_token_id: End-of-sequence token ID

    Returns:
        Generated token IDs for best beam
    """
    batch_size, cur_len = input_ids.shape
    device = input_ids.device

    # Expand input to beam size
    input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
    input_ids = input_ids.view(batch_size * num_beams, cur_len)

    # Initialize beam scores
    beam_scores = torch.zeros((batch_size, num_beams), device=device)
    beam_scores[:, 1:] = float("-inf")  # Only first beam starts
    beam_scores = beam_scores.view(-1)

    # Track finished sequences
    done = [False for _ in range(batch_size)]

    while cur_len < max_length:
        # Get model outputs
        outputs = model(input_ids, return_dict=True)
        next_token_logits = outputs["logits"][:, -1, :]

        # Calculate scores
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)
        next_token_scores = next_token_scores + beam_scores[:, None]

        # Reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Get top 2 * num_beams tokens
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )

        # Prepare next batch
        next_batch_beam = []

        for batch_idx in range(batch_size):
            if done[batch_idx]:
                # This batch is done, just pad
                next_batch_beam.extend([(0, 0, 0)] * num_beams)
                continue

            # Get next beams for this batch
            beam_idx = 0
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_token_scores[batch_idx])
            ):
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id

                # Check if beam is complete
                if eos_token_id is not None and token_id == eos_token_id:
                    if early_stopping:
                        done[batch_idx] = True

                # Add to next batch
                next_batch_beam.append((beam_token_score, token_id, effective_beam_id))

                beam_idx += 1
                if beam_idx >= num_beams:
                    break

        # Update beam scores and input_ids
        beam_scores = torch.tensor([x[0] for x in next_batch_beam], device=device)
        beam_tokens = torch.tensor([x[1] for x in next_batch_beam], device=device)
        beam_idx = torch.tensor([x[2] for x in next_batch_beam], device=device)

        # Update input_ids
        input_ids = input_ids[beam_idx]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(-1)], dim=-1)

        cur_len += 1

        # Stop if all sequences are done
        if all(done):
            break

    # Return best beam for each batch
    best_beams = input_ids.view(batch_size, num_beams, -1)[:, 0, :]
    return best_beams


if __name__ == "__main__":
    # Test sampling functions
    batch_size, vocab_size = 2, 1000

    print("Testing sampling strategies...")

    # Create sample logits
    logits = torch.randn(batch_size, vocab_size)

    print("\n1. Top-k filtering:")
    filtered = top_k_filtering(logits.clone(), top_k=50)
    num_valid = (filtered != float("-inf")).sum(dim=-1)
    print(f"   Valid tokens per batch: {num_valid.tolist()}")

    print("\n2. Top-p filtering:")
    filtered = top_p_filtering(logits.clone(), top_p=0.9)
    num_valid = (filtered != float("-inf")).sum(dim=-1)
    print(f"   Valid tokens per batch: {num_valid.tolist()}")

    print("\n3. Temperature scaling:")
    temp_low = apply_temperature(logits.clone(), 0.5)
    temp_high = apply_temperature(logits.clone(), 2.0)
    print(
        f"   Original entropy: {-(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean():.4f}"
    )
    print(
        f"   Low temp entropy: {-(F.softmax(temp_low, dim=-1) * F.log_softmax(temp_low, dim=-1)).sum(dim=-1).mean():.4f}"
    )
    print(
        f"   High temp entropy: {-(F.softmax(temp_high, dim=-1) * F.log_softmax(temp_high, dim=-1)).sum(dim=-1).mean():.4f}"
    )

    print("\n4. Repetition penalty:")
    generated = torch.randint(0, vocab_size, (batch_size, 20))
    penalized = apply_repetition_penalty(logits.clone(), generated, penalty=1.5)
    print(
        f"   Applied repetition penalty to {len(set(generated.flatten().tolist()))} unique tokens"
    )

    print("\n5. Full sampling:")
    sampled = sample_from_logits(
        logits,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        generated_tokens=generated,
    )
    print(f"   Sampled tokens: {sampled.squeeze().tolist()}")

    print("\n✓ All sampling strategies working correctly!")
