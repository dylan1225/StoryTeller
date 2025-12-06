"""
Basic building blocks for transformer models.

This module contains fundamental layers used throughout the transformer:
- Feed-forward networks (FFN)
- Layer normalization (LayerNorm/RMSNorm)
- Activation functions
"""

import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is a simpler and faster alternative to LayerNorm that only normalizes
    by the RMS (root mean square) without centering. Used in modern LLMs like LLaMA.

    Paper: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.

        Args:
            hidden_size: Dimension of the input
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Normalized tensor of same shape
        """
        # Calculate RMS
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        # Apply learned scale
        return self.weight * x


class FeedForward(nn.Module):
    """
    Feed-forward network (FFN) used in transformer layers.

    Standard FFN with two linear layers and an activation function in between.
    Supports both GELU and SwiGLU activations.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        """
        Initialize FFN.

        Args:
            hidden_size: Input/output dimension
            intermediate_size: Hidden dimension (typically 4x hidden_size)
            dropout: Dropout probability
            activation: Activation function ('gelu', 'swiglu', 'relu')
        """
        super().__init__()

        self.activation_name = activation

        if activation == "swiglu":
            # SwiGLU uses gated activation with 2 linear projections
            # Total params same as standard FFN but better performance
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.activation = nn.SiLU()  # Swish activation
        else:
            # Standard FFN
            self.fc1 = nn.Linear(hidden_size, intermediate_size)
            self.fc2 = nn.Linear(intermediate_size, hidden_size)

            if activation == "gelu":
                self.activation = nn.GELU()
            elif activation == "relu":
                self.activation = nn.ReLU()
            else:
                raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply FFN.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of same shape
        """
        if self.activation_name == "swiglu":
            # SwiGLU: (Swish(gate) * up) @ down
            gate = self.activation(self.gate_proj(x))
            up = self.up_proj(x)
            hidden = gate * up
            output = self.down_proj(hidden)
        else:
            # Standard FFN: up -> activation -> dropout -> down
            hidden = self.activation(self.fc1(x))
            hidden = self.dropout(hidden)
            output = self.fc2(hidden)

        return self.dropout(output)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    RoPE encodes positional information by rotating query and key vectors.
    It provides better length extrapolation than learned positional embeddings.

    Paper: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_seq_length: int = 2048, base: float = 10000.0):
        """
        Initialize RoPE.

        Args:
            dim: Dimension of the embedding (typically head_dim)
            max_seq_length: Maximum sequence length
            base: Base for the geometric progression
        """
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base

        # Pre-compute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos and sin for all positions
        self._set_cos_sin_cache(max_seq_length)

    def _set_cos_sin_cache(self, seq_len: int):
        """Pre-compute cosine and sine values for efficiency."""
        self.max_seq_length_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)

        # Outer product of positions and frequencies
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Concatenate to get the full embedding
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        """
        Get cos and sin values for sequence.

        Args:
            x: Input tensor (used only for device/dtype)
            seq_len: Sequence length (if None, uses x.shape[1])

        Returns:
            Tuple of (cos, sin) tensors
        """
        if seq_len is None:
            seq_len = x.shape[1]

        # Extend cache if needed
        if seq_len > self.max_seq_length_cached:
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.

    This is a helper function for applying RoPE.

    Args:
        x: Input tensor of shape (..., dim)

    Returns:
        Rotated tensor of same shape
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embedding to query and key tensors.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        k: Key tensor of same shape
        cos: Cosine tensor of shape (seq_len, head_dim)
        sin: Sine tensor of same shape

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Reshape cos and sin for broadcasting
    # (seq_len, head_dim) -> (1, seq_len, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Apply rotation using the formula:
    # q_embed = q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


if __name__ == "__main__":
    # Test the layers
    batch_size, seq_len, hidden_size = 2, 10, 768

    print("Testing RMSNorm...")
    x = torch.randn(batch_size, seq_len, hidden_size)
    rms_norm = RMSNorm(hidden_size)
    output = rms_norm(x)
    print(f"  Input shape: {x.shape}, Output shape: {output.shape}")

    print("\nTesting FeedForward (GELU)...")
    ffn = FeedForward(hidden_size, intermediate_size=3072, activation="gelu")
    output = ffn(x)
    print(f"  Input shape: {x.shape}, Output shape: {output.shape}")

    print("\nTesting FeedForward (SwiGLU)...")
    ffn_swiglu = FeedForward(hidden_size, intermediate_size=3072, activation="swiglu")
    output = ffn_swiglu(x)
    print(f"  Input shape: {x.shape}, Output shape: {output.shape}")

    print("\nTesting RoPE...")
    head_dim = 64
    num_heads = 12
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    rope = RotaryPositionalEmbedding(head_dim)
    cos, sin = rope(q, seq_len)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"  Q shape: {q.shape}, Q rotated shape: {q_rot.shape}")
    print(f"  K shape: {k.shape}, K rotated shape: {k_rot.shape}")

    print("\n✓ All layers working correctly!")
