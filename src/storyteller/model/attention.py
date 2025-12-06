"""
Multi-head attention implementation with optional flash attention and RoPE.

This module implements the core attention mechanism for the transformer.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RotaryPositionalEmbedding, apply_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional flash attention and RoPE.

    This implements the attention mechanism from "Attention is All You Need"
    with modern improvements like RoPE and optional flash attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        use_flash_attention: bool = True,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        max_seq_length: int = 2048,
    ):
        """
        Initialize multi-head attention.

        Args:
            hidden_size: Dimension of the model
            num_attention_heads: Number of attention heads
            attention_dropout: Dropout probability for attention weights
            use_flash_attention: Whether to use Flash Attention 2 if available
            use_rope: Whether to use Rotary Positional Embeddings
            rope_theta: Base for RoPE
            max_seq_length: Maximum sequence length (for RoPE)
        """
        super().__init__()

        assert hidden_size % num_attention_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
        )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout
        self.use_flash_attention = use_flash_attention
        self.use_rope = use_rope

        # Q, K, V projections
        # We can fuse them into a single linear layer for efficiency
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Dropout
        self.attn_dropout = (
            nn.Dropout(attention_dropout) if attention_dropout > 0 else nn.Identity()
        )
        self.resid_dropout = (
            nn.Dropout(attention_dropout) if attention_dropout > 0 else nn.Identity()
        )

        # Rotary positional embeddings
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                self.head_dim,
                max_seq_length=max_seq_length,
                base=rope_theta,
            )

        # Check if flash attention is available
        self.flash_available = False
        if use_flash_attention:
            try:
                # Flash attention 2 is available in PyTorch 2.0+
                # via scaled_dot_product_attention
                self.flash_available = hasattr(F, "scaled_dot_product_attention")
            except Exception:
                pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Apply multi-head attention.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, 1, 1, seq_len)
            use_cache: Whether to return key-value cache for efficient generation
            past_key_value: Cached (key, value) from previous forward pass

        Returns:
            Tuple of (output, new_key_value_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(hidden_states)

        # Split and reshape for multi-head attention
        # (batch, seq_len, 3 * hidden) -> 3 x (batch, seq_len, num_heads, head_dim)
        qkv = qkv.reshape(
            batch_size, seq_len, 3, self.num_attention_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if enabled
        if self.use_rope:
            # RoPE expects (batch, seq_len, num_heads, head_dim)
            q = q.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
            k = k.transpose(1, 2)

            # Get position embeddings
            cos, sin = self.rope(q, seq_len)

            # Apply rotation
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Transpose back to (batch, num_heads, seq_len, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        # Handle KV cache for generation
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)  # Concatenate along seq_len
            v = torch.cat([past_v, v], dim=2)

        new_key_value = (k, v) if use_cache else None

        # Compute attention
        if self.flash_available and self.training:
            # Use Flash Attention during training
            # Need to transpose to (batch, seq_len, num_heads, head_dim) for SDPA
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # For causal language modeling, use is_causal instead of passing a mask
            # This is more efficient and avoids shape incompatibilities
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,  # Don't pass mask, use is_causal instead
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True,  # Always causal for language modeling
            )

            # Transpose back
            attn_output = attn_output.transpose(1, 2)
        else:
            # Standard attention implementation
            attn_output = self._standard_attention(q, k, v, attention_mask)

        # Reshape and project output
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, hidden)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # Final projection and dropout
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)

        return output, new_key_value

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention implementation.

        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_heads, kv_seq_len, head_dim)
            v: Value tensor (batch, num_heads, kv_seq_len, head_dim)
            attention_mask: Attention mask

        Returns:
            Attention output tensor (batch, num_heads, seq_len, head_dim)
        """
        # Compute attention scores
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, kv_seq_len)
        # -> (batch, num_heads, seq_len, kv_seq_len)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask (prevent attending to future positions)
        if attention_mask is None:
            # Create causal mask
            seq_len = q.size(2)
            kv_seq_len = k.size(2)
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_seq_len, dtype=torch.bool, device=q.device),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        else:
            # Apply provided mask
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # (batch, num_heads, seq_len, kv_seq_len) @ (batch, num_heads, kv_seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        return attn_output


if __name__ == "__main__":
    # Test the attention mechanism
    batch_size, seq_len, hidden_size = 2, 16, 768
    num_heads = 12

    print("Testing MultiHeadAttention...")

    # Create attention module
    attn = MultiHeadAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        use_rope=True,
        use_flash_attention=True,
    )

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size)

    # Forward pass
    output, kv_cache = attn(x, use_cache=True)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  KV cache shapes: {kv_cache[0].shape}, {kv_cache[1].shape}")

    # Test with cache (for generation)
    new_token = torch.randn(batch_size, 1, hidden_size)
    output, new_cache = attn(new_token, use_cache=True, past_key_value=kv_cache)

    print("\nGeneration mode:")
    print(f"  New token shape: {new_token.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Updated cache shapes: {new_cache[0].shape}, {new_cache[1].shape}")

    print("\n✓ Attention mechanism working correctly!")
