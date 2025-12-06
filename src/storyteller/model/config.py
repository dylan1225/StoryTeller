"""
Model configuration for The Storyteller.

This module defines configuration classes for both base transformer and MoE models.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Base configuration for transformer models."""

    # Model architecture
    vocab_size: int = (
        50257  # GPT-2 tokenizer size, will be updated based on trained tokenizer
    )
    max_seq_length: int = 2048
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072  # Typically 4x hidden_size

    # Attention configuration
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    use_flash_attention: bool = True  # Use flash attention if available

    # Positional encoding
    positional_encoding: Literal["learned", "rope", "alibi"] = "rope"
    rope_theta: float = 10000.0  # Base for RoPE

    # Activation function
    activation: Literal["gelu", "swiglu", "relu"] = "gelu"

    # Normalization
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    norm_position: Literal["pre", "post"] = "pre"  # Pre-norm is more stable

    # Initialization
    initializer_range: float = 0.02

    # MoE specific (only used if use_moe=True)
    use_moe: bool = False
    num_experts: int = 8
    expert_capacity_factor: float = 1.25
    top_k_experts: int = 2
    moe_frequency: int = 2  # Apply MoE every N layers (1=all, 2=every other)
    load_balancing_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001

    # Training configuration
    gradient_checkpointing: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        )

        if self.use_moe:
            assert self.top_k_experts <= self.num_experts, (
                f"top_k_experts ({self.top_k_experts}) must be <= num_experts ({self.num_experts})"
            )
            assert self.moe_frequency >= 1, (
                f"moe_frequency must be >= 1, got {self.moe_frequency}"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads

    def num_parameters(self, count_embeddings: bool = True) -> int:
        """
        Estimate the number of parameters in the model.

        Args:
            count_embeddings: Whether to include embedding parameters

        Returns:
            Estimated parameter count
        """
        params = 0

        # Embeddings
        if count_embeddings:
            params += self.vocab_size * self.hidden_size  # Token embeddings
            if self.positional_encoding == "learned":
                params += self.max_seq_length * self.hidden_size  # Position embeddings

        # Per-layer parameters
        for layer_idx in range(self.num_layers):
            # Self-attention
            params += 4 * self.hidden_size * self.hidden_size  # Q, K, V, O projections
            params += 2 * self.hidden_size  # LayerNorm parameters (2x per layer)

            # FFN or MoE
            if self.use_moe and (layer_idx % self.moe_frequency == 0):
                # MoE layer
                params += self.num_experts * (
                    2
                    * self.hidden_size
                    * self.intermediate_size  # Up and down projections per expert
                )
                params += self.hidden_size * self.num_experts  # Router
            else:
                # Dense FFN
                params += 2 * self.hidden_size * self.intermediate_size

        # Final layer norm
        params += self.hidden_size

        # LM head (often tied with embeddings, but counting separately)
        params += self.vocab_size * self.hidden_size

        return params

    def active_parameters(self) -> int:
        """
        Estimate active parameters per forward pass (relevant for MoE).

        Returns:
            Estimated active parameter count
        """
        if not self.use_moe:
            return self.num_parameters()

        params = 0

        # Embeddings (always active)
        params += self.vocab_size * self.hidden_size
        if self.positional_encoding == "learned":
            params += self.max_seq_length * self.hidden_size

        # Per-layer parameters
        for layer_idx in range(self.num_layers):
            # Self-attention (always active)
            params += 4 * self.hidden_size * self.hidden_size
            params += 2 * self.hidden_size  # LayerNorms

            # FFN or MoE
            if layer_idx % self.moe_frequency == 0:
                # MoE layer: only top-k experts are active
                params += self.top_k_experts * (
                    2 * self.hidden_size * self.intermediate_size
                )
                params += self.hidden_size * self.num_experts  # Router
            else:
                # Dense FFN
                params += 2 * self.hidden_size * self.intermediate_size

        # Final layer norm and LM head
        params += self.hidden_size + self.vocab_size * self.hidden_size

        return params


# Predefined configurations
def get_small_config() -> ModelConfig:
    """Small model for fast experimentation (~125M params)."""
    return ModelConfig(
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )


def get_base_config() -> ModelConfig:
    """Base model configuration (~350M params)."""
    return ModelConfig(
        hidden_size=1024,
        num_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
    )


def get_moe_config() -> ModelConfig:
    """
    MoE model configuration (~500M total params, ~100M active).

    This is the recommended configuration for the Storyteller project.
    """
    return ModelConfig(
        hidden_size=1024,
        num_layers=16,
        num_attention_heads=16,
        intermediate_size=4096,
        use_moe=True,
        num_experts=8,
        top_k_experts=2,
        moe_frequency=2,  # MoE every other layer
        load_balancing_loss_weight=0.01,
        router_z_loss_weight=0.001,
    )


if __name__ == "__main__":
    # Print configuration statistics
    configs = {
        "Small": get_small_config(),
        "Base": get_base_config(),
        "MoE": get_moe_config(),
    }

    print("Model Configurations:\n")
    for name, config in configs.items():
        total_params = config.num_parameters() / 1e6
        active_params = config.active_parameters() / 1e6
        print(f"{name}:")
        print(f"  Total parameters: {total_params:.1f}M")
        print(f"  Active parameters: {active_params:.1f}M")
        print(f"  Use MoE: {config.use_moe}")
        if config.use_moe:
            print(f"  Experts: {config.num_experts}, Top-K: {config.top_k_experts}")
        print()
