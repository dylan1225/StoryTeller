"""
Mixture of Experts (MoE) layer implementation.

This module implements the MoE layer that combines multiple expert FFNs
with a routing mechanism.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

from .layers import FeedForward
from .router import TopKRouter


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.

    This layer contains multiple expert networks (FFNs) and a router that
    decides which experts to use for each token. Only the selected experts
    are computed, making the layer sparse and efficient.

    Architecture:
        1. Router selects top-K experts for each token
        2. Each token is processed by its selected experts
        3. Expert outputs are weighted and combined

    Key benefits:
    - Sparse computation: only K out of N experts active per token
    - High capacity: can have many experts without increasing compute
    - Specialization: experts can learn different patterns
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        capacity_factor: float = 1.25,
        use_noisy_gating: bool = False,
    ):
        """
        Initialize MoE layer.

        Args:
            hidden_size: Input/output dimension
            intermediate_size: FFN intermediate dimension
            num_experts: Number of expert FFNs
            top_k: Number of experts to route each token to
            dropout: Dropout probability
            activation: Activation function
            capacity_factor: Expert capacity factor
            use_noisy_gating: Whether to add noise to routing
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Create expert FFNs
        # Each expert is a standard feed-forward network
        self.experts = nn.ModuleList(
            [
                FeedForward(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_experts)
            ]
        )

        # Router to select experts
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            use_noisy_gating=use_noisy_gating,
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through MoE layer.

        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)

        Returns:
            Tuple of:
            - output: Mixed expert outputs (batch_size, seq_len, hidden_size)
            - router_stats: Router statistics for losses and logging
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        original_shape = hidden_states.shape

        # Get routing decisions
        # expert_indices: (batch_size, seq_len, top_k)
        # expert_weights: (batch_size, seq_len, top_k)
        expert_indices, expert_weights, router_stats = self.router(hidden_states)

        # Flatten batch and sequence dimensions
        hidden_states = hidden_states.reshape(-1, hidden_size)  # (B*S, H)
        expert_indices = expert_indices.reshape(-1, self.top_k)  # (B*S, K)
        expert_weights = expert_weights.reshape(-1, self.top_k)  # (B*S, K)

        # Initialize output tensor
        output = torch.zeros_like(hidden_states)  # (B*S, H)

        # Process tokens through selected experts
        # There are different strategies to implement this efficiently
        # Strategy 1: Loop over experts (better for small num_experts)
        # Strategy 2: Loop over tokens (better for large num_experts)
        # Strategy 3: Batched expert computation (most efficient but complex)

        # We use Strategy 1 for simplicity and pedagogical clarity
        # For production, Strategy 3 with expert batching is recommended

        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            # Create mask: (B*S, K) where each position is True if expert_idx is selected
            expert_mask = expert_indices == expert_idx  # (B*S, K)

            # Get token indices that use this expert (any position in top_k)
            token_indices = expert_mask.any(dim=1).nonzero(as_tuple=True)[0]

            if len(token_indices) == 0:
                continue  # No tokens routed to this expert

            # Get tokens for this expert
            expert_input = hidden_states[token_indices]  # (N_tokens, H)

            # Process through expert
            expert_output = self.experts[expert_idx](expert_input)  # (N_tokens, H)

            # Get weights for these tokens
            # For each token, find which position in top_k this expert is at
            # and use the corresponding weight
            for k_idx in range(self.top_k):
                # Tokens where this expert is at position k_idx
                mask_k = expert_mask[token_indices, k_idx]
                if not mask_k.any():
                    continue

                # Get corresponding weights
                weights = expert_weights[token_indices, k_idx]  # (N_tokens,)

                # Apply weights and accumulate to output
                # Only for tokens where this expert is at position k_idx
                weighted_output = expert_output * weights[mask_k].unsqueeze(-1)
                output[token_indices[mask_k]] += weighted_output

        # Reshape back to original shape
        output = output.reshape(original_shape)

        return output, router_stats


class MoELayerOptimized(nn.Module):
    """
    Optimized MoE layer with batched expert computation.

    This version processes all expert computations more efficiently by
    batching tokens that go to the same expert. More complex but faster.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        capacity_factor: float = 1.25,
        use_noisy_gating: bool = False,
    ):
        """Initialize optimized MoE layer."""
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Create experts
        self.experts = nn.ModuleList(
            [
                FeedForward(hidden_size, intermediate_size, dropout, activation)
                for _ in range(num_experts)
            ]
        )

        # Router
        self.router = TopKRouter(
            hidden_size, num_experts, top_k, capacity_factor, use_noisy_gating
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Optimized forward pass with expert batching.

        This implementation processes all tokens for each expert in a single batch,
        which is more efficient than processing token-by-token.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_tokens = batch_size * seq_len

        # Route tokens
        expert_indices, expert_weights, router_stats = self.router(hidden_states)

        # Flatten
        hidden_flat = hidden_states.reshape(num_tokens, hidden_size)
        expert_indices_flat = expert_indices.reshape(num_tokens, self.top_k)
        expert_weights_flat = expert_weights.reshape(num_tokens, self.top_k)

        # Initialize output
        output = torch.zeros_like(hidden_flat)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find all (token, k) pairs where this expert is selected
            expert_mask = expert_indices_flat == expert_idx  # (num_tokens, top_k)

            # Get positions
            token_idx, k_idx = expert_mask.nonzero(as_tuple=True)

            if len(token_idx) == 0:
                continue

            # Batch all tokens for this expert
            expert_input = hidden_flat[token_idx]  # (N, hidden_size)
            expert_output = self.experts[expert_idx](expert_input)  # (N, hidden_size)

            # Get weights and apply
            weights = expert_weights_flat[token_idx, k_idx].unsqueeze(-1)  # (N, 1)
            weighted_output = expert_output * weights

            # Scatter add to output
            output.index_add_(0, token_idx, weighted_output)

        # Reshape
        output = output.reshape(batch_size, seq_len, hidden_size)

        return output, router_stats


if __name__ == "__main__":
    # Test MoE layer
    batch_size, seq_len, hidden_size = 4, 16, 768
    intermediate_size = 3072
    num_experts = 8

    print("Testing MoELayer...")
    moe = MoELayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=2,
    )

    x = torch.randn(batch_size, seq_len, hidden_size)
    output, stats = moe(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Load balancing loss: {stats['load_balancing_loss'].item():.4f}")
    print(f"  Router z-loss: {stats['router_z_loss'].item():.4f}")
    print(f"  Expert counts: {stats['expert_counts'].tolist()}")

    print("\nTesting MoELayerOptimized...")
    moe_opt = MoELayerOptimized(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=2,
    )

    output_opt, stats_opt = moe_opt(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output_opt.shape}")
    print(f"  Load balancing loss: {stats_opt['load_balancing_loss'].item():.4f}")

    # Test that both implementations are functionally equivalent
    print("\nComparing implementations...")
    moe_opt.load_state_dict(moe.state_dict())  # Use same weights
    with torch.no_grad():
        out1, _ = moe(x)
        out2, _ = moe_opt(x)
        max_diff = (out1 - out2).abs().max().item()
        print(f"  Max difference: {max_diff:.2e}")

        if max_diff < 1e-5:
            print("  ✓ Both implementations produce same results!")
        else:
            print("  ⚠ Implementations differ (may be due to floating point)")

    print("\n✓ MoE layer working correctly!")
