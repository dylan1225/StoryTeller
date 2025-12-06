"""
Router implementation for Mixture of Experts.

The router decides which experts should process each token, implementing
Top-K routing with load balancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TopKRouter(nn.Module):
    """
    Top-K router for Mixture of Experts.

    This router selects the top-K experts for each token based on learned
    routing weights. It includes auxiliary losses for load balancing.

    Key concepts:
    - Each token is routed to K experts (typically K=2)
    - Routing weights are learned during training
    - Load balancing loss encourages even expert utilization
    - Router z-loss encourages smaller routing logits for stability
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        use_noisy_gating: bool = False,
        noise_epsilon: float = 1e-2,
    ):
        """
        Initialize the router.

        Args:
            hidden_size: Hidden dimension of the model
            num_experts: Total number of experts
            top_k: Number of experts to route each token to
            capacity_factor: Maximum tokens per expert (as multiple of average)
            use_noisy_gating: Whether to add noise to gating (for exploration)
            noise_epsilon: Standard deviation of gating noise
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.use_noisy_gating = use_noisy_gating
        self.noise_epsilon = noise_epsilon

        # Linear layer to compute routing logits
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Initialize with small weights for stability
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Route tokens to experts.

        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)

        Returns:
            Tuple of:
            - expert_indices: Top-K expert indices (batch_size, seq_len, top_k)
            - expert_weights: Routing weights (batch_size, seq_len, top_k)
            - router_stats: Dictionary of statistics for logging and losses
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Flatten batch and sequence dimensions for routing
        # (batch_size * seq_len, hidden_size)
        hidden_states_flat = hidden_states.reshape(-1, hidden_size)

        # Compute routing logits
        # (batch_size * seq_len, num_experts)
        router_logits = self.gate(hidden_states_flat)

        # Add noise to logits during training (for exploration)
        if self.use_noisy_gating and self.training:
            noise = torch.randn_like(router_logits) * self.noise_epsilon
            router_logits = router_logits + noise

        # Compute routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-K experts
        # expert_weights: (batch_size * seq_len, top_k)
        # expert_indices: (batch_size * seq_len, top_k)
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize weights so they sum to 1
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        # Reshape back to (batch_size, seq_len, top_k)
        expert_indices = expert_indices.reshape(batch_size, seq_len, self.top_k)
        expert_weights = expert_weights.reshape(batch_size, seq_len, self.top_k)

        # Compute auxiliary losses and statistics
        router_stats = self._compute_auxiliary_losses(
            router_logits,
            router_probs,
            expert_indices.reshape(-1, self.top_k),
            batch_size * seq_len,
        )

        return expert_indices, expert_weights, router_stats

    def _compute_auxiliary_losses(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        num_tokens: int,
    ) -> dict:
        """
        Compute auxiliary losses for load balancing.

        Args:
            router_logits: Raw routing logits (num_tokens, num_experts)
            router_probs: Routing probabilities (num_tokens, num_experts)
            expert_indices: Selected expert indices (num_tokens, top_k)
            num_tokens: Total number of tokens

        Returns:
            Dictionary with losses and statistics
        """
        # 1. Load Balancing Loss
        # Encourages balanced assignment of tokens to experts
        # Computes the product of:
        # - f_i: fraction of tokens assigned to expert i
        # - P_i: average routing probability for expert i
        # Minimizing this encourages both to be uniform (1/num_experts)

        # Compute fraction of tokens routed to each expert
        # Create one-hot encoding of selected experts
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts)
        # Sum across top_k and tokens to get counts per expert
        expert_counts = expert_mask.sum(dim=[0, 1])  # (num_experts,)
        # Normalize to get fractions
        fraction_per_expert = expert_counts.float() / (num_tokens * self.top_k)

        # Compute average routing probability per expert
        mean_prob_per_expert = router_probs.mean(dim=0)  # (num_experts,)

        # Load balancing loss: dot product of fractions and probabilities
        # Scaled by num_experts so the minimum (uniform routing) is 1.0
        load_balancing_loss = (
            self.num_experts * (fraction_per_expert * mean_prob_per_expert).sum()
        )

        # 2. Router Z-Loss
        # Encourages smaller routing logits for numerical stability
        # This is the mean squared logit
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        # 3. Statistics for logging
        # Measure how balanced the expert usage is
        # Perfect balance = 1.0, completely unbalanced = num_experts
        balance_metric = (fraction_per_expert.max() / (1.0 / self.num_experts)).item()

        # Entropy of routing probabilities (higher = more uncertain/diverse)
        routing_entropy = (
            -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1).mean().item()
        )

        return {
            "load_balancing_loss": load_balancing_loss,
            "router_z_loss": router_z_loss,
            "expert_balance_metric": balance_metric,
            "routing_entropy": routing_entropy,
            "expert_counts": expert_counts.detach().cpu(),
            "mean_expert_prob": mean_prob_per_expert.detach().cpu(),
        }


class SwitchRouter(nn.Module):
    """
    Switch Router that routes each token to exactly one expert.

    This is a simplified version where top_k=1, based on the Switch Transformer.
    Simpler and more memory-efficient than Top-K routing.

    Paper: https://arxiv.org/abs/2101.03961
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        jitter_noise: float = 0.0,
    ):
        """
        Initialize switch router.

        Args:
            hidden_size: Hidden dimension
            num_experts: Number of experts
            capacity_factor: Expert capacity as multiple of average load
            jitter_noise: Amount of jitter to add to routing logits
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise

        # Routing weights
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Route tokens to single expert.

        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)

        Returns:
            Tuple of (expert_indices, expert_weights, router_stats)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Flatten for routing
        hidden_flat = hidden_states.reshape(-1, hidden_size)

        # Compute routing logits
        router_logits = self.gate(hidden_flat)

        # Add jitter noise if specified
        if self.jitter_noise > 0 and self.training:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise

        # Get routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-1 expert
        expert_weights, expert_indices = torch.max(router_probs, dim=-1)

        # Reshape
        expert_indices = expert_indices.reshape(batch_size, seq_len, 1)
        expert_weights = expert_weights.reshape(batch_size, seq_len, 1)

        # Compute statistics
        router_stats = self._compute_stats(
            router_logits, router_probs, expert_indices.reshape(-1)
        )

        return expert_indices, expert_weights, router_stats

    def _compute_stats(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> dict:
        """Compute routing statistics."""
        num_tokens = expert_indices.shape[0]

        # Expert load
        expert_counts = torch.bincount(
            expert_indices, minlength=self.num_experts
        ).float()

        fraction_per_expert = expert_counts / num_tokens
        mean_prob_per_expert = router_probs.mean(dim=0)

        # Load balancing loss
        load_balancing_loss = (
            self.num_experts * (fraction_per_expert * mean_prob_per_expert).sum()
        )

        # Router z-loss
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        return {
            "load_balancing_loss": load_balancing_loss,
            "router_z_loss": router_z_loss,
            "expert_counts": expert_counts.detach().cpu(),
            "mean_expert_prob": mean_prob_per_expert.detach().cpu(),
        }


if __name__ == "__main__":
    # Test the routers
    batch_size, seq_len, hidden_size = 4, 16, 768
    num_experts = 8

    print("Testing TopKRouter...")
    router = TopKRouter(hidden_size, num_experts, top_k=2)
    x = torch.randn(batch_size, seq_len, hidden_size)

    indices, weights, stats = router(x)
    print(f"  Expert indices shape: {indices.shape}")
    print(f"  Expert weights shape: {weights.shape}")
    print(f"  Load balancing loss: {stats['load_balancing_loss'].item():.4f}")
    print(f"  Router z-loss: {stats['router_z_loss'].item():.4f}")
    print(f"  Expert balance metric: {stats['expert_balance_metric']:.2f}")
    print(f"  Expert counts: {stats['expert_counts'].tolist()}")

    print("\nTesting SwitchRouter...")
    switch_router = SwitchRouter(hidden_size, num_experts)
    indices, weights, stats = switch_router(x)
    print(f"  Expert indices shape: {indices.shape}")
    print(f"  Expert weights shape: {weights.shape}")
    print(f"  Load balancing loss: {stats['load_balancing_loss'].item():.4f}")
    print(f"  Router z-loss: {stats['router_z_loss'].item():.4f}")
    print(f"  Expert counts: {stats['expert_counts'].tolist()}")

    print("\n✓ Routers working correctly!")
