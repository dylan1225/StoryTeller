"""
Main transformer model for story generation.

This module implements the complete decoder-only transformer model.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from .attention import MultiHeadAttention
from .config import ModelConfig
from .layers import FeedForward, RMSNorm
from .moe_layer import MoELayer


class TransformerBlock(nn.Module):
    """
    A single transformer block with self-attention and feed-forward layers.

    Architecture:
        x -> LayerNorm -> SelfAttention -> Residual ->
             LayerNorm -> FeedForward -> Residual -> output
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        """
        Initialize transformer block.

        Args:
            config: Model configuration
            layer_idx: Layer index (used for MoE frequency)
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Layer normalization
        norm_class = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.ln1 = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = norm_class(config.hidden_size, eps=config.layer_norm_eps)

        # Self-attention
        self.attn = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout,
            use_flash_attention=config.use_flash_attention,
            use_rope=(config.positional_encoding == "rope"),
            rope_theta=config.rope_theta,
            max_seq_length=config.max_seq_length,
        )

        # Feed-forward network or MoE layer
        # Use MoE if enabled and this layer index matches the frequency
        self.use_moe = config.use_moe and (layer_idx % config.moe_frequency == 0)

        if self.use_moe:
            self.ffn = MoELayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                dropout=config.hidden_dropout,
                activation=config.activation,
                capacity_factor=config.expert_capacity_factor,
            )
        else:
            self.ffn = FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                dropout=config.hidden_dropout,
                activation=config.activation,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[dict]
    ]:
        """
        Forward pass through transformer block.

        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask
            use_cache: Whether to return key-value cache
            past_key_value: Past key-value cache

        Returns:
            Tuple of (output, new_key_value_cache, moe_stats)
        """
        # Pre-norm architecture: normalize before attention
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)

        # Self-attention with residual connection
        attn_output, new_kv = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value,
        )
        hidden_states = residual + attn_output

        # Feed-forward or MoE with residual connection
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)

        moe_stats = None
        if self.use_moe:
            ffn_output, moe_stats = self.ffn(hidden_states)
        else:
            ffn_output = self.ffn(hidden_states)

        hidden_states = residual + ffn_output

        return hidden_states, new_kv, moe_stats


class StorytellerModel(nn.Module):
    """
    The main Storyteller transformer model.

    This is a decoder-only transformer for autoregressive language modeling.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Positional embeddings (only for learned positions)
        if config.positional_encoding == "learned":
            self.position_embeddings = nn.Embedding(
                config.max_seq_length, config.hidden_size
            )
        else:
            self.position_embeddings = None

        # Dropout for embeddings
        self.emb_dropout = nn.Dropout(config.hidden_dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )

        # Final layer norm
        norm_class = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.ln_f = norm_class(config.hidden_size, eps=config.layer_norm_eps)

        # Language modeling head (often tied with token embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights between embeddings and lm_head (common in LLMs)
        self.lm_head.weight = self.token_embeddings.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, "weight"):
                torch.nn.init.ones_(module.weight)

    def get_input_embeddings(self):
        """Get token embeddings."""
        return self.token_embeddings

    def set_input_embeddings(self, new_embeddings):
        """Set token embeddings."""
        self.token_embeddings = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        return_dict: bool = True,
    ):
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Labels for language modeling loss (batch_size, seq_len)
            use_cache: Whether to use KV cache for generation
            past_key_values: Past key-value cache
            return_dict: Whether to return a dictionary

        Returns:
            ModelOutput with loss, logits, and optionally past_key_values
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get token embeddings
        hidden_states = self.token_embeddings(input_ids)

        # Add positional embeddings if using learned positions
        if self.position_embeddings is not None:
            past_length = (
                past_key_values[0][0].size(2) if past_key_values is not None else 0
            )
            position_ids = torch.arange(
                past_length, past_length + seq_len, dtype=torch.long, device=device
            )
            position_embeds = self.position_embeddings(position_ids)
            hidden_states = hidden_states + position_embeds

        # Apply dropout
        hidden_states = self.emb_dropout(hidden_states)

        # Prepare attention mask if provided
        if attention_mask is not None:
            # Convert to 4D mask (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask[:, None, None, :]
            # Convert 0s to -inf and 1s to 0.0
            attention_mask = (1.0 - attention_mask) * torch.finfo(
                hidden_states.dtype
            ).min

        # Pass through transformer blocks
        new_key_values = () if use_cache else None
        all_moe_stats = []

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None

            hidden_states, new_kv, moe_stats = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_kv,
            )

            if use_cache:
                new_key_values = new_key_values + (new_kv,)

            if moe_stats is not None:
                all_moe_stats.append(moe_stats)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        moe_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            # Add MoE auxiliary losses if using MoE
            if self.config.use_moe and all_moe_stats:
                # Average load balancing and z losses across MoE layers
                load_balancing_losses = torch.stack(
                    [stats["load_balancing_loss"] for stats in all_moe_stats]
                )
                router_z_losses = torch.stack(
                    [stats["router_z_loss"] for stats in all_moe_stats]
                )

                moe_loss = (
                    self.config.load_balancing_loss_weight
                    * load_balancing_losses.mean()
                    + self.config.router_z_loss_weight * router_z_losses.mean()
                )

                # Add MoE loss to total loss
                loss = loss + moe_loss

        if not return_dict:
            output = (logits,)
            if use_cache:
                output += (new_key_values,)
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": new_key_values if use_cache else None,
            "moe_loss": moe_loss,
            "moe_stats": all_moe_stats if all_moe_stats else None,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        do_sample: bool = True,
        **kwargs,  # Accept additional kwargs for compatibility
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling
            top_p: Nucleus (top-p) sampling
            repetition_penalty: Penalty for repeating tokens
            eos_token_id: Token ID for end of sequence
            pad_token_id: Token ID for padding (unused, for compatibility)
            do_sample: Whether to use sampling (unused, always samples)
            **kwargs: Additional arguments for compatibility

        Returns:
            Generated token IDs (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Get logits for next token
            outputs = self(input_ids, use_cache=False, return_dict=True)
            logits = outputs["logits"][:, -1, :]  # (batch_size, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    for token_id in set(input_ids[i].tolist()):
                        logits[i, token_id] /= repetition_penalty

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                for i in range(logits.shape[0]):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = float("-inf")

            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids


if __name__ == "__main__":
    # Test the model
    from .config import get_small_config

    print("Creating small model for testing...")
    config = get_small_config()
    config.num_layers = 4  # Reduce for faster testing
    config.max_seq_length = 128

    model = StorytellerModel(config)

    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    print("\nTesting forward pass...")
    outputs = model(input_ids, labels=labels, return_dict=True)

    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Logits shape: {outputs['logits'].shape}")

    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)

    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")

    print("\n✓ Model working correctly!")
