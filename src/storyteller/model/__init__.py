"""
Storyteller model package.

This package contains the transformer model implementation with Mixture of Experts.
"""

from .config import ModelConfig, get_small_config, get_base_config, get_moe_config
from .transformer import StorytellerModel, TransformerBlock
from .attention import MultiHeadAttention
from .layers import FeedForward, RMSNorm, RotaryPositionalEmbedding
from .moe_layer import MoELayer, MoELayerOptimized
from .router import TopKRouter, SwitchRouter

__all__ = [
    "ModelConfig",
    "get_small_config",
    "get_base_config",
    "get_moe_config",
    "StorytellerModel",
    "TransformerBlock",
    "MultiHeadAttention",
    "FeedForward",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "MoELayer",
    "MoELayerOptimized",
    "TopKRouter",
    "SwitchRouter",
]
