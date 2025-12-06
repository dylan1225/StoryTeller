"""
The Storyteller: A story-generating language model using Mixture of Experts.

This package provides tools for building, training, and deploying a small
language model for creative story generation.
"""

__version__ = "0.1.0"

from .model import (
    ModelConfig,
    StorytellerModel,
    get_small_config,
    get_base_config,
    get_moe_config,
)

__all__ = [
    "ModelConfig",
    "StorytellerModel",
    "get_small_config",
    "get_base_config",
    "get_moe_config",
]
