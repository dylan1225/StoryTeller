"""
Main training script for the Storyteller model.

Usage:
    storyteller-train --config configs/base_model.yaml
    storyteller-train --config configs/moe_model.yaml --resume checkpoints/moe_model/checkpoint_step_10000.pt
"""

import argparse
import os

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerFast

from storyteller.model import StorytellerModel, ModelConfig
from storyteller.data.dataset import StoryDataset, StoryDatasetPreloaded, get_dataloader
from storyteller.training.trainer import Trainer
from storyteller.utils.device_utils import (
    smart_select_device,
    custom_select_device,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config_dict: dict) -> StorytellerModel:
    """Create model from configuration dictionary."""
    # Extract model config and remove fields that aren't ModelConfig parameters
    model_cfg = config_dict["model"].copy()

    # Remove metadata fields that aren't part of ModelConfig
    model_cfg.pop("config_name", None)

    model_config = ModelConfig(**model_cfg)
    model = StorytellerModel(model_config)
    return model


def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Create optimizer with weight decay.

    Uses AdamW with separate weight decay for different parameter groups.
    """
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for biases and layer norms
        if "bias" in name or "ln" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": config["weight_decay"],
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config["learning_rate"],
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    num_training_steps: int,
):
    """Create learning rate scheduler."""
    warmup_steps = config.get("warmup_steps", 2000)
    scheduler_type = config.get("lr_scheduler", "cosine")

    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import LambdaLR
        import math

        # Create a lambda function that implements warmup + cosine decay
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(
                    max(1, num_training_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                # Scale to end at 0.1 of initial LR
                return 0.1 + (1.0 - 0.1) * cosine_decay

        scheduler = LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "linear":
        from torch.optim.lr_scheduler import LinearLR

        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_training_steps,
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def main():
    parser = argparse.ArgumentParser(description="Train Storyteller model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to trained tokenizer (default: uses config or 'data/tokenizers/storyteller-tokenizer')",
    )

    args = parser.parse_args()

    # Distributed setup (torchrun sets LOCAL_RANK/RANK/WORLD_SIZE)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if rank == 0:
            print(
                f"Distributed training enabled | backend={backend}, world_size={world_size}, rank={rank}, local_rank={local_rank}"
            )

    # Load configuration
    if rank == 0:
        print(f"Loading configuration from {args.config}...")
    config_dict = load_config(args.config)
    train_config = config_dict["training"]

    # Set device
    device_config = train_config.get("device", "smart")

    if distributed:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if rank == 0:
            print(f"Forcing distributed device assignment: {device}")
    else:
        if device_config == "smart":
            device = smart_select_device()
        elif device_config == "custom":
            device = custom_select_device()
        else:
            device = torch.device(device_config)
            print(f"Using device: {device}")

    # Determine tokenizer path (priority: CLI arg > config > default)
    if args.tokenizer_path is not None:
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = train_config.get(
            "tokenizer_path", "data/tokenizers/storyteller-tokenizer"
        )

    # Load tokenizer
    if rank == 0:
        print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    if rank == 0:
        print(f"  Vocabulary size: {len(tokenizer):,}")

    # Update vocab size in config
    config_dict["model"]["vocab_size"] = len(tokenizer)

    # Create datasets
    if rank == 0:
        print("\nCreating datasets...")

    # Determine which dataset class to use based on config
    use_cached = train_config.get("use_cached_dataset", False)

    if use_cached:
        cache_dir = train_config.get("cache_dir", "data/cache")
        if rank == 0:
            print(f"  Using cached dataset (cache_dir: {cache_dir})")

        train_dataset = StoryDatasetPreloaded(
            data_path=train_config["train_data_path"],
            tokenizer=tokenizer,
            max_seq_length=config_dict["model"]["max_seq_length"],
            cache_dir=cache_dir,
        )

        val_dataset = StoryDatasetPreloaded(
            data_path=train_config["val_data_path"],
            tokenizer=tokenizer,
            max_seq_length=config_dict["model"]["max_seq_length"],
            cache_dir=cache_dir,
        )
    else:
        if rank == 0:
            print("  Using standard dataset (no caching)")

        train_dataset = StoryDataset(
            data_path=train_config["train_data_path"],
            tokenizer=tokenizer,
            max_seq_length=config_dict["model"]["max_seq_length"],
        )

        val_dataset = StoryDataset(
            data_path=train_config["val_data_path"],
            tokenizer=tokenizer,
            max_seq_length=config_dict["model"]["max_seq_length"],
        )

    if rank == 0:
        print(f"  Train dataset: {len(train_dataset):,} examples")
        print(f"  Val dataset: {len(val_dataset):,} examples")

    # Configure pin_memory based on device
    # MPS doesn't support pin_memory, so disable it automatically
    pin_memory = train_config.get("pin_memory", True)
    if device.type == "mps" and pin_memory:
        if rank == 0:
            print("  Note: Disabling pin_memory (not supported on MPS)")
        pin_memory = False

    # Distributed samplers
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )

    # Create dataloaders
    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=not distributed,
        sampler=train_sampler,
        num_workers=train_config.get("num_workers", 0),
        pin_memory=pin_memory,
    )

    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=train_config.get("num_workers", 0),
        pin_memory=pin_memory,
    )

    # Create model
    if rank == 0:
        print("\nCreating model...")
    base_model = create_model_from_config(config_dict)

    # Move to device and wrap with DDP if needed
    base_model = base_model.to(device)
    model = (
        DDP(
            base_model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index,
            find_unused_parameters=False,
        )
        if distributed
        else base_model
    )

    # Enable gradient checkpointing if specified
    if config_dict["model"].get("gradient_checkpointing", False):
        print("  Enabling gradient checkpointing...")
        # This would need to be implemented in the model
        # model.gradient_checkpointing_enable()

    # Calculate training steps
    num_epochs = train_config["num_epochs"]
    gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
    num_training_steps = (
        len(train_dataloader) // gradient_accumulation_steps * num_epochs
    )
    if rank == 0:
        print("\nTraining configuration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {train_config['batch_size']}")
        print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        print(
            f"  Effective batch size per rank: {train_config['batch_size'] * gradient_accumulation_steps}"
        )
        if distributed:
            print(
                f"  Global effective batch size: {train_config['batch_size'] * gradient_accumulation_steps * world_size}"
            )
        print(f"  Total training steps (per rank): {num_training_steps:,}")

    # Create optimizer
    if rank == 0:
        print("\nCreating optimizer...")
    optimizer = create_optimizer(model, train_config)

    # Create scheduler
    if rank == 0:
        print("Creating learning rate scheduler...")
    scheduler = create_scheduler(optimizer, train_config, num_training_steps)

    # Get evaluation config
    eval_config = train_config.get("evaluation", {})

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=train_config.get("use_amp", True),
        amp_dtype=train_config.get("amp_dtype", "bfloat16"),
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        save_dir=train_config.get("save_dir", "checkpoints"),
        save_every_n_steps=train_config.get("save_every_n_steps", 5000),
        eval_every_n_steps=train_config.get("eval_every_n_steps", 1000),
        log_every_n_steps=train_config.get("log_every_n_steps", 100),
        keep_last_n_checkpoints=train_config.get("keep_last_n_checkpoints", 3),
        use_mlflow=train_config.get("use_mlflow", False),
        mlflow_experiment_name=train_config.get("mlflow_experiment_name"),
        mlflow_run_name=train_config.get("mlflow_run_name"),
        mlflow_tracking_uri=train_config.get("mlflow_tracking_uri"),
        mlflow_log_system_metrics=train_config.get("mlflow_log_system_metrics", True),
        tokenizer=tokenizer,
        num_eval_samples=eval_config.get("num_eval_samples", 50),
        eval_max_length=eval_config.get("eval_max_length", 512),
        eval_temperature=eval_config.get("eval_temperature", 1.0),
        eval_top_k=eval_config.get("eval_top_k", 50),
        eval_top_p=eval_config.get("eval_top_p", 0.95),
        rank=rank,
        world_size=world_size,
        is_distributed=distributed,
        is_main_process=(rank == 0),
        train_sampler=train_sampler,
        val_sampler=val_sampler,
    )

    # Resume from checkpoint if specified
    if args.resume:
        if rank == 0:
            print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    if rank == 0:
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60 + "\n")

    trainer.train(num_epochs=num_epochs)

    if rank == 0:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
