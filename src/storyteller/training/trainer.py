"""
Trainer class for training the Storyteller model.

This module implements a flexible trainer with support for:
- Mixed precision training
- Gradient accumulation
- Gradient checkpointing
- Learning rate scheduling
- Checkpoint management
- Logging (console + MLflow)
"""

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm

# Import AMP components based on PyTorch version
try:
    from torch.amp import autocast, GradScaler

    HAS_UNIFIED_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

    HAS_UNIFIED_AMP = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from storyteller.evaluation import StoryEvaluator


class Trainer:
    """
    Trainer for language models with modern optimization techniques.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        use_amp: bool = True,
        amp_dtype: str = "bfloat16",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_dir: str = "checkpoints",
        save_every_n_steps: int = 5000,
        eval_every_n_steps: int = 1000,
        log_every_n_steps: int = 100,
        keep_last_n_checkpoints: int = 3,
        use_mlflow: bool = False,
        mlflow_experiment_name: Optional[str] = None,
        mlflow_run_name: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_log_system_metrics: bool = True,
        tokenizer: Optional = None,
        num_eval_samples: int = 50,
        eval_max_length: int = 512,
        eval_temperature: float = 1.0,
        eval_top_k: int = 50,
        eval_top_p: float = 0.95,
        rank: int = 0,
        world_size: int = 1,
        is_distributed: bool = False,
        is_main_process: bool = True,
        train_sampler: Optional[DistributedSampler] = None,
        val_sampler: Optional[DistributedSampler] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: The model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            use_amp: Whether to use automatic mixed precision
            amp_dtype: AMP dtype ('float16' or 'bfloat16')
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            save_dir: Directory to save checkpoints
            save_every_n_steps: Save checkpoint every N steps
            eval_every_n_steps: Evaluate every N steps
            log_every_n_steps: Log metrics every N steps
            keep_last_n_checkpoints: Number of recent checkpoints to keep
            use_mlflow: Whether to use MLflow experiment tracking
            mlflow_experiment_name: MLflow experiment name
            mlflow_run_name: MLflow run name
            mlflow_tracking_uri: MLflow tracking server URI (optional)
            mlflow_log_system_metrics: Whether to log system metrics (CPU, GPU, memory)
            tokenizer: Tokenizer for story generation during evaluation
            num_eval_samples: Number of stories to generate during evaluation
            eval_max_length: Maximum length of generated stories
            eval_temperature: Sampling temperature for story generation
            eval_top_k: Top-k sampling parameter
            eval_top_p: Top-p (nucleus) sampling parameter
        """
        # Distributed context
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = is_distributed
        self.is_main_process = is_main_process
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

        # Device assignment
        self.device = torch.device(device) if isinstance(device, str) else device
        self.device_type = self.device.type

        # Keep DDP-wrapped models intact
        if isinstance(model, nn.parallel.DistributedDataParallel):
            self.model = model
        else:
            self.model = model.to(self.device)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Mixed precision configuration - handle MPS, CUDA, and CPU
        self.use_amp = self._configure_amp(use_amp, amp_dtype)
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16

        # GradScaler only for CUDA with float16
        # MPS and bfloat16 don't need/support GradScaler
        self.scaler = None
        if self.use_amp and self.device_type == "cuda" and amp_dtype == "float16":
            if HAS_UNIFIED_AMP:
                self.scaler = GradScaler(self.device_type)
            else:
                self.scaler = GradScaler()

        # Training config
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Checkpointing
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_steps = save_every_n_steps
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.saved_checkpoints = []

        # Logging
        self.eval_every_n_steps = eval_every_n_steps
        self.log_every_n_steps = log_every_n_steps

        # MLflow (only on main process)
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE and self.is_main_process
        if self.use_mlflow:
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)

            # Enable system metrics logging if requested
            if mlflow_log_system_metrics:
                try:
                    mlflow.enable_system_metrics_logging()
                    print("  MLflow system metrics logging enabled")
                except Exception as e:
                    print(f"  Warning: Could not enable system metrics logging: {e}")

            # Set experiment
            experiment_name = mlflow_experiment_name or "storyteller"
            mlflow.set_experiment(experiment_name)

            # Start run
            mlflow.start_run(run_name=mlflow_run_name)

            # Log model config and training params
            model_for_logging = model.module if hasattr(model, "module") else model
            if hasattr(model_for_logging, "config"):
                mlflow.log_params(
                    {
                        f"model/{k}": v
                        for k, v in model_for_logging.config.__dict__.items()
                        if isinstance(v, (int, float, str, bool))
                    }
                )

            mlflow.log_params(
                {
                    "batch_size": self.train_dataloader.batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "max_grad_norm": max_grad_norm,
                    "use_amp": use_amp,
                    "amp_dtype": amp_dtype,
                    "device": str(self.device),
                }
            )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Evaluation config
        self.tokenizer = tokenizer
        self.num_eval_samples = num_eval_samples
        self.eval_max_length = eval_max_length
        self.eval_temperature = eval_temperature
        self.eval_top_k = eval_top_k
        self.eval_top_p = eval_top_p
        self.story_evaluator = StoryEvaluator()

    def _configure_amp(self, use_amp: bool, amp_dtype: str) -> bool:
        """
        Configure automatic mixed precision based on device and dtype.

        Returns:
            Whether AMP should be enabled
        """
        if not use_amp:
            return False

        # CPU doesn't support AMP
        if self.device_type == "cpu":
            print("  Warning: AMP not supported on CPU, disabling...")
            return False

        # MPS supports AMP but with limitations
        if self.device_type == "mps":
            # MPS only supports float16, not bfloat16
            if amp_dtype == "bfloat16":
                print(
                    "  Warning: MPS doesn't support bfloat16, using float16 instead..."
                )
                self.amp_dtype = torch.float16
            return True

        # CUDA supports both float16 and bfloat16
        if self.device_type == "cuda":
            # Check if bfloat16 is supported
            if amp_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                print(
                    "  Warning: GPU doesn't support bfloat16, falling back to float16..."
                )
                self.amp_dtype = torch.float16
            return True

        return False

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        total_moe_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}", disable=not self.is_main_process)

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass with mixed precision
            # Use device_type for PyTorch 2.0+, otherwise old API
            autocast_kwargs = {"enabled": self.use_amp}
            if HAS_UNIFIED_AMP:
                autocast_kwargs["device_type"] = self.device_type
                autocast_kwargs["dtype"] = self.amp_dtype
            else:
                autocast_kwargs["dtype"] = self.amp_dtype

            with autocast(**autocast_kwargs):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                loss = outputs["loss"]

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.log_every_n_steps == 0:
                    self._log_metrics(
                        {
                            "train/loss": loss.item()
                            * self.gradient_accumulation_steps,
                            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                            "train/global_step": self.global_step,
                        },
                        outputs,
                    )

                # Evaluation
                if self.global_step % self.eval_every_n_steps == 0:
                    val_metrics = self.evaluate()
                    self._log_metrics(val_metrics, prefix="val")
                    self.model.train()

                # Checkpointing
                if self.global_step % self.save_every_n_steps == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            # Accumulate metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            if outputs.get("moe_loss") is not None:
                total_moe_loss += outputs["moe_loss"].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        return {
            "epoch/train_loss": total_loss / num_batches,
            "epoch/train_moe_loss": total_moe_loss / num_batches
            if total_moe_loss > 0
            else 0,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_moe_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating", disable=not self.is_main_process):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Use same autocast configuration as training
            autocast_kwargs = {"enabled": self.use_amp}
            if HAS_UNIFIED_AMP:
                autocast_kwargs["device_type"] = self.device_type
                autocast_kwargs["dtype"] = self.amp_dtype
            else:
                autocast_kwargs["dtype"] = self.amp_dtype

            with autocast(**autocast_kwargs):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )

            total_loss += outputs["loss"].item()
            if outputs.get("moe_loss") is not None:
                total_moe_loss += outputs["moe_loss"].item()
            num_batches += 1

        if self.is_distributed and dist.is_available() and dist.is_initialized():
            totals = torch.tensor(
                [total_loss, total_moe_loss, num_batches], device=self.device
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            total_loss, total_moe_loss, num_batches = totals.tolist()

        num_batches = max(num_batches, 1)
        avg_loss = total_loss / num_batches
        avg_moe_loss = total_moe_loss / num_batches if total_moe_loss > 0 else 0

        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        metrics = {
            "val/loss": avg_loss,
            "val/perplexity": perplexity,
        }

        if avg_moe_loss > 0:
            metrics["val/moe_loss"] = avg_moe_loss

        # Phase 1: Generate stories and compute quality metrics
        if (
            self.is_main_process
            and self.tokenizer is not None
            and self.num_eval_samples > 0
        ):
            generated_texts = self._generate_eval_stories()
            if generated_texts:
                # Compute Phase 1 metrics
                story_metrics = self.story_evaluator.evaluate_texts(generated_texts)
                metrics.update(story_metrics)

        return metrics

    def _generate_eval_stories(self) -> list:
        """
        Generate sample stories for evaluation.

        Returns:
            List of generated story strings
        """
        generated_texts = []

        # Sample prompts for story generation
        prompts = [
            "Once upon a time",
            "In a land far away",
            "There once was",
            "Long ago",
            "One day",
        ]

        model_for_gen = self.model.module if hasattr(self.model, "module") else self.model

        try:
            for i in range(self.num_eval_samples):
                # Cycle through prompts
                prompt = prompts[i % len(prompts)]

                # Encode prompt
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                    self.device
                )

                # Generate with sampling
                output_ids = model_for_gen.generate(
                    input_ids,
                    max_new_tokens=self.eval_max_length,
                    temperature=self.eval_temperature,
                    top_k=self.eval_top_k,
                    top_p=self.eval_top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Decode generated text
                text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                generated_texts.append(text)

        except Exception as e:
            print(f"Warning: Story generation failed during evaluation: {e}")
            return []

        return generated_texts

    def train(self, num_epochs: int):
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
        """
        if self.is_main_process:
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Device: {self.device_type} ({self.device})")
            if self.is_distributed:
                print(
                    f"Distributed: world_size={self.world_size}, rank={self.rank}, main_process={self.is_main_process}"
                )
            print(f"Mixed precision: {self.use_amp}")
            if self.use_amp:
                print(f"  AMP dtype: {self.amp_dtype}")
                print(f"  GradScaler: {'enabled' if self.scaler else 'disabled'}")
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            print(
                f"Effective batch size: {self.train_dataloader.batch_size * self.gradient_accumulation_steps}"
            )

        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch
            if isinstance(self.train_sampler, DistributedSampler):
                self.train_sampler.set_epoch(epoch)

            if self.is_main_process:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train epoch
            epoch_metrics = self.train_epoch()

            # Evaluate
            val_metrics = self.evaluate()

            # Log epoch metrics
            all_metrics = {**epoch_metrics, **val_metrics, "epoch": epoch}
            self._log_metrics(all_metrics)

            # Save best model
            if val_metrics["val/loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val/loss"]
                self.save_checkpoint("best_model.pt")
                if self.is_main_process:
                    print(f"✓ New best validation loss: {self.best_val_loss:.4f}")

        elapsed_time = time.time() - start_time
        if self.is_main_process:
            print(f"\nTraining completed in {elapsed_time / 3600:.2f} hours")

        # Save final model
        self.save_checkpoint("final_model.pt")

        # End MLflow run
        if self.use_mlflow:
            # Log final model
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.end_run()

    def _log_metrics(
        self,
        metrics: Dict[str, float],
        outputs: Optional[Dict] = None,
        prefix: str = "",
    ):
        """Log metrics to console and MLflow."""
        if not self.is_main_process:
            return
        # Console logging
        if self.global_step % self.log_every_n_steps == 0:
            log_str = f"Step {self.global_step}: "
            log_str += ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            if outputs and outputs.get("moe_stats"):
                # Log expert utilization for first MoE layer
                stats = outputs["moe_stats"][0]
                if "expert_balance_metric" in stats:
                    log_str += f", expert_balance: {stats['expert_balance_metric']:.2f}"

        # MLflow logging
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=self.global_step)

            # Log MoE statistics if available
            if outputs and outputs.get("moe_stats"):
                for i, stats in enumerate(outputs["moe_stats"]):
                    if "expert_balance_metric" in stats:
                        mlflow.log_metric(
                            f"moe/layer_{i}_balance",
                            stats["expert_balance_metric"],
                            step=self.global_step,
                        )
                    if "routing_entropy" in stats:
                        mlflow.log_metric(
                            f"moe/layer_{i}_entropy",
                            stats["routing_entropy"],
                            step=self.global_step,
                        )

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if not self.is_main_process:
            return

        checkpoint_path = self.save_dir / filename

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        checkpoint = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": model_to_save.config.__dict__,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")

        # Track saved checkpoints
        if "step" in filename:
            self.saved_checkpoints.append(checkpoint_path)

            # Remove old checkpoints
            if len(self.saved_checkpoints) > self.keep_last_n_checkpoints:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        # Use weights_only=False for compatibility with older PyTorch versions
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            # Older PyTorch versions don't have weights_only parameter
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if self.is_main_process:
            print(f"✓ Checkpoint loaded: {checkpoint_path}")
            print(f"  Resuming from step {self.global_step}, epoch {self.epoch}")
