# Storyteller Model Configurations

This directory contains YAML configuration files for training the Storyteller language model at various scales. Each config is optimized for different use cases, from rapid testing to production-quality story generation.

## Quick Start

```bash
# Train with a specific config
storyteller-train --config configs/<config_name>.yaml

# Example: Quick 2-5 minute test (FASTEST)
storyteller-train --config configs/ultra_tiny.yaml

# Example: Production-quality training
storyteller-train --config configs/gpt2_small.yaml
```

## Configuration Overview

| Config | Parameters | Training Time | Hardware | Use Case |
|--------|-----------|---------------|----------|----------|
| [ultra_tiny](#ultra_tinyyaml) | ~1-2M | **2-5 min** | Any (CPU ok) | **FASTEST** pipeline testing |
| [tiny_test](#tiny_testyaml) | ~11M | ~16 hours* | Any (CPU ok) | Testing & debugging |
| [fast_train](#fast_trainyaml) | ~10M | 10-20 min | 4-8GB VRAM | Educational demos |
| [min_model](#min_modelyaml) | ~50M | 6-12 hours | 16GB+ VRAM | First serious training |
| [gpt2_small](#gpt2_smallyaml) | ~124M | 1-2 days | 24GB+ VRAM | **RECOMMENDED** |
| [base_model](#base_modelyaml) | ~350M | 3-4 days | 32GB+ VRAM | High-quality stories |
| [moe_model](#moe_modelyaml) | ~500M (~100M active) | 4-5 days | 32GB VRAM | Sparse MoE research |

*Note: `tiny_test.yaml` takes longer than expected due to vocab embedding size. Use `ultra_tiny.yaml` for truly fast testing.

## Detailed Configuration Descriptions

### ultra_tiny.yaml

**~1-2M parameters - TRULY FAST testing config** ⚡ **FASTEST**

**Architecture:**
- 2 layers × 128 hidden × 2 heads
- 64 token context window
- Minimal decoder-only transformer

**When to use:**
- Rapid pipeline validation and debugging (2-5 minutes ACTUAL)
- Testing new features or code changes
- Verifying training loop works correctly
- CI/CD testing and automated testing
- Sanity checking before longer runs

**Hardware:**
- Runs on any device (CPU, MPS, CUDA)
- No GPU required

**Output quality:**
- Random text, not even basic generation
- ONLY for testing the pipeline

**Key settings:**
- `batch_size: 64` - Large batch for maximum throughput
- `num_epochs: 1` - Single epoch only
- `learning_rate: 1.0e-3` - High LR for fast convergence
- `eval_every_n_steps: 999999` - **Disabled eval during training (MAJOR SPEED BOOST)**
- `num_eval_samples: 0` - **No story generation (MAJOR SPEED BOOST)**
- `gradient_checkpointing: false` - Disabled for speed

**Speed optimizations:**
- Evaluation completely disabled during training
- No story generation (saves 7+ minutes per eval)
- Maximum batch size for throughput
- Minimal model size

---

### tiny_test.yaml

**~11M parameters - Testing config (slower than expected)**

**Architecture:**
- 4 layers × 192 hidden × 4 heads
- 128 token context window
- Simple decoder-only transformer

**When to use:**
- Quick pipeline validation and debugging (2-5 minutes)
- Testing new features or code changes
- Verifying data preprocessing works correctly
- Learning the training workflow
- CI/CD testing and automated testing

**Hardware:**
- Runs on any device (CPU, MPS, CUDA)
- No GPU required

**Output quality:**
- Very basic text generation
- NOT for actual story generation
- Use for testing only

**Key settings:**
- `batch_size: 32` - Large batch for throughput
- `num_epochs: 1` - Single epoch only
- `learning_rate: 8.0e-4` - Higher LR for fast convergence
- `log_every_n_steps: 5` - Frequent logging for immediate feedback
- `gradient_checkpointing: false` - Disabled for speed

---

### fast_train.yaml

**~10M parameters - Rapid training demonstration**

**Architecture:**
- 6 layers × 256 hidden × 4 heads
- 256 token context window
- Small decoder-only transformer

**When to use:**
- Educational demonstrations of LLM training dynamics (10-20 min)
- Quick experimentation with hyperparameters
- Understanding training metrics and loss curves
- Testing evaluation metrics implementation
- First-time users learning the training process

**Hardware:**
- Any device (CPU, MPS, CUDA)
- 4-8GB VRAM recommended

**Output quality:**
- Basic text patterns and simple sentences
- Not full stories but shows learning progress
- Good for demonstrating training dynamics

**Ideal for:**
- Students and educators to see a complete training cycle quickly

**Key settings:**
- `batch_size: 16` with `gradient_accumulation_steps: 2` (effective: 32)
- `num_epochs: 2` - Just 2 epochs for quick demo
- `learning_rate: 5.0e-4` - Higher LR for faster convergence
- `log_every_n_steps: 10` - Frequent logging for visibility

---

### min_model.yaml

**~50M parameters - Balanced testing and quality**

**Architecture:**
- 12 layers × 512 hidden × 8 heads
- 512 token context window
- Medium-scale decoder-only transformer

**When to use:**
- Initial full training experiments with reasonable quality (6-12 hours)
- Hyperparameter tuning and ablation studies
- Testing model changes before scaling up
- Resource-constrained environments (16GB VRAM)
- Faster iteration cycles during development

**Hardware:**
- 16GB+ VRAM (CUDA) or 16GB+ unified memory (MPS)

**Output quality:**
- Decent short stories with basic coherence and structure
- Good for prototyping and experimentation

**Ideal for:**
- First serious training run with actual story generation capability
- Projects that need quick turnaround but reasonable quality

**Key settings:**
- `batch_size: 8` with `gradient_accumulation_steps: 4` (effective: 32)
- `num_epochs: 5` - Balance between quality and time
- `max_seq_length: 512` - Medium context length
- `gradient_checkpointing: true` - Save memory

---

### gpt2_small.yaml

**~124M parameters - Production-quality story generation** ⭐ **RECOMMENDED**

**Architecture:**
- 12 layers × 768 hidden × 12 heads
- 1024 token context window (GPT2-small standard)
- Replicates proven GPT2 architecture

**When to use:**
- Serious story generation with high quality and coherence (1-2 days)
- Replicating proven GPT2 architecture for comparison
- Creating a baseline model for research experiments
- Production storytelling applications
- Projects requiring balance of quality and training speed

**Hardware:**
- 24GB+ VRAM (CUDA) or 32GB+ unified memory (MPS)

**Output quality:**
- Coherent, creative stories with good structure and narrative flow
- Best balance between training time and story quality

**Ideal for:**
- **RECOMMENDED for most users wanting quality results**
- Users with 1-2 days of training time available
- Production applications requiring good story generation

**Key settings:**
- `batch_size: 8` with `gradient_accumulation_steps: 4` (effective: 32)
- `num_epochs: 5` - Balance between quality and training time
- `learning_rate: 3.0e-4` - GPT2-small's default learning rate
- `max_seq_length: 1024` - GPT2-small's context window
- `warmup_steps: 500` - Gradual warmup for stability

---

### base_model.yaml

**~350M parameters - High-quality storytelling**

**Architecture:**
- 24 layers × 1024 hidden × 16 heads
- 2048 token context window (full-length stories)
- Large decoder-only transformer

**When to use:**
- High-quality story generation with rich vocabulary (3-4 days)
- Research requiring larger model capacity
- Longer context for complex narratives
- Comparison baseline for MoE experiments
- When you have time for longer training runs

**Hardware:**
- 32GB VRAM (CUDA) or 64GB+ unified memory (MPS)

**Output quality:**
- High-quality creative stories with complex plots and characters
- Rich vocabulary and sophisticated language use

**Ideal for:**
- Users wanting maximum quality from a dense transformer
- Research projects needing larger model capacity
- Production applications with longer training budgets

**Key settings:**
- `batch_size: 8` with `gradient_accumulation_steps: 4` (effective: 32)
- `num_epochs: 10` - More epochs for quality
- `max_seq_length: 2048` - Full-length story context
- `warmup_steps: 2000` - Longer warmup for stability
- `gradient_checkpointing: true` - Required for 32GB VRAM

---

### moe_model.yaml

**~500M total parameters, ~100M active - Sparse Mixture of Experts**

**Architecture:**
- 16 layers × 1024 hidden × 16 heads
- 2048 token context window
- **MoE Config:** 8 experts per layer, Top-2 routing, every other layer
- Sparse activation: Only ~100M params active per forward pass

**When to use:**
- Learning modern sparse MoE architectures (Mixtral, GPT-4 style) (4-5 days)
- Research on expert specialization and routing
- Maximum model capacity within 32GB VRAM constraint
- Exploring expert load balancing and utilization
- Educational projects on cutting-edge LLM techniques

**Hardware:**
- 32GB VRAM required (CUDA or MPS)
- Sparse activation keeps memory within budget

**Output quality:**
- Potentially higher quality than base_model with same active params
- Expert specialization can lead to interesting generation patterns

**Ideal for:**
- Advanced users exploring sparse models and expert routing
- Educational projects focused on learning MoE concepts
- Research on expert specialization

**Primary goal:**
- Educational - learning MoE concepts, not just story quality

**Key settings:**
- `use_moe: true` - Enable MoE architecture
- `num_experts: 8` - 8 experts per MoE layer
- `top_k_experts: 2` - Top-2 routing (2 experts active per token)
- `moe_frequency: 2` - Apply MoE every other layer
- `load_balancing_loss_weight: 0.01` - Ensure even expert utilization
- `batch_size: 4` with `gradient_accumulation_steps: 8` (effective: 32)
- `learning_rate: 2.0e-4` - Slightly lower for MoE stability
- `warmup_steps: 3000` - Longer warmup for MoE stability
- `log_expert_stats: true` - Monitor expert utilization

---

## Choosing the Right Configuration

### Decision Tree

**Need to test code changes or validate pipeline quickly?**
→ Use `ultra_tiny.yaml` ⚡ (2-5 minutes ACTUAL) - **FASTEST**

**Learning the training process or teaching LLMs?**
→ Use `fast_train.yaml` (10-20 minutes)

**First serious training run with limited time/hardware?**
→ Use `min_model.yaml` (6-12 hours, 16GB VRAM)

**Want production-quality stories with best time/quality balance?**
→ Use `gpt2_small.yaml` ⭐ (1-2 days, 24GB+ VRAM) - **RECOMMENDED**

**Need maximum quality from a dense model?**
→ Use `base_model.yaml` (3-4 days, 32GB VRAM)

**Exploring sparse MoE architectures and expert routing?**
→ Use `moe_model.yaml` (4-5 days, 32GB VRAM)

## Common Configuration Parameters

All configs share common structure with these key sections:

### Model Section
- `vocab_size`: Tokenizer vocabulary size (50000)
- `max_seq_length`: Context window length
- `hidden_size`: Model dimension
- `num_layers`: Number of transformer layers
- `num_attention_heads`: Number of attention heads
- `intermediate_size`: FFN intermediate dimension
- `positional_encoding`: "rope" (Rotary Position Embeddings)
- `activation`: "gelu" activation function
- `gradient_checkpointing`: Trade compute for memory

### Training Section
- `batch_size`: Per-device batch size
- `gradient_accumulation_steps`: Accumulation for larger effective batch
- `learning_rate`: Peak learning rate
- `num_epochs`: Number of training epochs
- `warmup_steps`: LR warmup steps
- `use_amp`: Enable automatic mixed precision
- `amp_dtype`: "float16" or "bfloat16"
- `device`: "smart" (auto-select best device)

### Evaluation Section
- `num_eval_samples`: Stories generated during evaluation
- `eval_max_length`: Maximum generation length
- `eval_temperature`: Sampling temperature
- `eval_top_k`: Top-k sampling parameter
- `eval_top_p`: Top-p (nucleus) sampling parameter

### Hardware Section
- `device: "smart"` - Intelligently selects: MPS > available CUDA GPU > CPU
- `use_cached_dataset: true` - Cache tokenized data for faster subsequent runs
- `num_workers`: DataLoader workers
- `pin_memory: true` - Pin memory for faster data transfer

## Platform Support

All configs use `device: "smart"` which automatically selects the best available hardware:

- **CUDA (NVIDIA GPUs)**: Supports float16 and bfloat16 AMP with GradScaler
- **MPS (Apple Silicon)**: Supports float16 AMP only (bfloat16 not supported)
- **CPU**: No AMP support (slower but functional)

The smart device selector:
1. Checks for MPS (Apple Silicon) availability
2. Checks for available CUDA GPU (< 20% memory usage)
3. Falls back to CPU

## MLflow Integration

All configs include MLflow experiment tracking:

```yaml
use_mlflow: true
mlflow_experiment_name: "storyteller"
mlflow_run_name: "<config_name>"
mlflow_tracking_uri: "http://localhost:8080"
mlflow_log_system_metrics: true
```

Start MLflow UI:
```bash
mlflow ui --port 8080
```

Then visit http://localhost:8080 to view training metrics, loss curves, and generated stories.

## Modifying Configurations

To create a custom config:

1. Copy an existing config that's closest to your needs
2. Modify the parameters (especially `hidden_size`, `num_layers`, `batch_size`)
3. Adjust training settings (`num_epochs`, `learning_rate`)
4. Update `config_name` and `mlflow_run_name` to match your filename

Example:
```yaml
model:
  config_name: "my_custom_model"  # Match your filename
  hidden_size: 512  # Customize architecture
  num_layers: 8
  # ... other settings

training:
  mlflow_run_name: "my_custom_model"  # Match config_name
  # ... other settings
```

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `batch_size`
- Increase `gradient_accumulation_steps` (keeps effective batch size constant)
- Enable `gradient_checkpointing: true`
- Reduce `max_seq_length`
- Use `amp_dtype: "float16"` instead of "bfloat16"

### Training Too Slow
- Disable `gradient_checkpointing` if you have enough memory
- Increase `batch_size` if memory allows
- Reduce `num_workers` if CPU-bound
- Use `use_cached_dataset: true` for faster data loading

### bfloat16 Not Supported
- Change `amp_dtype: "bfloat16"` to `amp_dtype: "float16"`
- Common on MPS (Apple Silicon) devices

### Smart Device Selection Issues
- Manually specify device: `device: "cuda:0"` or `device: "mps"` or `device: "cpu"`
- Check available devices with: `python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"`

## Additional Resources

- **Documentation**: See `docs/` directory for detailed guides
- **MLflow Guide**: See `docs/MLflow.md` for experiment tracking details
- **Metrics Guide**: See `docs/METRICS.md` for evaluation metrics documentation
- **Project Structure**: See main `CLAUDE.md` for architecture details

## Questions?

For issues or questions:
- Check the main README.md
- Review docs/ directory
- Open an issue on GitHub
