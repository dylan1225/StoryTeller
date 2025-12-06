# Storyteller Quick Start Guide

This guide will get you up and running with the Storyteller project in minutes.

## Installation

```bash
# Clone or navigate to the project directory
cd storyteller

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .
```

This installs the `storyteller` package and makes all CLI commands available:
- `storyteller-download` - Download story datasets
- `storyteller-preprocess` - Preprocess and clean data
- `storyteller-tokenizer` - Train custom tokenizer
- `storyteller-train` - Train the model
- `storyteller-generate` - Generate stories

## Workflow

### Step 1: Download and Prepare Data

```bash
# Download story datasets (TinyStories, WritingPrompts, etc.)
storyteller-download --output_dir data/raw

# Preprocess and clean the data
storyteller-preprocess --input_dir data/raw --output_dir data/processed

# Train a custom BPE tokenizer
storyteller-tokenizer \
    --input_file data/processed/train.txt \
    --output_dir data/tokenizers \
    --vocab_size 50000
```

### Step 2: Train a Model

**Option A: Fast Training Demo (~10M params) - Recommended for learning**
```bash
storyteller-train --config configs/fast_train.yaml
```
Trains in 10-20 minutes! Perfect for students to quickly see training dynamics, loss curves, and evaluation metrics in action. Great for understanding the system before longer training runs.

**Option B: Minimal Model (~50M params) - For testing**
```bash
storyteller-train --config configs/min_model.yaml
```
Fast to train, uses less memory, perfect for validating your setup.

**Option C: Base Model (350M params)**
```bash
storyteller-train --config configs/base_model.yaml
```

**Option D: MoE Model (500M total, 100M active)**
```bash
storyteller-train --config configs/moe_model.yaml
```

**Resume from checkpoint:**
```bash
storyteller-train \
    --config configs/moe_model.yaml \
    --resume checkpoints/moe_model/checkpoint_step_10000.pt
```

### Step 3: Generate Stories

**Interactive mode:**
```bash
storyteller-generate \
    --checkpoint checkpoints/best_model.pt \
    --interactive
```

**Generate from a prompt:**
```bash
storyteller-generate \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Once upon a time in a magical forest" \
    --max_length 512 \
    --temperature 0.9
```

**Batch generation:**
```bash
storyteller-generate \
    --checkpoint checkpoints/best_model.pt \
    --prompts_file prompts.txt \
    --output generated_stories.txt
```

## Project Structure

```
storyteller/
├── src/storyteller/         # Main package
│   ├── data/               # Data processing
│   ├── model/              # Model architecture
│   ├── training/           # Training code
│   ├── inference/          # Generation code
│   ├── evaluation/         # Metrics
│   └── utils/              # Utilities (device selection, etc.)
├── configs/                # YAML configs
├── docs/                   # Documentation
├── notebooks/              # Jupyter tutorials
└── tests/                  # Unit tests
```

## Configuration

Edit `configs/base_model.yaml` or `configs/moe_model.yaml` to customize:
- Model architecture (size, layers, attention heads)
- Training hyperparameters (learning rate, batch size)
- MoE settings (number of experts, routing strategy)
- Hardware settings (device, mixed precision)
- Dataset caching (use_cached_dataset for faster subsequent runs)

**Device Selection:**
Set `device: "smart"` to automatically select the best available compute:
- Priority: MPS (Apple Silicon) > available CUDA GPU > CPU
- Smart mode checks GPU memory usage in multi-GPU systems
- Manual options: "mps", "cuda", "cuda:0", "cuda:1", "cpu"

## Tips

1. **Start Small**: Use the fast training demo (`fast_train.yaml`) to quickly understand the system, then try minimal model (`min_model.yaml`) to validate your setup, and finally scale up to base or MoE models
2. **Monitor Training**: Use MLflow by setting `use_mlflow: true` in the config
   - Start tracking server: `mlflow ui --port 8080`
   - View at http://localhost:8080
   - System metrics (CPU, GPU, memory) are logged automatically
3. **Use Cached Datasets**: Set `use_cached_dataset: true` to avoid re-tokenizing on subsequent runs
   - First run tokenizes and caches (~11 minutes for full dataset)
   - Subsequent runs load from cache in seconds
4. **Smart Device Selection**: Set `device: "smart"` to automatically use the best available hardware
5. **Adjust Batch Size**: Reduce batch size if you run out of GPU memory
6. **Experiment with Sampling**: Try different temperature/top-k/top-p values for generation

## Troubleshooting

**Out of memory during training?**
- Reduce `batch_size` in config
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable gradient checkpointing (reduces memory at cost of speed)

**Poor generation quality?**
- Train for more epochs
- Increase model size
- Try different sampling parameters (temperature, top_p)
- Use the MoE model for better capacity

**Slow training?**
- Enable mixed precision: `use_amp: true`
- Use `amp_dtype: bfloat16` if supported
- Reduce `eval_every_n_steps` for less frequent evaluation

## Next Steps

- Read the [DESIGN.md](docs/DESIGN.md) for architecture details
- Check [TODO.md](docs/TODO.md) for upcoming features
- Explore the Jupyter notebooks for interactive tutorials
- Customize the model architecture for your use case

Happy storytelling! 📖✨
