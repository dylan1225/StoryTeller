# The Storyteller 📖

A story-generating language model built from scratch using Mixture of Experts (MoE) architecture, designed to run on consumer hardware (32GB VRAM).

## Overview

This project is a comprehensive curriculum for learning modern LLM concepts by building a small language model capable of generating creative stories. The model uses a sparse Mixture of Experts architecture to achieve high capacity while maintaining computational efficiency.

## Architecture

- **Model Size**: ~400-500M parameters (only ~100M active per forward pass)
- **Architecture**: Decoder-only transformer with MoE layers
- **MoE Configuration**: 8 experts per layer, Top-2 routing
- **Context Length**: 2048 tokens
- **Vocabulary**: ~50k tokens

## Project Structure

```
storyteller/
├── src/
│   └── storyteller/         # Main package
│       ├── data/            # Data preparation and preprocessing
│       ├── model/           # Core model architecture
│       ├── training/        # Training scripts and optimization
│       ├── inference/       # Generation and inference
│       ├── evaluation/      # Metrics and evaluation
│       └── utils/           # Utility functions (device selection, etc.)
├── configs/                 # Configuration files
├── docs/                    # Documentation
├── notebooks/               # Interactive tutorials
├── tests/                   # Unit tests
├── pyproject.toml           # Package configuration
└── README.md                # This file
```

## Setup

### Prerequisites

- Python 3.9+
- **Hardware** (one of the following):
  - CUDA-capable GPU with 32GB+ VRAM (recommended for training)
  - Apple Silicon Mac with 32GB+ unified memory (MPS supported)
  - CPU-only (works but significantly slower)
- 50GB+ disk space for datasets

### Installation

**Option 1: Using pip install (recommended)**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Optional: Install development dependencies
pip install -e ".[dev]"

# Optional: Install DeepSpeed for advanced optimization
pip install -e ".[deepspeed]"
```

**Option 2: Using requirements.txt**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt

# Optional: Install DeepSpeed
pip install -r requirements-deepspeed.txt

# Install package in editable mode
pip install -e .
```

This will install the `storyteller` package and all CLI tools:
- `storyteller-download` - Download story datasets
- `storyteller-preprocess` - Preprocess and clean data
- `storyteller-tokenizer` - Train custom tokenizer
- `storyteller-train` - Train the model
- `storyteller-generate` - Generate stories

## Quick Start

### 1. Prepare Data

```bash
# Download story datasets
storyteller-download --output_dir data/raw

# Preprocess and clean data
storyteller-preprocess --input_dir data/raw --output_dir data/processed

# Train custom tokenizer
storyteller-tokenizer --input_file data/processed/train.txt --output_dir data/tokenizers
```

### 2. Train Model

The project includes several pre-configured models for different use cases:

| Config | Params | Training Time | VRAM | Use Case |
|--------|--------|---------------|------|----------|
| `tiny_test.yaml` | ~3M | 2-5 min | 4GB+ | Pipeline testing, debugging |
| `fast_train.yaml` | ~10M | 10-20 min | 8GB+ | Learning, quick demos |
| `min_model.yaml` | ~50M | 6-12 hrs | 16GB+ | First serious training run |
| `gpt2_small.yaml` | ~124M | 1-2 days | 24GB+ | **RECOMMENDED** - Quality stories |
| `base_model.yaml` | ~350M | 3-4 days | 32GB+ | High-quality generation |
| `moe_model.yaml` | ~500M (100M active) | 4-5 days | 32GB+ | Advanced MoE learning |

```bash
# Fast training demo (~10M params) - 10-20 minutes, perfect for learning
storyteller-train --config configs/fast_train.yaml

# GPT2-small config (~124M params) - 1-2 days, RECOMMENDED for quality stories
storyteller-train --config configs/gpt2_small.yaml

# Train minimal model (~50M params) - first serious training run
storyteller-train --config configs/min_model.yaml

# Train base model (~350M params) - high-quality generation
storyteller-train --config configs/base_model.yaml

# Train MoE model (~500M total, ~100M active) - sparse architecture
storyteller-train --config configs/moe_model.yaml

# Resume from checkpoint
storyteller-train --config configs/moe_model.yaml --resume checkpoints/checkpoint_step_10000.pt
```

**Configuration Options:**

**Device Selection:**
In your config file, set:
- `device: "smart"` - Automatically selects best available (MPS > available CUDA GPU > CPU)
- `device: "mps"` - Force Apple Silicon
- `device: "cuda"` or `"cuda:0"` - Force specific CUDA GPU
- `device: "cpu"` - Force CPU

The smart mode checks GPU memory usage and selects an available GPU in multi-GPU systems.

**Tokenizer Selection:**
Specify which tokenizer to use in the config:
```yaml
training:
  tokenizer_path: "data/tokenizers/storyteller-tokenizer"  # Path to trained tokenizer
```

You can also override via CLI:
```bash
storyteller-train --config configs/gpt2_small.yaml --tokenizer_path path/to/custom/tokenizer
```

### 3. Generate Stories

```bash
# Interactive generation
storyteller-generate --checkpoint checkpoints/best_model.pt --interactive

# Generate from prompt
storyteller-generate --checkpoint checkpoints/best_model.pt --prompt "Once upon a time"

# Batch generation
storyteller-generate --checkpoint checkpoints/best_model.pt --prompts_file prompts.txt --output stories.txt
```

## Curriculum Modules

The project follows a 10-week curriculum:

1. **Foundations** (Week 1-2): Tokenization, transformer basics, data preparation
2. **Base Model** (Week 3-4): Simple decoder-only transformer, training loops
3. **Mixture of Experts** (Week 5-6): MoE architecture, routing, load balancing
4. **Advanced Training** (Week 7-8): Memory optimization, scaling
5. **Inference & Deployment** (Week 9-10): Sampling strategies, quantization

## Experiment Tracking with MLflow

The project uses MLflow for tracking experiments, metrics, and models.

### Start MLflow UI

```bash
mlflow ui --port 8080
```

Then open http://localhost:8080 to view your experiments, compare runs, and analyze metrics.

### Logged Metrics
- **Per-step metrics**: Training loss, learning rate, global step counter
- **Per-epoch metrics**: Average training loss per epoch
- **Validation metrics**: Loss and perplexity
- **MoE metrics**: Expert utilization, balance, and routing entropy (when using MoE)
- **System metrics**: CPU, GPU, memory, disk, network utilization - enabled by default
- **Artifacts**: Model checkpoints and final model

See [docs/MLflow.md](docs/MLflow.md) for detailed usage guide.

## Evaluation Strategy

### Train/Val Split (No Separate Test Set)

This project uses a **train/validation split without a separate test set**. This is a deliberate design choice appropriate for educational and research base models:

**Why This Approach:**

1. **Educational Focus**: This is a learning project for understanding LLM concepts, not a production system requiring publication-quality metrics
2. **Iterative Development**: The validation set serves multiple purposes during active development:
   - Monitoring training progress in real-time
   - Early stopping to prevent overfitting
   - Hyperparameter tuning across experiments
   - Quality metrics tracking (diversity, repetition, etc.)
3. **Resource Efficiency**: Holding out a third dataset would reduce training data, which is valuable for a base model
4. **Common for Base Models**: Many foundational models use train/val splits during development, reserving test sets for final evaluation when needed

**When You Would Need a Test Set:**

- Publishing research results requiring unbiased metrics
- Comparing multiple final models after extensive hyperparameter tuning
- Production deployment requiring strict performance guarantees
- Downstream task evaluation (fine-tuning, transfer learning)

**Our Validation Set Usage:**

The validation set in this project serves as a held-out set for:
- Computing unbiased loss and perplexity during training
- Generating sample stories for quality assessment
- Tracking Phase 1 metrics (diversity, repetition, length, vocabulary)
- Checkpoint selection (saving best model based on val loss)

**Data Split:**
- Training: ~7.8M stories (94.5%)
- Validation: ~435K stories (5.5%)
- Total: ~426M training tokens, ~23M validation tokens

If you decide to productionize this model or need formal benchmarking, you can easily create a test set by further splitting the validation data or holding out additional data during preprocessing.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Linting with ruff
python -m ruff check

# Auto-fix issues
python -m ruff check --fix

# Format code
python -m ruff format
```

### Type Checking

```bash
mypy model/ training/ inference/
```

## License

MIT

## Acknowledgments

- Design inspired by modern LLM architectures (GPT, Mixtral, Switch Transformer)
- Datasets: TinyStories, WritingPrompts, Project Gutenberg, BookCorpus
