# Storyteller Training Notebooks

Welcome to the Storyteller educational notebook series! These interactive Jupyter notebooks will guide you through building a complete story-generating language model from scratch using Mixture of Experts (MoE) architecture.

## 📚 Overview

This series takes you on a comprehensive journey from data preparation to generating creative stories with a trained transformer model. Each notebook builds on the previous one, teaching both theoretical concepts and practical implementation.

**Target Audience**: Students and practitioners interested in:
- Large language model training
- Transformer architectures
- Mixture of Experts (MoE)
- Text generation
- Deep learning best practices

**Time Commitment**: ~10-15 hours total (2-3 hours per notebook)

## 📖 Notebook Series

### 01. Data Preparation (`01_data_preparation.ipynb`)

**Duration**: ~2 hours | **Difficulty**: Beginner

**What You'll Learn**:
- How to download and explore story datasets (TinyStories, WritingPrompts)
- Text preprocessing and cleaning techniques
- Building a Byte-Pair Encoding (BPE) tokenizer from scratch
- Understanding vocabulary size, compression ratios, and token statistics
- Visualizing text distributions and tokenization patterns

**Key Concepts**:
- Tokenization fundamentals
- BPE algorithm
- Vocabulary construction
- Dataset statistics

**Prerequisites**: Basic Python, familiarity with text data

---

### 02. Understanding Transformers (`02_understanding_transformers.ipynb`)

**Duration**: ~3 hours | **Difficulty**: Intermediate

**What You'll Learn**:
- Scaled dot-product attention mechanism
- Multi-head attention and why it matters
- Positional encodings (learned vs. Rotary/RoPE)
- Causal masking for autoregressive generation
- Feed-forward networks in transformers
- Complete transformer decoder block construction
- Decoder-only architecture (GPT-style)

**Key Concepts**:
- Query, Key, Value paradigm
- Self-attention and cross-attention
- Residual connections and layer normalization
- Attention pattern visualization

**Prerequisites**: Linear algebra basics, neural network fundamentals

**Hands-On**:
- Implement attention from scratch
- Build a simple GPT model
- Visualize attention patterns across multiple heads

---

### 03. Building Mixture of Experts (`03_building_moe.ipynb`)

**Duration**: ~3 hours | **Difficulty**: Advanced

**What You'll Learn**:
- Why sparse models (MoE) are efficient
- Top-K routing and gating networks
- Expert specialization and load balancing
- Building complete MoE layers
- Integrating MoE into transformer blocks
- Analyzing parameter efficiency (total vs. active parameters)

**Key Concepts**:
- Conditional computation
- Sparse activation
- Expert routing strategies
- Load balancing loss
- Router entropy and utilization metrics

**Prerequisites**: Understanding of transformers (Notebook 02)

**Hands-On**:
- Implement Top-K router
- Create MoE layer with 8 experts
- Visualize expert routing patterns
- Compare MoE vs. dense model efficiency

---

### 04. Training Basics (`04_training_basics.ipynb`)

**Duration**: ~2.5 hours | **Difficulty**: Intermediate

**What You'll Learn**:
- Creating efficient data pipelines for language models
- Configuring AdamW optimizer with weight decay
- Learning rate scheduling (warmup + cosine decay)
- Mixed precision training (FP16/BF16)
- Gradient accumulation for larger effective batch sizes
- Gradient clipping for training stability
- Checkpointing and model saving
- Computing perplexity and validation metrics

**Key Concepts**:
- Next-token prediction objective
- Weight decay groups (which parameters to regularize)
- Warmup for training stability
- Automatic Mixed Precision (AMP)
- Gradient accumulation

**Prerequisites**: Basic deep learning training knowledge

**Hands-On**:
- Build a complete training loop
- Implement learning rate scheduling
- Train a small model
- Visualize training curves

---

### 05. Full Training and Inference (`05_full_training_and_inference.ipynb`)

**Duration**: ~3-4 hours | **Difficulty**: Advanced

**What You'll Learn**:
- End-to-end training pipeline for production
- MLflow experiment tracking and management
- Training the complete MoE Storyteller model
- Text generation strategies:
  - Greedy decoding
  - Sampling with temperature
  - Top-K sampling
  - Nucleus (Top-P) sampling
- Interactive story generation
- Comparing different checkpoints
- Evaluating generation quality
- Exporting models for production

**Key Concepts**:
- Experiment tracking best practices
- Generation hyperparameters and their effects
- Sampling strategies for text generation
- Model evaluation and comparison
- Production deployment preparation

**Prerequisites**: All previous notebooks

**Hands-On**:
- Train a full MoE model on real data
- Generate stories with different sampling strategies
- Use MLflow to track experiments
- Create an interactive story generator
- Export trained model for deployment

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# From the project root directory
cd storyteller

# Install dependencies
pip install -r requirements.txt
pip install jupyter ipykernel

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Prepare Data (Optional for first 2 notebooks)

```bash
# Download datasets
storyteller-download --datasets tinystories writingprompts

# Train tokenizer
storyteller-tokenizer

# Preprocess data
storyteller-preprocess
```

**Note**: Notebooks 01 and 02 can run without real data (use dummy data for learning). Notebooks 03-05 work best with actual datasets.

### 3. Start Jupyter

```bash
# Launch Jupyter from the notebooks directory
cd notebooks
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### 4. MLflow Setup (for Notebook 05)

```bash
# Start MLflow tracking server (in a separate terminal)
mlflow ui --port 8080

# Access at http://localhost:8080
```

---

## 📋 Recommended Learning Path

### Path 1: Complete Beginner (15 hours)
1. **01_data_preparation.ipynb** - Understand the data
2. **02_understanding_transformers.ipynb** - Learn architecture fundamentals
3. **03_building_moe.ipynb** - Master advanced concepts
4. **04_training_basics.ipynb** - Learn training best practices
5. **05_full_training_and_inference.ipynb** - Put it all together

### Path 2: Experienced Practitioners (8 hours)
- Skim notebooks 01-02 for project-specific context
- Focus on **03_building_moe.ipynb** for MoE implementation
- Deep dive **04_training_basics.ipynb** and **05_full_training_and_inference.ipynb**

### Path 3: Quick Start (3 hours)
- Review **01_data_preparation.ipynb** outputs only
- Study MoE implementation in **03_building_moe.ipynb**
- Jump to **05_full_training_and_inference.ipynb** for end-to-end training

---

## 💡 Tips for Success

### General Tips
- **Run cells in order**: Each notebook builds on previous cells
- **Experiment freely**: Modify hyperparameters and observe results
- **Read the markdown**: Detailed explanations are in text cells
- **Complete exercises**: Each notebook has hands-on exercises
- **Use GPU if available**: Training will be much faster

### Hardware Recommendations
- **Minimum**: 8GB RAM, CPU (for learning with small models)
- **Recommended**: 16GB RAM, GPU with 8GB+ VRAM
- **Optimal**: 32GB RAM, GPU with 16GB+ VRAM (for full model training)

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'storyteller'`
```bash
# Solution: Install the package in development mode
pip install -e .
```

**Issue**: Jupyter can't find the kernel
```bash
# Solution: Install ipykernel
python -m ipykernel install --user --name storyteller
```

**Issue**: Out of memory during training
```python
# Solution: Reduce batch size or use gradient accumulation
config['training']['batch_size'] = 2  # Smaller batch
config['training']['gradient_accumulation_steps'] = 8  # Accumulate more
```

**Issue**: MLflow tracking not working
```bash
# Solution: Start MLflow server first
mlflow ui --port 8080
# Then update the tracking URI in notebook 05
```

---

## 🎯 Learning Objectives by Topic

### Data & Preprocessing
- Dataset exploration and analysis
- Tokenization strategies (BPE, WordPiece)
- Vocabulary construction
- Text preprocessing best practices

### Model Architecture
- Transformer fundamentals
- Attention mechanisms
- Positional encodings
- Mixture of Experts (MoE)
- Sparse vs. dense models

### Training
- Optimization strategies (AdamW)
- Learning rate scheduling
- Mixed precision training
- Gradient accumulation and clipping
- Experiment tracking with MLflow

### Inference
- Text generation algorithms
- Sampling strategies
- Temperature and nucleus sampling
- Generation quality evaluation

---

## 📊 Expected Outcomes

After completing all notebooks, you will:

1. **Understand** the complete pipeline for training large language models
2. **Implement** transformer and MoE architectures from scratch
3. **Train** a 500M parameter MoE model for story generation
4. **Generate** creative stories using various sampling strategies
5. **Track** experiments professionally using MLflow
6. **Deploy** models in production-ready format

---

## 🔗 Additional Resources

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538) - Sparse MoE
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - Scaling MoE
- [RoFormer](https://arxiv.org/abs/2104.09864) - Rotary Position Embeddings

### Tutorials
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide
- [Hugging Face NLP Course](https://huggingface.co/course) - Comprehensive NLP course
- [Stanford CS224N](http://web.stanford.edu/class/cs224n/) - NLP with Deep Learning

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Tokenizers Documentation](https://huggingface.co/docs/tokenizers)

---

## 🤝 Contributing

Found an issue or have a suggestion? This is an educational project!

- **Issues**: Report bugs or unclear explanations
- **Improvements**: Suggest additional visualizations or exercises
- **Extensions**: Add new notebooks for advanced topics

---

## 📝 Notes

### Notebook Outputs
- Notebooks include example outputs for reference
- Your results may vary due to randomness
- Set seeds for reproducibility when needed

### Training Time
- Small models (notebooks 1-4): Minutes on CPU
- Full model (notebook 5): Hours on GPU
- Use gradient checkpointing to reduce memory

### Dataset Sizes
- **TinyStories**: ~2GB, quick to download
- **WritingPrompts**: ~5GB, more diverse stories
- Can start with TinyStories for faster iteration

---

## 🎓 Academic Use

These notebooks are designed for educational purposes and can be used in:
- University courses on NLP/Deep Learning
- Self-study and online courses
- Workshops and tutorials
- Research training programs

**Citation**: If you use these notebooks in academic work, please cite the Storyteller project.

---

## ⚖️ License

This educational material is provided as-is for learning purposes. See the main project LICENSE for details.

---

## 📧 Support

For questions or help:
1. Check the **Common Issues** section above
2. Review the notebook markdown cells for explanations
3. Consult the main project documentation
4. Refer to the Additional Resources

---

**Happy Learning! 🚀📚**

Start with `01_data_preparation.ipynb` and build your way up to training a complete story-generating AI!
