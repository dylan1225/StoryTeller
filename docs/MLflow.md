# MLflow Tracking Guide

The Storyteller project uses [MLflow](https://mlflow.org/) for experiment tracking, providing a comprehensive view of your training runs, metrics, and models.

## Quick Start

### 1. Start MLflow UI (Optional)

MLflow automatically logs to a local `mlruns` directory by default. To view the UI:

```bash
# From project root (default configs use port 8080)
mlflow ui --port 8080

# Or use default port 5000
mlflow ui
```

Then open http://localhost:8080 in your browser (or http://localhost:5000 if using default port).

### 2. Training with MLflow

MLflow logging is configured in your YAML files:

```yaml
training:
  use_mlflow: true
  mlflow_experiment_name: "storyteller"
  mlflow_run_name: "base_model"
  mlflow_tracking_uri: "http://localhost:8080"  # Match UI port
  mlflow_log_system_metrics: true  # Log CPU, GPU, memory (default: true)
```

### 3. Run Training

```bash
storyteller-train --config configs/base_model.yaml
```

## What Gets Logged

### Parameters
- **Model configuration**: vocab_size, hidden_size, num_layers, etc.
- **Training hyperparameters**: learning_rate, batch_size, etc.
- **Hardware settings**: device, use_amp, amp_dtype

### Metrics (per step)
- `train/loss` - Training loss (per batch/step)
- `train/learning_rate` - Current learning rate
- `train/global_step` - Global training step counter

### Metrics (per epoch)
- `epoch/train_loss` - Average training loss for entire epoch
- `epoch/train_moe_loss` - Average MoE auxiliary loss for epoch (if using MoE)
- `val/loss` - Validation loss
- `val/perplexity` - Validation perplexity

### MoE Metrics (if using Mixture of Experts)
- `moe/layer_X_balance` - Expert balance for MoE layers
- `moe/layer_X_entropy` - Routing entropy for MoE layers

### System Metrics (if enabled)
Automatically logged when `mlflow_log_system_metrics: true`:
- `system/cpu_utilization_percentage` - CPU usage
- `system/gpu_X_utilization_percentage` - GPU utilization per device
- `system/gpu_X_memory_usage_percentage` - GPU memory per device
- `system/gpu_X_memory_usage_megabytes` - GPU memory in MB
- `system/disk_usage_percentage` - Disk usage
- `system/network_receive_megabytes` - Network receive
- `system/network_transmit_megabytes` - Network transmit
- `system/system_memory_usage_percentage` - RAM usage
- `system/system_memory_usage_megabytes` - RAM in MB

### Artifacts
- **Final model**: Saved as MLflow model
- **Checkpoints**: Saved to checkpoint directory

## MLflow UI Features

### Experiment View
- Compare multiple runs side-by-side
- Filter and search runs
- Sort by metrics

### Run Details
- **Overview**: Run metadata and status
- **Parameters**: All hyperparameters
- **Metrics**: Interactive plots over time
- **Artifacts**: Models and files

### Charts
- Parallel coordinates plot
- Scatter plots
- Line charts for metrics

## Advanced Usage

### Remote Tracking Server

Set up a remote MLflow server:

```bash
# On server
mlflow server --host 0.0.0.0 --port 5000
```

Update your config:

```yaml
training:
  mlflow_tracking_uri: "http://your-server:5000"
```

### Comparing Runs

In the MLflow UI:
1. Select multiple runs (checkbox)
2. Click "Compare"
3. View side-by-side comparison

### Registering Models

Best models can be registered for deployment:

```python
import mlflow

# Register the best model
client = mlflow.tracking.MlflowClient()
client.create_registered_model("storyteller-production")

# Promote a run's model
result = mlflow.register_model(
    "runs:/<RUN_ID>/model",
    "storyteller-production"
)
```

### Querying Runs Programmatically

```python
import mlflow

# Search for best runs
runs = mlflow.search_runs(
    experiment_names=["storyteller"],
    filter_string="metrics.val/loss < 2.0",
    order_by=["metrics.val/loss ASC"],
    max_results=5
)

print(runs[["run_id", "metrics.val/loss", "params.learning_rate"]])
```

## Directory Structure

```
project/
├── mlruns/              # MLflow tracking data
│   ├── 0/              # Default experiment
│   ├── 1/              # Your experiments
│   └── ...
├── mlartifacts/        # Artifact storage
└── checkpoints/        # Model checkpoints (separate)
```

## Tips

1. **Experiment Names**: Use descriptive names like "storyteller-baseline", "storyteller-moe"
2. **Run Names**: Include key hyperparameters: "lr3e-4_bs32_base"
3. **Tags**: Add custom tags for organization
4. **Notes**: Document run purpose in MLflow UI

## Disabling MLflow

Set in config:
```yaml
training:
  use_mlflow: false
```

Or training will work fine if MLflow is not installed (graceful fallback).

## Example Workflow

```bash
# 1. Start UI
mlflow ui --port 8080 &

# 2. Train baseline
storyteller-train --config configs/base_model.yaml

# 3. Train MoE model
storyteller-train --config configs/moe_model.yaml

# 4. Open browser to http://localhost:8080

# 5. Compare runs and select best model

# 6. Export best model
mlflow models serve -m runs:/<best-run-id>/model -p 5001
```

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Models](https://mlflow.org/docs/latest/models.html)
