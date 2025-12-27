# HPO Checkpoint and Resume Support

## Overview

The HPO (Hyperparameter Optimization) process now supports checkpointing and resuming, allowing interrupted HPO runs to be resumed from the last checkpoint. This is particularly useful for long-running HPO processes on platforms like Google Colab (which have 12-24 hour session limits) and Kaggle.

## Features

- **Automatic checkpointing**: Save Optuna study state to SQLite database during HPO execution
- **Automatic resume**: Detect and load existing checkpoints to continue from where you left off
- **Platform-aware paths**: Automatically use appropriate storage locations for Colab (Drive mount) and Kaggle
- **Configurable**: Enable/disable checkpointing via HPO configuration YAML files
- **MLflow integration**: HPO runs are tracked in MLflow with checkpoint metadata for discoverability
- **Cross-platform tracking**: Optional Azure ML Workspace integration for unified tracking across platforms

## MLflow Integration

HPO runs are automatically tracked in MLflow, creating parent runs for each HPO sweep with checkpoint metadata. This enables:

- **Unified experiment tracking**: View all HPO runs across platforms in one place (with Azure ML Workspace)
- **Checkpoint discoverability**: Find checkpoint paths via MLflow UI search
- **Run metadata**: Track backbone, trial counts, best values, and resume status

### Setup MLflow Tracking

**Option 1: Config-Driven Setup (Recommended)**

The easiest way is to use the config-driven setup which reads from `config/mlflow.yaml`:

```python
from pathlib import Path
from shared.mlflow_setup import setup_mlflow_from_config

# Setup MLflow from config (automatically uses Azure ML if enabled in config)
setup_mlflow_from_config(
    experiment_name="your-experiment-name",
    config_dir=Path("config")
)
```

To enable Azure ML Workspace tracking, edit `config/mlflow.yaml`:
```yaml
azure_ml:
  enabled: true  # Set to true to enable Azure ML Workspace tracking
  workspace_name: "resume-ner-ws"  # Must match config/infrastructure.yaml
```

Set environment variables:
```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"
```

**Option 2: Manual Azure ML Workspace Setup**

For manual control over Azure ML client creation:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from shared.mlflow_setup import setup_mlflow_cross_platform
import os

# Initialize Azure ML client
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
    workspace_name="resume-ner-ws"
)

# Setup MLflow with Azure ML workspace
setup_mlflow_cross_platform(
    experiment_name="your-experiment-name",
    ml_client=ml_client
)
```

**Option 3: Local Tracking Only (Platform-Specific)**

For local tracking without Azure ML:

```python
from shared.mlflow_setup import setup_mlflow_cross_platform

# Setup MLflow with local SQLite backend
# Platform-aware: Colab → Drive (if mounted), Kaggle → /kaggle/working, Local → ./mlruns
setup_mlflow_cross_platform(
    experiment_name="your-experiment-name"
)
```

### MLflow Run Structure

Each HPO run creates a parent run in MLflow with:

**Parameters:**
- `backbone`: Model backbone name
- `max_trials`: Maximum number of trials
- `checkpoint_path`: Absolute path to SQLite checkpoint file (or `None` if disabled)
- `checkpoint_enabled`: Whether checkpointing is enabled
- `checkpoint_storage_type`: Storage type (`"sqlite"` or `None`)
- `study_name`: Optuna study name
- `objective_metric`: Metric being optimized
- `resumed_from_checkpoint`: Whether this run resumed from a checkpoint

**Metrics:**
- `best_value`: Best trial value found
- `best_{metric}`: Best value for the objective metric
- `n_trials`: Total number of trials
- `n_completed_trials`: Number of completed trials

**Example MLflow Run:**
```
Experiment: resume_ner_baseline-hpo-distilbert
└── Run: hpo_distilbert_20241226_143022
    ├── Parameters:
    │   ├── backbone: distilbert
    │   ├── max_trials: 20
    │   ├── checkpoint_path: /path/to/study.db
    │   ├── checkpoint_enabled: true
    │   └── resumed_from_checkpoint: false
    └── Metrics:
        ├── best_value: 0.85
        └── n_completed_trials: 18
```

### Finding Checkpoints via MLflow

You can search for checkpoints in MLflow UI:

```python
import mlflow

# Search for HPO runs with checkpoints
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="params.checkpoint_enabled = 'true'",
    order_by=["start_time DESC"]
)

# Get checkpoint path from latest run
latest_run = runs.iloc[0]
checkpoint_path = latest_run["params.checkpoint_path"]
print(f"Checkpoint location: {checkpoint_path}")
```

### Troubleshooting MLflow

**MLflow tracking fails silently:**
- HPO will continue without MLflow tracking if setup fails
- Check that MLflow is installed: `pip install mlflow`
- For Azure ML: Ensure `azureml-mlflow` is installed and credentials are valid

**No runs appear in MLflow:**
- Verify MLflow tracking URI is set correctly
- Check that experiment name matches what you're searching for
- For Azure ML: Ensure workspace access is configured

## Configuration

### Basic Setup

Add a `checkpoint` section to your HPO configuration file (`config/hpo/smoke.yaml` or `config/hpo/prod.yaml`):

```yaml
checkpoint:
  enabled: true                    # Enable checkpointing
  storage_path: "{backbone}/study.db"  # Path relative to output_dir, {backbone} placeholder
  auto_resume: true                 # Automatically resume if checkpoint exists
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable checkpointing. If `false`, HPO runs in-memory (no persistence). |
| `storage_path` | string | `"{backbone}/study.db"` | Relative path from `output_dir` for checkpoint file. Supports `{backbone}` placeholder. |
| `auto_resume` | boolean | `true` | Automatically resume from checkpoint if it exists (only applies when `enabled=true`). |

### Example Configurations

**Minimal (default location):**
```yaml
checkpoint:
  enabled: true
  # Uses default: {backbone}/study.db relative to output_dir
```

**Custom path:**
```yaml
checkpoint:
  enabled: true
  storage_path: "checkpoints/hpo_{backbone}.db"
```

**Disable checkpointing:**
```yaml
checkpoint:
  enabled: false
  # HPO runs in-memory, no persistence
```

## Platform-Specific Behavior

### Google Colab

When running on Google Colab:
- **Drive mount preferred**: If Google Drive is mounted at `/content/drive`, checkpoints are automatically saved to `/content/drive/MyDrive/resume-ner-checkpoints/` for persistence across sessions
- **Fallback**: If Drive is not mounted, checkpoints are saved to `/content/` (session-only)

**Example Colab setup:**
```python
from google.colab import drive
drive.mount('/content/drive')

# HPO will automatically use Drive for checkpointing
# Checkpoint location: /content/drive/MyDrive/resume-ner-checkpoints/hpo/{backbone}/study.db
```

### Kaggle

When running on Kaggle:
- **Automatic persistence**: Checkpoints are saved to `/kaggle/working/` which is automatically persisted
- **No manual backup needed**: Kaggle automatically saves outputs

**Example Kaggle checkpoint location:**
```
/kaggle/working/resume-ner-azureml/outputs/hpo/{backbone}/study.db
```

### Local Execution

When running locally:
- Checkpoints are saved to the specified `output_dir` (default: `./outputs/hpo/{backbone}/study.db`)

## Usage Examples

### Basic Usage

1. **Enable checkpointing in config:**
   ```yaml
   # config/hpo/prod.yaml
   checkpoint:
     enabled: true
   ```

2. **Setup MLflow tracking (optional but recommended):**
   ```python
   from pathlib import Path
   from shared.mlflow_setup import setup_mlflow_from_config
   
   # Setup MLflow from config (automatically uses Azure ML if enabled)
   setup_mlflow_from_config(
       experiment_name="my-experiment",
       config_dir=Path("config")
   )
   ```

3. **Run HPO normally:**
   ```python
   from orchestration.jobs.local_sweeps import run_local_hpo_sweep
   
   study = run_local_hpo_sweep(
       dataset_path="dataset",
       config_dir=Path("config"),
       backbone="distilbert",
       hpo_config=hpo_config,  # Contains checkpoint config
       train_config=train_config,
       output_dir=Path("outputs/hpo"),
       mlflow_experiment_name="my-experiment",
   )
   ```

4. **If interrupted, simply re-run the same command:**
   - The system will automatically detect the checkpoint
   - Load existing study with completed trials
   - Calculate remaining trials needed
   - Continue from where it left off

### Resume Example

**First run (interrupted after 5 trials):**
```python
# HPO config: max_trials=10
# Runs 5 trials, then session timeout/interruption
# Checkpoint saved: outputs/hpo/distilbert/study.db (5 trials)
```

**Resume run:**
```python
# Same command, same config
# System detects checkpoint with 5 completed trials
# Calculates: remaining_trials = 10 - 5 = 5
# Runs 5 more trials to complete
```

### Manual Checkpoint Management

You can also manually check checkpoint status:

```python
import optuna

# Load existing study
study = optuna.load_study(
    study_name="hpo_distilbert",
    storage="sqlite:///outputs/hpo/distilbert/study.db"
)

# Check completed trials
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"Completed trials: {len(completed)}")
print(f"Best value: {study.best_value}")
```

## How It Works

### Checkpoint Creation

1. When `checkpoint.enabled=true`, the system:
   - Resolves checkpoint storage path (platform-aware)
   - Creates SQLite database file
   - Configures Optuna study with persistent storage

2. During HPO execution:
   - Each trial result is automatically saved to SQLite database
   - No manual save operations needed

### Resume Detection

1. On HPO start, if `auto_resume=true`:
   - Checks if checkpoint file exists
   - Attempts to load existing study
   - Counts completed trials
   - Calculates remaining trials needed

2. If checkpoint found:
   - Loads study with all previous trials
   - Calculates: `remaining_trials = max_trials - completed_trials`
   - Runs only remaining trials

3. If checkpoint not found or corrupted:
   - Falls back to creating new study
   - Logs warning message
   - Continues normally

### Trial Counting

Only **COMPLETE** trials are counted when calculating remaining trials. Trials in other states (FAILED, PRUNED, etc.) are not counted, ensuring accurate resume behavior.

## Troubleshooting

### Checkpoint Not Found

**Symptom**: HPO starts new study even though checkpoint should exist

**Solutions**:
- Verify `checkpoint.enabled=true` in config
- Check that checkpoint file path is correct
- Ensure checkpoint file wasn't deleted or moved
- Check file permissions

### Corrupted Checkpoint

**Symptom**: Error loading checkpoint, falls back to new study

**Solutions**:
- Check SQLite database file integrity
- If corrupted, delete checkpoint file and start fresh
- Ensure no concurrent access to checkpoint file

### Platform Detection Issues

**Symptom**: Checkpoints saved to wrong location

**Solutions**:
- Verify environment variables: `COLAB_GPU` (Colab) or `KAGGLE_KERNEL_RUN_TYPE` (Kaggle)
- Check that Drive is mounted in Colab: `/content/drive` should exist
- For Kaggle, ensure working directory is `/kaggle/working`

### All Trials Already Completed

**Symptom**: HPO reports "All trials already completed" and exits

**This is expected behavior**: If you've already completed all trials, HPO will skip optimization. To run more trials, increase `max_trials` in your HPO config.

## Best Practices

1. **Enable checkpointing for long runs**: Always enable checkpointing for HPO runs that may take hours or days
2. **Use Azure ML Workspace for unified tracking**: Set up Azure ML Workspace to view experiments across all platforms
3. **Use Drive mount in Colab**: Mount Google Drive to ensure checkpoints persist across sessions
4. **Monitor checkpoint size**: SQLite databases grow with number of trials; monitor disk space
5. **Backup important checkpoints**: Copy checkpoint files to backup location for important experiments
6. **Use descriptive storage paths**: Use meaningful paths in `storage_path` to organize multiple experiments
7. **Check MLflow for checkpoint locations**: Use MLflow UI to find checkpoint paths instead of manually searching

## Technical Details

### Storage Format

- **Format**: SQLite database (`.db` file)
- **Storage URI**: `sqlite:///absolute/path/to/study.db`
- **Optuna API**: Uses `optuna.create_study(..., storage=uri, load_if_exists=True)`

### Study Naming

- **Format**: `hpo_{backbone}` (e.g., `hpo_distilbert`)
- **Uniqueness**: Each backbone gets its own study in the same storage
- **Isolation**: Different backbones don't interfere with each other

### Performance Considerations

- **SQLite overhead**: Minimal for typical HPO workloads (<1000 trials)
- **Load time**: Fast study loading even with hundreds of trials
- **Concurrent access**: SQLite supports concurrent reads, but writes should be serialized

## Related Documentation

- [K-Fold Cross-Validation](./K_FOLD_CROSS_VALIDATION.md) - Checkpoint compatibility with k-fold CV
- [Model Selection Strategy](./MODEL_SELECTION_STRATEGY.md) - Using HPO results for model selection
- [Troubleshooting Guide](./TROUBLESHOOTING.md) - General troubleshooting

