# Local Training Workflow Documentation

This document describes the local training workflow that supports execution on local machines and Google Colab, as an alternative to the Azure ML-based workflow.

## Overview

The local training workflow (`notebooks/01_orchestrate_training_local.ipynb`) provides a complete training pipeline that:

- Downloads datasets directly from Kaggle (no Azure ML data assets required)
- Executes training locally or on Google Colab GPU
- Uses Optuna for hyperparameter optimization
- Tracks experiments with MLflow (local file store)
- Produces ONNX models ready for deployment

## Differences from Azure ML Workflow

| Aspect | Azure ML Workflow | Local Workflow |
|--------|------------------|----------------|
| **Orchestration Notebook** | `01_orchestrate_training.ipynb` | `01_orchestrate_training_local.ipynb` |
| **Dataset Source** | Azure ML Data Assets | Kaggle (via `kagglehub`) |
| **Compute** | Azure ML Compute Clusters | Local machine / Google Colab GPU |
| **HPO** | Azure ML Sweep Jobs | Optuna studies |
| **Environment** | Azure ML Environments (Docker) | Local Python environment |
| **Tracking** | Azure ML MLflow | Local MLflow file store |
| **Model Storage** | Azure ML Model Registry | Local filesystem |

## Prerequisites

### Local Machine Setup

1. **Python Environment**: Python 3.10+
2. **Dependencies**: Install from `config/environment/conda.yaml` plus:
   - `kagglehub` - For dataset download
   - `optuna` - For hyperparameter optimization
   - `mlflow` - For experiment tracking

3. **GPU (Optional but Recommended)**:
   - NVIDIA GPU with CUDA support
   - PyTorch with CUDA enabled

### Google Colab Setup

1. **Runtime Configuration**:
   - Runtime > Change runtime type > Hardware accelerator = **GPU**
   - Runtime > Change runtime type > Runtime shape = **High-RAM** (recommended)

2. **Repository Setup**:
   ```python
   # Clone repository in Colab
   !git clone https://github.com/longdang193/resume-ner-azureml.git
   %cd resume-ner-azureml
   ```

3. **Kaggle Authentication** (if required):
   - Upload `kaggle.json` to Colab or set up Kaggle API credentials

## Workflow Steps

### Step P1-3.1: Load Centralized Configs

Loads and validates all configuration files. Same as Azure ML workflow.

**Key Files:**
- `config/experiment/resume_ner_baseline.yaml`
- `config/data/resume_tiny.yaml`
- `config/model/*.yaml`
- `config/train.yaml`
- `config/hpo/smoke.yaml` or `config/hpo/prod.yaml`

### Step P1-3.2: Download Dataset from Kaggle & Create dataset-tiny

1. Downloads dataset using `kagglehub.dataset_download("yashpwrr/resume-ner-training-dataset")`
2. Copies `train.json` to local `dataset/` directory
3. Creates `dataset_tiny/` with filtered short samples (8 train + 2 validation)
4. Updates `config/data/resume_tiny.yaml` if needed

**Output:**
- `dataset/train.json` - Full dataset
- `dataset_tiny/train.json` - Tiny train set
- `dataset_tiny/validation.json` - Tiny validation set

### Step P1-3.3: Setup Local Environment

1. Installs dependencies from `config/environment/conda.yaml`
2. Verifies GPU availability (CUDA/Colab GPU)
3. Sets up MLflow tracking with local file store

**MLflow Tracking:**
- Tracking URI: `file://<project_root>/mlruns`
- Experiment name: From experiment config

### Step P1-3.4: The Dry Run

Runs a minimal training job (1 epoch, small batch) to validate:
- Data loading
- Model initialization
- Training loop
- Checkpoint saving

**Output:** `outputs/dry_run/checkpoint/`

### Step P1-3.5: The Sweep (HPO) - Local with Optuna

Runs hyperparameter optimization using Optuna:

1. **Search Space**: Translated from `config/hpo/*.yaml` to Optuna distributions
2. **Sampling**: Random sampling (configurable)
3. **Early Stopping**: Median pruner (if configured)
4. **Trial Execution**: Each trial runs `src/train.py` with sampled hyperparameters
5. **Metric Tracking**: Results logged to MLflow

**Configuration:**
- Max trials: From `hpo_config["sampling"]["max_trials"]`
- Timeout: From `hpo_config["sampling"]["timeout_minutes"]`
- Objective: From `hpo_config["objective"]` (default: `macro-f1`, maximize)

**Output:**
- Optuna study objects for each backbone
- Trial checkpoints: `outputs/hpo/<backbone>/trial_<n>/`
- MLflow runs with metrics and parameters

### Step P1-3.6: Best Configuration Selection

Selects the best configuration across all backbone studies:

1. Extracts best trial from each Optuna study
2. Compares across backbones using objective metric
3. Builds configuration dictionary matching Azure ML format

**Output:**
- Best configuration saved to `notebooks/best_configuration_cache.json`

### Step P1-3.7: Final Training

Trains final production model using best HPO configuration:

- Uses combined train+validation data
- Full epoch budget
- Fixed random seed (42)
- Early stopping disabled

**Output:** `outputs/final_training/checkpoint/`

### Step P1-4: Model Conversion & Optimization

Converts PyTorch checkpoint to ONNX:

1. Loads checkpoint from final training
2. Exports to ONNX (FP32)
3. Applies int8 quantization (if requested)
4. Runs smoke inference test

**Output:**
- `outputs/onnx_model/model.onnx` (FP32)
- `outputs/onnx_model/model_int8.onnx` (Quantized, if successful)

## File Structure

```
resume-ner-azureml/
├── notebooks/
│   ├── 01_orchestrate_training.ipynb          # Azure ML workflow
│   ├── 01_orchestrate_training_local.ipynb     # Local workflow (NEW)
│   └── 00_make_tiny_dataset.ipynb              # Dataset preparation
├── src/
│   ├── orchestration/
│   │   └── jobs/
│   │       ├── local_sweeps.py                 # Optuna HPO (NEW)
│   │       └── local_selection.py             # Best config selection (NEW)
│   ├── train.py                                # Training script (reused)
│   └── convert_to_onnx.py                      # ONNX conversion (reused)
├── config/                                     # Configuration files (reused)
├── dataset/                                    # Full dataset
├── dataset_tiny/                              # Tiny dataset for smoke tests
├── outputs/                                    # Training outputs
│   ├── dry_run/
│   ├── hpo/
│   ├── final_training/
│   └── onnx_model/
└── mlruns/                                     # MLflow tracking data
```

## Usage

### Running Locally

1. **Setup environment:**
   ```bash
   pip install -r requirements.txt  # If available
   # Or install from config/environment/conda.yaml
   ```

2. **Open notebook:**
   ```bash
   jupyter notebook notebooks/01_orchestrate_training_local.ipynb
   ```

3. **Run all cells** sequentially

### Running on Google Colab

1. **Open Colab notebook:**
   - Upload `notebooks/01_orchestrate_training_local.ipynb` to Colab
   - Or clone repository and open from there

2. **Configure runtime:**
   - Runtime > Change runtime type > GPU

3. **Run all cells**

## Troubleshooting

### GPU Not Available

**Symptoms:** Training runs on CPU (very slow)

**Solutions:**
- Local: Install PyTorch with CUDA support
- Colab: Ensure GPU runtime is selected (Runtime > Change runtime type)

### Dataset Download Fails

**Symptoms:** `kagglehub.dataset_download()` fails

**Solutions:**
- Verify Kaggle dataset name: `yashpwrr/resume-ner-training-dataset`
- Check internet connection
- For Colab: May need to authenticate with Kaggle API

### MLflow Metrics Not Found

**Symptoms:** HPO trials complete but metrics are 0.0

**Solutions:**
- Check that `src/train.py` is logging metrics to MLflow
- Verify MLflow tracking URI is set correctly
- Check `mlruns/` directory for logged runs

### Out of Memory Errors

**Symptoms:** CUDA out of memory during training

**Solutions:**
- Reduce batch size in HPO config
- Use smaller backbone model (e.g., `distilbert` instead of `deberta`)
- Use `dataset_tiny` for testing

### Optuna Study Fails

**Symptoms:** HPO sweep fails or hangs

**Solutions:**
- Check that training script (`src/train.py`) runs successfully
- Verify dataset path is correct
- Check timeout settings in HPO config
- Review trial logs in `outputs/hpo/`

## Performance Considerations

### Local Machine

- **CPU Training**: Very slow (hours for small dataset)
- **GPU Training**: Recommended, much faster
- **Memory**: Ensure sufficient RAM (16GB+ recommended)

### Google Colab

- **Free Tier**: Limited GPU hours, may disconnect
- **Pro Tier**: Better GPU access, longer sessions
- **Session Limits**: Save checkpoints frequently

## Migration from Azure ML Workflow

To switch from Azure ML to local workflow:

1. **Use local notebook**: `01_orchestrate_training_local.ipynb`
2. **No Azure credentials needed**: Removes dependency on Azure ML workspace
3. **Same training code**: `src/train.py` works for both workflows
4. **Same configs**: Reuse all YAML configuration files

## Next Steps

After completing local training:

1. **Review Results:**
   - Check MLflow UI: `mlflow ui --backend-store-uri mlruns/`
   - Review best configuration in `best_configuration_cache.json`

2. **Test ONNX Model:**
   - Load and test `outputs/onnx_model/model_int8.onnx`
   - Verify inference works correctly

3. **Deploy Model:**
   - Use ONNX model for Phase 2 inference
   - Or register in Azure ML Model Registry (if needed)

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [KaggleHub Documentation](https://github.com/Kaggle/kagglehub)
- [ONNX Runtime](https://onnxruntime.ai/)

