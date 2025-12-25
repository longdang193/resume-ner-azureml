# Testing Guide

## Prerequisites

Before running tests, ensure you have the required dependencies installed:

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Or install individually
pip install transformers onnxruntime numpy pytest pytest-cov pytest-mock optuna mlflow
```

**Note**: If using conda, activate your environment first:

**For PowerShell (Windows):**

Since `conda activate` requires initialization, use one of these methods:

```powershell
# Option 1: Use conda run (no initialization needed, runs command in environment)
& "C:\Users\HOANG PHI LONG DANG\Miniconda3\Scripts\conda.exe" run -n resume-ner-training python --version

# Option 2: Use the environment's Python directly (no activation needed)
& "C:\Users\HOANG PHI LONG DANG\Miniconda3\envs\resume-ner-training\python.exe" -m pytest tests/

# Option 3: Initialize conda for PowerShell (one-time setup, then restart PowerShell)
& "C:\Users\HOANG PHI LONG DANG\Miniconda3\Scripts\conda.exe" init powershell
# After restarting PowerShell, you can use:
conda activate resume-ner-training

# Option 4: Use Anaconda Prompt (GUI option - has conda pre-initialized)
# Open "Anaconda Prompt" from Start Menu, then:
conda activate resume-ner-training
```

**Note**: For running test scripts, you can use Option 2 directly without activation:

```powershell
& "C:\Users\HOANG PHI LONG DANG\Miniconda3\envs\resume-ner-training\python.exe" tests/e2e/test_e2e_workflow.py
```

**For Bash/Linux/Mac:**

```bash
conda activate resume-ner-training
pip install pytest pytest-cov pytest-mock
```

## Running Tests

### End-to-End Workflow Tests

These tests validate the complete training pipeline from config loading to final training.

#### End-to-End Workflow Test

Validates the complete training pipeline using tiny datasets (mimics the Colab notebook workflow):

```bash
# Run full workflow (HPO + final training)
python tests/e2e/test_e2e_workflow.py

# Skip final training (HPO only)
python tests/e2e/test_e2e_workflow.py --skip-training

# Custom output directory
python tests/e2e/test_e2e_workflow.py --output-dir outputs/my_test
```

**What it tests:**

- Config loading and validation
- Dataset verification
- MLflow setup
- HPO sweep execution
- Best configuration selection
- Final training (optional)
- Output validation

**Prerequisites:**

- Tiny dataset must exist at `dataset_tiny/seed0/` (create with `notebooks/00_make_tiny_dataset.ipynb`)
- Experiment config must use `data/resume_tiny.yaml`

#### HPO Pipeline Tests with Tiny Datasets

Validates HPO pipeline with various tiny dataset configurations:

```bash
# Run all test suites with default config
python tests/e2e/test_hpo_with_tiny_datasets.py

# Test specific seeds
python tests/e2e/test_hpo_with_tiny_datasets.py --seeds 0 1 2

# Test specific backbones
python tests/e2e/test_hpo_with_tiny_datasets.py --backbones distilbert

# Skip specific test suites
python tests/e2e/test_hpo_with_tiny_datasets.py --skip-deterministic
python tests/e2e/test_hpo_with_tiny_datasets.py --skip-random-seeds
python tests/e2e/test_hpo_with_tiny_datasets.py --skip-kfold
python tests/e2e/test_hpo_with_tiny_datasets.py --skip-edge-cases

# Combine multiple skip options
python tests/e2e/test_hpo_with_tiny_datasets.py --skip-kfold --skip-edge-cases

# Custom output directory
python tests/e2e/test_hpo_with_tiny_datasets.py --output-dir outputs/my_hpo_tests

# Specify custom log file
python tests/e2e/test_hpo_with_tiny_datasets.py --log-file outputs/my_hpo_tests/test.log
```

**What it tests:**

- HPO completion with tiny datasets
- K-fold cross-validation with small datasets
- Edge cases (k > n_samples, batch size issues)
- Random seed variants
- Multiple backbone support

**Prerequisites:**

- Tiny datasets must exist at `dataset_tiny/seed{N}/` for each seed
- Configuration loaded from `config/test/hpo_pipeline.yaml`

**Logging:**

- Log files are automatically created in the output directory with timestamp: `test_hpo_YYYYMMDD_HHMMSS.log`
- Use `--log-file` to specify a custom log file path
- All output (both console and log) includes timestamps and log levels

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test files
pytest tests/unit/api/test_inference_performance.py -v
pytest tests/unit/api/test_inference_fixes.py -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

### Integration Tests

#### HPO Pipeline Integration Tests (pytest)

```bash
# Run HPO pipeline integration tests
pytest tests/integration/test_hpo_pipeline.py -v -m integration

# Run with specific markers
pytest tests/integration/test_hpo_pipeline.py -v -m "integration and slow"
```

#### API Integration Tests

```bash
# Run API integration tests
pytest tests/integration/api/ -v

# Test inference performance with actual model
pytest tests/integration/api/test_inference_performance.py \
    --onnx-model outputs/final_training/distilroberta/distilroberta_model.onnx \
    --checkpoint outputs/final_training/distilroberta/checkpoint \
    -v

# Test ONNX inference speed
pytest tests/integration/api/test_onnx_inference.py \
    --onnx-model <path> --checkpoint <path> -v

# Test tokenization speed
pytest tests/integration/api/test_tokenization_speed.py --checkpoint <path> -v
```

**Note**: On Windows, use quotes for paths:

```bash
pytest tests/integration/api/test_inference_performance.py --onnx-model "outputs/final_training/distilroberta/distilroberta_model.onnx" --checkpoint "outputs/final_training/distilroberta/checkpoint" -v
```

## Test Structure

- **`tests/e2e/`**: End-to-end workflow tests (standalone scripts, not pytest-based)
  - **`test_e2e_workflow.py`**: Full training pipeline validation (config → HPO → training)
  - **`test_hpo_with_tiny_datasets.py`**: HPO pipeline validation with tiny datasets
  - **Note**: These are standalone Python scripts, not pytest tests. Run them directly with `python`.
- **`tests/unit/`**: Unit tests with mocked dependencies (pytest-based)
  - **`api/`**: API unit tests
  - **`training/`**: Training unit tests
- **`tests/integration/`**: Integration tests requiring actual models/files (pytest-based)
  - **`test_hpo_pipeline.py`**: pytest wrappers for HPO tests
  - **`api/`**: API integration tests
  - **`orchestrators/`**: Test orchestration logic
  - **`services/`**: Test service modules
- **`tests/fixtures/`**: Test data and fixtures

## Test Coverage

The test suite covers:

- ✅ End-to-end training pipeline (config → HPO → training)
- ✅ HPO pipeline with tiny datasets
- ✅ K-fold cross-validation
- ✅ Edge cases (small datasets, k > n_samples)
- ✅ Tokenization performance
- ✅ Entity extraction with offset mapping
- ✅ Batch processing performance
- ✅ Error handling

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:

1. **Activate the correct conda environment**:

   ```bash
   conda activate resume-ner-training
   ```

2. **Install missing dependencies**:

   ```bash
   pip install transformers onnxruntime numpy pytest optuna mlflow
   ```

3. **Verify installation**:

   ```bash
   pip list | grep transformers
   ```

### Test Failures

- **Dataset not found**: Run `notebooks/00_make_tiny_dataset.ipynb` to create tiny datasets
- **Config errors**: Verify `config/test/hpo_pipeline.yaml` exists and is valid
- **Model files missing**: Ensure model files exist for integration tests
- **MLflow errors**: Check that `mlruns/` directory is writable

### Common Issues

- **"Dataset directory not found"**: Create tiny datasets first or check dataset path in config
- **"HPO config not found"**: Verify `config/hpo/smoke.yaml` exists
- **"Training script not found"**: Ensure `src/training/train.py` exists
