# Testing Guide

## Prerequisites

Before running tests, ensure you have the required dependencies installed:

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Or install individually
pip install transformers onnxruntime numpy pytest pytest-cov pytest-mock optuna mlflow fastapi uvicorn httpx python-multipart pyyaml
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

## Test Structure

Tests are organized by **feature/workflow** rather than by test type, following the production code structure:

- **`tests/fixtures/`**: Shared test fixtures and helpers (NEW)
  - **`datasets.py`**: Dataset fixtures (`tiny_dataset`, `create_dataset_structure`)
  - **`mlflow.py`**: MLflow mocking fixtures (`mock_mlflow_tracking`, `mock_mlflow_client`, `mock_mlflow_run`)
  - **`configs.py`**: Config fixtures (HPO, selection, acquisition, conversion)
  - **`validators.py`**: Validation helpers (`validate_path_structure`, `validate_run_name`, `validate_tags`)

- **`tests/workflows/`**: End-to-end workflow tests (pytest-based)
  - **`test_notebook_01_e2e.py`**: Tests notebook 01 (HPO + Benchmarking workflow)
  - **`test_notebook_02_e2e.py`**: Tests notebook 02 (Best config selection → final training → conversion)
  - **`test_full_workflow_e2e.py`**: Tests complete workflow (01 → 02 end-to-end)

- **`tests/hpo/`**: Hyperparameter optimization tests
  - **`unit/`**: Fast, isolated HPO tests (search space, trial selection, checkpoint resume)
  - **`integration/`**: HPO with real components (sweep execution, refit training, early termination, path structure)
  - **`e2e/`**: Full HPO workflow (`test_hpo_workflow.py`)

- **`tests/benchmarking/`**: Benchmarking tests
  - **`unit/`**: Config option tests
  - **`integration/`**: Benchmark workflow, orchestrator, utils, edge cases

- **`tests/selection/`**: Best model selection tests
  - **`unit/`**: Config options, cache tests
  - **`integration/`**: MLflow selection, artifact acquisition, workflow, edge cases

- **`tests/final_training/`**: Final training tests
  - **`unit/`**: Config tests
  - **`integration/`**: Component tests, logging intervals

- **`tests/conversion/`**: Model conversion tests
  - **`unit/`**: Config and options tests
  - **`integration/`**: Config integration tests

- **`tests/tracking/`**: MLflow tracking tests
  - **`unit/`**: Naming policy, tags registry, MLflow config, Azure ML artifact upload
  - **`integration/`**: Tracking config enabled, naming integration, Azure ML artifact upload integration
  - **`scripts/`**: Manual test scripts and verification tools

- **`tests/config/`**: Configuration loading tests
  - **`unit/`**: Config loader, experiment/data/model configs, paths/naming/mlflow YAML tests, fingerprints
  - **`integration/`**: Config integration tests

- **`tests/training/`**: Training unit tests (kept existing structure)
    - **`test_trainer.py`**: Training loop and data loader tests
    - **`test_checkpoint_loader.py`**: Checkpoint path resolution and validation tests
    - **`test_data_combiner.py`**: Dataset combination strategy tests
  - **`test_cv_utils.py`**: Cross-validation utilities

- **`tests/api/`**: API tests (kept existing structure)
  - **`unit/`**: API unit tests
  - **`integration/`**: API integration tests

- **`tests/shared/`**: Shared utility tests
  - **`unit/`**: MLflow setup, drive backup tests

- **`tests/docs/`**: Coverage analysis and limitations documentation (NEW)
  - Coverage analysis documents for all YAML configs
  - Implementation status and limitations

- **`tests/test_data/`**: Test data fixtures (kept existing)

## Running Tests

### Workflow E2E Tests

These tests validate complete notebook workflows end-to-end:

#### Notebook 01 E2E Test

Tests the complete workflow from `01_orchestrate_training_colab.ipynb`:

```bash
# Core workflow with mocked training (default, CI-compatible)
pytest tests/workflows/test_notebook_01_e2e.py -v

# Full workflow
E2E_TEST_SCOPE=full pytest tests/workflows/test_notebook_01_e2e.py -v

# Real training execution (slower)
E2E_USE_REAL_TRAINING=true pytest tests/workflows/test_notebook_01_e2e.py -v
```

**What it tests:**

- Environment detection
- Config loading
- Dataset verification
- MLflow setup
- HPO sweep execution (mocked)
- Benchmarking execution (mocked)
- Path, naming, and tag validation

#### Notebook 02 E2E Test

Tests the complete workflow from `02_best_config_selection.ipynb`:

```bash
# Default: CI-compatible mode (mocked training)
pytest tests/workflows/test_notebook_02_e2e.py -v
```

**What it tests:**

- Best model selection
- Artifact acquisition
- Final training (mocked)
- Model conversion (mocked)
- Path, naming, and tag validation

#### Full Workflow E2E Test

Tests the complete workflow from notebook 01 through notebook 02:

```bash
# Default: CI-compatible mode (mocked training)
pytest tests/workflows/test_full_workflow_e2e.py -v

# Real training execution (slower)
E2E_USE_REAL_TRAINING=true pytest tests/workflows/test_full_workflow_e2e.py -v
```

**What it tests:**

- Complete pipeline: HPO → Benchmarking → Selection → Final Training → Conversion
- Path, naming, and tag validation throughout
- Lineage tracking validation

### Feature-Specific Tests

Tests are organized by feature. Run tests for a specific feature:

```bash
# HPO tests
pytest tests/hpo/ -v

# Benchmarking tests
pytest tests/benchmarking/ -v

# Selection tests
pytest tests/selection/ -v

# Final training tests
pytest tests/final_training/ -v

# Conversion tests
pytest tests/conversion/ -v

# Tracking tests
pytest tests/tracking/ -v

# Azure ML artifact upload tests (unit tests)
pytest tests/tracking/unit/test_azureml_artifact_upload.py -v

# Azure ML artifact upload integration tests (requires Azure ML)
pytest tests/tracking/integration/test_azureml_artifact_upload_integration.py -v --run-azureml-tests

# Verify artifact upload fixes are in place
python tests/tracking/scripts/verify_artifact_upload_fix.py

# Manual artifact upload test (requires active MLflow run in Azure ML)
python tests/tracking/scripts/test_artifact_upload_manual.py

# Config tests
pytest tests/config/ -v
```

### Training Tests

```bash
# Run all training tests
pytest tests/training/ -v

# Run specific tests
pytest tests/training/test_checkpoint_loader.py -v
pytest tests/training/test_data_combiner.py -v
pytest tests/training/test_trainer.py -v
```

### API Tests

```bash
# Run all API tests
pytest tests/api/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Shared Fixtures

All tests can use shared fixtures from `tests/fixtures/`:

```python
from fixtures import (
    tiny_dataset,
    mock_mlflow_tracking,
    validate_path_structure,
    validate_run_name,
    validate_tags,
)
```

See `tests/fixtures/` for available fixtures and helpers.

### Azure ML Artifact Upload Tests

Tests for Azure ML artifact upload fixes, including monkey-patch for compatibility issues and refit run completion:

```bash
# Run all unit tests (9 tests, all passing)
pytest tests/tracking/unit/test_azureml_artifact_upload.py -v

# Run specific test class
pytest tests/tracking/unit/test_azureml_artifact_upload.py::TestAzureMLArtifactBuilderPatch -v

# Verify fixes are in place (quick check)
python tests/tracking/scripts/verify_artifact_upload_fix.py

# Manual test in real Azure ML environment
python tests/tracking/scripts/test_artifact_upload_manual.py
```

**What these tests verify:**
- ✅ Monkey-patch for `azureml_artifacts_builder` to handle `tracking_uri` TypeError
- ✅ Artifact upload to refit runs (child runs) instead of parent runs
- ✅ Refit runs are marked as FINISHED after successful artifact upload
- ✅ Refit runs are marked as FAILED after failed artifact upload
- ✅ Azure ML compatibility between MLflow 3.5.0 and azureml-mlflow 1.61.0.post1

**Documentation:**
- See `tests/tracking/README_artifact_upload_tests.md` for detailed documentation
- See `tests/tracking/TEST_SUMMARY.md` for quick reference

## Configuration

### Test Configuration

All test configuration is centralized in `config/test/hpo_pipeline.yaml`:

```yaml
hpo_pipeline_tests:
  defaults:
    random_seed: 42
    minimal_k_folds: 2
    backbones: ["distilbert", "distilroberta"]  # List of backbones to test
    metric_decimal_places: 4
    separator_width: 60
    very_small_validation_threshold: 2
  
  # HPO overrides (applied after loading HPO config from configs.hpo_config)
  hpo_overrides:
    max_trials: 2  # Override max_trials (null = use value from HPO config file)
  
  datasets:
    deterministic_path: "dataset_tiny"
    random_seeds: [0, 1, 2, 3, 4]
  
  output:
    base_dir: "outputs/hpo_tests"
    mlflow_dir: "mlruns"
  
  configs:
    hpo_config: "hpo/smoke.yaml"
    train_config: "train.yaml"
```

### Configuration Loading

Configuration is loaded consistently across all test components:

1. **Config Loader** (`src/testing/fixtures/config/test_config_loader.py`):
   - `get_test_config(root_dir)` - Returns cached config dictionary
   - `load_hpo_test_config(config_path, root_dir)` - Loads config from file
   - Constants are loaded on module import: `DEFAULT_RANDOM_SEED`, `DEFAULT_BACKBONE`, etc.

2. **Environment Setup** (`src/testing/setup/environment_setup.py`):
   - `setup_test_environment(root_dir, ...)` - Loads all configs and sets up paths
   - Resolves paths from config (HPO config, train config, output dir, MLflow dir, dataset paths)
   - Initializes MLflow tracking URI
   - Returns environment dictionary with all configs and paths

3. **Configuration Precedence**:
   - CLI arguments (highest priority)
   - Config file values (default)
   - Hardcoded fallbacks (lowest priority)

### Changing Configuration

#### Example 1: Change Random Seeds to Test

1. Edit `config/test/hpo_pipeline.yaml`:

   ```yaml
   datasets:
     random_seeds: [0, 1, 2]  # Changed from [0]
   ```

2. The test script automatically uses the new value:
   - The `--seeds` argument in `test_hpo_with_tiny_datasets.py` uses config value as default

#### Example 2: Override Number of HPO Trials

To change the number of trials without editing the HPO config file:

1. Edit `config/test/hpo_pipeline.yaml`:

   ```yaml
   hpo_overrides:
     max_trials: 5  # Override max_trials (was 2 in hpo/smoke.yaml)
   ```

2. The override is applied when the HPO config is loaded:
   - The HPO config file (`hpo/smoke.yaml`) is loaded first
   - Then `hpo_overrides.max_trials` (if not null) replaces the value
   - Set to `null` to use the value from the HPO config file

## Test Coverage

The test suite covers:

- ✅ End-to-end training pipeline (config → HPO → training)
- ✅ HPO pipeline with tiny datasets
- ✅ K-fold cross-validation
- ✅ Continued training (checkpoint loading, data combination)
- ✅ Edge cases (small datasets, k > n_samples)
- ✅ Tokenization performance
- ✅ Entity extraction with offset mapping
- ✅ Batch processing performance
- ✅ Error handling
- ✅ Azure ML artifact upload (monkey-patch, child run uploads, refit run completion)

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:

1. **Activate the correct conda environment**:

   ```bash
   conda activate resume-ner-training
   ```

2. **Install missing dependencies**:

   ```bash
   pip install transformers onnxruntime numpy pytest optuna mlflow fastapi uvicorn httpx
   ```

3. **Verify installation**:

   ```bash
   pip list | grep transformers
   ```

4. **Ensure `src/` is in Python path**: Test helper modules are in `src/testing/`, so make sure `src/` is in your Python path when running tests.

### Test Failures

- **Dataset not found**: Run `notebooks/00_make_tiny_dataset.ipynb` to create tiny datasets
- **Config errors**: Verify `config/test/hpo_pipeline.yaml` exists and is valid
- **Model files missing**: Ensure model files exist for integration tests
- **MLflow errors**: Check that `mlruns/` directory is writable

### Common Issues

- **"Dataset directory not found"**: Create tiny datasets first or check dataset path in config
- **"HPO config not found"**: Verify `config/hpo/smoke.yaml` exists
- **"Training script not found"**: Ensure `src/training/train.py` exists
- **"ModuleNotFoundError: No module named 'testing'"**: Ensure `src/` is in Python path (usually handled automatically by test scripts)
- **"Azure ML builder not registered"**: This is expected if not using Azure ML workspace. Azure ML-specific tests will be skipped.
- **"No active MLflow run"**: For manual artifact upload tests, ensure you have an active MLflow run in Azure ML environment.
