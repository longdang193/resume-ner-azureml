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

- **`tests/e2e/`**: End-to-end workflow tests (standalone scripts, not pytest-based)
  - **`test_e2e_workflow.py`**: Full training pipeline validation (config → HPO → training)
  - **`test_hpo_with_tiny_datasets.py`**: HPO pipeline validation with tiny datasets
  - **Note**: These are standalone Python scripts, not pytest tests. Run them directly with `python`.
- **`tests/unit/`**: Unit tests with mocked dependencies (pytest-based)
  - **`api/`**: API unit tests
  - **`training/`**: Training unit tests
    - **`test_trainer.py`**: Training loop and data loader tests
    - **`test_checkpoint_loader.py`**: Checkpoint path resolution and validation tests
    - **`test_data_combiner.py`**: Dataset combination strategy tests
- **`tests/integration/`**: Integration tests requiring actual models/files (pytest-based)
  - **`api/`**: API integration tests
- **`src/testing/`**: Test helper utilities (not test scripts)
  - **`orchestrators/`**: Test orchestration logic
  - **`services/`**: Test service modules (HPO execution, k-fold validation, edge case detection)
  - **`setup/`**: Environment setup and configuration loading
  - **`validators/`**: Dataset validation utilities
  - **`aggregators/`**: Result aggregation utilities
  - **`comparators/`**: Result comparison utilities
  - **`fixtures/`**: Test fixtures and configuration loaders

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

# Run training unit tests (including continued training)
pytest tests/unit/training/ -v

# Run specific continued training tests
pytest tests/unit/training/test_checkpoint_loader.py -v
pytest tests/unit/training/test_data_combiner.py -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

#### Training Unit Tests

**Checkpoint Loader Tests** (`test_checkpoint_loader.py`):
- Validates checkpoint directory structure and required files
- Tests checkpoint path resolution from environment variables, config files, and cache files
- Tests pattern resolution (`{backbone}`, `{run_id}`)
- Tests priority order (env var > config > cache)
- Edge cases: missing files, invalid paths, non-existent checkpoints

**Data Combiner Tests** (`test_data_combiner.py`):
- Tests all three data combination strategies (`new_only`, `combined`, `append`)
- Validates dataset merging and shuffling
- Tests validation split creation
- Edge cases: missing validation sets, empty datasets, invalid strategies
- Error handling for missing dataset paths

### Integration Tests

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

#### FastAPI Local Server Tests

These tests start a real FastAPI server process and test all endpoints with actual model files. This is different from `test_api.py` which uses mocked `TestClient`.

**Prerequisites:**

- Trained model files in `outputs/final_training/distilroberta/`:
  - `distilroberta_model.onnx`
  - `checkpoint/` directory
- Test data files in `tests/test_data/` (see `tests/test_data/README.md`)
- Required Python packages:

  ```bash
  pip install python-multipart httpx pyyaml
  ```

**Running Tests:**

```bash
# Run all local server tests
pytest tests/integration/api/test_api_local_server.py -v

# Run specific test class
pytest tests/integration/api/test_api_local_server.py::TestServerLifecycle -v

# Run with custom model paths
pytest tests/integration/api/test_api_local_server.py \
    --onnx-model outputs/final_training/distilroberta/distilroberta_model.onnx \
    --checkpoint outputs/final_training/distilroberta/checkpoint \
    -v
```

**Test Categories:**

- **Server Lifecycle**: Startup, shutdown, failure handling
- **Health & Info Endpoints**: `/health`, `/info`
- **Single Text Prediction** (`/predict`):
  - Valid inputs: Normal text, various entity types, special characters
  - Edge cases: Empty string, very long text, unicode characters, whitespace-only text
  - Error cases: Missing text field, invalid JSON, non-string text value
- **Batch Text Prediction** (`/predict/batch`):
  - Valid inputs: Small, medium, large batches
  - Edge cases: Empty batch, mixed valid/invalid texts, batch with empty text
  - Error cases: Batch exceeding MAX_BATCH_SIZE, missing texts field, non-list value
- **File Upload** (`/predict/file`):
  - Valid inputs: PDF files, PNG images, larger PDF files
  - Edge cases: Small PDF files
  - Error cases: Invalid file type, missing file in request
- **Batch File Upload** (`/predict/file/batch`):
  - Valid inputs: Small and medium file batches, mixed file types
  - Edge cases: Empty batch
  - Error cases: Batch exceeding MAX_BATCH_SIZE
- **Debug Endpoint**: `/predict/debug` with detailed token-level information
- **Error Handling**: Invalid inputs, missing fields, malformed requests
- **Performance**: Latency validation against thresholds (P50, P95, max)
- **Stability**: Consistency across repeated runs, performance degradation detection

**Configuration:**

Test configuration is in `config/test/api_local_server.yaml`:

- Server settings (host, port, timeouts)
- Performance thresholds (latency, throughput)
- Request timeouts

**Test Data:**

Test data fixtures are defined in `tests/test_data/fixtures.py` and documented in `tests/test_data/README.md`. If test files are missing, generate them:

```bash
cd tests/test_data
python generate_test_files.py
```

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
