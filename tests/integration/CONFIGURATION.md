# Configuration Loading and Application

This document describes how configuration is consistently loaded and applied across the HPO pipeline test suite.

## Configuration Source

All test configuration is centralized in `config/test/hpo_pipeline.yaml`:

```yaml
hpo_pipeline_tests:
  defaults:
    random_seed: 42
    minimal_k_folds: 2
    backbones: ["distilbert", "deberta"]  # List of backbones to test
    metric_decimal_places: 4
    separator_width: 60
    very_small_validation_threshold: 2
  
  # HPO overrides (applied after loading HPO config from configs.hpo_config)
  hpo_overrides:
    max_trials: null  # Override max_trials (null = use value from HPO config file)
  
  datasets:
    deterministic_path: "dataset_tiny"
    random_seeds: [0]
  
  output:
    base_dir: "outputs/hpo_tests"
    mlflow_dir: "mlruns"
  
  configs:
    hpo_config: "hpo/smoke.yaml"
    train_config: "train.yaml"
```

## Configuration Loading Flow

### 1. Config Loader (`tests/fixtures/config/test_config_loader.py`)

**Responsibility**: Load and cache configuration from YAML file.

- `get_test_config(root_dir)` - Returns cached config dictionary
- `load_hpo_test_config(config_path, root_dir)` - Loads config from file
- Constants are loaded on module import: `DEFAULT_RANDOM_SEED`, `DEFAULT_BACKBONE`, etc.

### 2. Environment Setup (`tests/integration/orchestrators/test_orchestrator.py`)

**Function**: `setup_test_environment(root_dir, ...)`

**What it does**:
1. Loads test config via `get_test_config(root_dir)`
2. Resolves paths from config:
   - HPO config path: `configs.hpo_config` → `config/hpo/smoke.yaml`
   - Train config path: `configs.train_config` → `config/train.yaml`
   - Output directory: `output.base_dir` → `outputs/hpo_tests`
   - MLflow directory: `output.mlflow_dir` → `mlruns`
   - Deterministic dataset: `datasets.deterministic_path` → `dataset_tiny`
3. Loads HPO and train config YAML files
4. Initializes MLflow tracking URI
5. Returns environment dictionary with all configs and paths

### 3. Notebook (`tests/test_hpo_with_tiny_datasets.ipynb`)

**Configuration loading**:
1. Cell 3: Imports `get_test_config` and loads `RANDOM_SEEDS_TO_TEST` from config
2. Cell 4: Calls `setup_test_environment()` which loads all configs
3. All test functions receive configs from the environment setup

**Configuration values used**:
- `RANDOM_SEEDS_TO_TEST` - From `datasets.random_seeds`
- `BACKBONES_TO_TEST` - From `defaults.backbones` (via `BACKBONES_LIST` constant)
- `DEFAULT_BACKBONE` - Derived from first item in `defaults.backbones` (via config loader constants)
- `DEFAULT_RANDOM_SEED` - From `defaults.random_seed` (via config loader constants)
- All paths and configs from `setup_test_environment()`

### 4. CLI (`tests/integration/cli/main.py`)

**Configuration loading**:
1. Loads test config via `get_test_config()` for CLI help text defaults
2. Calls `run_all_tests()` which internally calls `setup_test_environment()`
3. CLI arguments can override config values, but defaults come from config

**Configuration precedence**:
1. CLI arguments (highest priority)
2. Config file values (default)
3. Hardcoded fallbacks (lowest priority)

### 5. Pytest Fixtures (`tests/integration/conftest.py`)

**Configuration loading**:
- All fixtures use `get_test_config(root_dir)` to load config
- Each fixture resolves its specific path/value from config:
  - `hpo_config` fixture: Uses `configs.hpo_config`
  - `train_config` fixture: Uses `configs.train_config`
  - `deterministic_dataset` fixture: Uses `datasets.deterministic_path`
  - `test_output_dir` fixture: Uses `output.base_dir`
  - `mlflow_tracking_uri` fixture: Uses `output.mlflow_dir`

## Consistency Guarantees

### All Components Use Same Config Source

✅ **Notebook**: Loads via `get_test_config()` and `setup_test_environment()`
✅ **CLI**: Loads via `get_test_config()` and `setup_test_environment()`
✅ **Pytest**: Loads via `get_test_config()` in fixtures
✅ **Services**: Use constants from `test_config_loader` module

### Configuration Application

All test functions receive configuration through:
1. **Environment setup** (`setup_test_environment()`) - Provides HPO config, train config, paths
2. **Config constants** - Provides default values (backbone, random_seed, etc.)
3. **Function parameters** - Can override defaults if needed

### Path Resolution

All paths are resolved consistently:
- Relative paths in config are resolved relative to `root_dir`
- Config paths are relative to `config_dir` (which is `root_dir / "config"`)
- Dataset paths are relative to `root_dir`
- Output paths are relative to `root_dir`

## Example: Changing Configuration

### Example 1: Change Random Seeds to Test

1. Edit `config/test/hpo_pipeline.yaml`:
   ```yaml
   datasets:
     random_seeds: [0, 1, 2]  # Changed from [0]
   ```

2. All components automatically use the new value:
   - Notebook: `RANDOM_SEEDS_TO_TEST` loads from config
   - CLI: Default `--seeds` argument uses config value
   - `run_all_tests()`: Uses config value if default `[0]` is passed

### Example 2: Override Number of HPO Trials

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

## Verification

To verify configuration is loaded correctly:

1. **Notebook**: Check Cell 4 output - shows all loaded config values
2. **CLI**: Use `--verbose` flag to see configuration
3. **Pytest**: Fixtures automatically load from config

