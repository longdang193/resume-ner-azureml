# Testing Guide

This document describes the testing strategy and how to run tests for the Resume NER training pipeline.

## Overview

The test suite is organized into three main categories:

1. **Unit Tests** (`tests/unit/`) - Fast, isolated tests for individual components
2. **Integration Tests** (`tests/integration/`) - Tests for component interactions and workflows
3. **Troubleshooting Tests** - Tests that verify fixes for common issues documented in TROUBLESHOOTING.md

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures (loads from YAML configs)
├── config_loader.py               # YAML configuration loader for tests
├── unit/
│   ├── training/                 # Training component tests
│   │   ├── test_metrics.py
│   │   ├── test_cv_utils.py
│   │   ├── test_data.py
│   │   ├── test_config.py
│   │   ├── test_trainer_oom_prevention.py
│   │   └── test_tokenizer_offset_mapping.py
│   ├── orchestration/            # Orchestration tests
│   │   ├── test_config_loader.py
│   │   ├── test_data_asset_references.py
│   │   └── jobs/
│   │       ├── test_local_selection.py
│   │       ├── test_local_sweeps.py
│   │       └── test_conversion_checkpoint_resolution.py
│   ├── platform_adapters/        # Platform adapter tests
│   │   ├── test_logging_adapter.py
│   │   ├── test_outputs.py
│   │   ├── test_adapters.py
│   │   └── test_mlflow_context.py
│   └── shared/                   # Shared utility tests
│       ├── test_yaml_utils.py
│       └── test_json_cache.py
└── integration/
    └── test_troubleshooting_fixes.py

config/
└── test/                          # YAML test configuration files
    ├── fixtures.yaml              # Test data fixtures
    ├── execution.yaml             # Execution settings and thresholds
    ├── mocks.yaml                 # Mock configurations
    └── environments.yaml          # Environment-specific settings
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/training/test_metrics.py

# Specific test class or function
pytest tests/unit/training/test_metrics.py::TestComputeF1ForLabel
```

### Run with Coverage

```bash
# Terminal output
pytest --cov=src --cov-report=term-missing

# HTML report
pytest --cov=src --cov-report=html
# Then open htmlcov/index.html in your browser
```

### Run in Parallel

```bash
pytest -n auto  # Requires pytest-xdist
```

## Test Coverage

### Core Components (Phase 1)

- **Metrics Computation** (`test_metrics.py`)
  - F1 score calculation for individual labels
  - Macro-averaged F1 across all labels
  - Complete metrics dictionary generation

- **Cross-Validation** (`test_cv_utils.py`)
  - K-fold splitting
  - Fold data extraction
  - Split serialization/deserialization

- **Data Loading** (`test_data.py`)
  - Dataset loading from JSON
  - Text normalization
  - Annotation encoding
  - Dataset class with fast/slow tokenizer support

- **Configuration Loading** (`test_config_loader.py`)
  - Experiment config loading
  - Config hashing
  - Immutability validation

### Important Components (Phase 2)

- **Training Configuration** (`test_config.py`)
  - Config building and merging
  - Argument overrides

- **Platform Adapters** (`test_platform_adapters/`)
  - Logging adapters (Azure ML vs Local)
  - Output path resolution
  - Platform detection
  - MLflow context management

- **HPO Utilities** (`test_local_selection.py`, `test_local_sweeps.py`)
  - Best configuration selection
  - Search space translation

- **Shared Utilities** (`test_shared/`)
  - YAML loading
  - JSON caching

### Troubleshooting Prevention Tests

These tests verify that fixes for common issues (from TROUBLESHOOTING.md) are in place:

1. **OOM Prevention** (`test_trainer_oom_prevention.py`)
   - Verifies DeBERTa batch size is automatically capped
   - Tests batch size limits for different model types

2. **Data Asset References** (`test_data_asset_references.py`)
   - Ensures `azureml:name:version` format is used (not manual datastore paths)
   - Prevents `ScriptExecution.StreamAccess.NotFound` errors

3. **MLflow Context Management** (`test_mlflow_context.py`)
   - Verifies Azure ML does NOT create nested runs
   - Ensures local execution creates MLflow runs properly
   - Prevents metrics from being logged to wrong runs

4. **Tokenizer Offset Mapping** (`test_tokenizer_offset_mapping.py`)
   - Tests fast tokenizer offset mapping support
   - Verifies slow tokenizer fallback to 'O' labels
   - Prevents `NotImplementedError` with slow tokenizers

5. **Checkpoint Resolution** (`test_conversion_checkpoint_resolution.py`)
   - Tests checkpoint path resolution
   - Verifies asset reference format
   - Ensures nested checkpoint directories are found

6. **Integration Tests** (`test_troubleshooting_fixes.py`)
   - End-to-end verification of fixes
   - Config directory resolution
   - Data asset reference format in job creation

## Writing New Tests

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<function_name>_<scenario>`

### Example Test Structure

```python
"""Tests for module_name."""

import pytest
from module import function


class TestFunctionName:
    """Tests for function_name function."""

    def test_basic_usage(self):
        """Test basic functionality."""
        result = function("input")
        assert result == "expected"

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            function(None)
```

### Using Fixtures

Common fixtures are available in `conftest.py` (loaded from YAML configurations):

- `temp_dir` - Temporary directory for test files
- `sample_resume_data` - Sample resume JSON data (from `config/test/fixtures.yaml`)
- `mock_configs` - Mock configuration files (from `config/test/fixtures.yaml`)
- `mock_fast_tokenizer` - Mock fast tokenizer (settings from `config/test/mocks.yaml`)
- `mock_slow_tokenizer` - Mock slow tokenizer (settings from `config/test/mocks.yaml`)
- `label2id` - Label to ID mapping (from `config/test/fixtures.yaml`)

Example:

```python
def test_load_dataset(sample_resume_data, temp_dir):
    """Test dataset loading."""
    # Use fixtures
    train_file = temp_dir / "train.json"
    # ... test code
```

### YAML Test Configuration

Test data and settings are managed through YAML configuration files in `config/test/`:

#### Configuration Files

1. **`fixtures.yaml`** - Test data fixtures
   - Sample resume data entries
   - Label mappings
   - Entity type definitions
   - Mock configuration templates

2. **`execution.yaml`** - Test execution settings
   - Coverage thresholds (overall and per-module)
   - Test markers and categories
   - Performance thresholds (timeouts, slow test thresholds)
   - Parallel execution settings

3. **`mocks.yaml`** - Mock configurations
   - Tokenizer mock settings (max_length, truncation behavior)
   - Model mock configurations
   - Platform adapter mock settings

4. **`environments.yaml`** - Environment-specific settings
   - CI/CD test settings
   - Local development settings
   - Platform-specific configurations

#### Using the Configuration Loader

The `tests/config_loader.py` module provides functions to access test configurations:

```python
from config_loader import (
    get_sample_resume_data,
    get_label_mapping,
    get_coverage_threshold,
    get_mock_config_template,
    get_tokenizer_mock_settings,
    get_environment_settings,
)

# Get sample resume data
resume_data = get_sample_resume_data()

# Get label mapping
label_map = get_label_mapping("standard")

# Get coverage threshold for a module
threshold = get_coverage_threshold("training")  # Returns 85

# Get mock configuration template
experiment_config = get_mock_config_template("experiment")

# Get tokenizer mock settings
tokenizer_settings = get_tokenizer_mock_settings("fast")

# Get environment-specific settings
env_settings = get_environment_settings("ci")  # or "local", or None for default
```

#### Environment-Specific Configuration

Set the `TEST_ENV` environment variable to use environment-specific settings:

```bash
# Use CI settings
TEST_ENV=ci pytest

# Use local settings (default)
TEST_ENV=local pytest
```

The environment settings are merged with defaults, so you only need to override what differs.

#### Modifying Test Data

To modify test data, edit the YAML files in `config/test/`:

- **Add new resume entries**: Edit `config/test/fixtures.yaml` → `sample_data.resume_entries`
- **Change label mappings**: Edit `config/test/fixtures.yaml` → `label_mappings.standard`
- **Update coverage thresholds**: Edit `config/test/execution.yaml` → `coverage.module_thresholds`
- **Adjust mock settings**: Edit `config/test/mocks.yaml`

The configuration loader caches loaded YAML files for performance. To reload configurations during testing, call `clear_cache()`:

```python
from config_loader import clear_cache

clear_cache()  # Force reload on next access
```

## Continuous Integration

Tests should be run automatically in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    conda activate resume-ner-training
    pytest --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Coverage Goals

- **Target**: 70%+ code coverage for core components
- **Critical Components**: 80%+ coverage
  - Training metrics
  - Data loading
  - Configuration management
  - Platform adapters

## Troubleshooting Test Failures

### Common Issues

1. **Import Errors**
   - Ensure test dependencies are installed: `conda env update -f config/environment/conda.yaml`
   - Check that `src/` is in Python path

2. **Fixture Not Found**
   - Verify fixture is defined in `conftest.py`
   - Check fixture name matches exactly

3. **Mock Not Working**
   - Ensure you're patching the correct import path
   - Use `unittest.mock.patch` for standard library
   - Use `pytest-mock` for pytest integration

4. **Temporary Files Not Cleaned Up**
   - Use `temp_dir` fixture which automatically cleans up
   - Don't create files in project root

## Related Documentation

- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
- [Clean Code Principles](rules/CLEAN_CODE.md) - Code quality guidelines
- [Testing Plan](../.cursor/plans/testing-plan-for-ml-pipeline-e4dd6e57.plan.md) - Original testing plan
