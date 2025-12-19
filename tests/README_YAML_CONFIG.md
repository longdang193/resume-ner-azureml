# YAML Test Configuration Usage

This document explains how the YAML test configuration files are used in the test suite.

## Configuration Files

All test configuration files are located in `config/test/`:

- **`execution.yaml`**: Coverage thresholds, markers, performance settings
- **`fixtures.yaml`**: Test data fixtures (sample data, label mappings, entity types)
- **`mocks.yaml`**: Mock component settings (tokenizers, models, platform adapters)
- **`environments.yaml`**: Environment-specific settings (CI, local, default)

## Automatic Integration

The YAML configurations are automatically loaded and applied when pytest runs via `tests/pytest_configure.py`:

1. **Coverage Thresholds**: Automatically enforced from `execution.yaml`
2. **Markers**: Registered from `execution.yaml`
3. **Environment Settings**: Applied based on `TEST_ENV` environment variable
4. **Test Fixtures**: Load data from `fixtures.yaml` via `conftest.py`

## Usage in Tests

### Accessing Configuration Data

```python
from config_loader import (
    get_sample_resume_data,
    get_label_mapping,
    get_entity_types,
    get_mock_config_template,
    get_tokenizer_mock_settings,
    get_coverage_threshold,
    get_execution_settings,
)

# Get test data
resume_data = get_sample_resume_data()
label_map = get_label_mapping("standard")
entity_types = get_entity_types("standard")

# Get mock settings
tokenizer_settings = get_tokenizer_mock_settings("fast")

# Get coverage threshold
threshold = get_coverage_threshold("training")  # Returns 85
overall = get_coverage_threshold()  # Returns 80
```

### Using Fixtures (Automatic)

The `conftest.py` fixtures automatically use YAML configs:

```python
def test_example(sample_resume_data, label2id, mock_fast_tokenizer):
    # These fixtures automatically load from YAML configs
    assert len(sample_resume_data) > 0
    assert "PERSON" in label2id
```

## Running Tests with Coverage

Coverage thresholds are automatically enforced:

```bash
# Run with coverage (thresholds from execution.yaml)
pytest --cov=src --cov-report=term-missing --cov-report=html

# The overall threshold (80%) is automatically applied
# Module-specific thresholds are tracked but not enforced automatically
```

## Environment-Specific Settings

Set `TEST_ENV` environment variable to use environment-specific settings:

```bash
# Use CI settings (less verbose, fail under 80%)
TEST_ENV=ci pytest --cov=src

# Use local settings (verbose, show all output)
TEST_ENV=local pytest

# Use default settings
pytest
```

## Verifying Configuration is Loaded

When pytest runs, you should see in the header:

```
YAML Test Configuration: Loaded from config/test/
  Overall Coverage Threshold: 80%
  Module Thresholds:
    - training: 85%
    - orchestration: 75%
    ...
```

If you don't see this, check that:
1. `tests/pytest_configure.py` exists
2. `config/test/*.yaml` files exist
3. `tests/config_loader.py` is importable

## Updating Configuration

To update thresholds or settings:

1. Edit the appropriate YAML file in `config/test/`
2. Changes take effect immediately (no restart needed)
3. Use `clear_cache()` if you need to force reload:

```python
from config_loader import clear_cache
clear_cache()
```

