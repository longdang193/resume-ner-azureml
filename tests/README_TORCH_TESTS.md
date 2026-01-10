# Running Tests That Require PyTorch

Tests that require PyTorch are marked with the `@pytest.mark.torch` marker. These tests should be run in the `resume-ner-training` conda environment.

## Activating the Environment

```bash
conda activate resume-ner-training
```

## Running Torch Tests

### Run all torch-requiring tests:
```bash
pytest -m torch
```

### Run specific torch test file:
```bash
pytest tests/unit/training/test_trainer.py
```

### Run tests excluding torch tests (for environments without torch):
```bash
pytest -m "not torch"
```

## Test Files That Require Torch

The following test files are marked with `pytestmark = pytest.mark.torch`:

- `tests/config/unit/test_data_config.py` - Imports `training.data` which requires torch
- `tests/unit/training/test_trainer.py` - Directly uses PyTorch
- `tests/unit/training/test_checkpoint_loader.py` - Uses PyTorch models
- `tests/workflows/test_full_workflow_e2e.py` - End-to-end tests with PyTorch
- `tests/workflows/test_notebook_01_e2e.py` - End-to-end tests with PyTorch
- `tests/workflows/test_notebook_02_e2e.py` - End-to-end tests with PyTorch
- `tests/final_training/unit/test_final_training_config_critical.py` - Uses PyTorch
- `tests/selection/integration/test_artifact_acquisition_workflow.py` - Uses PyTorch

## Automatic Skipping

If you run torch-requiring tests without the `resume-ner-training` environment activated, they will be automatically skipped with a message indicating the required environment.

## Running All Tests

To run all tests (including torch tests), ensure you're in the `resume-ner-training` environment:

```bash
conda activate resume-ner-training
pytest
```

To run only non-torch tests (useful for CI/CD environments without torch):

```bash
pytest -m "not torch"
```


