# HPO Pipeline Integration Tests

This directory contains integration tests for the HPO (Hyperparameter Optimization) pipeline, refactored to follow Single Responsibility Principle (SRP).

## Structure

```
tests/integration/
├── services/              # Business logic services
│   ├── hpo_executor.py           # HPO sweep execution
│   ├── kfold_validator.py        # K-fold CV validation
│   └── edge_case_detector.py     # Edge case detection
├── orchestrators/         # Test orchestration
│   └── test_orchestrator.py      # Coordinates test execution
├── aggregators/          # Result aggregation
│   └── result_aggregator.py      # Aggregates test results
├── cli/                  # CLI interface
│   └── main.py                   # Command-line entry point
├── conftest.py           # Pytest fixtures
└── test_hpo_pipeline.py # Pytest test wrappers
```

## Usage

### CLI

Run tests from command line:

```bash
python -m tests.integration.cli.main --seeds 0 1 2
```

Options:
- `--seeds`: List of seed numbers to test (default: from config)
- `--root-dir`: Project root directory
- `--output-dir`: Test output directory
- `--hpo-config`: Path to HPO config file
- `--train-config`: Path to train config file
- `--skip-deterministic`: Skip deterministic dataset test
- `--skip-random`: Skip random seed variant tests
- `--skip-kfold`: Skip k-fold validation tests
- `--skip-edge-cases`: Skip edge case tests
- `--verbose`: Verbose output

### Pytest

Run tests with pytest:

```bash
pytest tests/integration/test_hpo_pipeline.py -v -m integration
```

### Notebook

Use the notebook wrapper:

```python
from tests.integration.orchestrators.test_orchestrator import run_all_tests

results = run_all_tests(
    root_dir=Path("."),
    random_seeds=[0, 1, 2],
)
```

## Configuration

Test configuration is loaded from `config/test/hpo_pipeline.yaml`. See that file for default settings.

## Module Responsibilities

- **Services**: Pure business logic (HPO execution, validation, detection)
- **Orchestrators**: Coordinate test execution and setup
- **Aggregators**: Transform and aggregate results
- **CLI**: Parse arguments and invoke orchestrator
- **Pytest wrappers**: Thin wrappers for pytest integration

