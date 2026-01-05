<!-- db2320cb-ad35-4d05-988f-888f50ead5fd 22874291-569a-4d92-9b77-c1097fdf94ad -->
# Refactor Testing Scripts and Functions for Single Responsibility Principle

## Current State Analysis

The main SRP violations are in `tests/integration/orchestrators/test_orchestrator.py`, which currently handles:

1. **Environment setup** (config loading, path resolution, MLflow initialization)
2. **Test orchestration** (calling services)
3. **Presentation** (calling print functions)
4. **Business logic** (dataset existence checks, result comparison)

Other modules already follow SRP well:

- `result_aggregator.py` - Only aggregates results
- `hpo_executor.py` - Only executes HPO sweeps
- `kfold_validator.py` - Only validates k-fold splits
- `edge_case_detector.py` - Only detects edge cases
- `result_formatters.py` - Only formats/prints results
- `test_config_loader.py` - Only loads configs

## Refactoring Strategy

### 1. Extract Environment Setup Module

**File**: `tests/integration/setup/environment_setup.py` (new)

**Responsibilities**:

- Load and resolve configuration files
- Setup directory paths
- Initialize MLflow tracking
- Return environment dictionary

**Functions to extract**:

- `setup_test_environment()` from `test_orchestrator.py`
- Split into smaller functions:
  - `load_configs()` - Load HPO and train configs
  - `resolve_paths()` - Resolve all directory paths
  - `initialize_mlflow()` - Setup MLflow tracking URI

### 2. Extract Dataset Validation Module

**File**: `tests/integration/validators/dataset_validator.py` (new)

**Responsibilities**:

- Validate dataset existence
- Check dataset structure
- Return validation results

**Functions to extract**:

- Dataset existence checks from `test_orchestrator.py` (currently using `check_dataset_exists` from presenters)
- Move dataset validation logic out of test functions

### 3. Extract Result Comparison Module

**File**: `tests/integration/comparators/result_comparator.py` (new)

**Responsibilities**:

- Compare test results (deterministic vs random seed variants)
- Calculate differences and variability
- Return comparison data structure

**Functions to extract**:

- Comparison logic from `test_orchestrator.py` (currently in `test_random_seed_variants` and `run_all_tests`)
- Move `print_comparison` call logic to orchestrator, but extract comparison computation

### 4. Refactor Test Orchestrator

**File**: `tests/integration/orchestrators/test_orchestrator.py` (refactor)

**New Responsibilities** (SRP-compliant):

- **Only orchestrate** test execution flow
- Call services in correct order
- Pass results to aggregators and presenters
- **No** environment setup, validation, or presentation logic

**Changes**:

- Remove `setup_test_environment()` - import from `environment_setup.py`
- Remove dataset validation calls - use `dataset_validator.py`
- Remove comparison logic - use `result_comparator.py`
- Remove all `print_*` function calls - return results only, let callers handle presentation
- Keep only orchestration logic: calling services and passing data between them

### 5. Update Presentation Layer

**File**: `tests/fixtures/presenters/result_formatters.py` (update)

**Changes**:

- Move `check_dataset_exists()` to `dataset_validator.py` (it's validation, not presentation)
- Keep only formatting/printing functions

### 6. Update CLI and Notebooks

**Files**:

- `tests/integration/cli/main.py`
- `tests/test_hpo_with_tiny_datasets.ipynb`

**Changes**:

- Import new modules as needed
- Handle presentation at the CLI/notebook level (call presenters after orchestration)
- Update imports to use new module structure

## Implementation Steps

1. **Create environment setup module** (`tests/integration/setup/environment_setup.py`)

   - Extract and split `setup_test_environment()`
   - Add unit tests for path resolution and config loading

2. **Create dataset validator module** (`tests/integration/validators/dataset_validator.py`)

   - Move `check_dataset_exists()` from presenters
   - Add dataset structure validation

3. **Create result comparator module** (`tests/integration/comparators/result_comparator.py`)

   - Extract comparison logic from orchestrator
   - Return structured comparison data

4. **Refactor test orchestrator** (`tests/integration/orchestrators/test_orchestrator.py`)

   - Remove environment setup, validation, and presentation
   - Keep only orchestration logic
   - Update all test functions to be pure orchestration

5. **Update presenters** (`tests/fixtures/presenters/result_formatters.py`)

   - Remove `check_dataset_exists()` (moved to validators)
   - Keep only formatting functions

6. **Update CLI** (`tests/integration/cli/main.py`)

   - Import new modules
   - Add presentation calls after orchestration

7. **Update notebook** (`tests/test_hpo_with_tiny_datasets.ipynb`)

   - Update imports
   - Add presentation calls after test execution

8. **Update pytest test file** (`tests/integration/test_hpo_pipeline.py`)

   - Ensure imports work with new structure
   - Verify fixtures still work

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Test Entry Points                         │
│  (CLI, Notebook, pytest)                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Test Orchestrator                                │
│  (test_orchestrator.py)                                       │
│  - Orchestrates test execution flow                          │
│  - Calls services in order                                   │
│  - Passes results to aggregators                             │
└──────┬──────────────┬──────────────┬────────────────────────┘
       │              │              │
       ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Environment  │ │  Dataset     │ │   Result     │
│   Setup      │ │  Validator   │ │  Comparator  │
└──────────────┘ └──────────────┘ └──────────────┘
       │              │              │
       └──────────────┴──────────────┘
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ HPO Executor │ │ KFold        │ │ Edge Case    │
│              │ │ Validator    │ │ Detector     │
└──────────────┘ └──────────────┘ └──────────────┘
       │              │              │
       └──────────────┴──────────────┘
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Result     │ │  Result      │ │   Config     │
│  Aggregator  │ │  Formatters  │ │   Loader     │
└──────────────┘ └──────────────┘ └──────────────┘
```

## Benefits

1. **Clear separation of concerns**: Each module has a single, well-defined responsibility
2. **Improved testability**: Each component can be tested independently
3. **Better maintainability**: Changes to one concern don't affect others
4. **Easier extension**: New test types or validators can be added without modifying orchestrator
5. **Reusability**: Setup, validation, and comparison logic can be reused across different test entry points

## Files to Create

- `tests/integration/setup/__init__.py`
- `tests/integration/setup/environment_setup.py`
- `tests/integration/validators/__init__.py`
- `tests/integration/validators/dataset_validator.py`
- `tests/integration/comparators/__init__.py`
- `tests/integration/comparators/result_comparator.py`

## Files to Modify

- `tests/integration/orchestrators/test_orchestrator.py` (major refactor)
- `tests/fixtures/presenters/result_formatters.py` (remove `check_dataset_exists`)
- `tests/integration/cli/main.py` (update imports and add presentation)
- `tests/test_hpo_with_tiny_datasets.ipynb` (update imports and add presentation)
- `tests/integration/test_hpo_pipeline.py` (verify imports work)

## Testing Strategy

- Unit tests for each new module (environment setup, dataset validator, result comparator)
- Integration tests to verify orchestrator still works correctly
- Verify all existing tests still pass after refactoring