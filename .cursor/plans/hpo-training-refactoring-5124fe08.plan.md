<!-- 5124fe08-c91b-4312-a495-fdc8e87c0be3 0095b559-2537-4b03-a4eb-2c5c871d8c0f -->
# HPO, Training, and Training Exec Refactoring Plan

## Overview

This plan identifies and eliminates redundancies across `hpo`, `training`, and `training_exec` modules, reorganizes code to follow DRY principles, improves Single Responsibility Principle (SRP) adherence, and maintains feature-level cohesion.

## Current State Analysis

### Key Redundancies Identified

1. **Subprocess Command Building** (3+ locations)

                                                - `hpo/execution/local/trial.py::TrialExecutor._build_command()` - builds training CLI args
                                                - `hpo/execution/local/refit.py::_build_refit_command()` - duplicated function (appears twice!)
                                                - `training_exec/executor.py::execute_final_training()` - builds training CLI args inline
                                                - All build similar `python -m training.train` commands with hyperparameters

2. **Environment Setup** (3+ locations)

                                                - `hpo/execution/local/trial.py::TrialExecutor._setup_environment()` - sets PYTHONPATH, MLflow vars, output dirs
                                                - `hpo/execution/local/refit.py::_setup_refit_environment()` - duplicated function (appears twice!)
                                                - `training_exec/executor.py::execute_final_training()` - sets similar environment variables inline
                                                - All set PYTHONPATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, output directories

3. **Subprocess Execution** (4+ locations)

                                                - `hpo/execution/local/trial.py::TrialExecutor.execute()` - runs subprocess.run()
                                                - `hpo/execution/local/refit.py::run_refit_training()` - runs subprocess.run() (appears twice!)
                                                - `training_exec/executor.py::execute_final_training()` - runs subprocess.run()
                                                - Similar error handling, logging, and return code checking

4. **MLflow Run Creation** (3+ locations)

                                                - `hpo/execution/local/refit.py::_create_refit_mlflow_run()` - duplicated function
                                                - `training_exec/executor.py::execute_final_training()` - creates MLflow runs inline
                                                - Similar patterns for experiment creation, run naming, tag building

5. **Config Building Overlap**

                                                - `training/config.py::build_training_config()` - builds config from args + files
                                                - `training_exec/jobs.py::build_final_training_config()` - builds final training config
                                                - Some overlap in config loading and merging logic

6. **Project Root Resolution** (3+ locations)

                                                - `hpo/execution/local/trial.py` - finds project root
                                                - `hpo/execution/local/refit.py::_find_project_root()` - duplicated function
                                                - `training_exec/executor.py` - uses root_dir parameter
                                                - All resolve project root differently

### Module Responsibilities

**hpo**: Hyperparameter optimization orchestration

- Search space management
- Trial execution (subprocess-based)
- Refit training (subprocess-based)
- Study management

**training**: Core training logic

- Model creation
- Training loop
- Data loading
- Metrics computation
- Direct execution (not subprocess)

**training_exec**: Final training job execution

- Subprocess orchestration
- MLflow setup
- Lineage tracking
- Azure ML job creation

## Refactoring Strategy

### Phase 1: Extract Common Subprocess Infrastructure

#### 1.1 Create `training/execution/subprocess_runner.py`

**Purpose**: Centralize subprocess execution patterns for training runs.

**New Module**: `src/training/execution/subprocess_runner.py`

**Functions**:

- `build_training_command()` - Unified command builder
                                - Parameters: backbone, dataset_path, config_dir, hyperparameters, training_options
                                - Returns: List[str] command arguments
                                - Handles: learning_rate, batch_size, dropout, weight_decay, epochs, fold_idx, etc.

- `setup_training_environment()` - Unified environment setup
                                - Parameters: root_dir, src_dir, output_dir, mlflow_config, optional_fold_config
                                - Returns: Dict[str, str] environment variables
                                - Sets: PYTHONPATH, MLFLOW_ *vars, AZURE_ML_OUTPUT_* vars, fold configs

- `execute_training_subprocess()` - Unified subprocess execution
                                - Parameters: command, cwd, env, capture_output=True
                                - Returns: subprocess.CompletedProcess
                                - Handles: error checking, logging, return code validation

**Benefits**:

- Eliminates 3+ duplicate command builders
- Eliminates 3+ duplicate environment setters
- Standardizes subprocess execution patterns
- Single place to update training CLI invocation

#### 1.2 Update Existing Code to Use New Module

**Files to Update**:

- `hpo/execution/local/trial.py::TrialExecutor` - Replace `_build_command()` and `_setup_environment()` with calls to new module
- `hpo/execution/local/refit.py` - Replace duplicated `_build_refit_command()` and `_setup_refit_environment()` functions
- `training_exec/executor.py::execute_final_training()` - Extract command building and environment setup to use new module

**Migration Pattern**:

```python
# Before
args = self._build_command(trial_params, dataset_path, backbone, train_config, fold_idx)
env = self._setup_environment(output_dir, root_dir, src_dir, mlflow_experiment_name, ...)

# After
from training.execution.subprocess_runner import build_training_command, setup_training_environment
args = build_training_command(
    backbone=backbone,
    dataset_path=dataset_path,
    config_dir=self.config_dir,
    hyperparameters=trial_params,
    training_options={"fold_idx": fold_idx, "hpo_epochs": train_config.get("training", {}).get("hpo_epochs", 1)}
)
env = setup_training_environment(
    root_dir=root_dir,
    src_dir=src_dir,
    output_dir=output_dir,
    mlflow_config={"experiment_name": mlflow_experiment_name, "parent_run_id": parent_run_id}
)
```

### Phase 2: Consolidate MLflow Setup

#### 2.1 Enhance `tracking/mlflow/setup.py` or Create `training/execution/mlflow_setup.py`

**Purpose**: Centralize MLflow run creation patterns used by training execution.

**New/Enhanced Module**: `src/training/execution/mlflow_setup.py`

**Functions**:

- `create_training_mlflow_run()` - Unified MLflow run creation
                                - Parameters: experiment_name, run_name, tags, parent_run_id, context
                                - Returns: run_id, run object
                                - Handles: experiment creation/get, run creation, tag application

- `setup_mlflow_tracking_env()` - Extract MLflow environment variable setup
                                - Parameters: experiment_name, tracking_uri, parent_run_id, additional_vars
                                - Returns: Dict[str, str] environment variables
                                - Consolidates MLflow env var logic from multiple locations

**Files to Update**:

- `hpo/execution/local/refit.py::_create_refit_mlflow_run()` - Use new function
- `training_exec/executor.py::execute_final_training()` - Extract MLflow setup to use new function

### Phase 3: Eliminate Duplication in `hpo/execution/local/refit.py`

#### 3.1 Remove Duplicate Functions

**Issue**: `refit.py` contains duplicate implementations of:

- `_find_project_root()` (appears twice)
- `_build_refit_command()` (appears twice)
- `_setup_refit_environment()` (appears twice)
- `_create_refit_mlflow_run()` (appears twice)
- `_verify_refit_environment()` (appears twice)
- `_read_refit_metrics()` (appears twice)
- `_log_refit_metrics_to_mlflow()` (appears twice)

**Action**:

- Keep single implementation of each function
- Remove duplicate code blocks
- Ensure single `run_refit_training()` function uses consolidated implementations

**Estimated Reduction**: ~400-500 lines of duplicate code removed

### Phase 4: Clarify Module Boundaries

#### 4.1 Refine `training` Module Responsibility

**Current**: Mixes direct execution (`orchestrator.py`) with core training logic

**Proposed Structure**:

- **Core Training Logic** (keep as-is):
                                - `trainer.py` - Training loop
                                - `model.py` - Model creation
                                - `data.py` - Data loading
                                - `metrics.py` - Metrics computation
                                - `evaluator.py` - Evaluation logic
                                - `config.py` - Config building (for direct execution)

- **Execution Infrastructure** (new submodule):
                                - `training/execution/` - New subdirectory
                                                                - `subprocess_runner.py` - Subprocess execution (from Phase 1)
                                                                - `mlflow_setup.py` - MLflow setup (from Phase 2)
                                                                - `__init__.py` - Exports

**Rationale**: Separates "how to train" (core) from "how to execute training" (execution), improving SRP while maintaining cohesion.

#### 4.2 Clarify `training_exec` Module Responsibility

**Current**: Mixes subprocess orchestration with job creation

**Proposed Structure**:

- **Keep**: Subprocess orchestration for final training
- **Keep**: Lineage extraction and tagging
- **Keep**: Azure ML job creation (`jobs.py`)
- **Refactor**: Extract subprocess logic to use `training/execution/subprocess_runner.py`

**Rationale**: `training_exec` becomes a thin orchestration layer that coordinates final training execution, while delegating subprocess details to shared infrastructure.

#### 4.3 Clarify `hpo` Module Responsibility

**Current**: Contains trial execution, refit execution, and orchestration

**Proposed Structure**:

- **Keep**: HPO-specific logic (search spaces, study management, trial orchestration)
- **Refactor**: Trial execution delegates to `training/execution/subprocess_runner.py`
- **Refactor**: Refit execution delegates to shared infrastructure

**Rationale**: `hpo` focuses on hyperparameter optimization orchestration, not subprocess execution details.

### Phase 5: Config Building Consolidation

#### 5.1 Review Config Building Overlap

**Files**:

- `training/config.py::build_training_config()` - For direct execution
- `training_exec/jobs.py::build_final_training_config()` - For final training jobs

**Action**:

- Review if `build_final_training_config()` can leverage `build_training_config()` 
- Or document clear separation: one for CLI args, one for programmatic config
- Ensure no duplicate YAML loading logic

**Decision Point**: If significant overlap exists, extract common config loading to `training/config.py` and have both use it.

## Implementation Order

1. **Phase 1** - Extract subprocess infrastructure (highest impact, eliminates most duplication)
2. **Phase 3** - Remove duplicate functions in refit.py (quick win)
3. **Phase 2** - Consolidate MLflow setup (medium impact)
4. **Phase 4** - Reorganize module structure (structural improvement)
5. **Phase 5** - Review config building (lowest priority, may not need changes)

## Expected Outcomes

### Code Reduction

- **~500-700 lines** of duplicate code eliminated
- **3+ duplicate functions** removed from `refit.py`
- **3+ command builders** consolidated to 1
- **3+ environment setters** consolidated to 1

### Improved Maintainability

- Single source of truth for training subprocess execution
- Changes to training CLI invocation happen in one place
- MLflow setup patterns standardized
- Clearer module boundaries

### Maintained Cohesion

- Training execution infrastructure grouped together
- HPO orchestration remains cohesive
- Final training execution remains cohesive
- No over-fragmentation (execution logic stays together)

## Risk Mitigation

1. **Backward Compatibility**: All changes are internal refactorings - public APIs remain unchanged
2. **Testing**: Existing tests should continue to pass after refactoring
3. **Incremental Migration**: Each phase can be done independently with tests passing after each
4. **Rollback**: Each phase is isolated and can be rolled back independently

## Files to Create

- `src/training/execution/__init__.py`
- `src/training/execution/subprocess_runner.py`
- `src/training/execution/mlflow_setup.py` (or enhance existing `tracking/mlflow/setup.py`)

## Files to Modify

- `src/hpo/execution/local/trial.py`
- `src/hpo/execution/local/refit.py` (remove duplicates)
- `src/training_exec/executor.py`
- `src/training/config.py` (if Phase 5 consolidation needed)

## Files to Review (No Changes Expected)

- `src/training/orchestrator.py` (direct execution, different from subprocess)
- `src/training_exec/jobs.py` (Azure ML job creation, different concern)

## Proposed Module Structure

### Current Structure

```
src/
├── hpo/
│   ├── __init__.py
│   ├── execution/
│   │   ├── local/
│   │   │   ├── trial.py          # TrialExecutor with _build_command, _setup_environment
│   │   │   ├── refit.py          # Duplicate functions (2x implementations)
│   │   │   └── sweep.py
│   │   └── azureml/
│   ├── core/
│   ├── checkpoint/
│   └── utils/
├── training/
│   ├── __init__.py
│   ├── orchestrator.py           # Direct execution (run_training)
│   ├── trainer.py                # Core training loop
│   ├── model.py                  # Model creation
│   ├── data.py                   # Data loading
│   ├── metrics.py                # Metrics computation
│   ├── config.py                 # Config building
│   └── ... (other core files)
└── training_exec/
    ├── __init__.py
    ├── executor.py               # Subprocess execution with inline command/env setup
    ├── jobs.py                   # Azure ML job creation
    ├── lineage.py
    └── tags.py
```

### Proposed Structure

```
src/
├── hpo/
│   ├── __init__.py
│   ├── execution/
│   │   ├── local/
│   │   │   ├── trial.py          # TrialExecutor delegates to training.execution
│   │   │   ├── refit.py          # Single implementation, delegates to training.execution
│   │   │   └── sweep.py
│   │   └── azureml/
│   ├── core/                     # HPO-specific: search spaces, study management
│   ├── checkpoint/               # HPO-specific: checkpoint storage
│   └── utils/                    # HPO-specific utilities
├── training/
│   ├── __init__.py
│   ├── orchestrator.py           # Direct execution (run_training) - unchanged
│   ├── trainer.py                # Core training loop - unchanged
│   ├── model.py                  # Model creation - unchanged
│   ├── data.py                   # Data loading - unchanged
│   ├── metrics.py                # Metrics computation - unchanged
│   ├── evaluator.py              # Evaluation logic - unchanged
│   ├── config.py                 # Config building - unchanged
│   ├── checkpoint_loader.py      # Checkpoint resolution - unchanged
│   ├── distributed.py            # DDP setup - unchanged
│   ├── cv_utils.py               # Cross-validation utilities - unchanged
│   ├── data_combiner.py          # Data combination - unchanged
│   ├── logging.py                # Logging utilities - unchanged
│   ├── utils.py                  # General utilities - unchanged
│   ├── cli.py                    # CLI parsing - unchanged
│   ├── train.py                  # CLI entry point - unchanged
│   └── execution/                # NEW: Subprocess execution infrastructure
│       ├── __init__.py           # Exports: build_training_command, setup_training_environment, execute_training_subprocess
│       ├── subprocess_runner.py  # NEW: Unified subprocess execution
│       └── mlflow_setup.py       # NEW: MLflow run creation for training execution
└── training_exec/
    ├── __init__.py
    ├── executor.py               # Refactored: delegates to training.execution
    ├── jobs.py                   # Azure ML job creation - unchanged
    ├── lineage.py                # Lineage extraction - unchanged
    └── tags.py                   # Tag application - unchanged
```

### Detailed Proposed Structure

#### New Module: `training/execution/`

**Purpose**: Centralized infrastructure for executing training as subprocesses. Used by HPO trials, refit training, and final training execution.

**File: `training/execution/subprocess_runner.py`**

```python
"""Subprocess execution infrastructure for training runs.

This module provides unified functions for building training commands,
setting up execution environments, and running training subprocesses.
Used by HPO trials, refit training, and final training execution.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import os
import sys

def build_training_command(
    backbone: str,
    dataset_path: str | Path,
    config_dir: Path,
    hyperparameters: Dict[str, Any],
    training_options: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Build command arguments for training subprocess.
    
    Args:
        backbone: Model backbone name
        dataset_path: Path to dataset directory
        config_dir: Configuration directory
        hyperparameters: Dict with learning_rate, batch_size, dropout, weight_decay
        training_options: Optional dict with:
   - fold_idx: Fold index for cross-validation
   - epochs: Number of epochs (defaults from config)
   - hpo_epochs: Override for HPO (speed optimization)
   - early_stopping_enabled: Enable/disable early stopping
   - use_combined_data: Use combined dataset
   - random_seed: Random seed
    
    Returns:
        List of command arguments for subprocess
    """
    ...

def setup_training_environment(
    root_dir: Path,
    src_dir: Path,
    output_dir: Path,
    mlflow_config: Dict[str, Any],
    fold_config: Optional[Dict[str, Any]] = None,
    trial_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Set up environment variables for training subprocess.
    
    Args:
        root_dir: Project root directory
        src_dir: Source directory (root_dir / "src")
        output_dir: Output directory for checkpoints
        mlflow_config: Dict with:
   - experiment_name: MLflow experiment name
   - tracking_uri: Optional tracking URI
   - parent_run_id: Optional parent run ID
   - trial_number: Optional trial number
        fold_config: Optional dict with:
   - fold_idx: Fold index
   - fold_splits_file: Path to fold splits file
        trial_config: Optional dict with:
   - skip_artifact_logging: Skip MLflow artifact logging (for HPO)
    
    Returns:
        Dictionary of environment variables
    """
    ...

def execute_training_subprocess(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    capture_output: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """
    Execute training subprocess with error handling.
    
    Args:
        command: Command arguments
        cwd: Working directory
        env: Environment variables
        capture_output: Capture stdout/stderr
        text: Return text output
    
    Returns:
        CompletedProcess result
    
    Raises:
        RuntimeError: If subprocess fails
    """
    ...
```

**File: `training/execution/mlflow_setup.py`**

```python
"""MLflow setup utilities for training execution.

This module provides functions for creating MLflow runs and setting up
MLflow tracking for training subprocesses.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient

def create_training_mlflow_run(
    experiment_name: str,
    run_name: str,
    tags: Optional[Dict[str, str]] = None,
    parent_run_id: Optional[str] = None,
    context: Optional[Any] = None,
) -> tuple[str, Any]:
    """
    Create MLflow run for training execution.
    
    Args:
        experiment_name: MLflow experiment name
        run_name: Run name
        tags: Optional tags dictionary
        parent_run_id: Optional parent run ID
        context: Optional naming context
    
    Returns:
        Tuple of (run_id, run_object)
    """
    ...

def setup_mlflow_tracking_env(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    trial_number: Optional[int] = None,
    additional_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Set up MLflow tracking environment variables.
    
    Args:
        experiment_name: MLflow experiment name
        tracking_uri: Optional tracking URI
        parent_run_id: Optional parent run ID
        trial_number: Optional trial number
        additional_vars: Optional additional environment variables
    
    Returns:
        Dictionary of environment variables
    """
    ...
```

**File: `training/execution/__init__.py`**

```python
"""Training execution infrastructure.

This module provides utilities for executing training as subprocesses.
"""

from .subprocess_runner import (
    build_training_command,
    setup_training_environment,
    execute_training_subprocess,
)
from .mlflow_setup import (
    create_training_mlflow_run,
    setup_mlflow_tracking_env,
)

__all__ = [
    "build_training_command",
    "setup_training_environment",
    "execute_training_subprocess",
    "create_training_mlflow_run",
    "setup_mlflow_tracking_env",
]
```

### Module Responsibility Matrix

| Module | Responsibility | Delegates To |

|--------|---------------|--------------|

| `hpo/execution/local/trial.py` | HPO trial orchestration | `training.execution` for subprocess execution |

| `hpo/execution/local/refit.py` | Refit training orchestration | `training.execution` for subprocess execution |

| `training/orchestrator.py` | Direct training execution | Core training modules (no subprocess) |

| `training_exec/executor.py` | Final training orchestration | `training.execution` for subprocess execution |

| `training/execution/` | Subprocess execution infrastructure | None (leaf module) |

| `training/trainer.py` | Core training loop | None (leaf module) |

| `training/model.py` | Model creation | None (leaf module) |

| `training/data.py` | Data loading | None (leaf module) |

### Dependency Flow

```
hpo/execution/local/trial.py
    └─> training.execution.subprocess_runner
            └─> training.train (subprocess)

hpo/execution/local/refit.py
    └─> training.execution.subprocess_runner
            └─> training.train (subprocess)

training_exec/executor.py
    └─> training.execution.subprocess_runner
            └─> training.train (subprocess)

training/orchestrator.py
    └─> training.trainer (direct import)
            └─> training.model, training.data (direct imports)
```

### Key Design Decisions

1. **New `training/execution/` submodule**: Groups execution infrastructure together while keeping it under `training` module for cohesion.

2. **Separation of concerns**:

                                                - `training/` = Core training logic + execution infrastructure
                                                - `hpo/` = HPO orchestration (delegates execution)
                                                - `training_exec/` = Final training orchestration (delegates execution)

3. **No over-fragmentation**: Execution infrastructure stays together in one submodule rather than scattered across multiple modules.

4. **Backward compatibility**: Public APIs remain unchanged - changes are internal refactorings.

5. **Single Responsibility**:

                                                - `training.execution` = How to execute training as subprocess
                                                - `training.trainer` = How to train a model
                                                - `hpo.execution` = How to orchestrate HPO trials
                                                - `training_exec` = How to orchestrate final training