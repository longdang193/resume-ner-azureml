<!-- a286ef66-0f21-487e-831f-4896761305a5 d0bef3fa-073e-4306-89a8-42e261ad8db3 -->
# HPO Checkpoint/Resume Support Implementation Plan

## Overview

### Purpose

This plan introduces checkpoint and resume functionality to the Hyperparameter Optimization (HPO) process, enabling interrupted HPO runs to be resumed from the last checkpoint. This solves the problem of losing progress when HPO processes are interrupted (session timeouts, crashes, or manual stops), particularly critical in Google Colab (12-24 hour session limits) and Kaggle environments.

### Scope

**In scope**

- Optuna study persistence using SQLite storage backend
- Automatic checkpoint creation during HPO execution
- Automatic resume detection and loading of existing studies
- Platform-aware path resolution for Colab (Drive mount) and Kaggle
- Configuration-driven checkpoint behavior via HPO YAML config
- Backward compatibility (checkpointing is opt-in via config)

**Out of scope**

- Distributed HPO across multiple machines
- Checkpoint migration between platforms
- Manual checkpoint management CLI tools
- Checkpoint encryption or compression

### Guiding Principles

- Single Responsibility Principle (SRP)
- Clean Code & modular design
- Config-driven behavior (YAML / env-based)
- Backward compatibility
- Testability, observability, and reproducibility
- Platform-agnostic design with platform-specific optimizations

## Goals & Success Criteria

### Goals

- G1: Enable HPO checkpoint persistence using Optuna SQLite storage
- G2: Support automatic resume from checkpoints when available
- G3: Work seamlessly on both Google Colab (with Drive mount) and Kaggle
- G4: Make checkpoint behavior configurable via HPO config YAML

### Success Criteria

- [ ] HPO can be interrupted and resumed without losing completed trials
- [ ] Checkpoint files are created in platform-appropriate locations (Colab Drive, Kaggle working)
- [ ] Resume automatically detects and loads existing studies
- [ ] Configuration controls checkpoint enable/disable and auto-resume behavior
- [ ] All existing tests pass without modification
- [ ] New tests verify checkpoint/resume functionality
- [ ] Documentation updated with checkpoint usage examples

## Current State Analysis

### Existing Behavior

The HPO process currently:

1. Creates an in-memory Optuna study in `run_local_hpo_sweep()` (line 556)
2. Runs all trials in a single execution (lines 606-612)
3. Stores trial outputs in `output_dir / backbone / trial_{number}/`
4. Has no persistence mechanism - if interrupted, all progress is lost

### Pain Points / Limitations

- L1: **No checkpoint persistence** - Interrupted HPO runs lose all completed trials
- L2: **Session timeout vulnerability** - Colab sessions timeout after 12-24 hours, losing HPO progress
- L3: **No incremental execution** - Cannot run HPO in batches and resume later
- L4: **Platform-specific paths not handled** - No automatic detection of Colab Drive or Kaggle paths

### Architectural / SRP Issues

- Platform detection logic exists only in notebooks, not in core code
- No centralized utility for platform-aware path resolution
- Checkpoint logic would be embedded in HPO orchestration (needs separation)

## High-Level Design

### Architecture Overview

```
Entry Points (CLI / Notebook / Tests)
|
v
Orchestration Layer (run_local_hpo_sweep)
|
v
Platform Detection Utility (new)
|     |
|     v
|  Checkpoint Manager (new)
|     |
|     v
|  Optuna Study (with SQLite storage)
|
v
Domain Services (HPO objective, trial execution)
```

### Responsibility Breakdown

| Layer | Responsibility |

|-------|---------------|

| Entry Points | Pass checkpoint config to HPO orchestration |

| Orchestration | Create/load study with checkpoint support |

| Platform Detection | Detect Colab/Kaggle and resolve appropriate paths |

| Checkpoint Manager | Handle study storage path resolution and validation |

| Optuna Study | Persist trials to SQLite database |

| Domain Logic | Execute trials (unchanged) |

## Module & File Structure

### New Files to Create

- `src/shared/platform_detection.py` — Platform detection utilities (Colab/Kaggle)
- `src/orchestration/jobs/checkpoint_manager.py` — Checkpoint path resolution and validation

### Files to Modify

- `src/orchestration/jobs/local_sweeps.py`
  - Add checkpoint storage parameter to `run_local_hpo_sweep()`
  - Add resume logic to load existing studies
  - Calculate remaining trials based on completed trials
  - Add checkpoint path resolution using platform detection

- `config/hpo/smoke.yaml` and `config/hpo/prod.yaml`
  - Add optional `checkpoint` section with `enabled`, `storage_path`, `auto_resume` fields

- `tests/e2e/test_e2e_workflow.py`
  - Update `run_hpo_sweep()` to pass checkpoint config
  - Verify checkpoint files are created

- `tests/e2e/test_hpo_with_tiny_datasets.py`
  - Add tests for checkpoint/resume functionality

### Files Explicitly Not Touched

- `src/training/train.py` — Training logic unchanged
- `src/orchestration/jobs/training.py` — Training orchestration unchanged
- MLflow integration — Unchanged

## Detailed Design per Component

### Component: Platform Detection Utility

**Responsibility (SRP)**

- Detect execution environment (Colab, Kaggle, local)
- Provide platform-specific path resolution

**Inputs**

- Environment variables (COLAB_GPU, KAGGLE_KERNEL_RUN_TYPE)
- Base path for resolution

**Outputs**

- Platform identifier ("colab", "kaggle", "local")
- Resolved paths for checkpoint storage

**Public API**

```python
def detect_platform() -> str:
    """Detect execution platform: 'colab', 'kaggle', or 'local'"""

def resolve_checkpoint_path(base_path: Path, relative_path: str) -> Path:
    """Resolve checkpoint path with platform-specific optimizations"""
```

**Implementation Notes**

- Colab: Prefer Drive mount path if available (`/content/drive/MyDrive/...`)
- Kaggle: Use `/kaggle/working/` (automatically persisted)
- Local: Use provided base path
- Fallback to base path if platform-specific path unavailable

### Component: Checkpoint Manager

**Responsibility (SRP)**

- Resolve checkpoint storage paths
- Validate checkpoint file existence and integrity
- Provide storage URI for Optuna

**Inputs**

- Base output directory
- Checkpoint configuration from HPO config
- Platform identifier

**Outputs**

- SQLite storage URI string (or None for in-memory)
- Resolved checkpoint file path

**Public API**

```python
def resolve_storage_path(
    output_dir: Path,
    checkpoint_config: Dict[str, Any],
    platform: str,
    backbone: str
) -> Optional[Path]:
    """Resolve checkpoint storage path with platform awareness"""

def get_storage_uri(storage_path: Optional[Path]) -> Optional[str]:
    """Convert storage path to Optuna storage URI"""
```

**Implementation Notes**

- Default checkpoint location: `output_dir / backbone / "study.db"`
- Support configurable `storage_path` with `{backbone}` placeholder
- Create parent directories if needed
- Return None if checkpointing disabled

### Component: HPO Orchestration (Modified)

**Responsibility (SRP)**

- Create or load Optuna study with checkpoint support
- Calculate remaining trials based on completed trials
- Run optimization for remaining trials only

**Inputs**

- Checkpoint configuration from HPO config
- Platform detection result

**Outputs**

- Optuna study (new or loaded)

**Public API**

```python
def run_local_hpo_sweep(
    ...,
    checkpoint_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Run HPO with optional checkpoint/resume support"""
```

**Implementation Notes**

- Load existing study if checkpoint exists and `auto_resume=True`
- Calculate `remaining_trials = max_trials - completed_trials`
- Only run `n_trials=remaining_trials` if resuming
- Print clear messages about resume status and trial counts
- Handle corrupted checkpoint files gracefully (fallback to new study)

## Configuration & Controls

### Configuration Sources

- HPO YAML config files (`config/hpo/*.yaml`)
- Environment variables (for platform detection)

### Example Configuration

```yaml
# config/hpo/smoke.yaml
checkpoint:
  enabled: true
  storage_path: "outputs/hpo/{backbone}/study.db"  # Relative to output_dir, {backbone} placeholder
  auto_resume: true  # Automatically resume if checkpoint exists

sampling:
  max_trials: 10
  # ... rest of config
```

### Validation Rules

- `checkpoint.enabled`: Optional boolean, defaults to `false` (backward compatible)
- `checkpoint.storage_path`: Optional string, supports `{backbone}` placeholder
- `checkpoint.auto_resume`: Optional boolean, defaults to `true` when checkpointing enabled
- If `enabled=false` or missing, HPO runs in-memory (current behavior)
- Storage path defaults to `"{backbone}/study.db"` relative to output_dir if not specified

## Implementation Steps

1. **Create platform detection utility** (`src/shared/platform_detection.py`)

   - Implement `detect_platform()` function
   - Implement `resolve_checkpoint_path()` function
   - Add unit tests

2. **Create checkpoint manager** (`src/orchestration/jobs/checkpoint_manager.py`)

   - Implement storage path resolution
   - Implement storage URI conversion
   - Add validation logic
   - Add unit tests

3. **Modify HPO orchestration** (`src/orchestration/jobs/local_sweeps.py`)

   - Add checkpoint config parameter
   - Integrate platform detection
   - Add study load logic with resume support
   - Calculate remaining trials
   - Update `study.optimize()` to use remaining trials
   - Add informative logging

4. **Update HPO config files** (`config/hpo/smoke.yaml`, `config/hpo/prod.yaml`)

   - Add checkpoint section with sensible defaults
   - Document checkpoint options

5. **Update calling code** (`tests/e2e/test_e2e_workflow.py`)

   - Extract checkpoint config from HPO config
   - Pass to `run_local_hpo_sweep()`

6. **Add integration tests** (`tests/e2e/test_hpo_with_tiny_datasets.py`)

   - Test checkpoint creation
   - Test resume from checkpoint
   - Test remaining trials calculation
   - Test platform-specific paths

7. **Update documentation**

   - Add checkpoint/resume section to HPO documentation
   - Add Colab/Kaggle usage examples
   - Update troubleshooting guide

## Testing Strategy

### Unit Tests

- Platform detection: Mock environment variables, test all platforms
- Checkpoint manager: Test path resolution, URI conversion, validation
- Study load/save: Mock Optuna study operations

### Integration Tests

- End-to-end HPO with checkpoint enabled
- Interrupt and resume HPO run
- Verify no duplicate trials
- Verify remaining trials calculation
- Test on different platforms (mock platform detection)

### Edge Cases

- Checkpoint file corrupted (fallback to new study)
- Checkpoint file missing but resume requested (create new study)
- All trials already completed (skip optimization)
- Platform detection fails (fallback to local paths)
- Drive not mounted in Colab (fallback to /content/)

### Performance / Load Tests

- Checkpoint file size with many trials
- Study load time with large number of completed trials
- SQLite database performance

## Backward Compatibility & Migration

- **Backward compatible**: Checkpointing is opt-in via config (`enabled: false` by default)
- **Existing behavior preserved**: If `checkpoint` section missing or `enabled: false`, HPO runs in-memory
- **No migration required**: Existing HPO configs continue to work unchanged
- **Gradual adoption**: Users can enable checkpointing by adding config section

## Documentation Updates

### New Documentation

- `docs/HPO_CHECKPOINT_RESUME.md` — Checkpoint/resume guide with Colab/Kaggle examples

### Updated Documentation

- `docs/K_FOLD_CROSS_VALIDATION.md` — Mention checkpoint compatibility
- `config/hpo/*.yaml` — Add inline comments for checkpoint options
- `README.md` — Add checkpoint feature to HPO section

## Rollout & Validation Checklist

- [ ] Platform detection utility implemented and tested
- [ ] Checkpoint manager implemented and tested
- [ ] HPO orchestration modified with checkpoint support
- [ ] HPO config files updated with checkpoint section
- [ ] Unit tests added and passing
- [ ] Integration tests added and passing
- [ ] Backward compatibility verified (existing configs work)
- [ ] Documentation updated
- [ ] Tested on Colab environment (mock or real)
- [ ] Tested on Kaggle environment (mock or real)
- [ ] CI passing

## Appendix

### Platform-Specific Path Examples

**Google Colab:**

- Base: `/content/resume-ner-azureml`
- Drive mount: `/content/drive/MyDrive/resume-ner-checkpoints`
- Checkpoint: `/content/drive/MyDrive/resume-ner-checkpoints/hpo/distilbert/study.db`

**Kaggle:**

- Base: `/kaggle/working/resume-ner-azureml`
- Checkpoint: `/kaggle/working/resume-ner-azureml/outputs/hpo/distilbert/study.db`

**Local:**

- Base: `./` (project root)
- Checkpoint: `./outputs/hpo/distilbert/study.db`

### Optuna Storage URI Format

- SQLite: `sqlite:///path/to/study.db` (use 3 slashes for absolute paths)
- In-memory: `None` (default when checkpointing disabled)
- Study creation: `optuna.create_study(..., storage=storage_uri, load_if_exists=True)` loads existing or creates new

### Example Resume Flow

```
1. First run: max_trials=10, completes 5 trials, session timeout
2. Checkpoint saved: outputs/hpo/distilbert/study.db (5 trials)
3. Resume run: auto_resume=true detects checkpoint
4. Loads study with 5 completed trials
5. Calculates remaining_trials = 10 - 5 = 5
6. Runs 5 more trials to complete
```

### To-dos

- [ ] Create platform detection utility (src/shared/platform_detection.py) with detect_platform() and resolve_checkpoint_path() functions
- [ ] Create checkpoint manager (src/orchestration/jobs/checkpoint_manager.py) with storage path resolution and URI conversion
- [ ] Modify run_local_hpo_sweep() to support checkpoint/resume: add storage parameter, load existing studies, calculate remaining trials
- [ ] Add checkpoint section to config/hpo/smoke.yaml and config/hpo/prod.yaml with enabled, storage_path, auto_resume options
- [ ] Update test_e2e_workflow.py to extract checkpoint config from HPO config and pass to run_local_hpo_sweep()
- [ ] Add integration tests for checkpoint creation, resume functionality, and remaining trials calculation
- [ ] Create docs/HPO_CHECKPOINT_RESUME.md and update existing HPO documentation with checkpoint usage examples