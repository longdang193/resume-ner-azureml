<!-- 55de7dca-e6f4-4d11-949e-1acc547c4752 7b268e2a-34b7-4bb2-bfcb-1d72be03c688 -->
# Final Training Configuration Implementation Plan

## Overview

### Purpose

This plan introduces a YAML-based final training configuration system to replace the current inline `build_final_training_config()` function. This enables centralized control of all final training parameters (dataset, checkpoint, variant, hyperparameters, MLflow) through a single config file, improving maintainability and reproducibility while preserving backward compatibility.

### Scope

**In scope**

- Create `config/final_training.yaml` with comprehensive training configuration options
- Implement YAML loader with checkpoint auto-resolution using fingerprints/metadata
- Replace inline config building in notebooks with YAML-based approach
- Support dataset selection via data config file or auto-detect
- Integrate with existing fingerprint-based naming system
- Maintain backward compatibility with existing inline config approach

**Out of scope**

- Dataset combination strategies (removed per user request)
- Changes to training script itself (only orchestration/config loading)
- MLflow backend changes (only configuration)

### Guiding Principles

- Single Responsibility Principle (SRP)
- Config-driven behavior (YAML-based)
- Backward compatibility with deprecation path
- Integration with fingerprint-based naming system
- Auto-detection where possible (checkpoint, dataset, variant)

## Goals & Success Criteria

### Goals

- G1: Centralize final training configuration in YAML file
- G2: Enable checkpoint auto-resolution from fingerprints/metadata
- G3: Support dataset selection via config file or auto-detect
- G4: Maintain backward compatibility with existing notebooks

### Success Criteria

- [ ] `config/final_training.yaml` created with all training parameters
- [ ] YAML loader resolves checkpoints from fingerprints/metadata
- [ ] Notebook successfully uses YAML config instead of inline function
- [ ] Backward compatibility maintained (inline config still works with deprecation warning)
- [ ] Unit tests for config loader
- [ ] Integration tests verify end-to-end workflow

## Current State Analysis

### Existing Behavior

Currently, final training configuration is built inline in notebooks using `build_final_training_config()`:

```python
def build_final_training_config(
    best_config: dict,
    train_config: dict,
    random_seed: int = 42,
) -> dict:
    # Returns dict with backbone, learning_rate, batch_size, etc.
```

This dict is then:

1. Used to compute fingerprints (`spec_fp`, `exec_fp`)
2. Passed as CLI arguments to training script
3. Used for MLflow run naming

Checkpoint resolution currently happens in notebook cells using:

- `resolve_checkpoint_path()` from `src/training/checkpoint_loader.py`
- Manual metadata loading from cache files
- Environment variable `CHECKPOINT_PATH`

### Pain Points / Limitations

- L1: Configuration scattered across notebook cells
- L2: No centralized way to override training parameters
- L3: Checkpoint resolution logic duplicated in notebooks
- L4: Variant management requires manual increment logic
- L5: Dataset selection hardcoded to `resume_v1.yaml`

### Architectural / SRP Issues

- Configuration logic embedded in notebooks
- Checkpoint resolution not integrated with fingerprint system
- No single source of truth for final training parameters

## High-Level Design

### Architecture Overview

```
Notebook / CLI
    |
    v
Final Training Config Loader (new)
    |-- Loads config/final_training.yaml
    |-- Resolves checkpoint from fingerprints/metadata
    |-- Resolves dataset config (auto-detect or explicit)
    |-- Computes variant number (auto-increment)
    |-- Merges with best_config and train.yaml defaults
    |
    v
Resolved Config Dict
    |-- Used for fingerprint computation
    |-- Used for CLI argument construction
    |-- Used for MLflow tracking setup
    |
    v
Training Script Execution
```

### Responsibility Breakdown

| Layer | Responsibility |

|-------|---------------|

| Config Loader | Load YAML, resolve checkpoints, merge configs |

| Checkpoint Resolver | Auto-detect from fingerprints/metadata/index |

| Variant Manager | Auto-increment variant numbers |

| Notebook | Use loader, construct CLI args, execute training |

## Module & File Structure

### New Files to Create

- `config/final_training.yaml` — Final training configuration template
- `src/orchestration/final_training_config.py` — YAML loader and resolver
- `tests/unit/orchestration/test_final_training_config.py` — Unit tests

### Files to Modify

- `notebooks/01_orchestrate_training_colab.ipynb`
        - Replace `build_final_training_config()` usage with YAML loader
        - Update checkpoint resolution to use loader
        - Update variant management to use loader
- `src/orchestration/__init__.py`
        - Export new `load_final_training_config` function

### Files Explicitly Not Touched

- `src/training/config.py` — Training script config loading unchanged
- `src/training/checkpoint_loader.py` — Keep for backward compatibility
- `config/train.yaml` — Base training config unchanged

## Detailed Design per Component

### Component: `config/final_training.yaml`

**Responsibility (SRP)**

- Single source of truth for final training configuration

**Structure**

```yaml
# Final Training Configuration
# Controls all aspects of final training: dataset, checkpoint, variant, hyperparameters

# Run mode configuration (unified behavior control)
run:
  # Run mode determines overall behavior:
  # - reuse_if_exists: Skip if complete (default), reuse variant if exists
  # - force_new: Always create new variant, ignore existing
  # - resume_if_incomplete: Resume same run if incomplete, otherwise new variant
  # - continue_from_previous: Continue from previous final_training (requires source.type=final_training)
  mode: reuse_if_exists

# Source configuration (explicit checkpoint source intent)
source:
  # Source type clarifies checkpoint intent:
  # - scratch: No checkpoint, start from scratch
  # - hpo_best: Use HPO best checkpoint (if available)
  # - final_training: Continue from previous final training run
  type: hpo_best
  
  # Parent specification (used only when type=final_training)
  # - null: Auto-detect from latest final_training by spec_fp/exec_fp
  # - "outputs/final_training/local/distilbert/spec_..._exec_.../v1": Explicit path
  # - {spec_fp: "...", exec_fp: "...", variant: 1}: Load by fingerprints
  parent: null

# Dataset configuration
dataset:
  # Override the default data_config from experiment config
  # - null/not set: Auto-detect from experiment_config.data_config (default)
  # - "data/resume_v1.yaml": Use a specific dataset config
  data_config: null  # e.g., "data/resume_v2.yaml"
  
  # Optional: Override dataset path directly (if you want to use a dataset
  # that's not in a config file)
  local_path_override: null

# Checkpoint configuration
checkpoint:
  # Whether to load a checkpoint (low-level toggle, set automatically based on source.type)
  # - false: When source.type=scratch
  # - true: When source.type=hpo_best or final_training
  load: false
  
  # Checkpoint source (if load is true)
  # Options:
  # - null: Auto-detect from latest final_training by spec_fp/exec_fp
  # - "outputs/final_training/local/distilbert/spec_..._exec_.../v1/checkpoint": Explicit path
  # - {spec_fp: "...", exec_fp: "...", variant: 1}: Load by fingerprints
  source: null
  
  # Whether to validate checkpoint exists
  validate: true

# Variant configuration
variant:
  # Which variant to use
  # - null: Auto-increment based on run.mode
  # - 1, 2, 3, etc.: Explicit variant number (ignored if run.mode=force_new)
  number: null

# Identity controls (fingerprint computation configuration)
identity:
  # Control which factors are included in exec_fp computation
  # This allows evolving fingerprint computation without code refactoring
  include_code_fp: true      # Include code fingerprint (git commit, dependencies)
  include_precision_fp: true  # Include precision (fp32/fp16) in exec_fp
  include_determinism_fp: false  # Include determinism settings in exec_fp

# Random seed configuration
# This affects spec_fp computation and training reproducibility
seed:
  # Random seed for training
  # - null: Use from train.yaml or best_configuration (default: 42)
  # - 42, 123, etc.: Explicit seed value
  random_seed: null

# Training hyperparameter overrides
# These override values from train.yaml and best_configuration
# Set to null to use defaults
training:
  # Core hyperparameters
  learning_rate: null  # Override learning rate
  epochs: null  # Override epochs
  batch_size: null  # Override batch size
  dropout: null
  weight_decay: null
  
  # Training optimization
  gradient_accumulation_steps: null
  warmup_steps: null
  max_grad_norm: null
  
  # Early stopping configuration
  early_stopping:
    enabled: null  # null = use from train.yaml
    patience: null
    min_delta: null

# Output configuration
output:
  # Whether to skip training if already completed
  # This is the default behavior for run.mode=reuse_if_exists
  skip_if_complete: true

# MLflow tracking configuration (optional)
# NOTE: MLflow names/tags are metadata only and do NOT affect spec_fp or exec_fp
mlflow:
  # Override experiment name (if null, uses default: {experiment_name}-{STAGE_TRAINING}-{backbone})
  # Stored in metadata.json only, not in fingerprints
  experiment_name: null
  
  # Override run name (if null, uses default: {backbone}_{trial_name})
  # Stored in metadata.json only, not in fingerprints
  run_name: null
  
  # Additional tags to add to the run
  # Stored in metadata.json only, not in fingerprints
  tags: {}
```

**Validation Rules**

- `dataset.data_config` must be null or a valid path relative to `config/`
- `checkpoint.source` can be null, string path, or dict with `spec_fp`/`exec_fp`/`variant`
- `variant.number` must be null or >= 1
- `seed.random_seed` must be null or integer
- All `training.*` fields can be null (use defaults) or valid values

### Component: `src/orchestration/final_training_config.py`

**Responsibility (SRP)**

- Load and resolve final training configuration from YAML

**Public API**

```python
def load_final_training_config(
    root_dir: Path,
    config_dir: Path,
    best_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    train_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load and resolve final training configuration from YAML.
    
    This function:
 1. Loads config/final_training.yaml
 2. Resolves dataset config (auto-detect or explicit)
 3. Resolves checkpoint path (auto-detect from fingerprints/metadata or explicit)
 4. Computes variant number (auto-increment or explicit)
 5. Merges with best_config and train.yaml defaults
 6. Returns resolved config dict compatible with existing code
    
    Args:
        root_dir: Project root directory.
        config_dir: Config directory (root_dir / "config").
        best_config: Best configuration from HPO (contains backbone, hyperparameters).
        experiment_config: Experiment configuration (contains data_config, etc.).
        train_config: Optional base training config (loads train.yaml if None).
    
    Returns:
        Resolved final training config dict with all parameters set.
    """
```

**Implementation Notes**

1. **Dataset Resolution**:

            - If `dataset.data_config` is null, use `experiment_config.data_config`
            - If `dataset.data_config` is set, load that config file
            - If `dataset.local_path_override` is set, use that path directly

2. **Checkpoint Resolution** (when `checkpoint.load_from_checkpoint` is true):

            - If `checkpoint.source` is null:
                    - Try to find latest training by `spec_fp`/`exec_fp` from index files
                    - Fall back to metadata lookup by fingerprints
                    - Fall back to latest training cache
            - If `checkpoint.source` is a string path, validate and use it
            - If `checkpoint.source` is a dict with `spec_fp`/`exec_fp`/`variant`, resolve using those
            - Set `CHECKPOINT_PATH` environment variable for training script

3. **Variant Resolution**:

            - If `variant.number` is null:
                    - Check existing trainings with same `spec_fp`/`exec_fp`
                    - Find highest variant number, increment by 1
            - If `variant.force_new` is true, always increment
            - If `variant.number` is set, use that value

4. **Config Merging**:

            - Start with `train.yaml` defaults
            - Apply `best_config` hyperparameters
            - Apply `final_training.yaml` overrides (non-null values only)
            - Return merged dict compatible with existing `build_final_training_config()` output

5. **Backward Compatibility**:

            - If `config/final_training.yaml` doesn't exist, fall back to inline config building
            - Issue deprecation warning when using inline config

**Dependencies**

- `orchestration.fingerprints` — For computing `spec_fp`/`exec_fp`
- `orchestration.metadata_manager` — For loading training metadata
- `orchestration.index_manager` — For fast lookup of existing trainings
- `training.checkpoint_loader` — For checkpoint validation
- `shared.yaml_utils` — For loading YAML files

### Component: Checkpoint Auto-Resolution

**Responsibility (SRP)**

- Auto-detect checkpoint path from fingerprints and metadata

**Resolution Priority** (when `checkpoint.source` is null):

1. **Index Lookup**: Use `index_manager.find_by_spec_and_env()` to find latest training
2. **Metadata Lookup**: Use `metadata_manager.find_metadata_by_spec_fp()` to find training
3. **Cache Lookup**: Use `paths.load_cache_file()` to find latest training cache
4. **Fallback**: Return None (no checkpoint)

**Implementation**

```python
def resolve_checkpoint_from_fingerprints(
    root_dir: Path,
    config_dir: Path,
    spec_fp: str,
    exec_fp: str,
    environment: str,
) -> Optional[Path]:
    """
    Resolve checkpoint path from fingerprints.
    
    Tries multiple lookup strategies in priority order.
    Returns None if no checkpoint found.
    """
```

### Component: Variant Auto-Increment

**Responsibility (SRP)**

- Compute next available variant number for a given `spec_fp`/`exec_fp`

**Implementation**

```python
def compute_next_variant(
    root_dir: Path,
    config_dir: Path,
    spec_fp: str,
    exec_fp: str,
    environment: str,
    force_new: bool = False,
) -> int:
    """
    Compute next available variant number.
    
    Scans existing training directories to find highest variant,
    then returns next number (or 1 if none exist).
    """
```

## Configuration & Controls

### Configuration Sources

- Primary: `config/final_training.yaml`
- Fallback: Inline `build_final_training_config()` (deprecated)
- Base defaults: `config/train.yaml`
- HPO results: `best_configuration` dict

### Example Configuration

```yaml
# Minimal config (all auto-detect)
dataset:
  data_config: null  # Use experiment default

checkpoint:
  load_from_checkpoint: false

variant:
  number: null  # Auto-increment

# Explicit config
dataset:
  data_config: "data/resume_v2.yaml"

checkpoint:
  load_from_checkpoint: true
  source: null  # Auto-detect from fingerprints

variant:
  number: 2  # Use variant 2

training:
  learning_rate: 1e-5  # Override for fine-tuning
  epochs: 3
```

### Validation Rules

- All paths must be relative to `config_dir` or absolute
- Checkpoint paths must exist and be valid (if `validate: true`)
- Variant numbers must be >= 1
- Random seed must be integer or null

## Implementation Steps

1. **Create `config/final_training.yaml` template**

            - Define all configuration sections
            - Add comments and examples

2. **Implement `load_final_training_config()` in `src/orchestration/final_training_config.py`**

            - Load YAML file
            - Implement dataset resolution
            - Implement checkpoint auto-resolution
            - Implement variant auto-increment
            - Implement config merging logic

3. **Add checkpoint resolution helpers**

            - `resolve_checkpoint_from_fingerprints()` — Auto-detect from fingerprints
            - Integrate with `index_manager` and `metadata_manager`

4. **Add variant management helpers**

            - `compute_next_variant()` — Auto-increment logic
            - Scan existing training directories

5. **Update notebook `01_orchestrate_training_colab.ipynb`**

            - Replace `build_final_training_config()` with `load_final_training_config()`
            - Update checkpoint resolution to use loader
            - Update variant logic to use loader
            - Keep backward compatibility path with deprecation warning

6. **Add unit tests**

            - Test YAML loading
            - Test checkpoint resolution (with mocks)
            - Test variant computation
            - Test config merging

7. **Add integration tests**

            - End-to-end test with real config file
            - Test backward compatibility path

8. **Update exports in `src/orchestration/__init__.py`**

            - Export `load_final_training_config`

9. **Update documentation**

            - Add `config/final_training.yaml` to config documentation
            - Update notebook comments

## Testing Strategy

### Unit Tests

- **YAML Loading**: Test loading valid/invalid YAML files
- **Dataset Resolution**: Test auto-detect vs explicit config
- **Checkpoint Resolution**: Mock index/metadata lookups, test priority order
- **Variant Computation**: Mock directory scanning, test increment logic
- **Config Merging**: Test merging with best_config and train.yaml

### Integration Tests

- **End-to-End**: Load config, resolve checkpoint, compute variant, merge configs
- **Backward Compatibility**: Test fallback to inline config when YAML missing
- **Real Metadata**: Test with actual metadata/index files

### Edge Cases

- Missing YAML file (fallback)
- Invalid checkpoint path (validation)
- No existing variants (start at 1)
- Multiple variants exist (increment correctly)

## Backward Compatibility & Migration

### What Remains Compatible

- Existing notebooks using `build_final_training_config()` will continue to work
- Training script behavior unchanged
- Metadata/index files unchanged

### Deprecated Behavior

- Inline `build_final_training_config()` function (deprecated, but still works)
- Manual checkpoint resolution in notebooks (deprecated, use loader)

### Migration Steps

1. Create `config/final_training.yaml` with desired settings
2. Update notebook to use `load_final_training_config()`
3. Remove inline `build_final_training_config()` call
4. Test training execution

## Documentation Updates

### New Documentation

- `docs/config/final_training.md` — Configuration reference

### Updated Documentation

- `notebooks/01_orchestrate_training_colab.ipynb` — Update comments
- `README.md` — Mention new config file (if applicable)

## Rollout & Validation Checklist

- [ ] `config/final_training.yaml` created
- [ ] `load_final_training_config()` implemented
- [ ] Checkpoint auto-resolution implemented
- [ ] Variant auto-increment implemented
- [ ] Unit tests added and passing
- [ ] Integration tests added and passing
- [ ] Notebook updated to use new loader
- [ ] Backward compatibility verified
- [ ] Deprecation warnings added
- [ ] Documentation updated

### To-dos

- [ ] Create config/final_training.yaml with all configuration sections (dataset, checkpoint, variant, seed, training, output, mlflow)
- [ ] Implement load_final_training_config() in src/orchestration/final_training_config.py with YAML loading and basic structure
- [ ] Implement dataset resolution logic (auto-detect from experiment_config or load explicit config file)
- [ ] Implement checkpoint auto-resolution from fingerprints/metadata/index (resolve_checkpoint_from_fingerprints helper)
- [ ] Implement variant auto-increment logic (compute_next_variant helper that scans existing directories)
- [ ] Implement config merging logic (train.yaml defaults -> best_config -> final_training.yaml overrides)
- [ ] Update notebooks/01_orchestrate_training_colab.ipynb to use load_final_training_config() instead of build_final_training_config()
- [ ] Add unit tests in tests/unit/orchestration/test_final_training_config.py for loader, resolution, and merging
- [ ] Add integration tests for end-to-end config loading and backward compatibility
- [ ] Update src/orchestration/__init__.py to export load_final_training_config