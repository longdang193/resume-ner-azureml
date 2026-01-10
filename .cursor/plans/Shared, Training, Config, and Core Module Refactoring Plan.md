<!-- 20e7fec0-22b8-404c-9707-d45c7258c6e3 8ec9309d-0f36-4cd7-ac3c-ab7460c79144 -->
# Shared, Training, Config, and Core Module Refactoring Plan

## Current State Analysis

### Identified Redundancies

1. **Config Loading Wrappers**

- `training/config.py::load_config_file()` - thin wrapper around `shared/yaml_utils.load_yaml()`
- `config/loader.py::_load_yaml()` - thin wrapper around `shared/yaml_utils.load_yaml()`
- Both modules add minimal value, creating unnecessary indirection

2. **Checkpoint Path Resolution (Name Collision)**

- `shared/platform_detection.py::resolve_checkpoint_path()` - platform-specific checkpoint path resolution (Colab/Kaggle/Azure)
- `training/checkpoint_loader.py::resolve_checkpoint_path()` - config-based checkpoint resolution from training config
- Same function name, different purposes - causes confusion and potential import conflicts

3. **Config Building/Merging Logic**

- `training/config.py::build_training_config()` - builds config from CLI args + YAML files
- `config/training.py::load_final_training_config()` + `_merge_configs()` - builds final training config with precedence rules
- Overlap in config merging patterns and argument override application

4. **Normalization Functions**

- `core/normalize.py::normalize_for_name()` / `normalize_for_path()` - naming/path normalization
- `training/data.py::normalize_text()` - text normalization for tokenization
- Different purposes but both are normalization utilities

## Refactoring Strategy

### Phase 1: Eliminate Config Loading Wrappers

**Goal**: Remove unnecessary wrapper functions and use `shared/yaml_utils.load_yaml()` directly.

**Changes**:

- Remove `training/config.py::load_config_file()` wrapper
- Update all callers in `training/` to use `shared.yaml_utils.load_yaml()` directly
- Remove `config/loader.py::_load_yaml()` wrapper  
- Update `config/loader.py` to import and use `shared.yaml_utils.load_yaml()` directly
- Update `training/__init__.py` to remove `load_config_file` export

**Files to modify**:

- `src/training/config.py` - remove wrapper, update imports
- `src/config/loader.py` - remove wrapper, update imports  
- `src/training/__init__.py` - remove export
- All files importing `load_config_file` from training module

**Impact**: Reduces indirection, makes dependencies clearer

### Phase 2: Resolve Checkpoint Path Name Collision

**Goal**: Rename functions to reflect their distinct purposes and prevent confusion.

**Changes**:

- Rename `shared/platform_detection.py::resolve_checkpoint_path()` → `resolve_platform_checkpoint_path()`
- Rename `training/checkpoint_loader.py::resolve_checkpoint_path()` → `resolve_training_checkpoint_path()`
- Update all callers to use new names
- Update `shared/__init__.py` and `training/__init__.py` exports

**Files to modify**:

- `src/shared/platform_detection.py` - rename function
- `src/training/checkpoint_loader.py` - rename function
- `src/shared/__init__.py` - update export
- `src/training/__init__.py` - update export
- All files calling these functions (8 files identified)

**Impact**: Eliminates name collision, improves code clarity

### Phase 3: Consolidate Config Building Logic

**Goal**: Extract common config merging patterns into shared utilities while maintaining feature-specific logic.

**Changes**:

- Create `config/merging.py` with shared config merging utilities:
- `merge_configs_with_precedence()` - generic config merger with precedence rules
- `apply_argument_overrides()` - extract from `training/config.py::_apply_argument_overrides()`
- Refactor `training/config.py::build_training_config()` to use shared merging utilities
- Refactor `config/training.py::_merge_configs()` to use shared merging utilities
- Keep domain-specific logic (final training config building) in `config/training.py`

**Files to create**:

- `src/config/merging.py` - new module for shared config merging logic

**Files to modify**:

- `src/training/config.py` - use shared merging utilities
- `src/config/training.py` - use shared merging utilities
- `src/config/__init__.py` - export new utilities

**Impact**: Reduces duplication in config merging logic, improves maintainability

### Phase 4: Organize Normalization Functions

**Goal**: Clarify separation of concerns for normalization utilities.

**Changes**:

- Keep `core/normalize.py` for naming/path normalization (used by naming system)
- Keep `training/data.py::normalize_text()` for text normalization (training-specific)
- Add clear docstrings distinguishing purposes
- Consider renaming `training/data.py::normalize_text()` → `normalize_text_for_tokenization()` if clarity improves

**Files to modify**:

- `src/training/data.py` - improve docstring, consider rename
- `src/core/normalize.py` - improve docstrings

**Impact**: Clarifies purpose of each normalization function

## Module Responsibilities (Post-Refactoring)

### `shared/` Module

- **Purpose**: Cross-cutting utilities used across multiple domains
- **Responsibilities**:
- YAML/JSON file I/O (`yaml_utils.py`, `json_cache.py`)
- Logging utilities (`logging_utils.py`)
- Platform detection (`platform_detection.py`)
- MLflow setup (`mlflow_setup.py`)
- CLI argument parsing (`argument_parsing.py`)
- Metrics reading (`metrics_utils.py`)
- Tokenization utilities (`tokenization_utils.py`)
- **No changes to structure**, only cleanup of exports

### `training/` Module  

- **Purpose**: Training-specific functionality
- **Responsibilities**:
- Training execution (`execution/` submodule)
- Model training (`trainer.py`, `train.py`)
- Data loading (`data.py`, `data_combiner.py`)
- Metrics computation (`metrics.py`)
- Checkpoint management (`checkpoint_loader.py`)
- Config building from CLI args (`config.py`)
- **Changes**: Remove wrapper functions, use shared utilities directly

### `config/` Module

- **Purpose**: Configuration loading, validation, and domain-specific config building
- **Responsibilities**:
- Experiment config loading (`loader.py`)
- Domain-specific config builders (`training.py`, `conversion.py`, `environment.py`)
- Config validation (`validation.py`)
- **NEW**: Shared config merging utilities (`merging.py`)
- **Changes**: Add merging utilities, remove wrapper functions

### `core/` Module

- **Purpose**: Low-level utilities with no external dependencies (naming system foundation)
- **Responsibilities**:
- Token definitions (`tokens.py`)
- Normalization for names/paths (`normalize.py`)
- Placeholder extraction (`placeholders.py`)
- **No changes** - already well-organized

## Implementation Order

1. **Phase 1** (Config Loading Wrappers) - Low risk, high clarity gain
2. **Phase 2** (Checkpoint Path Renaming) - Medium risk, eliminates confusion
3. **Phase 3** (Config Merging Consolidation) - Higher risk, requires careful testing
4. **Phase 4** (Normalization Documentation) - Low risk, documentation improvement

## Testing Strategy

- Update imports in all affected files
- Run existing test suite to ensure no regressions
- Verify config loading works identically after wrapper removal
- Verify checkpoint path resolution works with new names
- Test config merging with various precedence scenarios

## Risk Mitigation

- **Backward Compatibility**: Update all imports immediately to avoid breaking changes
- **Incremental Approach**: Complete each phase before moving to next
- **Test Coverage**: Ensure tests pass after each phase
- **Documentation**: Update docstrings to clarify function purposes

## Expected Outcomes

- Reduced code duplication (eliminate 2 wrapper functions)
- Clearer function naming (resolve checkpoint path collision)
- Shared config merging utilities (reduce duplication in 2 modules)
- Improved maintainability through clearer module boundaries
- Better adherence to DRY and SRP principles