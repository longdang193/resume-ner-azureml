# Consolidate Paths and Naming - Detailed Implementation Checklist

This document provides a detailed, phase-by-phase implementation checklist with important clarifications for the consolidation plan.

## Important Clarifications

### extract_placeholders() Location
- **Clarification**: `extract_placeholders()` is currently in `orchestration/tokens.py` (line 60), NOT in `paths.py`
- **Action**: Move `extract_placeholders()` from `tokens.py` to `core/placeholders.py` (no extraction from paths.py needed)
- **Note**: `tokens.py` currently defines `extract_placeholders` in the same file

### PROCESS_PATTERN_KEYS Constant
- **Clarification**: This constant does NOT exist in current codebase - needs to be created
- **Action**: Define `PROCESS_PATTERN_KEYS = {"final_training": "final_training_v2", "conversion": "conversion_v2", "hpo": "hpo_v2", "benchmarking": "benchmarking_v2"}` in `paths/resolve.py`
- **Usage**: Use in `paths/config.py` validator to enforce keys exist for schema v2

### resolve_output_path_v2() Callers
- **Current state**: Function exists in `paths.py` (line 662) and is exported from `orchestration/__init__.py`
- **Callers found**: Only used in docstring examples within `paths.py` itself; no external callers found
- **Action**: Audit all potential callers before removing; add wrapper in `orchestration/paths.py` only if external callers exist

### Token Expansion Locations
- **Found locations**:
  - `naming_centralized.py`: Multiple places computing `trial8 = context.trial_key_hash[:8]` (lines 263, 283, 296, 467)
  - `paths.py`: `study8`, `trial8` parsing in `parse_hpo_path_v2()` (line 737)
  - `jobs/hpo/local/refit/executor.py`: `trial8` computation (lines 145, 626)
  - `jobs/hpo/local_sweeps.py`: `trial8` computation (line 262)
  - `jobs/selection/trial_finder.py`: Hash slicing for display (lines 201, 203)
  - `jobs/selection/study_summary.py`: Hash slicing (line 190)
  - `jobs/selection/artifact_acquisition.py`: Hash slicing (lines 125, 146)
  - `jobs/hpo/local/cv/orchestrator.py`: Multiple `trial8` computations (lines 173, 210, 234)
- **Action**: Consolidate ALL hash slicing logic into `naming/context_tokens.py` with `build_token_values()`

### MLflow Config Dependencies
- **Current dependencies**: 
  - `tags.py` imports `get_naming_config` from `orchestration.jobs.tracking.config.loader`
  - `run_names.py` imports `get_naming_config`, `load_mlflow_config` from same
- **Decision needed**: 
  - **Option A**: Move minimal config loader (`get_naming_config`, `load_mlflow_config`) to `naming/mlflow/config.py` and move MLflow naming to `naming/mlflow/`
  - **Option B**: Keep MLflow naming in `orchestration/jobs/tracking/naming/` and create thin facade in `naming/mlflow/` that re-exports
- **Recommendation**: Option A if config loader is lightweight (just YAML loading), Option B if it has heavy dependencies

### environment.py Clarification
- **Note**: Current `orchestration/environment.py` is about Azure ML training environments (conda, docker), NOT path environment overrides
- **Action**: Plan is correct - no path-specific environment.py exists to remove; path overrides are handled via `env_overrides` in paths.yaml

## Detailed Implementation Checklist

### Phase 1: Pre-Implementation Analysis

- [ ] **Audit resolve_output_path_v2() callers**
  - [ ] Search codebase for all imports/uses of `resolve_output_path_v2`
  - [ ] Document all call sites
  - [ ] Determine if wrapper needed in `orchestration/paths.py` or can be removed entirely

- [ ] **Audit token expansion logic**
  - [ ] Document all locations computing `spec8`, `trial8`, `study8`, `bench8`, `conv8` from full hashes
  - [ ] List all files that need to be updated to use `naming/context_tokens.py`
  - [ ] Verify no token expansion logic remains after consolidation

- [ ] **Analyze MLflow config dependencies**
  - [ ] Read `orchestration/jobs/tracking/config/loader.py` fully
  - [ ] List all dependencies of `get_naming_config()` and `load_mlflow_config()`
  - [ ] Decide: Option A (move to naming/mlflow/) or Option B (facade)
  - [ ] Document decision and rationale

- [ ] **Verify ExperimentConfig usage**
  - [ ] Find all callers of `get_stage_config()` in current `naming.py`
  - [ ] Verify they can work with dict interface instead of `ExperimentConfig`
  - [ ] Document any needed changes

### Phase 2: Create Core Module

- [ ] **Create src/core/ directory structure**
  - [ ] Create `src/core/` directory
  - [ ] Create `src/core/__init__.py`

- [ ] **Move tokens.py to core/tokens.py**
  - [ ] Copy `orchestration/tokens.py` to `core/tokens.py`
  - [ ] Remove `extract_placeholders()` function (will move to placeholders.py)
  - [ ] Update imports to use relative imports if needed
  - [ ] Keep `is_token_known()`, `is_token_allowed()`, `TOKENS` registry
  - [ ] Export public functions in `core/__init__.py`

- [ ] **Create core/placeholders.py**
  - [ ] Move `extract_placeholders()` from `tokens.py` to `core/placeholders.py`
  - [ ] Update function to not depend on tokens.py
  - [ ] Add docstring explaining it's for parsing {placeholder} patterns
  - [ ] Export in `core/__init__.py`

- [ ] **Move normalize.py to core/normalize.py**
  - [ ] Copy `orchestration/normalize.py` to `core/normalize.py`
  - [ ] Keep `normalize_for_path()` and `normalize_for_name()` functions
  - [ ] Verify no dependencies on orchestration modules
  - [ ] Export in `core/__init__.py`

- [ ] **Verify core/ has no circular dependencies**
  - [ ] Check all imports in core/ modules
  - [ ] Ensure no imports from `paths/`, `naming/`, or `orchestration/`
  - [ ] Run import tests

### Phase 3: Create Paths Module

- [ ] **Create src/paths/ directory structure**
  - [ ] Create `src/paths/` directory
  - [ ] Create `src/paths/__init__.py`

- [ ] **Create paths/config.py**
  - [ ] Move `load_paths_config()` from `orchestration/paths.py`
  - [ ] Move `apply_env_overrides()` from `orchestration/paths.py`
  - [ ] Move `validate_paths_config()` from `orchestration/paths.py`
  - [ ] Move `_get_default_paths()` from `orchestration/paths.py`
  - [ ] **Add config caching**: Implement `@lru_cache` with manual mtime check
    - [ ] Cache key: `(config_dir, storage_env, mtime(paths.yaml))`
    - [ ] Check mtime before returning cached value
    - [ ] Clear cache if file modified
  - [ ] **Add PROCESS_PATTERN_KEYS validation**: Import from resolve.py and validate in `validate_paths_config()` for schema v2
  - [ ] Update imports to use `core/` modules
  - [ ] Export in `paths/__init__.py`

- [ ] **Create paths/resolve.py**
  - [ ] Move `resolve_output_path()` from `orchestration/paths.py` (legacy function)
  - [ ] Move `build_output_path()` from `orchestration/naming_centralized.py` (v2 entrypoint)
  - [ ] Move `_build_output_path_fallback()` from `naming_centralized.py`
  - [ ] **Create PROCESS_PATTERN_KEYS constant**: `{"final_training": "final_training_v2", "conversion": "conversion_v2", "hpo": "hpo_v2", "benchmarking": "benchmarking_v2"}`
  - [ ] **Update build_output_path()**: Use `naming/context_tokens.build_token_values()` for token expansion
  - [ ] Call `paths/validation.validate_output_path()` (don't define private `_validate_*`)
  - [ ] Use `paths/config.py` for config loading
  - [ ] Use `core/normalize.py` for path normalization
  - [ ] Use `core/placeholders.py` for pattern parsing
  - [ ] **Do NOT include `resolve_output_path_v2()`** (only in orchestration facade if needed)
  - [ ] Export in `paths/__init__.py` (but NOT resolve_output_path_v2)

- [ ] **Create paths/validation.py**
  - [ ] Move `validate_path_before_mkdir()` from `orchestration/path_resolution.py`
  - [ ] Create `validate_output_path()` public function (called by resolve.py)
  - [ ] Keep filesystem-specific checks only (forbidden chars, length, mkdir safety)
  - [ ] **Do NOT include naming pattern validation** (that's in naming/display_policy.py)
  - [ ] Export in `paths/__init__.py`

- [ ] **Create paths/cache.py**
  - [ ] Move `get_cache_file_path()` from `orchestration/paths.py`
  - [ ] Move `get_timestamped_cache_filename()` from `orchestration/paths.py`
  - [ ] Move `get_cache_strategy_config()` from `orchestration/paths.py`
  - [ ] Move `save_cache_with_dual_strategy()` from `orchestration/paths.py`
  - [ ] Move `load_cache_file()` from `orchestration/paths.py`
  - [ ] Update imports to use `paths/config.py` if needed
  - [ ] Export in `paths/__init__.py`

- [ ] **Create paths/drive.py**
  - [ ] Move `get_drive_backup_base()` from `orchestration/paths.py`
  - [ ] Move `get_drive_backup_path()` from `orchestration/paths.py`
  - [ ] Use `paths/config.py` for Drive config
  - [ ] Export in `paths/__init__.py`

- [ ] **Create paths/parse.py**
  - [ ] Move `parse_hpo_path_v2()` from `orchestration/paths.py`
  - [ ] Move `is_v2_path()` from `orchestration/paths.py`
  - [ ] Move `find_study_by_hash()` from `orchestration/paths.py`
  - [ ] Move `find_trial_by_hash()` from `orchestration/paths.py`
  - [ ] Update to use `naming/context_tokens.py` for token expansion if needed
  - [ ] Export in `paths/__init__.py`

- [ ] **Create paths/__init__.py**
  - [ ] Export all public functions from submodules
  - [ ] Maintain backward compatibility signatures
  - [ ] **Do NOT export `resolve_output_path_v2()`**

- [ ] **Verify paths/ dependencies**
  - [ ] Check all imports - should only depend on `core/` and `naming/context_tokens.py`
  - [ ] Ensure no circular dependencies
  - [ ] Run import tests

### Phase 4: Create Naming Module

- [ ] **Create src/naming/ directory structure**
  - [ ] Create `src/naming/` directory
  - [ ] Create `src/naming/__init__.py`

- [ ] **Create naming/context.py**
  - [ ] Move `NamingContext` dataclass from `orchestration/naming_centralized.py`
  - [ ] Move `create_naming_context()` factory from `naming_centralized.py`
  - [ ] Keep context validation logic in `__post_init__`
  - [ ] Keep it "dumb" (data + lightweight validation only)
  - [ ] Depend on `core/normalize.py` only
  - [ ] Export in `naming/__init__.py`

- [ ] **Create naming/context_tokens.py**
  - [ ] Create `build_token_values(context: NamingContext) -> dict` function
  - [ ] Consolidate ALL token expansion logic:
    - [ ] `spec8 = context.spec_fp[:8] if context.spec_fp else ""`
    - [ ] `exec8 = context.exec_fp[:8] if context.exec_fp else ""`
    - [ ] `study8 = context.study_key_hash[:8] if context.study_key_hash else ""`
    - [ ] `trial8 = context.trial_key_hash[:8] if context.trial_key_hash else ""`
    - [ ] `bench8 = context.benchmark_config_hash[:8] if context.benchmark_config_hash else ""`
    - [ ] `conv8 = context.conv_fp[:8] if context.conv_fp else ""`
    - [ ] `spec_hash = context.spec_fp[:8]` (for names)
    - [ ] `exec_hash = context.exec_fp[:8]` (for names)
    - [ ] `study_hash = context.study_key_hash` (full hash for names)
    - [ ] `trial_hash = context.trial_key_hash` (full hash for names)
    - [ ] `conv_hash = context.conv_fp[:8]` (for names)
    - [ ] `bench_hash = context.benchmark_config_hash` (full hash for names)
  - [ ] Return dict with all token values
  - [ ] Used by both `paths/build_output_path()` and `naming/format_run_name()`
  - [ ] Export in `naming/__init__.py`

- [ ] **Create naming/experiments.py**
  - [ ] Move `get_stage_config()` from `orchestration/naming.py`
  - [ ] **Change signature**: `get_stage_config(experiment_cfg: dict, stage: str) -> dict` (dict-based, NOT ExperimentConfig)
  - [ ] Move `build_aml_experiment_name()` from `orchestration/naming.py`
  - [ ] Move `build_mlflow_experiment_name()` from `orchestration/naming.py`
  - [ ] **Remove dependency on `orchestration.config_loader.ExperimentConfig`**
  - [ ] If ExperimentConfig needed, move minimal loader to `naming/` or `core/`
  - [ ] Export in `naming/__init__.py`

- [ ] **Create naming/display_policy.py**
  - [ ] Move `load_naming_policy()` from `orchestration/jobs/tracking/naming/policy.py`
  - [ ] Move `format_run_name()` from `orchestration/jobs/tracking/naming/policy.py`
  - [ ] Move `parse_parent_training_id()` from `policy.py` if exists
  - [ ] Move `validate_naming_policy()` internal validation from `policy.py`
  - [ ] **Add config caching**: Implement `@lru_cache` with manual mtime check
    - [ ] Cache key: `(config_dir, mtime(naming.yaml))`
    - [ ] Check mtime before returning cached value
  - [ ] Use `core/tokens.py` for token validation
  - [ ] Use `naming/context_tokens.py` for token expansion
  - [ ] Keep validation internal (no separate validation.py module)
  - [ ] Export in `naming/__init__.py`

- [ ] **Handle MLflow naming (Decision-based)**
  - [ ] **If Option A (move to naming/mlflow/)**:
    - [ ] Create `naming/mlflow/` directory
    - [ ] Create `naming/mlflow/__init__.py`
    - [ ] Move `get_naming_config()` and `load_mlflow_config()` to `naming/mlflow/config.py`
    - [ ] Move `run_keys.py` from `orchestration/jobs/tracking/naming/`
    - [ ] Move `run_names.py` and update imports
    - [ ] Move `tags.py` and update imports
    - [ ] Move `hpo_keys.py`
    - [ ] Move `refit_keys.py`
    - [ ] Move `policy.py` → rename to `mlflow_policy.py`
    - [ ] Move `tags_registry.py`
    - [ ] Update all imports to use relative imports within mlflow/
    - [ ] Update imports to use `naming/mlflow/config.py`
  - [ ] **If Option B (facade)**:
    - [ ] Create `naming/mlflow/` directory
    - [ ] Create `naming/mlflow/__init__.py` as thin facade
    - [ ] Re-export functions from `orchestration/jobs/tracking/naming/`
    - [ ] Keep original modules in `orchestration/jobs/tracking/naming/`
  - [ ] Export MLflow functions in `naming/__init__.py`

- [ ] **Create naming/__init__.py**
  - [ ] Export all public functions from submodules
  - [ ] Re-export MLflow naming functions (if moved)
  - [ ] Maintain backward compatibility signatures
  - [ ] **Do NOT export `build_output_path()`** (that's in paths/)

- [ ] **Verify naming/ dependencies**
  - [ ] Check all imports - should depend on `core/` and `paths/` only
  - [ ] Ensure no circular dependencies
  - [ ] Run import tests

### Phase 5: Update Orchestration Facades

- [ ] **Create orchestration/paths.py (legacy facade)**
  - [ ] Re-export all public functions from `paths/`
  - [ ] **Add `resolve_output_path_v2()` wrapper** (only if external callers found in audit)
    - [ ] Thin wrapper: `def resolve_output_path_v2(root_dir, context, base_outputs="outputs"): return paths.build_output_path(root_dir, context, base_outputs)`
  - [ ] Add deprecation warnings: `warnings.warn("Use 'from paths import ...' instead", DeprecationWarning)`
  - [ ] Keep for 1-2 releases before removal

- [ ] **Create orchestration/naming.py (legacy facade)**
  - [ ] Re-export all public functions from `naming/`
  - [ ] Add deprecation warnings: `warnings.warn("Use 'from naming import ...' instead", DeprecationWarning)`
  - [ ] Keep for 1-2 releases before removal

- [ ] **Update orchestration/__init__.py**
  - [ ] Change imports to use `paths` and `naming` modules
  - [ ] Re-export for backward compatibility
  - [ ] Keep existing public API stable
  - [ ] Update `resolve_output_path_v2` import if wrapper created

### Phase 6: Update Internal Imports

- [ ] **Update imports in src/orchestration/jobs/**
  - [ ] Update all files importing from `orchestration.paths` → use `paths` directly
  - [ ] Update all files importing from `orchestration.naming_centralized` → use `naming` directly
  - [ ] Update all files importing from `orchestration.naming` → use `naming` directly
  - [ ] Update all files importing from `orchestration.tokens` → use `core.tokens` or `core` directly
  - [ ] Update all files importing from `orchestration.normalize` → use `core.normalize` or `core` directly
  - [ ] Update MLflow naming imports if moved to `naming/mlflow/`
  - [ ] **Update token expansion logic**: Replace all `hash[:8]` computations with `naming.context_tokens.build_token_values(context)`

- [ ] **Update imports in src/orchestration/ (non-jobs)**
  - [ ] Update `drive_backup.py` imports
  - [ ] Update `metadata_manager.py` imports
  - [ ] Update `final_training_config.py` imports
  - [ ] Update `conversion_config.py` imports
  - [ ] Update `benchmark_utils.py` imports
  - [ ] Update any other files using paths/naming

- [ ] **Update token expansion call sites**
  - [ ] `naming_centralized.py`: Replace `trial8 = context.trial_key_hash[:8]` with `build_token_values()`
  - [ ] `paths.py` parse functions: Use token expansion if needed
  - [ ] `jobs/hpo/local/refit/executor.py`: Replace hash slicing
  - [ ] `jobs/hpo/local_sweeps.py`: Replace hash slicing
  - [ ] `jobs/selection/trial_finder.py`: Replace hash slicing
  - [ ] `jobs/selection/study_summary.py`: Replace hash slicing
  - [ ] `jobs/selection/artifact_acquisition.py`: Replace hash slicing
  - [ ] `jobs/hpo/local/cv/orchestrator.py`: Replace hash slicing

- [ ] **Keep notebooks unchanged**
  - [ ] Verify notebooks continue using `orchestration.*` imports
  - [ ] No changes needed to notebooks (stable API)

### Phase 7: Testing and Verification

- [ ] **Run existing tests**
  - [ ] Run all path resolution tests
  - [ ] Run all naming tests
  - [ ] Run all integration tests
  - [ ] Fix any test failures

- [ ] **Test path resolution in all environments**
  - [ ] Test local environment
  - [ ] Test Colab environment (using env_overrides)
  - [ ] Test Kaggle environment (using env_overrides)
  - [ ] Test AzureML environment (using env_overrides)
  - [ ] Verify env_overrides work correctly

- [ ] **Test naming functions**
  - [ ] Test with various NamingContext configurations
  - [ ] Test experiment name building
  - [ ] Test MLflow run name formatting
  - [ ] Test display policy loading

- [ ] **Verify backward compatibility**
  - [ ] Test that `orchestration.paths.*` imports still work
  - [ ] Test that `orchestration.naming.*` imports still work
  - [ ] Test that `resolve_output_path_v2()` wrapper works (if created)
  - [ ] Test that notebooks work without changes

- [ ] **Verify no circular dependencies**
  - [ ] Run dependency checker
  - [ ] Verify `core/` has no dependencies on `paths/` or `naming/`
  - [ ] Verify `paths/` depends only on `core/` and `naming/context_tokens.py`
  - [ ] Verify `naming/` depends only on `core/` and `paths/`

- [ ] **Verify single authority for paths**
  - [ ] Search codebase for any other `build_output_path()` definitions
  - [ ] Verify only `paths/resolve.py` builds filesystem paths
  - [ ] Verify `naming/` calls `paths/` when it needs directories

- [ ] **Test config caching**
  - [ ] Test that paths.yaml caching works (mtime check)
  - [ ] Test that naming.yaml caching works (mtime check)
  - [ ] Test cache invalidation when files change
  - [ ] Verify performance improvement

- [ ] **Test token expansion**
  - [ ] Test `build_token_values()` with various contexts
  - [ ] Verify no duplicate token expansion logic remains
  - [ ] Search codebase for remaining `hash[:8]` patterns (should be none)

- [ ] **Test each module independently (SRP validation)**
  - [ ] Test `core/` modules in isolation
  - [ ] Test `paths/` modules in isolation
  - [ ] Test `naming/` modules in isolation

### Phase 8: Documentation and Cleanup

- [ ] **Document new import patterns**
  - [ ] Update README or docs with new import examples
  - [ ] Document deprecation timeline for `orchestration.*` imports
  - [ ] Document migration path for internal code

- [ ] **Code cleanup**
  - [ ] Remove unused imports
  - [ ] Fix any linter warnings
  - [ ] Ensure consistent code style

- [ ] **Final verification**
  - [ ] Run full test suite
  - [ ] Verify no regressions
  - [ ] Check code coverage

### Phase 9: Future Cleanup (After 1-2 Releases)

- [ ] **Remove old modules** (after deprecation period)
  - [ ] Remove `src/orchestration/paths.py` (replaced by facade)
  - [ ] Remove `src/orchestration/path_resolution.py` (moved to paths/validation.py)
  - [ ] Remove `src/orchestration/naming.py` (replaced by facade)
  - [ ] Remove `src/orchestration/naming_centralized.py` (moved to naming/)
  - [ ] Remove `src/orchestration/tokens.py` (moved to core/)
  - [ ] Remove `src/orchestration/normalize.py` (moved to core/)
  - [ ] Consider removing `orchestration/jobs/tracking/naming/` if fully migrated to `naming/mlflow/`

## Quick Reference: File Mapping

### Core Module
- `orchestration/tokens.py` → `core/tokens.py` (minus extract_placeholders)
- `orchestration/tokens.py::extract_placeholders()` → `core/placeholders.py`
- `orchestration/normalize.py` → `core/normalize.py`

### Paths Module
- `orchestration/paths.py::load_paths_config()` → `paths/config.py`
- `orchestration/paths.py::apply_env_overrides()` → `paths/config.py`
- `orchestration/paths.py::validate_paths_config()` → `paths/config.py`
- `orchestration/paths.py::resolve_output_path()` → `paths/resolve.py`
- `orchestration/naming_centralized.py::build_output_path()` → `paths/resolve.py`
- `orchestration/path_resolution.py::validate_path_before_mkdir()` → `paths/validation.py`
- `orchestration/paths.py::get_cache_file_path()` → `paths/cache.py`
- `orchestration/paths.py::get_drive_backup_base()` → `paths/drive.py`
- `orchestration/paths.py::parse_hpo_path_v2()` → `paths/parse.py`

### Naming Module
- `orchestration/naming_centralized.py::NamingContext` → `naming/context.py`
- `orchestration/naming_centralized.py::create_naming_context()` → `naming/context.py`
- `orchestration/naming.py::get_stage_config()` → `naming/experiments.py` (dict-based)
- `orchestration/jobs/tracking/naming/policy.py::load_naming_policy()` → `naming/display_policy.py`
- `orchestration/jobs/tracking/naming/policy.py::format_run_name()` → `naming/display_policy.py`
- `orchestration/jobs/tracking/naming/*` → `naming/mlflow/*` (if Option A)

## Notes

- All token expansion logic (`hash[:8]`) should be consolidated into `naming/context_tokens.py`
- `PROCESS_PATTERN_KEYS` constant needs to be created in `paths/resolve.py`
- `extract_placeholders()` is in `tokens.py`, not `paths.py`
- MLflow naming location decision (Option A vs B) should be made in Phase 1
- Keep `orchestration.*` imports working for backward compatibility
