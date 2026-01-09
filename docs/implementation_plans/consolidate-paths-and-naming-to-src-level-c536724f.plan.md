<!-- c536724f-85bf-4dcd-be80-096c3f6c651b 066fbcc5-22f5-446c-8482-9bbf5debde29 -->
# Consolidate Paths and Naming (Final Optimized Plan)

## Goal

Create single source of truth for paths and naming at `src/` level, organized into focused modules following SRP. **Paths owns all filesystem layout; naming owns display/run names.** Keep `orchestration.*` as stable public API for notebooks. Includes optimizations: remove redundancies, add config caching, centralize mappings, avoid circular dependencies.

## Current State Analysis

### Path-related modules (to consolidate):

- `src/orchestration/paths.py` - Main path resolution, cache management, Drive backup
- `src/orchestration/path_resolution.py` - Path validation, Colab-specific resolution
- Path parsing utilities in `paths.py` (parse_hpo_path_v2, is_v2_path)

### Naming-related modules (to consolidate):

- `src/orchestration/naming.py` - Simple naming (get_stage_config, build_aml_experiment_name, build_mlflow_experiment_name)
- `src/orchestration/naming_centralized.py` - Complex naming (NamingContext, build_output_path, create_naming_context)
- `src/orchestration/jobs/tracking/naming/` - MLflow-specific naming (run keys, run names, tags, hashing)

### Shared utilities (to extract):

- `src/orchestration/tokens.py` - Token validation
- `src/orchestration/normalize.py` - Path normalization
- Placeholder extraction (currently in paths.py)

## Proposed Structure (Final Optimized)

```
src/
├── core/                      # Shared utilities (no circular deps)
│   ├── __init__.py
│   ├── tokens.py              # Token validation and checking
│   ├── normalize.py           # Path/naming normalization
│   └── placeholders.py        # extract_placeholders() utility
│
├── paths/                     # Filesystem path management (single authority)
│   ├── __init__.py            # Public API exports
│   ├── config.py              # Load + apply env_overrides + validate schema (cached)
│   ├── resolve.py             # resolve_output_path + build_output_path (v2 only)
│   ├── validation.py          # Filesystem/path safety validation
│   ├── cache.py               # Cache file path management
│   ├── drive.py               # Drive backup path mapping
│   └── parse.py               # Path parsing/detection helpers
│
├── naming/                    # Display names and MLflow naming
│   ├── __init__.py            # Public API exports
│   ├── context.py             # NamingContext dataclass + factory
│   ├── context_tokens.py      # Expand NamingContext into token dict
│   ├── experiments.py         # Experiment/stage naming helpers (dict-based)
│   ├── display_policy.py      # naming.yaml parsing for display names (cached)
│   └── mlflow/                # MLflow-specific naming (if standalone)
│       ├── __init__.py
│       ├── run_keys.py
│       ├── run_names.py
│       ├── tags.py
│       ├── hpo_keys.py
│       ├── refit_keys.py
│       ├── mlflow_policy.py   # Renamed from policy.py
│       └── tags_registry.py
│
└── orchestration/             # Legacy facade (backward compatibility)
    ├── __init__.py
    ├── paths.py               # Re-export from paths/* (with resolve_output_path_v2 wrapper)
    └── naming.py              # Re-export from naming/*
```

## Key Design Principles

### Single Authority Rule

- **paths/** owns ALL filesystem layout: `build_output_path()`, `resolve_output_path()`, patterns, env overrides, absolute/relative semantics, path normalization, parsing
- **naming/** owns display/run names: experiment names, MLflow run keys/tags/policy, NamingContext (shared data object)
- **naming/** calls into **paths/** when it needs directories (no duplicate path building)

### No Redundancy

- Only ONE place builds output dirs: `paths/resolve.py`
- Only ONE v2 entrypoint: `paths.build_output_path()` (no `resolve_output_path_v2()` in paths/)
- Only ONE validation per concern: `paths/validation.py` (filesystem), validation internal to `naming/display_policy.py` (naming patterns)
- Only ONE pattern key mapping: `PROCESS_PATTERN_KEYS` constant in `paths/resolve.py`
- Only ONE YAML loader per file: `paths/config.py` and `naming/display_policy.py` with caching

### Dependency Direction

- `core/` has no dependencies on `paths/` or `naming/`
- `paths/` depends on `core/` only
- `naming/` depends on `core/` and `paths/` (calls paths for directories)
- `orchestration/` depends on `paths/` and `naming/` (re-exports)
- **naming/experiments.py does NOT depend on orchestration.config_loader** (uses dict interface)

## Implementation Steps

### Step 1: Create src/core/ module (shared utilities)

1. **core/tokens.py** - Responsibility: Token validation and checking

   - Move from `orchestration/tokens.py`
   - `is_token_known()`, `is_token_allowed()`, token validation logic
   - No dependencies on paths or naming

2. **core/normalize.py** - Responsibility: Path/naming normalization

   - Move from `orchestration/normalize.py`
   - `normalize_for_path()`, normalization rules
   - No dependencies on paths or naming

3. **core/placeholders.py** - Responsibility: Placeholder extraction

   - Extract `extract_placeholders()` from paths.py
   - Utility for parsing {placeholder} patterns
   - No dependencies on paths or naming

4. **core/init.py** - Public API

   - Export all public functions

### Step 2: Create src/paths/ module (filesystem authority)

1. **paths/config.py** - Responsibility: Load and manage paths.yaml configuration

   - `load_paths_config()` - Load config from file (cached by config_dir + storage_env + mtime)
   - `apply_env_overrides()` - Apply environment-specific overrides (replaces environment.py)
   - `validate_paths_config()` - Validate config schema (enforces PROCESS_PATTERN_KEYS for v2)
   - `_get_default_paths()` - Default config fallback
   - **Config caching**: Cache by `(config_dir, storage_env, mtime(paths.yaml))` using functools.lru_cache with manual mtime check

2. **paths/resolve.py** - Responsibility: Resolve all output paths (single authority)

   - `resolve_output_path()` - Legacy path resolution for named buckets (hpo, cache, etc.)
   - `build_output_path()` - V2 path resolution using NamingContext - **ONLY v2 entrypoint**
   - `_build_output_path_fallback()` - Fallback path building logic
   - `PROCESS_PATTERN_KEYS` - Centralized constant: `{"final_training": "final_training_v2", "conversion": "conversion_v2", "hpo": "hpo_v2", "benchmarking": "benchmarking_v2"}`
   - Calls `paths/validation.validate_output_path()` (never defines private `_validate_*`)
   - Uses `paths/config.py` for effective config
   - Uses `core/normalize.py` for path normalization
   - Uses `core/placeholders.py` for pattern parsing
   - Uses `naming/context_tokens.py` for token expansion

3. **paths/validation.py** - Responsibility: Filesystem/path safety validation

   - `validate_path_before_mkdir()` - Validate paths before directory creation
   - `validate_output_path()` - Public validation function (called by resolve.py)
   - Filesystem-specific checks (forbidden chars, length, mkdir safety)
   - NOT naming pattern validation (that's internal to naming/display_policy.py)

4. **paths/cache.py** - Responsibility: Cache file path management

   - `get_cache_file_path()` - Get cache file paths
   - `get_timestamped_cache_filename()` - Generate timestamped filenames
   - `get_cache_strategy_config()` - Get cache strategy config
   - `save_cache_with_dual_strategy()` - Save with dual strategy
   - `load_cache_file()` - Load cache files

5. **paths/drive.py** - Responsibility: Drive backup path mapping

   - `get_drive_backup_base()` - Get Drive backup base directory
   - `get_drive_backup_path()` - Convert local to Drive path
   - Uses `paths/config.py` for Drive config

6. **paths/parse.py** - Responsibility: Path parsing and detection

   - `parse_hpo_path_v2()` - Parse HPO v2 paths
   - `is_v2_path()` - Detect v2 path pattern
   - `find_study_by_hash()` - Find study by hash
   - `find_trial_by_hash()` - Find trial by hash

7. **paths/init.py** - Public API

   - Export all public functions from submodules
   - Maintain backward compatibility signatures
   - **Do NOT export `resolve_output_path_v2()`** (only in legacy facade if needed)

### Step 3: Create src/naming/ module (display names authority)

1. **naming/context.py** - Responsibility: NamingContext dataclass and factory

   - `NamingContext` dataclass definition (moved from naming_centralized.py)
   - `create_naming_context()` - Factory function with auto-detection
   - Context validation logic
   - Keep it "dumb" (data + lightweight validation)
   - Depends on `core/normalize.py` only

2. **naming/context_tokens.py** - Responsibility: Expand NamingContext into token dict

   - `build_token_values(context: NamingContext) -> dict` - Expand context into tokens (spec8, trial8, study8, etc.)
   - Used by both `paths/build_output_path()` and `naming/format_run_name()`
   - Prevents duplicate "spec8 = spec_fp[:8]" logic scattered across modules
   - Single place for all derived fields

3. **naming/experiments.py** - Responsibility: Experiment and stage naming

   - `get_stage_config(experiment_cfg: dict, stage: str) -> dict` - Get stage configuration (dict-based interface)
   - `build_aml_experiment_name()` - Build AML experiment names
   - `build_mlflow_experiment_name()` - Build MLflow experiment names
   - **Avoid dependency on `orchestration.config_loader`** - use dict interface instead
   - If ExperimentConfig is needed, move minimal loader to `naming/` or `core/`

4. **naming/display_policy.py** - Responsibility: Naming policy loading and display name formatting

   - `load_naming_policy(..., validate=True)` - Load naming.yaml with validation (cached by config_dir + mtime)
   - `format_run_name()` - Format run names according to policy
   - `parse_parent_training_id()` - Parse parent training IDs
   - Internal validation helpers (naming pattern validation, MLflow-safe, token validation)
   - Uses `core/tokens.py` for token validation
   - Uses `naming/context_tokens.py` for token expansion
   - **Config caching**: Cache by `(config_dir, mtime(naming.yaml))` using functools.lru_cache with manual mtime check
   - **Note**: No separate `naming/validation.py` - validation is internal to policy loading

5. **naming/mlflow/** - Responsibility: MLflow-specific naming (if standalone)

   - **Decision point**: Check if MLflow naming depends on `orchestration.jobs.tracking.config`
   - **Option A**: If dependencies are minimal, move to `naming/mlflow/` and move minimal config with it
   - **Option B**: If dependencies are heavy, keep under `orchestration/jobs/tracking/naming/` and expose thin facade in `naming/`
   - Move from `orchestration/jobs/tracking/naming/*`:
     - `run_keys.py` - MLflow run key building
     - `run_names.py` - MLflow run name building
     - `tags.py` - MLflow tag building
     - `hpo_keys.py` - HPO key building
     - `refit_keys.py` - Refit key building
     - `mlflow_policy.py` - MLflow naming policy (renamed from policy.py to avoid collision with display_policy.py)
     - `tags_registry.py` - Tags registry management
   - Update imports to use relative imports within mlflow/

6. **naming/init.py** - Public API

   - Export all public functions from submodules
   - Re-export MLflow naming functions (if moved)
   - Maintain backward compatibility signatures
   - **Note**: Does NOT export `build_output_path()` - that's in paths/

### Step 4: Update orchestration/ for backward compatibility

1. **orchestration/paths.py** - Legacy facade

   - Re-export all public functions from `paths/`
   - **Add `resolve_output_path_v2()` as thin wrapper** (if needed for backward compat, only here)
   - Add deprecation warnings: "Use 'from paths import ...' instead"
   - Keep for 1-2 releases before removal

2. **orchestration/naming.py** - Legacy facade

   - Re-export all public functions from `naming/`
   - Add deprecation warnings: "Use 'from naming import ...' instead"
   - Keep for 1-2 releases before removal

3. **orchestration/init.py** - Update exports

   - Change imports to use `paths` and `naming` modules
   - Re-export for backward compatibility
   - Keep existing public API stable

### Step 5: Update all internal imports

1. Update files in `src/orchestration/jobs/` to import from `paths` and `naming`
2. Update files in `src/orchestration/` that use paths/naming
3. Update imports to use new module structure
4. **Keep notebooks using `orchestration.*` imports** (stable API, no changes needed)

### Step 6: Handle MLflow naming dependencies

1. Analyze dependencies of `orchestration/jobs/tracking/naming/` modules
2. Check if they depend on `orchestration.jobs.tracking.config`
3. **Decision**:

   - If minimal dependencies: Move to `naming/mlflow/` and move minimal config with it
   - If heavy dependencies: Keep under `tracking/` and expose thin facade in `naming/`

4. Ensure no circular dependencies

### Step 7: Remove redundancies and optimize

1. **Remove `resolve_output_path_v2()` from paths/**:

   - Keep only `build_output_path()` as v2 entrypoint
   - If backward compat needed, add thin wrapper only in `orchestration/paths.py`

2. **Consolidate validation**:

   - Ensure `validate_output_path()` exists only in `paths/validation.py` (public function)
   - Ensure naming validation is internal to `naming/display_policy.py` (no separate validation.py)
   - No duplicate validation functions

3. **Centralize pattern key mapping**:

   - Ensure `PROCESS_PATTERN_KEYS` constant exists only in `paths/resolve.py`
   - Use in `paths/config.py` validator to enforce keys exist for schema v2

4. **Add config caching**:

   - `paths/config.py`: Cache by `(config_dir, storage_env, mtime(paths.yaml))`
   - `naming/display_policy.py`: Cache by `(config_dir, mtime(naming.yaml))`
   - Use functools.lru_cache with manual mtime check

5. **Add token expansion helper**:

   - Create `naming/context_tokens.py` with `build_token_values(context) -> dict`
   - Use in both `paths/build_output_path()` and `naming/format_run_name()`
   - Prevents duplicate "spec8 = spec_fp[:8]" logic

6. **Avoid naming/ depending on orchestration.config_loader**:

   - Use dict-based interface in `naming/experiments.py`
   - Move minimal config loader to `naming/` or `core/` if needed

### Step 8: Cleanup (after 1-2 releases)

1. Remove old modules:

   - `src/orchestration/paths.py` (after deprecation period)
   - `src/orchestration/path_resolution.py`
   - `src/orchestration/naming.py`
   - `src/orchestration/naming_centralized.py`
   - `src/orchestration/tokens.py` (moved to core/)
   - `src/orchestration/normalize.py` (moved to core/)

2. Consider removing `orchestration/jobs/tracking/naming/` if fully migrated to `naming/mlflow/`

## Key Design Decisions

### Single Authority Principle

- **paths/** is the ONLY authority for filesystem layout
- **naming/** calls into **paths/** when it needs directories
- No duplicate path building logic

### No Environment-Specific Modules

- Remove `paths/environment.py` - use `env_overrides` in config instead
- All environment handling via `paths/config.py` applying overrides
- No "if colab then..." branching logic

### Shared Core Utilities

- `core/` provides shared utilities with no circular dependencies
- Both `paths/` and `naming/` depend on `core/`
- Prevents duplication and circular dependencies

### Stable Notebook API

- Keep `orchestration.*` as stable public API for notebooks
- Notebooks continue using `orchestration.paths` and `orchestration.naming`
- Internal code can migrate to `paths.*`/`naming.*` gradually
- No breaking changes for notebooks

### Validation Separation

- `paths/validation.py` = filesystem safety (mkdir, forbidden chars, length) - public function
- Naming validation = internal to `naming/display_policy.py` (no separate module)
- Clear separation of concerns

### Config Caching

- Both `paths/config.py` and `naming/display_policy.py` cache by config_dir + mtime
- Big runtime win, prevents duplicate YAML loads

### Token Expansion

- Single `naming/context_tokens.py` expands NamingContext into token dict
- Used by both paths and naming, prevents duplicate logic

### Avoid Circular Dependencies

- `naming/experiments.py` uses dict interface, not `orchestration.config_loader`
- Move minimal config loader if needed, don't depend on orchestration internals

## Testing Strategy

1. Run existing tests to ensure no regressions
2. Test path resolution in all environments (local, Colab, Kaggle) using env_overrides
3. Test naming functions with various contexts
4. Verify notebook imports work correctly (using orchestration.*)
5. Check backward compatibility imports still work
6. Test each module independently (SRP validation)
7. Verify no circular dependencies
8. Test that only paths/ builds filesystem paths
9. Test config caching (verify mtime checks work)
10. Test token expansion (verify no duplicate logic)

## Migration Checklist

- [ ] Create `src/core/` module structure (tokens, normalize, placeholders)
- [ ] Create `src/paths/` module structure (config with caching, resolve with PROCESS_PATTERN_KEYS, validation, cache, drive, parse)
- [ ] Move `build_output_path()` from naming_centralized.py to paths/resolve.py
- [ ] Create `naming/context_tokens.py` for token expansion
- [ ] Create `src/naming/` module structure (context, context_tokens, experiments with dict interface, display_policy with caching)
- [ ] Rename naming/policy.py to naming/display_policy.py
- [ ] Rename naming/mlflow/policy.py to naming/mlflow/mlflow_policy.py
- [ ] Decide on MLflow naming location (mlflow/ vs tracking/)
- [ ] Update `orchestration/__init__.py` and create legacy facades
- [ ] Add resolve_output_path_v2 wrapper only in orchestration/paths.py (if needed)
- [ ] Update all internal imports in `src/orchestration/`
- [ ] Keep notebooks using `orchestration.*` imports (no changes needed)
- [ ] Test all path resolution scenarios
- [ ] Test all naming scenarios
- [ ] Verify backward compatibility
- [ ] Verify no circular dependencies
- [ ] Verify single authority for path building
- [ ] Test config caching
- [ ] Test token expansion
- [ ] Document new import patterns
- [ ] (After 1-2 releases) Remove old modules

### To-dos

- [ ] Create src/core/ directory structure with __init__.py and submodules (tokens, normalize, placeholders)
- [ ] Move orchestration/tokens.py to core/tokens.py with is_token_known, is_token_allowed functions
- [ ] Move orchestration/normalize.py to core/normalize.py with normalize_for_path function
- [ ] Create core/placeholders.py with extract_placeholders() extracted from paths.py
- [ ] Create core/__init__.py to export all public functions
- [ ] Create src/paths/ directory structure with __init__.py and all submodules
- [ ] Create paths/config.py with load_paths_config (cached by config_dir + storage_env + mtime), apply_env_overrides, validate_paths_config (enforces PROCESS_PATTERN_KEYS)
- [ ] Create paths/resolve.py with resolve_output_path, build_output_path (ONLY v2 entrypoint), PROCESS_PATTERN_KEYS constant, calls paths/validation.validate_output_path()
- [ ] Create paths/validation.py with validate_path_before_mkdir and validate_output_path (public function, filesystem safety only)
- [ ] Create paths/cache.py with cache file path management functions
- [ ] Create paths/drive.py with Drive backup path mapping functions
- [ ] Create paths/parse.py with path parsing and detection helper functions
- [ ] Create paths/__init__.py to export all public functions (NOT resolve_output_path_v2)
- [ ] Create src/naming/ directory structure with __init__.py and all submodules (context, context_tokens, experiments, display_policy, mlflow/)
- [ ] Create naming/context.py with NamingContext dataclass and create_naming_context factory (moved from naming_centralized.py, keep it dumb)
- [ ] Create naming/context_tokens.py with build_token_values(context) -> dict to expand context into tokens (spec8, trial8, etc.)
- [ ] Create naming/experiments.py with get_stage_config(experiment_cfg: dict, stage: str) using dict interface (NOT orchestration.config_loader)
- [ ] Create naming/display_policy.py (renamed from policy.py) with load_naming_policy (cached by config_dir + mtime), format_run_name, internal validation helpers
- [ ] Analyze MLflow naming dependencies and decide: move to naming/mlflow/ (if standalone) or keep in tracking/ with facade
- [ ] Move orchestration/jobs/tracking/naming/ to naming/mlflow/ (if decided) with mlflow_policy.py renamed, or create facade in naming/
- [ ] Create naming/__init__.py to export all public functions (NOT build_output_path - that is in paths/)
- [ ] Update paths/resolve.py to use naming/context_tokens.py for token expansion
- [ ] Create orchestration/paths.py and orchestration/naming.py as legacy facades re-exporting from paths/ and naming/, add resolve_output_path_v2 wrapper only here if needed
- [ ] Update orchestration/__init__.py to import from paths and naming modules, re-exporting for backward compatibility
- [ ] Update all imports in src/orchestration/ and src/orchestration/jobs/ to use paths and naming modules directly
- [ ] Run tests and verify path resolution, naming functions, backward compatibility, no circular deps, single authority for paths, config caching, token expansion