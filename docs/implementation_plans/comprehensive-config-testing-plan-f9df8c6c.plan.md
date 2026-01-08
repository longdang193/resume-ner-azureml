<!-- f9df8c6c-c8e9-4b37-a874-fd51f37c50f2 8ac8a164-caca-4d8d-adaa-3922d3037500 -->
# Comprehensive Configuration Testing Plan

## Overview

This plan creates a comprehensive test suite covering all path resolution, naming, path building, tag scenarios, and MLflow configuration for all processes (HPO, benchmarking, final_training, conversion, best_configurations) using the four configuration files: `mlflow.yaml`, `naming.yaml`, `paths.yaml`, and `tags.yaml`. This includes all configuration options, edge cases, and integration scenarios.

## Test Structure

### 1. Path Resolution Tests (`test_paths_comprehensive.py`)

#### 1.1 Configuration Loading

- **Test `load_paths_config()` scenarios:**
- Load from existing `paths.yaml` file
- Fallback to defaults when file missing
- Schema version validation (v1, v2+)
- Required pattern keys validation for v2+
- Placeholder validation (known/allowed tokens)
- Invalid config handling (missing base.outputs, invalid schema_version)

#### 1.2 Environment Overrides

- **Test `apply_env_overrides()` scenarios:**
- Colab environment override (`/content/drive/MyDrive/...`)
- Azure ML environment override (`/mnt/outputs`)
- Kaggle environment override (`/kaggle/working/outputs`)
- Local (no override)
- Shallow merge behavior (only base, outputs sections)
- Missing override (returns original config)

#### 1.3 Output Path Resolution

- **Test `resolve_output_path()` scenarios:**
- Simple paths: `hpo`, `benchmarking`, `final_training`, `conversion`
- Cache subdirectories: `cache` with `subcategory` parameter
- Pattern replacement: `final_training` with `backbone`, `run_id`
- Absolute vs relative base paths
- All output categories from config

#### 1.4 V2 Path Building

- **Test `build_output_path()` scenarios for each process type:**
- **HPO:**
- With study_key_hash and trial_key_hash (v2 pattern)
- Without hashes (fallback to legacy)
- With storage_env override
- **HPO Refit:**
- With hashes (v2 pattern with `/refit` suffix)
- Without hashes (fallback)
- **Benchmarking:**
- With study_key_hash, trial_key_hash, benchmark_config_hash
- Without bench hash (optional)
- Without study/trial hashes (fallback)
- **Final Training:**
- With spec_fp, exec_fp, variant=1
- Multiple variants (v1, v2, v3)
- Path normalization (special characters)
- **Conversion:**
- With parent_training_id and conv_fp
- Path parsing from parent_training_id
- **Best Configurations:**
- With spec_fp
- Cache directory structure

#### 1.5 Cache File Paths

- **Test `get_cache_file_path()` scenarios:**
- Latest file path (`file_type="latest"`)
- Index file path (`file_type="index"`)
- Specific filename override
- All cache types: `best_configurations`, `final_training`, `best_model_selection`

#### 1.6 Cache Strategy Operations

- **Test `save_cache_with_dual_strategy()` scenarios:**
- Creates timestamped file
- Updates latest pointer file
- Updates index file
- Includes cache_metadata
- All cache types
- Max entries limit enforcement

- **Test `load_cache_file()` scenarios:**
- Load latest (`use_latest=True`)
- Load by specific timestamp
- Load by specific identifier (from index)
- Returns None when not found
- Fallback to index when latest missing

#### 1.7 Timestamped Cache Filenames

- **Test `get_timestamped_cache_filename()` scenarios:**
- Best config pattern: `best_config_{backbone}_{trial}_{timestamp}.json`
- Final training pattern: `final_training_{backbone}_{run_id}_{timestamp}.json`
- Best model selection pattern
- Path normalization applied
- All placeholders replaced correctly

#### 1.8 Drive Backup Paths

- **Test `get_drive_backup_path()` scenarios:**
- Converts local output path to Drive path
- Mirrors exact directory structure
- Returns None for paths outside `outputs/`
- Returns None when Drive not configured
- Handles all output categories

#### 1.9 Path Parsing and Detection

- **Test `parse_hpo_path_v2()` scenarios:**
- Extracts study8, trial8, storage_env, model from v2 paths
- Returns None for non-v2 paths
- Handles full paths and relative fragments

- **Test `is_v2_path()` scenarios:**
- Detects v2 pattern (`study-{hash}/trial-{hash}`)
- Returns False for legacy paths

- **Test `find_study_by_hash()` and `find_trial_by_hash()` scenarios:**
- Finds study folder by study_key_hash
- Finds trial folder by study_key_hash + trial_key_hash
- Returns None when not found
- Searches across all storage_env directories

### 2. Naming Tests (`test_naming_comprehensive.py`)

#### 2.1 NamingContext Validation

- **Test `NamingContext` validation for all process types:**
- Valid contexts for: `hpo`, `hpo_refit`, `benchmarking`, `final_training`, `conversion`, `best_configurations`
- Invalid process_type raises ValueError
- Invalid environment raises ValueError
- Invalid variant (< 1) raises ValueError
- `final_training` requires spec_fp and exec_fp
- `conversion` requires parent_training_id and conv_fp
- `best_configurations` requires spec_fp
- `storage_env` defaults to `environment` if not provided

#### 2.2 Context Creation

- **Test `create_naming_context()` scenarios:**
- Auto-detects environment when None
- Defaults storage_env to environment
- Handles all process types
- Preserves all optional fields

#### 2.3 Run Name Building

- **Test `build_mlflow_run_name()` scenarios for all process types:**
- **HPO Trial:**
- Pattern: `{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}{version}`
- With study_key_hash and trial_number
- Without study_key_hash (raises error or fallback)
- With fold_idx (hpo_trial_fold pattern)
- Auto-increment version suffix
- **HPO Sweep:**
- Pattern: `{env}_{model}_hpo_study-{study_hash}{semantic_suffix}{version}`
- With semantic_suffix from study_name
- Without semantic_suffix
- Auto-increment version suffix
- **HPO Refit:**
- Pattern: `{env}_{model}_hpo_refit_study-{study_hash}_trial-{trial_hash}_t{trial_number}{version}`
- With all hashes and trial_number
- Auto-increment version suffix
- **Final Training:**
- Pattern: `{env}_{model}_final_training_spec-{spec_hash}_exec-{exec_hash}_v{variant}{version}`
- With spec_fp, exec_fp, variant
- Shortened fingerprints in name
- No auto-increment (uses variant)
- **Benchmarking:**
- Pattern: `{env}_{model}_benchmark_study-{study_hash}_trial-{trial_hash}_bench-{bench_hash}{version}`
- With all hashes
- Without bench_hash (optional)
- Auto-increment version suffix
- **Conversion:**
- Pattern: `{env}_{model}_conversion_spec-{spec_hash}_exec-{exec_hash}_v{variant}_conv-{conv_hash}{version}`
- Parses parent_training_id for spec/exec/variant
- With conv_fp

#### 2.4 Run Name Policy

- **Test policy-driven formatting:**
- Loads patterns from `naming.yaml`
- Component extraction (study_hash, trial_number, etc.)
- Separator policy (field: `_`, component: `-`, version: `_`)
- Normalization rules (env, model)
- Validation (max_length, forbidden_chars)
- Fallback to legacy when policy unavailable

#### 2.5 Auto-Increment Versioning

- **Test version suffix scenarios:**
- Enabled for HPO processes (hpo, benchmarking)
- Disabled for final_training, conversion
- Format: `{separator}{number}` (e.g., `_1`, `_2`)
- Atomic version reservation (prevents race conditions)
- Counter key generation

### 3. Path Building Tests (`test_path_building_comprehensive.py`)

#### 3.1 HPO Paths

- **Test HPO v2 path structure:**
- `outputs/hpo/{storage_env}/{model}/study-{study8}/trial-{trial8}/`
- With all required hashes
- Without hashes (fallback to legacy: `trial_{n}_{run_id}`)
- All storage environments (local, colab, kaggle, azureml)

#### 3.2 HPO Refit Paths

- **Test HPO refit path structure:**
- `outputs/hpo/{storage_env}/{model}/study-{study8}/trial-{trial8}/refit/`
- Appends `/refit` subdirectory
- Inherits study/trial structure from parent

#### 3.3 Benchmarking Paths

- **Test benchmarking v2 path structure:**
- `outputs/benchmarking/{storage_env}/{model}/study-{study8}/trial-{trial8}/bench-{bench8}/`
- With benchmark_config_hash
- Without bench_hash (optional, no `/bench-{bench8}`)
- Fallback to legacy when hashes missing

#### 3.4 Final Training Paths

- **Test final training path structure:**
- `outputs/final_training/{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}/`
- All variants (v1, v2, v3, ...)
- Path normalization (special characters in model/env)
- Full fingerprints vs short hashes

#### 3.5 Conversion Paths

- **Test conversion path structure:**
- `outputs/conversion/{storage_env}/{model}/{parent_training_id}/conv-{conv8}/`
- Parses parent_training_id (spec-{spec8}_exec-{exec8}/v{variant})
- Multiple conversions per training run

#### 3.6 Best Configurations Paths

- **Test best config path structure:**
- `outputs/cache/best_configurations/{model}/spec-{spec8}/`
- Special cache directory handling
- Model-specific organization

#### 3.7 Path Normalization

- **Test path normalization scenarios:**
- Special character replacement (`/`, `\`, `-`,  ``, `:`, `*`, `?`, `"`, `<`, `>`, `|`)
- Max component length (255 chars)
- Max path length (260 chars)
- Case preservation (lowercase: false)

### 4. Tag Tests (`test_tags_comprehensive.py`)

#### 4.1 Tags Registry

- **Test `load_tags_registry()` scenarios:**
- Loads from `tags.yaml` file
- Falls back to defaults when file missing
- Merges loaded data with defaults (loaded takes precedence)
- Module-level caching (returns same instance)
- Schema version handling (defaults to 0 if missing)
- Required keys validation (grouping.study_key_hash, etc.)

- **Test `TagsRegistry.key()` scenarios:**
- Access all tag keys from all sections:
- `grouping`: study_key_hash, trial_key_hash, parent_run_id, etc.
- `process`: stage, project, backbone, model, env, etc.
- `training`: trained_on_full_data, refit, etc.
- `hpo`: trial_number, best_trial_run_id, etc.
- `lineage`: source, hpo_study_key_hash, etc.
- `paths`: spec_fp, exec_fp, conv_fp, etc.
- `azureml`: run_type, sweep
- `mlflow`: run_type, parent_run_id
- `legacy`: trial_number, trial_id, variant, etc.
- Raises `TagKeyError` for missing keys
- Handles invalid section types
- Handles invalid key value types

#### 4.2 Tag Building

- **Test `build_mlflow_tags()` scenarios for all process types:**
- **Minimal tags (always set):**
- `code.stage` (process_type or "unknown")
- `code.model` (model or "unknown")
- `code.env` (environment or detected)
- `code.storage_env` (storage_env or environment)
- `code.execution_platform` (environment)
- `code.created_by` (user/system)
- `code.project` (from config or default)

- **HPO process tags:**
- `code.stage` = "hpo" or "hpo_sweep" or "hpo_trial"
- `code.hpo.trial_number` (if trial_number provided)
- `code.study_key_hash` (if provided)
- `code.trial_key_hash` (if provided)
- `code.grouping_schema_version` = "1" (if grouping tags present)

- **HPO Refit process tags:**
- `code.stage` = "hpo_refit"
- `code.run_type` = "refit"
- All HPO tags plus refit-specific

- **Benchmarking process tags:**
- `code.stage` = "benchmarking"
- `code.study_key_hash`, `code.trial_key_hash`
- `code.benchmark_config_hash` (if provided)
- `code.grouping_schema_version` = "1"

- **Final Training process tags:**
- `code.stage` = "final_training"
- `code.spec_fp` (specification fingerprint)
- `code.exec_fp` (execution fingerprint)
- `code.variant` (variant number)
- `code.output_dir` (if output_dir provided)

- **Conversion process tags:**
- `code.stage` = "conversion"
- `code.parent_training_id` (parsed from context)
- `code.conv_fp` (conversion fingerprint)
- `code.lineage.parent_training_run_id` (if parent provided)

- **Optional tags (set when provided):**
- `code.parent_run_id` (parent MLflow run)
- `code.group_id` (group identifier)
- `code.refit_protocol_fp` (refit protocol fingerprint)
- `code.run_key_hash` (run key hash)

#### 4.3 Tag Sanitization

- **Test `sanitize_tag_value()` scenarios:**
- Truncates values exceeding max_length (default 250)
- Adds "..." indicator when truncated
- Preserves values within limit
- Handles empty strings
- Uses max_length from config (mlflow.yaml)

#### 4.4 Tag Key Resolution

- **Test `get_tag_key()` scenarios:**
- Loads from registry (tags.yaml)
- Falls back to provided fallback constant
- Raises TagKeyError when key missing and no fallback
- Handles registry loading failures gracefully

### 5. Integration Tests (`test_config_integration.py`)

#### 5.1 End-to-End Scenarios

- **Test complete workflows:**
- HPO trial: context → path → name → tags
- HPO refit: context → path → name → tags
- Final training: context → path → name → tags
- Benchmarking: context → path → name → tags
- Conversion: context → path → name → tags

#### 5.2 Configuration Consistency

- **Test config file interactions:**
- Paths from `paths.yaml` match patterns in `naming.yaml`
- Tag keys from `tags.yaml` used in `build_mlflow_tags()`
- Naming patterns from `naming.yaml` used in `build_mlflow_run_name()`
- MLflow config from `mlflow.yaml` affects tracking behavior

#### 5.3 Cross-Process Consistency

- **Test consistency across processes:**
- Same model/environment produces consistent paths
- Tag keys consistent across all processes
- Naming conventions consistent (separators, normalization)

## Test Files to Create

1. `tests/unit/orchestration/test_paths_comprehensive.py` - All path resolution tests
2. `tests/unit/orchestration/test_naming_comprehensive.py` - All naming tests
3. `tests/unit/orchestration/test_path_building_comprehensive.py` - All path building tests
4. `tests/unit/orchestration/jobs/tracking/naming/test_tags_comprehensive.py` - All tag tests
5. `tests/integration/orchestration/test_config_integration.py` - Integration tests

## Test Coverage Goals

- **Path Resolution:** 100% function coverage, all environment scenarios
- **Naming:** All process types, all patterns from naming.yaml, all edge cases
- **Path Building:** All process types, v2 and legacy patterns, all storage environments
- **Tags:** All tag sections, all process types, all optional tags, sanitization

## Dependencies

- Existing test infrastructure (pytest fixtures, tmp_path)
- Config files: `mlflow.yaml`, `naming.yaml`, `paths.yaml`, `tags.yaml`
- Test utilities: `shared.yaml_utils`, `shared.json_cache`

## Implementation Notes

- Use pytest fixtures for config directory setup
- Use parameterized tests for multiple scenarios
- Test both v2 patterns and legacy fallbacks
- Test all storage environments (local, colab, kaggle, azureml)
- Test error handling and edge cases
- Ensure backward compatibility with existing tests

### To-dos

- [ ] Create tests for paths.yaml configuration loading: load_paths_config(), schema validation, environment overrides, placeholder validation
- [ ] Create tests for path resolution: resolve_output_path() for all categories, cache subdirectories, pattern replacement, absolute/relative paths
- [ ] Create tests for v2 path building: build_output_path() for all process types (HPO, refit, benchmarking, final_training, conversion, best_config), with/without hashes, fallbacks
- [ ] Create tests for cache operations: get_cache_file_path(), save_cache_with_dual_strategy(), load_cache_file(), get_timestamped_cache_filename() for all cache types
- [ ] Create tests for Drive backup paths: get_drive_backup_path(), get_drive_backup_base(), path mirroring, all output categories
- [ ] Create tests for path parsing: parse_hpo_path_v2(), is_v2_path(), find_study_by_hash(), find_trial_by_hash()
- [ ] Create tests for NamingContext: validation for all process types, required fields, invalid inputs, storage_env defaults
- [ ] Create tests for run name building: build_mlflow_run_name() for all process types (hpo_trial, hpo_trial_fold, hpo_sweep, hpo_refit, final_training, benchmarking, conversion), policy-driven formatting, auto-increment
- [ ] Create tests for naming policy: load_naming_policy(), format_run_name(), validate_run_name(), component extraction, separators, normalization
- [ ] Create tests for path building for all processes: HPO v2, HPO refit, benchmarking v2, final training, conversion, best configurations, path normalization
- [ ] Create tests for tags registry: load_tags_registry(), TagsRegistry.key() for all sections, caching, defaults, validation, error handling
- [ ] Create tests for tag building: build_mlflow_tags() for all process types, all tag sections (grouping, process, training, hpo, lineage, paths, legacy), optional tags
- [ ] Create tests for tag sanitization: sanitize_tag_value(), max_length handling, truncation, get_tag_key() with fallbacks
- [ ] Create integration tests: end-to-end scenarios (context → path → name → tags) for all processes, configuration consistency, cross-process consistency