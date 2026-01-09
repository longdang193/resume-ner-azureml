# data/*.yaml Coverage Analysis

This document summarizes test coverage for data configuration files in `config/data/*.yaml`.

## Coverage Status: ✅ Complete

All data configuration options are now covered by tests.

## Test Files

1. **`tests/unit/orchestration/test_data_config.py`** - Complete coverage of all data config options (NEW)
   - 29 tests passing
   - 1 skipped (conditional on real files existing)

## Coverage by Section

### 1. Top-level Options

- ✅ **`name`** - Tested in `test_name_option`, `test_load_complete_data_config`
- ✅ **`version`** - Tested in `test_version_option`, `test_load_complete_data_config`
- ✅ **`description`** - Tested in `test_description_option`, `test_load_complete_data_config`
- ✅ **`local_path`** - Tested in `test_local_path_option`, `test_load_complete_data_config`
- ✅ **`seed`** - Tested in `test_seed_option`, `test_load_complete_data_config`

### 2. splitting Section

- ✅ **`splitting.train_test_ratio`** - Tested in `test_splitting_train_test_ratio`, `test_load_complete_data_config`
- ✅ **`splitting.stratified`** - Tested in `test_splitting_stratified_true`, `test_splitting_stratified_false`, `test_load_complete_data_config`
- ✅ **`splitting.random_seed`** - Tested in `test_splitting_random_seed`, `test_load_complete_data_config`

### 3. schema Section

- ✅ **`schema.format`** - Tested in `test_schema_format`, `test_load_complete_data_config`
- ✅ **`schema.annotation_format`** - Tested in `test_schema_annotation_format`, `test_load_complete_data_config`
- ✅ **`schema.entity_types`** - Tested in `test_schema_entity_types`, `test_load_complete_data_config`, `test_build_label_list_from_data_config`

### 4. schema.stats Section

- ✅ **`schema.stats.median_sentence_length`** - Tested in `test_schema_stats_median_sentence_length`, `test_load_complete_data_config`
- ✅ **`schema.stats.mean_sentence_length`** - Tested in `test_schema_stats_mean_sentence_length`, `test_load_complete_data_config`
- ✅ **`schema.stats.p95_sentence_length`** - Tested in `test_schema_stats_p95_sentence_length`, `test_load_complete_data_config`
- ✅ **`schema.stats.suggested_sequence_length`** - Tested in `test_schema_stats_suggested_sequence_length`, `test_load_complete_data_config`
- ✅ **`schema.stats.entity_density`** - Tested in `test_schema_stats_entity_density`, `test_load_complete_data_config`

## Test Coverage Details

### TestDataConfigLoading (1 test)
- Tests loading complete data config matching resume_tiny.yaml structure

### TestDataConfigOptions (18 tests)
- Tests each individual configuration option
- Verifies correct values are loaded from YAML

### TestDataConfigIntegration (2 tests)
- Tests data config loading via `ExperimentConfig` and `load_all_configs()`
- Tests integration with `build_label_list()` function

### TestDataConfigValidation (7 tests)
- Tests edge cases: missing sections, partial sections
- Tests type preservation (numeric, boolean, list)
- Tests validation of required vs optional fields

### TestDataConfigRealFiles (3 tests)
- Tests loading actual data config files from `config/data/`
- Verifies all real configs have required sections
- Tests resume_tiny.yaml and resume_v1.yaml

## Test Statistics

- **Total test file**: 1 (`test_data_config.py`)
- **Total tests**: 29 passing, 1 skipped
- **Coverage**: 100% of all config options in data/*.yaml files

## Configuration Options Summary

### Required Options
- `name` (string) - Dataset name identifier
- `local_path` (string) - Relative path to dataset directory

### Optional Options
- `version` (string) - Dataset version identifier
- `description` (string) - Dataset description
- `seed` (int) - Seed number for dataset subdirectory (seed0, seed1, etc.)

### Optional Sections

#### splitting (dict)
- `train_test_ratio` (float) - Ratio for train/test split (e.g., 0.8)
- `stratified` (bool) - Whether to use stratified splitting
- `random_seed` (int) - Random seed for splitting

#### schema (dict)
- `format` (string) - Dataset format (e.g., "json")
- `annotation_format` (string) - Annotation format (e.g., "character_spans")
- `entity_types` (list) - List of entity type strings (e.g., ["SKILL", "EDUCATION", ...])

#### schema.stats (dict)
- `median_sentence_length` (int) - Median sentence length in characters
- `mean_sentence_length` (int/float) - Mean sentence length in characters
- `p95_sentence_length` (int) - 95th percentile sentence length
- `suggested_sequence_length` (int) - Suggested sequence length for model
- `entity_density` (float) - Entity density (ratio of tokens that are entities)

## Implementation Notes

1. **Data configs are loaded via `load_config_file()`** in `training/config.py` with path pattern `data/{filename}.yaml`

2. **Data configs are also loaded via `load_all_configs()`** in `orchestration/config_loader.py` which loads from `ExperimentConfig.data_config`

3. **Integration with training**: Data configs are used by:
   - `build_label_list()` in `training/data.py` to extract entity types from `schema.entity_types`
   - `split_train_test()` in `training/data.py` to use `splitting.*` options
   - Dataset loading logic to resolve `local_path`

4. **Type preservation**: YAML loading preserves numeric types (int/float) and boolean types correctly.

5. **Entity types**: The `schema.entity_types` list is used to build label lists for model training, with "O" (outside) added as the first label.

6. **Stats section**: The `schema.stats` section contains EDA (Exploratory Data Analysis) insights that can inform model configuration (e.g., `suggested_sequence_length`).

## No Known Limitations

All configuration options in data/*.yaml files are:

- ✅ Properly loaded from the config files
- ✅ Used in the codebase where applicable (training/data.py, training/config.py)
- ✅ Comprehensively tested
- ✅ Have correct type handling (int, float, bool, string, list)
- ✅ Integration with label list building works correctly

