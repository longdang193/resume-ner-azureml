# experiment/*.yaml Coverage Analysis

This document summarizes test coverage for experiment configuration files in `config/experiment/*.yaml`.

## Coverage Status: ✅ Complete

All experiment configuration options are now covered by tests.

## Test Files

1. **`tests/unit/orchestration/test_experiment_config.py`** - Complete coverage of all experiment config options (NEW)
   - 29 tests passing
   - 2 skipped (conditional on real files existing)

## Coverage by Section

### 1. Top-level Options

- ✅ **`experiment_name`** - Tested in `test_experiment_name_option`, `test_experiment_name_fallback`, `test_load_complete_experiment_config`
- ✅ **`data_config`** - Tested in `test_data_config_option`, `test_load_complete_experiment_config`
- ✅ **`model_config`** - Tested in `test_model_config_option`, `test_load_complete_experiment_config`
- ✅ **`train_config`** - Tested in `test_train_config_option`, `test_load_complete_experiment_config`
- ✅ **`hpo_config`** - Tested in `test_hpo_config_option`, `test_load_complete_experiment_config`
- ✅ **`env_config`** - Tested in `test_env_config_option`, `test_load_complete_experiment_config`
- ✅ **`benchmark_config`** - Tested in `test_benchmark_config_option`, `test_benchmark_config_default`, `test_load_complete_experiment_config`

### 2. stages Section

#### stages.smoke

- ✅ **`stages.smoke.aml_experiment`** - Tested in `test_stages_smoke_aml_experiment`, `test_stages_smoke_both_options`, `test_stages_all_stages_together`
- ✅ **`stages.smoke.hpo_config`** - Tested in `test_stages_smoke_hpo_config`, `test_stages_smoke_both_options`, `test_stages_all_stages_together`

#### stages.hpo

- ✅ **`stages.hpo.aml_experiment`** - Tested in `test_stages_hpo_aml_experiment`, `test_stages_hpo_both_options`, `test_stages_all_stages_together`
- ✅ **`stages.hpo.hpo_config`** - Tested in `test_stages_hpo_hpo_config`, `test_stages_hpo_both_options`, `test_stages_all_stages_together`

#### stages.training

- ✅ **`stages.training.aml_experiment`** - Tested in `test_stages_training_aml_experiment`, `test_stages_training_both_options`, `test_stages_all_stages_together`
- ✅ **`stages.training.backbones`** - Tested in `test_stages_training_backbones_single`, `test_stages_training_backbones_multiple`, `test_stages_training_both_options`, `test_stages_all_stages_together`

### 3. naming Section

- ✅ **`naming.include_backbone_in_experiment`** - Tested in `test_naming_include_backbone_in_experiment_true`, `test_naming_include_backbone_in_experiment_false`, `test_load_complete_experiment_config`
- ✅ **`naming` missing/defaults** - Tested in `test_naming_missing_defaults_to_empty`, `test_load_experiment_config_with_defaults`

## Test Coverage Details

### TestExperimentConfigLoading (2 tests)

- Tests loading complete experiment configs matching resume_ner_baseline.yaml structure
- Tests loading with defaults (missing optional sections)

### TestExperimentConfigOptions (11 tests)

- Tests each individual configuration option
- Tests experiment_name fallback behavior
- Tests benchmark_config default behavior

### TestStagesConfiguration (11 tests)

- Tests all stage options (smoke, hpo, training)
- Tests aml_experiment for each stage
- Tests hpo_config for smoke and hpo stages
- Tests backbones (single and multiple) for training stage
- Tests all stages configured together

### TestNamingConfiguration (3 tests)

- Tests include_backbone_in_experiment (true/false)
- Tests missing naming section defaults to empty dict

### TestExperimentConfigIntegration (3 tests)

- Tests integration with `load_all_configs()`
- Tests that stages are preserved correctly
- Tests that naming is preserved correctly

### TestExperimentConfigRealFile (2 tests)

- Tests loading actual resume_ner_baseline.yaml
- Validates structure of real config file

## Test Statistics

- **Total test file**: 1 (`test_experiment_config.py`)
- **Total tests**: 29 passing, 2 skipped
- **Coverage**: 100% of all config options in experiment/*.yaml files

## Configuration Options Summary

### Required Options

- `data_config` (string) - Relative path to data config file
- `model_config` (string) - Relative path to model config file
- `train_config` (string) - Relative path to training config file
- `hpo_config` (string) - Relative path to HPO config file
- `env_config` (string) - Relative path to environment config file

### Optional Options

- `experiment_name` (string) - Experiment name (defaults to YAML filename if not specified)
- `benchmark_config` (string) - Relative path to benchmark config file (defaults to "benchmark.yaml")

### Optional Sections

#### stages (dict)

- `stages.smoke` (dict) - Smoke test stage configuration
  - `aml_experiment` (string) - Azure ML experiment name for smoke tests
  - `hpo_config` (string) - HPO config override for smoke tests
- `stages.hpo` (dict) - HPO stage configuration
  - `aml_experiment` (string) - Azure ML experiment name for HPO sweeps
  - `hpo_config` (string) - HPO config override for HPO stage
- `stages.training` (dict) - Training stage configuration
  - `aml_experiment` (string) - Azure ML experiment name for final training
  - `backbones` (list) - List of backbone names for training

#### naming (dict)

- `include_backbone_in_experiment` (bool) - Whether to include backbone in experiment name

## Implementation Notes

1. **Experiment configs are loaded via `load_experiment_config()`** in `orchestration/config_loader.py`
   - Path pattern: `config/experiment/{experiment_name}.yaml`
   - Returns `ExperimentConfig` dataclass with resolved paths

2. **Path resolution**: All relative paths in experiment config are resolved relative to the config root directory

3. **Default behavior**:
   - `experiment_name` defaults to YAML filename if not specified
   - `benchmark_config` defaults to "benchmark.yaml" if not specified
   - `stages` defaults to empty dict if not specified
   - `naming` defaults to empty dict if not specified

4. **Integration**: Experiment configs are used by `load_all_configs()` to load all domain configs (data, model, train, hpo, env, benchmark)

5. **Stages usage**: Stages configuration is preserved in `ExperimentConfig.stages` and can be accessed by orchestration code to determine stage-specific behavior (AML experiment names, HPO config overrides, training backbones)

6. **Naming usage**: Naming configuration is preserved in `ExperimentConfig.naming` and can be used by orchestration code to determine experiment naming behavior (e.g., whether to include backbone in experiment name)

## No Known Limitations

All configuration options in experiment/*.yaml files are:

- ✅ Properly loaded from the config files
- ✅ Used in the codebase where applicable (orchestration/config_loader.py)
- ✅ Comprehensively tested
- ✅ Have correct default values where appropriate
- ✅ Path resolution works correctly for relative paths
