# naming.yaml Explicit Coverage Summary

This document confirms that all configuration options in `config/naming.yaml` (lines 1-156) are explicitly tested.

## Coverage Status: ✅ 100% Complete

All sections and options are covered by tests in:
1. **`tests/unit/orchestration/test_naming_comprehensive.py`** - Integration tests for all process types
2. **`tests/unit/orchestration/jobs/tracking/naming/test_naming_policy_details.py`** - Detailed policy tests
3. **`tests/unit/orchestration/test_naming_yaml_explicit_coverage.py`** - Explicit tests for every config option (NEW)

## Explicit Coverage by Section

### 1. schema_version (line 5)
- ✅ **Test**: `TestSchemaVersion.test_schema_version_is_loaded`
- **Value**: `1`

### 2. separators (lines 8-11)
- ✅ **Tests**: `TestSeparatorPolicy.test_separator_field`, `test_separator_component`, `test_separator_version`
- **Values**:
  - `field: "_"`
  - `component: "-"`
  - `version: "_"`

### 3. run_names (lines 15-135)

All 7 process types are explicitly tested:

#### hpo_trial (lines 17-28)
- ✅ **Test**: `TestRunNamesComponentOptions.test_hpo_trial_component_options`
- **Pattern**: `"{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}{version}"`
- **Components**:
  - `study_hash`: `length: 8`, `source: "study_key_hash"`, `default: "unknown"`
  - `trial_number`: `format: "{number}"`, `zero_pad: 2`, `source: "trial_number"`, `default: "unknown"`

#### hpo_trial_fold (lines 31-46)
- ✅ **Test**: `TestRunNamesComponentOptions.test_hpo_trial_fold_component_options`
- **Pattern**: `"{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}_fold{fold_idx}"`
- **Components**:
  - `study_hash`: `length: 8`, `source: "study_key_hash"`, `default: "unknown"`
  - `trial_number`: `format: "{number}"`, `zero_pad: 2`, `source: "trial_number"`, `default: "unknown"`
  - `fold_idx`: `format: "{number}"`, `source: "fold_idx"`, `default: "0"`

#### hpo_refit (lines 49-64)
- ✅ **Test**: `TestRunNamesComponentOptions.test_hpo_refit_component_options`
- **Pattern**: `"{env}_{model}_hpo_refit_study-{study_hash}_trial-{trial_hash}_t{trial_number}{version}"`
- **Components**:
  - `study_hash`: `length: 8`, `source: "study_key_hash"`, `default: "unknown"`
  - `trial_hash`: `length: 8`, `source: "trial_key_hash"`, `default: "unknown"`
  - `trial_number`: `format: "{number}"`, `zero_pad: 2`, `source: "trial_number"`, `default: "unknown"`

#### hpo_sweep (lines 67-78)
- ✅ **Test**: `TestRunNamesComponentOptions.test_hpo_sweep_semantic_suffix_options`
- **Pattern**: `"{env}_{model}_hpo_study-{study_hash}{semantic_suffix}{version}"`
- **Components**:
  - `study_hash`: `length: 8`, `source: "study_key_hash"`, `default: "unknown"`
  - `semantic_suffix`: `enabled: true`, `max_length: 30`, `source: "study_name"`, `default: ""`

#### final_training (lines 81-95)
- ✅ **Test**: `TestRunNamesComponentOptions.test_final_training_component_options`
- **Pattern**: `"{env}_{model}_final_training_spec-{spec_hash}_exec-{exec_hash}_v{variant}{version}"`
- **Components**:
  - `spec_hash`: `length: 8`, `source: "spec_fp"`, `default: "unknown"`
  - `exec_hash`: `length: 8`, `source: "exec_fp"`, `default: "unknown"`
  - `variant`: `format: "{number}"`, `source: "variant"`, `default: "1"`

#### benchmarking (lines 98-112)
- ✅ **Test**: `TestRunNamesComponentOptions.test_benchmarking_component_options`
- **Pattern**: `"{env}_{model}_benchmark_study-{study_hash}_trial-{trial_hash}_bench-{bench_hash}{version}"`
- **Components**:
  - `study_hash`: `length: 8`, `source: "study_key_hash"`, `default: "unknown"`
  - `trial_hash`: `length: 8`, `source: "trial_key_hash"`, `default: "unknown"`
  - `bench_hash`: `length: 8`, `source: "benchmark_config_hash"`, `default: "unknown"`

#### conversion (lines 115-134)
- ✅ **Test**: `TestRunNamesComponentOptions.test_conversion_component_options`
- **Pattern**: `"{env}_{model}_conversion_spec-{spec_hash}_exec-{exec_hash}_v{variant}_conv-{conv_hash}{version}"`
- **Components**:
  - `spec_hash`: `length: 8`, `source: "parent_training_id"`, `default: "unknown"`
  - `exec_hash`: `length: 8`, `source: "parent_training_id"`, `default: "unknown"`
  - `variant`: `format: "{number}"`, `source: "parent_training_id"`, `default: "1"`
  - `conv_hash`: `length: 8`, `source: "conv_fp"`, `default: "unknown"`

### 4. version (lines 137-139)
- ✅ **Tests**: `TestVersionFormatExplicit.test_version_format`, `test_version_separator`
- **Values**:
  - `format: "{separator}{number}"`
  - `separator: "_"`

### 5. normalize (lines 142-148)
- ✅ **Tests**: 
  - `TestNormalizeExplicit.test_normalize_env_replace`
  - `TestNormalizeExplicit.test_normalize_env_lowercase`
  - `TestNormalizeExplicit.test_normalize_model_replace`
  - `TestNormalizeExplicit.test_normalize_model_lowercase`
- **Values**:
  - `env.replace`: `{"/": "_", "-": "_", " ": "_"}`
  - `env.lowercase`: `false`
  - `model.replace`: `{"/": "_", "-": "_", " ": "_"}`
  - `model.lowercase`: `false`

### 6. validate (lines 151-154)
- ✅ **Tests**: 
  - `TestValidateExplicit.test_validate_max_length`
  - `TestValidateExplicit.test_validate_forbidden_chars`
  - `TestValidateExplicit.test_validate_warn_length`
- **Values**:
  - `max_length: 256`
  - `forbidden_chars: ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]`
  - `warn_length: 150`

## Component Option Coverage

All component options are explicitly tested:
- ✅ `length` - Hash truncation (tested for all hash components)
- ✅ `format` - Number formatting (tested for trial_number, variant, fold_idx)
- ✅ `zero_pad` - Zero padding (tested for trial_number in hpo_trial, hpo_trial_fold, hpo_refit)
- ✅ `source` - Source field mapping (tested for all components)
- ✅ `default` - Default values (tested for all components)
- ✅ `enabled` - Semantic suffix toggle (tested for hpo_sweep.semantic_suffix)
- ✅ `max_length` - Semantic suffix truncation (tested for hpo_sweep.semantic_suffix)

## Test Statistics

- **Total explicit tests**: 20 in `test_naming_yaml_explicit_coverage.py`
- **Total comprehensive tests**: 46 in `test_naming_policy_details.py` + existing tests
- **Coverage**: 100% of all config options in `naming.yaml` (lines 1-156)

## Verification

All tests pass:
```bash
pytest tests/unit/orchestration/test_naming_yaml_explicit_coverage.py -v
# Result: 20 passed
```

## Conclusion

✅ **All configuration options in `naming.yaml` (lines 1-156) are explicitly tested.**

Every single config value, option, and component configuration is covered by at least one explicit test, ensuring:
- Config loading works correctly
- All values are accessible
- All options are used as intended
- No configuration drift goes unnoticed

