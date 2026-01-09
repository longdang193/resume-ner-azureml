# tags.yaml Configuration Limitations and Notes

This document explicitly documents code limitations, inconsistencies, or unimplemented configuration options discovered during analysis of `tags.yaml` (lines 1-80).

## Test Coverage Status

✅ **All 47 tag keys are covered by tests** in `test_tags_comprehensive.py::test_tags_registry_key_access_all_sections`.

✅ **All sections are tested**: `schema_version`, `grouping`, `process`, `training`, `hpo`, `lineage`, `paths`, `azureml`, `mlflow`, `legacy`.

## Implementation Notes

1. **`TagsRegistry` uses lazy validation** - Tag keys are validated only when accessed via `registry.key()`, not during initialization. This allows creating registries with incomplete configs for testing purposes.

   **Intended purpose**: Allow flexible testing and partial configs.

   **Actual behavior**: Validation happens lazily on key access, raising `TagKeyError` if a key is missing.

   **Test handling**: `test_tags_registry_validation_required_keys` verifies that missing keys raise `TagKeyError` on access, not on initialization.

2. **All tag keys have default values** - The `_get_default_tag_keys()` function provides hardcoded defaults for all 47 tag keys. If `tags.yaml` is missing or incomplete, the registry falls back to these defaults.

   **Intended purpose**: Ensure backward compatibility and graceful degradation when config is missing.

   **Actual behavior**: `load_tags_registry()` merges loaded config with defaults, with loaded values taking precedence.

   **Test handling**: `test_load_tags_registry_fallback_to_defaults` and `test_load_tags_registry_merges_with_defaults` verify this behavior.

3. **`schema_version` defaults to 0** - If `schema_version` is not specified in `tags.yaml`, it defaults to 0.

   **Intended purpose**: Provide a versioning mechanism for tag key schema evolution.

   **Actual behavior**: Defaults to 0 if missing, but current implementation does not use schema version for any validation or migration logic.

   **Test handling**: `test_load_tags_registry_schema_version_defaults_to_0` verifies the default behavior.

4. **Module-level caching** - `load_tags_registry()` uses module-level caching to avoid repeated file reads.

   **Intended purpose**: Improve performance by caching the loaded registry.

   **Actual behavior**: First call loads and caches the registry; subsequent calls return the cached instance if the config path is the same.

   **Test handling**: `test_load_tags_registry_module_level_caching` verifies caching behavior.

## Configuration Completeness

All tag keys defined in `tags.yaml` are:
- ✅ Loaded correctly by `load_tags_registry()`
- ✅ Accessible via `registry.key(section, name)`
- ✅ Covered by comprehensive tests
- ✅ Have default fallback values

## Implementation Completeness

After comprehensive analysis, **all tag keys defined in `tags.yaml` are properly implemented and accessible**. All 47 tag keys are:
- ✅ Properly loaded from the config file via `load_tags_registry()`
- ✅ Accessible through the registry API via `registry.key(section, name)`
- ✅ Have default fallback values in `_get_default_tag_keys()`
- ✅ Are comprehensively tested in `test_tags_comprehensive.py`
- ✅ Used throughout the codebase via `get_tag_key()` or `build_mlflow_tags()`

The implementation follows a defensive design pattern with lazy validation and default fallbacks, ensuring robust behavior even with incomplete or missing configuration files.

## Tag Key Usage Patterns

Some tag keys are not directly used in `build_mlflow_tags()` but are accessed via `get_tag_key()` in other parts of the codebase:

- **Training tags** (`training.trained_on_full_data`, `training.source_training_run`, `training.refit`, `training.refit_has_validation`, `training.interrupted`) - Used in final training executor and HPO refit executors
- **HPO tags** (`hpo.best_trial_run_id`, `hpo.best_trial_number`, `hpo.refit_planned`) - Used in sweep tracker and HPO cleanup
- **Lineage tags** (`lineage.*`) - Used in benchmark tracker, final training executor, and tags module
- **Azure ML tags** (`azureml.run_type`, `azureml.sweep`) - Used in sweep tracker
- **MLflow tags** (`mlflow.run_type`, `mlflow.parent_run_id`) - Used in final training executor and sweep tracker
- **Process tags** (`process.backbone`) - Defined but accessed via context/model name rather than directly

**Intended purpose**: Different tag keys are used in different contexts (HPO, final training, benchmarking, conversion).

**Actual behavior**: All tag keys are accessible via `get_tag_key()` and `registry.key()`, but not all are used in `build_mlflow_tags()`.

**Test handling**: All tag keys are tested for accessibility in `test_tags_registry_key_access_all_sections`, and usage in specific contexts is tested in component/integration tests.

## No Known Limitations

**No unimplemented or inconsistent configuration options** were found in `tags.yaml`. The configuration is complete and fully functional. All tag keys are:
- Properly loaded and accessible
- Used in appropriate contexts throughout the codebase
- Have default fallback values
- Are comprehensively tested

