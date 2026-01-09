# mlflow.yaml Configuration Limitations and Notes

This document explicitly documents code limitations, inconsistencies, or unimplemented configuration options discovered during analysis of `mlflow.yaml` (lines 1-92).

## Test Coverage Status

✅ **All 29 config options are covered by tests** in `test_mlflow_config_comprehensive.py` and `test_mlflow_yaml_explicit_coverage.py`.

✅ **All sections are tested**: `azure_ml`, `tracking`, `naming`, `index`, `run_finder`.

## Implementation Notes and Limitations

1. **`tracking.benchmark.enabled` is now implemented and controls MLflow run creation for benchmarking.**

   **Intended purpose**: Allow disabling MLflow tracking for benchmarking runs.

   **Actual behavior**: `MLflowBenchmarkTracker.start_benchmark_run()` checks `tracking.benchmark.enabled` from config. If `false`, the tracker returns `None` without creating a run.

   **Test handling**: Tests verify that the configuration option exists and is parsed correctly (`test_tracking_benchmark_enabled`, `test_tracking_benchmark_disabled`). Implementation now respects this value.

2. **`tracking.training.enabled` is now implemented and controls MLflow run creation for training.**

   **Intended purpose**: Allow disabling MLflow tracking for training runs.

   **Actual behavior**: `MLflowTrainingTracker.start_training_run()` checks `tracking.training.enabled` from config. If `false`, the tracker returns `None` without creating a run.

   **Test handling**: Tests verify that the configuration option exists and is parsed correctly (`test_tracking_training_enabled`, `test_tracking_training_disabled`). Implementation now respects this value.

3. **`tracking.conversion.enabled` is now implemented and controls MLflow run creation for conversion.**

   **Intended purpose**: Allow disabling MLflow tracking for conversion runs.

   **Actual behavior**: `MLflowConversionTracker.start_conversion_run()` checks `tracking.conversion.enabled` from config. If `false`, the tracker returns `None` without creating a run.

   **Test handling**: Tests verify that the configuration option exists and is parsed correctly (`test_tracking_conversion_enabled`, `test_tracking_conversion_disabled`). Implementation now respects this value.

4. **`tracking.benchmark.log_artifacts` is now implemented and controls artifact logging for benchmarks.**

   **Intended purpose**: Control whether `benchmark.json` artifact is logged to MLflow.

   **Actual behavior**: `MLflowBenchmarkTracker.log_benchmark_results()` checks `tracking.benchmark.log_artifacts` from config. If `false`, the artifact is not logged.

   **Test handling**: Tests verify that the configuration option exists and is parsed (`test_tracking_benchmark_log_artifacts`). Implementation now respects this value.

5. **`tracking.training.log_checkpoint` is now implemented and controls checkpoint artifact logging for training.**

   **Intended purpose**: Control whether checkpoint directory is logged as artifact to MLflow.

   **Actual behavior**: `MLflowTrainingTracker.log_training_artifacts()` checks `tracking.training.log_checkpoint` from config. If `false`, the checkpoint is not logged.

   **Test handling**: Tests verify that the configuration option exists and is parsed (`test_tracking_training_log_checkpoint`). Implementation now respects this value.

6. **`tracking.training.log_metrics_json` is now implemented and controls metrics.json artifact logging for training.**

   **Intended purpose**: Control whether `metrics.json` file is logged as artifact to MLflow.

   **Actual behavior**: `MLflowTrainingTracker.log_training_artifacts()` checks `tracking.training.log_metrics_json` from config. If `false`, the metrics.json file is not logged.

   **Test handling**: Tests verify that the configuration option exists and is parsed (`test_tracking_training_log_metrics_json`). Implementation now respects this value.

7. **`tracking.conversion.log_onnx_model` is now implemented and controls ONNX model artifact logging for conversion.**

   **Intended purpose**: Control whether converted ONNX model is logged as artifact to MLflow.

   **Actual behavior**: `MLflowConversionTracker.log_conversion_results()` checks `tracking.conversion.log_onnx_model` from config. If `false`, the ONNX model is not logged.

   **Test handling**: Tests verify that the configuration option exists and is parsed (`test_tracking_conversion_log_onnx_model`). Implementation now respects this value.

8. **`tracking.conversion.log_conversion_log` is now implemented and controls conversion log artifact logging.**

   **Intended purpose**: Control whether conversion log file is logged as artifact to MLflow.

   **Actual behavior**: `MLflowConversionTracker.log_conversion_results()` checks `tracking.conversion.log_conversion_log` from config. If `false`, the conversion log is not logged.

   **Test handling**: Tests verify that the configuration option exists and is parsed (`test_tracking_conversion_log_conversion_log`). Implementation now respects this value.

9. **`local` section (lines 16-22) is comment-only and has no configuration options.**

   **Intended purpose**: Document platform-specific local tracking paths (Colab, Kaggle, Local).

   **Actual behavior**: Paths are resolved automatically by platform detection code, not from config. No config options exist for this section.

   **Test handling**: No tests needed as this section contains no configurable options.

10. **`naming.run_name.max_length` is defined in the configuration but is not enforced by the codebase.**

    **Intended purpose**: Provide guidance for maximum run name length (100 chars) for readability.

    **Actual behavior**: The config value is loaded and available, but run name building does not enforce this limit. The comment in config states "not enforced, just guidance".

    **Test handling**: Tests verify that the configuration option exists and is parsed (`test_naming_run_name_max_length`), and explicitly document that it is guidance-only, not enforced.

## Configuration Completeness

All config options defined in `mlflow.yaml` are:

- ✅ Properly loaded from the config file via `load_mlflow_config()`
- ✅ Accessible through the appropriate getter functions (`get_naming_config()`, `get_index_config()`, `get_run_finder_config()`, `get_auto_increment_config()`)
- ✅ Covered by comprehensive tests
- ✅ Have default fallback values where appropriate

## Summary

**Total config options**: 29

- **Fully implemented**: 21 (azure_ml, naming.*, index.*, run_finder.*)
- **Defined but not used**: 8 (tracking.*.enabled, tracking.*.log_*)

The `tracking.*` section options are loaded from config but not used to control runtime behavior. This appears to be intentional design where tracking is always enabled, and the config options may be reserved for future use or documentation purposes.
