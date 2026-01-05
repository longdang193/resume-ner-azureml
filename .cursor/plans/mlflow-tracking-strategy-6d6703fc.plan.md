<!-- 6d6703fc-6652-4245-801e-932234adb524 f8af6bea-187c-483f-88f8-17d561a96ab6 -->
# MLflow Tracking Strategy Implementation

## Overview

Extend MLflow tracking to cover all training pipeline stages: benchmarking, final training, and model conversion. Follow the existing HPO tracking pattern (`MLflowSweepTracker`) for consistency.

## Architecture

### Tracking Structure

```
Parent Run (Pipeline/Experiment Level)
├── HPO Sweep Run (existing)
│   └── Trial Child Runs (existing)
├── Benchmark Run (new)
├── Final Training Run (new)
└── Conversion Run (new)
```

### Components

1. **New Tracker Classes** (similar to `MLflowSweepTracker`):

   - `MLflowBenchmarkTracker` - For benchmarking stage
   - `MLflowTrainingTracker` - For final training stage  
   - `MLflowConversionTracker` - For model conversion stage

2. **Integration Points**:

   - `src/orchestration/benchmark_utils.py` - Add MLflow tracking
   - `src/training/trainer.py` - Add MLflow tracking to `train_model()`
   - `src/model_conversion/convert_to_onnx.py` - Add MLflow tracking
   - `src/orchestration/jobs/selection/local_selection.py` - Optional: log selection decision

## Implementation Details

### 1. Benchmarking Stage Tracking

**File**: `src/orchestration/jobs/tracking/mlflow_tracker.py` (extend)

**New Class**: `MLflowBenchmarkTracker`

**Methods**:

- `start_benchmark_run()` - Context manager for benchmark run
- `log_benchmark_results()` - Log metrics and artifacts

**Parameters to Log**:

```python
mlflow.log_param("benchmark_batch_sizes", str(batch_sizes))
mlflow.log_param("benchmark_iterations", iterations)
mlflow.log_param("benchmark_warmup_iterations", warmup_iterations)
mlflow.log_param("benchmark_max_length", max_length)
mlflow.log_param("benchmark_device", device)
```

**Metrics to Log**:

```python
# Per batch size metrics
for batch_size in batch_sizes:
    mlflow.log_metric(f"latency_batch_{batch_size}_ms", mean_latency)
    mlflow.log_metric(f"latency_batch_{batch_size}_std_ms", std_latency)
    mlflow.log_metric(f"latency_batch_{batch_size}_min_ms", min_latency)
    mlflow.log_metric(f"latency_batch_{batch_size}_max_ms", max_latency)
    mlflow.log_metric(f"latency_batch_{batch_size}_p50_ms", p50)
    mlflow.log_metric(f"latency_batch_{batch_size}_p95_ms", p95)
    mlflow.log_metric(f"latency_batch_{batch_size}_p99_ms", p99)

mlflow.log_metric("throughput_samples_per_sec", throughput)
```

**Tags**:

```python
mlflow.set_tag("benchmark_source", "hpo_trial" or "final_training")
mlflow.set_tag("benchmarked_model", backbone)
mlflow.set_tag("mlflow.runType", "benchmark")
```

**Artifacts**:

```python
mlflow.log_artifact(benchmark_json_path, artifact_path="benchmark.json")
```

**Integration**: Modify `src/orchestration/benchmark_utils.py::run_benchmarking()` to accept optional tracker and log results.

### 2. Final Training Stage Tracking

**File**: `src/orchestration/jobs/tracking/mlflow_tracker.py` (extend)

**New Class**: `MLflowTrainingTracker`

**Methods**:

- `start_training_run()` - Context manager for training run
- `log_training_parameters()` - Log training config
- `log_training_metrics()` - Log evaluation metrics
- `log_training_artifacts()` - Log checkpoint and metrics.json

**Parameters to Log**:

```python
# Training config
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("dropout", dropout)
mlflow.log_param("weight_decay", weight_decay)
mlflow.log_param("epochs", epochs)
mlflow.log_param("backbone", backbone)

# Data config
mlflow.log_param("data_version", data_version)
mlflow.log_param("dataset_path", dataset_path)  # optional

# New parameters
mlflow.log_param("training_type", "final" or "continued")
mlflow.log_param("source_checkpoint", checkpoint_path)  # if continued
mlflow.log_param("data_strategy", "combined" or "new_only" or "append")  # if continued
mlflow.log_param("random_seed", seed)
```

**Metrics to Log**:

```python
mlflow.log_metric("macro-f1", macro_f1)
mlflow.log_metric("loss", loss)
mlflow.log_metric("macro-f1-span", macro_f1_span)

# Per-entity metrics (if available)
for entity, metrics in per_entity_metrics.items():
    mlflow.log_metric(f"{entity}_precision", metrics["precision"])
    mlflow.log_metric(f"{entity}_recall", metrics["recall"])
    mlflow.log_metric(f"{entity}_f1", metrics["f1"])
    mlflow.log_metric(f"{entity}_support", metrics["support"])
```

**Tags**:

```python
mlflow.set_tag("mlflow.runName", f"{backbone}_{run_id}")
mlflow.set_tag("mlflow.runType", "training")
mlflow.set_tag("training_type", "final" or "continued")
```

**Artifacts**:

```python
mlflow.log_artifacts(checkpoint_dir, artifact_path="checkpoint")
mlflow.log_artifact(metrics_json_path, artifact_path="metrics.json")
```

**Integration**:

- Modify `src/training/trainer.py::train_model()` to accept optional tracker parameter
- Add tracking calls at appropriate points (start run, log params, log metrics, log artifacts)

### 3. Model Conversion Stage Tracking

**File**: `src/orchestration/jobs/tracking/mlflow_tracker.py` (extend)

**New Class**: `MLflowConversionTracker`

**Methods**:

- `start_conversion_run()` - Context manager for conversion run
- `log_conversion_parameters()` - Log conversion config
- `log_conversion_results()` - Log metrics and artifacts

**Parameters to Log**:

```python
mlflow.log_param("conversion_source", checkpoint_path)
mlflow.log_param("conversion_target", "onnx_int8" or "onnx_fp32")
mlflow.log_param("quantization", "int8" or "none")
mlflow.log_param("onnx_opset_version", opset_version)
mlflow.log_param("conversion_backbone", backbone)
```

**Metrics to Log**:

```python
mlflow.log_metric("conversion_success", 1 or 0)
mlflow.log_metric("onnx_model_size_mb", model_size_mb)
mlflow.log_metric("compression_ratio", original_size / onnx_size)
mlflow.log_metric("smoke_test_passed", 1 or 0)
```

**Tags**:

```python
mlflow.set_tag("conversion_type", "onnx_int8" or "onnx_fp32")
mlflow.set_tag("source_training_run", training_run_id)
mlflow.set_tag("mlflow.runType", "conversion")
```

**Artifacts**:

```python
mlflow.log_artifact(onnx_model_path, artifact_path="model_int8.onnx" or "model_fp32.onnx")
mlflow.log_artifact(conversion_log_path, artifact_path="conversion_log.txt")
```

**Integration**:

- Modify `src/model_conversion/convert_to_onnx.py::main()` to accept optional tracker
- Add tracking calls after conversion completes

### 4. Best Configuration Selection (Optional)

**File**: `src/orchestration/jobs/selection/local_selection.py`

**Optional Enhancement**: Log selection decision to parent run or create a lightweight run

**What to Log**:

- Selected backbone
- Selected hyperparameters
- Selection rationale (accuracy-speed tradeoff details)
- Link to source HPO runs

## Configuration

### New Config Options

**File**: `config/mlflow.yaml` (if exists) or add to stage-specific configs

```yaml
# Optional: Control tracking per stage
tracking:
  benchmark:
    enabled: true
    log_artifacts: true
  training:
    enabled: true
    log_checkpoint: true
    log_metrics_json: true
  conversion:
    enabled: true
    log_onnx_model: true
    log_conversion_log: true
```

## Implementation Steps

1. **Extend `mlflow_tracker.py`**:

   - Add `MLflowBenchmarkTracker` class
   - Add `MLflowTrainingTracker` class
   - Add `MLflowConversionTracker` class
   - Follow same pattern as `MLflowSweepTracker` (experiment setup, context managers, error handling)

2. **Integrate Benchmarking**:

   - Modify `src/orchestration/benchmark_utils.py::run_benchmarking()`
   - Parse `benchmark.json` output and log to MLflow
   - Add tracker parameter with default None for backward compatibility

3. **Integrate Final Training**:

   - Modify `src/training/trainer.py::train_model()`
   - Add tracker parameter (optional)
   - Log parameters at start, metrics during/after training, artifacts at end
   - Ensure compatibility with existing training code

4. **Integrate Model Conversion**:

   - Modify `src/model_conversion/convert_to_onnx.py::main()`
   - Add tracker parameter (optional)
   - Log conversion results after successful conversion
   - Handle smoke test results

5. **Update Notebook/Orchestration**:

   - Update `notebooks/01_orchestrate_training_colab.ipynb` to use new trackers
   - Create tracker instances and pass to respective functions
   - Ensure proper run hierarchy (parent run if needed)

6. **Testing**:

   - Test each tracker independently
   - Test integration with existing code
   - Verify artifacts upload correctly (especially for Azure ML)
   - Test backward compatibility (tracker=None should work)

## Error Handling

- Follow HPO pattern: log warnings but don't fail the stage if MLflow tracking fails
- Use try-except blocks around all MLflow calls
- Provide fallback behavior when MLflow unavailable

## Backward Compatibility

- All tracker parameters should be optional (default None)
- Existing code should work without changes
- Tracking is opt-in via configuration

## Files to Modify

1. `src/orchestration/jobs/tracking/mlflow_tracker.py` - Add new tracker classes
2. `src/orchestration/benchmark_utils.py` - Integrate benchmark tracking
3. `src/training/trainer.py` - Integrate training tracking
4. `src/model_conversion/convert_to_onnx.py` - Integrate conversion tracking
5. `notebooks/01_orchestrate_training_colab.ipynb` - Use new trackers
6. `config/mlflow.yaml` or stage configs - Add tracking configuration (optional)

### To-dos

- [ ] Extend mlflow_tracker.py with MLflowBenchmarkTracker, MLflowTrainingTracker, and MLflowConversionTracker classes following MLflowSweepTracker pattern
- [ ] Modify benchmark_utils.py to accept optional tracker and log benchmark results (parameters, metrics, tags, artifacts) to MLflow
- [ ] Modify trainer.py to accept optional tracker and log training parameters, metrics, and artifacts (checkpoint, metrics.json) to MLflow
- [ ] Modify convert_to_onnx.py to accept optional tracker and log conversion parameters, metrics, and artifacts (ONNX model, conversion log) to MLflow
- [ ] Update notebook to instantiate and use new trackers for benchmarking, training, and conversion stages
- [ ] Add optional tracking configuration to control MLflow logging per stage (benchmark, training, conversion)