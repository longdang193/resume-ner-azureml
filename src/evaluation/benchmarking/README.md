# Inference Performance Benchmarking

This directory contains tools for benchmarking NER model inference performance, measuring actual latency and throughput to replace parameter-count proxies with real performance data.

## Overview

The benchmarking tools measure:
- **Latency**: Mean, median, P95, P99 inference times (milliseconds)
- **Throughput**: Documents processed per second
- **Batch performance**: Performance across different batch sizes

Results are saved as `benchmark.json` files that can be automatically used by model selection logic.

## Usage

### Basic Benchmarking

Benchmark a specific model checkpoint:

```bash
python -m src.benchmarking.cli \
  --checkpoint outputs/hpo/distilbert/trial_0/checkpoint \
  --test-data dataset/test.json \
  --batch-sizes 1 8 16 \
  --iterations 100 \
  --output outputs/hpo/distilbert/trial_0/benchmark.json
```

### Arguments

- `--checkpoint`: Path to model checkpoint directory (required)
- `--test-data`: Path to test data JSON file (required)
- `--batch-sizes`: List of batch sizes to test (default: 1 8 16)
- `--iterations`: Number of iterations per batch size (default: 100)
- `--warmup`: Number of warmup iterations before measurement (default: 10)
- `--output`: Path to output JSON file (required)
- `--device`: Device to use ('cuda' or 'cpu', default: auto-detect)
- `--max-length`: Maximum sequence length (default: 512)

### Test Data Format

The test data file should be a JSON file containing either:

1. **List of strings**:
```json
[
  "John Smith worked at Microsoft.",
  "Jane Doe is a software engineer.",
  ...
]
```

2. **List of dictionaries with 'text' field**:
```json
[
  {"text": "John Smith worked at Microsoft."},
  {"text": "Jane Doe is a software engineer."},
  ...
]
```

## Output Format

The benchmark script generates a JSON file with the following structure:

```json
{
  "batch_1": {
    "mean_ms": 18.5,
    "median_ms": 17.2,
    "p95_ms": 25.1,
    "p99_ms": 30.5,
    "throughput_docs_per_sec": 54.05
  },
  "batch_8": {
    "mean_ms": 45.2,
    "median_ms": 43.1,
    "p95_ms": 58.3,
    "p99_ms": 65.7,
    "throughput_docs_per_sec": 176.99
  },
  "batch_16": {
    "mean_ms": 78.9,
    "median_ms": 75.4,
    "p95_ms": 95.2,
    "p99_ms": 108.3,
    "throughput_docs_per_sec": 202.79
  },
  "device": "cuda",
  "timestamp": "2025-01-20T10:30:00Z"
}
```

## Integration with Model Selection

After running HPO, benchmark the best trials:

1. **Identify best trial per backbone** (from HPO results)
2. **Run benchmarking** on each best trial
3. **Model selection automatically uses benchmark data** when available

The model selection logic will:
- Check for `benchmark.json` in each trial directory
- Use actual latency if available, fall back to parameter proxy if not
- Include `speed_data_source` in selection output ("benchmark" or "parameter_proxy")

## Workflow Example

```bash
# 1. After HPO completes, identify best trials
#    (e.g., distilbert/trial_0, deberta/trial_1)

# 2. Benchmark DistilBERT best trial
python -m src.benchmarking.cli \
  --checkpoint outputs/hpo/distilbert/trial_0/checkpoint \
  --test-data dataset/test.json \
  --batch-sizes 1 8 16 \
  --iterations 100 \
  --output outputs/hpo/distilbert/trial_0/benchmark.json

# 3. Benchmark DeBERTa best trial
python -m src.benchmarking.cli \
  --checkpoint outputs/hpo/deberta/trial_1/checkpoint \
  --test-data dataset/test.json \
  --batch-sizes 1 8 16 \
  --iterations 100 \
  --output outputs/hpo/deberta/trial_1/benchmark.json

# 4. Run model selection (will automatically use benchmark data)
python -c "
from pathlib import Path
from orchestration.jobs.local_selection import select_best_configuration_across_studies
# ... selection code ...
"
```

## Comparing Models

Use the utility functions to compare multiple models:

```python
from src.benchmarking import compare_models
from pathlib import Path

benchmark_files = [
    Path("outputs/hpo/distilbert/trial_0/benchmark.json"),
    Path("outputs/hpo/deberta/trial_1/benchmark.json"),
]

comparison = compare_models(
    benchmark_files,
    model_names=["DistilBERT", "DeBERTa"]
)
print(comparison)
```

## Programmatic Usage

You can also use the benchmarking functions programmatically:

```python
from pathlib import Path
from src.benchmarking import benchmark_model, load_test_texts

# Load test data
test_texts = load_test_texts(Path("dataset/test.json"))

# Run benchmark
results = benchmark_model(
    checkpoint_dir=Path("outputs/hpo/distilbert/trial_0/checkpoint"),
    test_texts=test_texts,
    batch_sizes=[1, 8, 16],
    num_iterations=100,
    warmup_iterations=10,
)

# Use results
print(f"Batch 1 mean latency: {results['batch_1']['mean_ms']} ms")
```

## Best Practices

1. **Use representative test data**: Use actual test set or validation set texts
2. **Sufficient iterations**: Use at least 100 iterations for stable statistics
3. **Warmup**: Always include warmup iterations to avoid cold start effects
4. **Consistent environment**: Run benchmarks on the same hardware for fair comparison
5. **Multiple batch sizes**: Test different batch sizes to understand scaling behavior

## Notes

- Benchmarking automatically detects GPU availability (CUDA) or falls back to CPU
- Results are device-specific - benchmarks run on different devices are not directly comparable
- The `batch_1` mean latency is typically used as the speed score in model selection
- Benchmark files are saved alongside `metrics.json` in trial directories

## Module Structure

The benchmarking module is organized with Single Responsibility Principle:

- `cli.py`: CLI entry point and argument parsing
- `model_loader.py`: Model and tokenizer loading
- `data_loader.py`: Test data loading utilities
- `execution.py`: Inference execution and latency measurement
- `statistics.py`: Statistics calculation from measurements
- `formatting.py`: Result formatting and comparison utilities
- `orchestrator.py`: High-level orchestration for HPO trials
- `utils.py`: Subprocess wrapper and MLflow logging

## Phase 3: Champion-Based Benchmarking

The benchmarking module now supports **champion-based benchmarking** (Phase 3), which uses complete champion data from Phase 2 selection to avoid redundant MLflow lookups and checkpoint finding.

### Primary Entry Point: `benchmark_champions()`

The recommended way to benchmark champions is using `benchmark_champions()`:

```python
from evaluation.benchmarking.orchestrator import benchmark_champions, filter_missing_benchmarks
from evaluation.selection.trial_finder import select_champions_for_backbones

# Step 1: Select champions (Phase 2)
champions = select_champions_for_backbones(...)

# Step 2: Filter missing benchmarks (idempotency check)
champions_to_benchmark = filter_missing_benchmarks(
    champions=champions,
    benchmark_experiment=benchmark_experiment,
    benchmark_config=benchmark_config,
    data_fingerprint=data_fp,
    eval_fingerprint=eval_fp,
    root_dir=ROOT_DIR,
    environment=environment,
    mlflow_client=mlflow_client,
    run_mode=run_mode,
)

# Step 3: Benchmark champions (Phase 3 - uses complete data, no redundant lookups)
benchmark_results = benchmark_champions(
    champions=champions_to_benchmark,
    test_data_path=test_data_path,
    root_dir=ROOT_DIR,
    environment=environment,
    data_config=data_config,
    hpo_config=hpo_config,
    benchmark_config=benchmark_config,
    ...
)
```

### Benefits of Champion-Based Benchmarking

1. **No Redundant MLflow Lookups**: Champions already have all run_ids (`trial_run_id`, `refit_run_id`, `sweep_run_id`)
2. **No Redundant Checkpoint Finding**: Champions already have `checkpoint_path` set
3. **No Redundant Hash Computation**: Champions already have `study_key_hash` and `trial_key_hash`
4. **Idempotent**: Uses stable benchmark keys and respects `run_mode` configuration

### Legacy Support

The `benchmark_best_trials()` function still supports the legacy `best_trials` format for backward compatibility. However, it now detects champion mode via the `_is_champion` flag and skips redundant operations when using champions.

### Deprecated Functions

The following functions are deprecated but still work for legacy code:

- `find_checkpoint_in_trial_dir()`: Only needed for legacy `best_trials` format
- `compute_grouping_tags()`: Only needed for legacy `best_trials` format

**Migration**: Use `benchmark_champions()` with champions from Phase 2 selection instead.



