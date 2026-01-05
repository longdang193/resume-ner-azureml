# Model Selection Strategy: Accuracy-Speed Tradeoff

## Introduction

This document explains the model selection strategy implemented for balancing accuracy and inference speed during hyperparameter optimization (HPO). The strategy addresses the limitation where model selection was purely accuracy-based, ignoring significant speed differences between model architectures.

## Problem Statement

### Current Limitations

The original implementation (`src/orchestration/jobs/local_selection.py`) had several limitations:

1. **Accuracy-only selection**: Models were selected purely based on macro-f1 score, ignoring inference speed differences
2. **Memory dependency**: Selection required Optuna study objects in memory, making it impossible to re-select after notebook restart
3. **No tradeoff mechanism**: Could not balance accuracy gains against speed costs
4. **Limited transparency**: Selection reasoning was not logged or transparent

### Why This Matters

- **Production constraints**: Faster models (e.g., DistilBERT with ~66M parameters) may be preferable if accuracy difference is small (<2%)
- **Cost efficiency**: Smaller models reduce inference costs, latency, and resource requirements
- **Flexibility**: Different use cases may prioritize accuracy vs speed differently
- **Reproducibility**: Need to re-select models from saved logs without re-running expensive HPO processes

### Real-World Context

In our HPO experiments, we compare:

- **DistilBERT**: ~66M parameters, ~2.8x faster inference, typically 1-3% lower accuracy
- **DeBERTa-v3-base**: ~184M parameters, slower inference, typically higher accuracy

A pure accuracy-based selection might always choose DeBERTa, even when the accuracy gain is marginal (<1.5%) and doesn't justify the 2.8x slower inference speed.

## Strategy Design

### Core Principle: Accuracy-First with Speed Tiebreaker

The strategy prioritizes accuracy, but uses speed as a tiebreaker when accuracy differences are within a configurable threshold.

**Key features**:

- **Relative threshold**: Uses percentage of best accuracy (more robust across accuracy ranges)
- **Dual-mode operation**: Works with both in-memory Optuna studies and saved disk logs
- **Configurable tradeoff**: Adjustable threshold and minimum accuracy gain requirements
- **Transparent reasoning**: Selection logic is logged with candidate comparison

### Selection Logic Flow

```
1. Load best trial from each backbone (from studies or disk)
2. Sort candidates by accuracy (descending)
3. If accuracy_threshold is set:
   a. Check if any faster model is within threshold
   b. Verify minimum accuracy gain requirement (if set)
   c. Select faster model if conditions met
4. Otherwise: Select best accuracy model
```

## Design Rationale

### Why Relative Thresholds Over Absolute?

**Problem**: An absolute threshold (e.g., 0.015) means different things at different accuracy levels:

- At accuracy 0.5: 1.5% difference = 3% relative difference
- At accuracy 0.9: 1.5% difference = 1.67% relative difference

**Solution**: Relative thresholds (percentage of best accuracy) are more robust:

- 1.5% relative threshold means the same proportional difference regardless of baseline accuracy
- More intuitive: "Is the faster model within 1.5% of the best?"

**Example**:

```python
# Absolute threshold (problematic)
best_acc = 0.5, threshold = 0.015 → effective = 0.015
best_acc = 0.9, threshold = 0.015 → effective = 0.015 (same, but different meaning)

# Relative threshold (better)
best_acc = 0.5, threshold = 0.015 → effective = 0.5 * 0.015 = 0.0075
best_acc = 0.9, threshold = 0.015 → effective = 0.9 * 0.015 = 0.0135 (proportional)
```

### Why Accuracy-First Rather Than Composite Scoring?

**Composite scoring** (weighted combination of accuracy and speed) has drawbacks:

- Requires arbitrary weight selection (e.g., 70% accuracy, 30% speed)
- Hard to interpret: What does a composite score of 0.65 mean?
- Doesn't prioritize accuracy when differences are significant

**Accuracy-first approach**:

- Clear priority: Accuracy matters most
- Speed only matters when accuracy differences are small
- More intuitive: "If models are close in accuracy, prefer faster"
- Easier to tune: Single threshold parameter

### Why Disk-Based Selection?

**Problem**: Optuna study objects are not persisted to disk, only `metrics.json` files are saved.

**Solution**: Reconstruct selection from saved `metrics.json` files:

- Enables selection after notebook restart
- Works with existing HPO outputs
- Handles k-fold CV by aggregating fold metrics
- No need to re-run expensive HPO processes

**Implementation**: The `select_best_from_disk()` function:

1. Scans HPO output directories for `metrics.json` files
2. Aggregates k-fold CV metrics (if applicable)
3. Finds best trial per backbone
4. Applies accuracy-speed tradeoff logic

### Speed Score Methodology

**Benchmark-based measurement (recommended)**: We use actual inference latency measurements from benchmarking:

- Benchmark results are saved as `benchmark.json` in each trial directory
- Uses `batch_1.mean_ms` (mean latency for batch size 1) as the speed score
- Speed scores are normalized relative to the fastest model (fastest = 1.0)
- Automatically detected and used by model selection when available

**Fallback: Parameter count proxy**: If benchmark data is not available, falls back to parameter count:

- **DistilBERT**: ~66M parameters → speed_score = 1.0 (baseline)
- **DeBERTa-v3-base**: ~184M parameters → speed_score = 2.79 (~2.79x slower)

**Rationale for benchmarking**:

- Actual measurements are more accurate than parameter count estimates
- Accounts for architecture efficiency differences
- Reflects real-world performance on your hardware
- Enables fair comparison across different model architectures

**How it works**:

1. After HPO completes, run benchmarking on best trials (see `benchmarks/README.md`)
2. Benchmark results saved as `benchmark.json` in trial directories
3. Model selection automatically detects and uses benchmark data
4. Falls back to parameter proxy if benchmark data not available (backward compatible)
5. Selection output includes `speed_data_source` field ("benchmark" or "parameter_proxy")

## Configuration Guide

### Configuration File: `config/hpo/prod.yaml`

```yaml
selection:
  # Accuracy threshold for speed tradeoff (0.015 = 1.5% relative)
  accuracy_threshold: 0.015
  
  # Use relative threshold (recommended: true)
  use_relative_threshold: true
  
  # Minimum accuracy gain to justify slower model (optional)
  min_accuracy_gain: 0.02
```

### Configuration Options

#### `accuracy_threshold` (float, optional)

- **Default**: `null` (accuracy-only selection)
- **Recommended**: `0.015` (1.5% relative)
- **Meaning**: If two models are within this threshold, prefer faster model
- **Example**: With threshold 0.015, if DeBERTa is 0.014 better than DistilBERT, prefer DistilBERT

#### `use_relative_threshold` (bool)

- **Default**: `true` (recommended)
- **Meaning**:
  - `true`: Threshold is percentage of best accuracy (e.g., 1.5% of 0.8 = 0.012)
  - `false`: Threshold is absolute difference (e.g., 0.015 absolute)

#### `min_accuracy_gain` (float, optional)

- **Default**: `null` (disabled)
- **Recommended**: `0.02` (2% relative)
- **Meaning**: Require minimum accuracy gain to justify slower model
- **Example**: With min_gain 0.02, DeBERTa must be >2% better than DistilBERT to be selected

### Configuration Strategies

#### 1. Accuracy-Only (Default)

```yaml
selection:
  accuracy_threshold: null
```

- Pure accuracy-based selection
- Always selects highest accuracy model
- Use when accuracy is critical and speed doesn't matter

#### 2. Balanced (Recommended)

```yaml
selection:
  accuracy_threshold: 0.015
  use_relative_threshold: true
```

- Prefers faster model if accuracy difference < 1.5%
- Good balance for most production use cases
- Recommended starting point

#### 3. Speed-Preferring

```yaml
selection:
  accuracy_threshold: 0.02
  use_relative_threshold: true
  min_accuracy_gain: 0.03
```

- Prefers faster model if accuracy difference < 2%
- Requires 3% gain to justify slower model
- Use when speed/latency is critical

#### 4. Accuracy-Preferring

```yaml
selection:
  accuracy_threshold: 0.01
  use_relative_threshold: true
```

- Only prefers speed if accuracy difference < 1%
- Use when accuracy is more important than speed

## Usage Examples

### Example 1: In-Memory Selection (Notebook Running)

```python
from orchestration.jobs.local_selection import select_best_configuration_across_studies

# HPO studies are in memory
best_configuration = select_best_configuration_across_studies(
    studies=hpo_studies,
    hpo_config=hpo_config,
    dataset_version=dataset_version,
    # Uses threshold from hpo_config["selection"] if not specified
)

print(f"Selected: {best_configuration['backbone']}")
print(f"Reason: {best_configuration['selection_criteria']['reason']}")
```

### Example 2: Disk-Based Selection (After Notebook Restart)

```python
from pathlib import Path
from orchestration.jobs.local_selection import select_best_configuration_across_studies

# Load from saved HPO outputs
HPO_OUTPUT_DIR = Path("outputs/hpo")

best_configuration = select_best_configuration_across_studies(
    studies=None,  # No in-memory studies
    hpo_config=hpo_config,
    dataset_version=dataset_version,
    hpo_output_dir=HPO_OUTPUT_DIR,  # Load from disk
    accuracy_threshold=0.015,  # Override config if needed
)

print(f"Selected: {best_configuration['backbone']}")
print(f"Reason: {best_configuration['selection_criteria']['reason']}")

# Show all candidates
if 'all_candidates' in best_configuration['selection_criteria']:
    for c in best_configuration['selection_criteria']['all_candidates']:
        print(f"  {c['backbone']}: acc={c['accuracy']:.4f}, speed={c['speed_score']:.2f}x")
```

### Example 3: Comparing Different Thresholds

```python
# Try different thresholds to understand tradeoffs
thresholds = [None, 0.01, 0.015, 0.02]

for threshold in thresholds:
    config = select_best_configuration_across_studies(
        studies=None,
        hpo_config=hpo_config,
        dataset_version=dataset_version,
        hpo_output_dir=HPO_OUTPUT_DIR,
        accuracy_threshold=threshold,
    )
    print(f"Threshold {threshold}: {config['backbone']} "
          f"(acc: {config['selection_criteria']['best_value']:.4f})")
```

### Example 4: Notebook Integration (Colab)

```python
# Step P1-3.6: Best Configuration Selection

from pathlib import Path
from shared.json_cache import save_json
from orchestration.jobs.local_selection import select_best_configuration_across_studies

BEST_CONFIG_CACHE_FILE = ROOT_DIR / "notebooks" / "best_configuration_cache.json"
HPO_OUTPUT_DIR = ROOT_DIR / "outputs" / "hpo"

# Try in-memory first, fall back to disk
if 'hpo_studies' in locals() and hpo_studies:
    best_configuration = select_best_configuration_across_studies(
        studies=hpo_studies,
        hpo_config=hpo_config,
        dataset_version=dataset_version,
    )
else:
    # Load from disk (works after notebook restart)
    best_configuration = select_best_configuration_across_studies(
        studies=None,
        hpo_config=hpo_config,
        dataset_version=dataset_version,
        hpo_output_dir=HPO_OUTPUT_DIR,
    )

# Save to cache
save_json(BEST_CONFIG_CACHE_FILE, best_configuration)

# Display results
print(f"Best configuration selected:")
print(f"  Backbone: {best_configuration.get('backbone')}")
print(f"  Trial: {best_configuration.get('trial_name')}")
print(f"  Best {hpo_config['objective']['metric']}: "
      f"{best_configuration.get('selection_criteria', {}).get('best_value'):.4f}")
print(f"  Reason: {best_configuration.get('selection_criteria', {}).get('reason', 'N/A')}")
```

## Best Practices

### Recommended Thresholds

1. **General Production**: `accuracy_threshold: 0.015` (1.5% relative)
   - Good balance between accuracy and speed
   - Works well for most use cases

2. **Low-Latency Requirements**: `accuracy_threshold: 0.02` with `min_accuracy_gain: 0.03`
   - Prioritizes speed when latency is critical
   - Requires significant accuracy gain to justify slower model

3. **High-Accuracy Requirements**: `accuracy_threshold: 0.01`
   - Only prefers speed if accuracy difference is very small (<1%)
   - Use when accuracy is more important than speed

### When to Use Disk-Based Selection

- **After notebook restart**: HPO studies are no longer in memory
- **Re-evaluating selection**: Want to try different thresholds without re-running HPO
- **Standalone scripts**: Scripts that don't have access to Optuna studies
- **CI/CD pipelines**: Automated selection from saved HPO outputs

### Monitoring Selection

Always check the `selection_criteria` in the returned configuration:

- `reason`: Explains why this model was selected
- `all_candidates`: Shows all models considered
- `accuracy_diff_from_best`: Shows how much accuracy was traded for speed

## Implementation Details

### File Structure

- **Core implementation**: `src/orchestration/jobs/local_selection.py`
  - `MODEL_SPEED_SCORES`: Model speed characteristics
  - `load_best_trial_from_disk()`: Load best trial from disk
  - `select_best_from_disk()`: Disk-based selection logic
  - `select_best_configuration_across_studies()`: Enhanced selection with tradeoff

- **Configuration**: `config/hpo/prod.yaml`
  - `selection` section with threshold and tradeoff options

### K-Fold CV Support

When loading from disk with k-fold CV:

1. Groups trials by base name (e.g., `trial_0_fold0`, `trial_0_fold1` → `trial_0`)
2. Calculates average metric across folds
3. Selects trial with highest average metric

### Backward Compatibility

- If `accuracy_threshold` is `null` or not set: Pure accuracy-based selection (original behavior)
- Existing code continues to work without changes
- New features are opt-in via configuration

## After HPO: Benchmarking and Selection Workflow

### Complete Workflow

```
1. Run HPO
   ↓
   outputs/hpo/{backbone}/trial_X/metrics.json (includes per_entity metrics)

2. Identify Best Trial per Backbone
   ↓
   Review HPO results, find highest macro-f1 per backbone

3. Run Benchmarking
   ↓
   python benchmarks/benchmark_inference.py \
     --checkpoint outputs/hpo/{backbone}/trial_X/checkpoint \
     --test-data dataset/test.json \
     --output outputs/hpo/{backbone}/trial_X/benchmark.json

4. Run Model Selection
   ↓
   Automatically uses benchmark.json if available
   Falls back to parameter proxy if not
   Returns best model with speed_data_source indicator
```

### Per-Entity Metrics Usage

Per-entity metrics are automatically included in `metrics.json` files:

```json
{
  "macro-f1": 0.8017,
  "macro-f1-span": 0.3154,
  "per_entity": {
    "PERSON": {"precision": 0.85, "recall": 0.90, "f1": 0.875, "support": 100},
    "ORG": {"precision": 0.75, "recall": 0.80, "f1": 0.775, "support": 50}
  }
}
```

Use per-entity metrics to:
- Identify which entities each model struggles with
- Compare entity-level performance across models
- Make informed decisions when accuracy differences are small
- Prioritize improvements for specific entity types

### Benchmark Data Integration

When benchmark data is available, selection output includes:

```python
{
    "selection_criteria": {
        "speed_data_source": "benchmark",  # or "parameter_proxy"
        "all_candidates": [
            {
                "backbone": "deberta",
                "accuracy": 0.8050,
                "speed_score": 2.29,  # Normalized from actual latency
                "speed_data_source": "benchmark",
                "benchmark_latency_ms": 42.3  # Actual measurement
            }
        ]
    }
}
```

## Future Enhancements

1. **Pareto frontier analysis**: Show all non-dominated solutions for manual selection
2. **Statistical significance**: Use CV std dev to check if accuracy differences are significant
3. **Model size consideration**: Factor in model size (disk/memory) in addition to speed
4. **Cost-based selection**: Consider inference cost (e.g., API pricing) in selection
5. **Per-entity weighted scoring**: Allow entity priorities to influence selection

## References

- HPO Configuration: `config/hpo/prod.yaml`
- Implementation: `src/orchestration/jobs/local_selection.py`
- Related Documentation: `docs/K_FOLD_CROSS_VALIDATION.md`
