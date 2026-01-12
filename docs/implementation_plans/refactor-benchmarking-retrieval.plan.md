# Refactor Benchmarking: Deterministic Best Trial Retrieval & Idempotent Benchmarking

## Overview

### Purpose

Refactor benchmarking to:
1. **Make "best" trial selection deterministic and explicit** (stop mixing sources implicitly)
2. **Support multi-variant HPO cleanly** (overall best by default, per-variant optional)
3. **Separate retrieval from benchmarking** (explicit retrieval step for better debugging)
4. **Make benchmarking idempotent** (skip if already benchmarked for exact trial identity)

### Why This Matters

**Current Issues:**
- Best trial selection mixes MLflow, Optuna study, and disk sources implicitly
- No clear handling of multiple HPO variants (v1, v2, v3)
- Retrieval and benchmarking are coupled, making debugging harder
- No idempotency check - re-runs waste compute on already-benchmarked trials
- Ambiguous "best" when multiple sources disagree

**Benefits:**
- **Deterministic**: Same inputs → same best trial selection (regardless of environment)
- **Debuggable**: Explicit retrieval step shows exactly which trials were selected and why
- **Efficient**: Skip already-benchmarked trials using stable identity keys
- **Flexible**: Support both "overall best" and "per-variant" benchmarking modes
- **Future-proof**: MLflow-first approach scales to distributed/multi-machine workflows

### Scope

**In scope:**
- Create explicit `retrieve_best_trials()` function with deterministic source priority
- Add `selection_scope` parameter (overall vs per-variant)
- Separate retrieval step from benchmarking in notebooks
- Add idempotency check using stable benchmark keys
- Update `benchmark_best_trials()` to accept pre-retrieved trials
- Add benchmark existence check (MLflow + disk)

**Out of scope:**
- Changing how HPO runs are executed
- Changing benchmark metrics or scoring logic
- Changing MLflow tracking structure (only reading from it)

## Goals & Success Criteria

### Goals

- **G1**: Deterministic best trial selection with explicit source priority (MLflow > study > disk)
- **G2**: Support multi-variant HPO with configurable scope (overall vs per-variant)
- **G3**: Explicit retrieval step separate from benchmarking
- **G4**: Idempotent benchmarking using stable identity keys
- **G5**: Clear logging when sources disagree or benchmarks are skipped

### Success Criteria

- [ ] `retrieve_best_trials()` function with MLflow-first hybrid approach
- [ ] `selection_scope` parameter supports "overall" (default) and "per_variant"
- [ ] Notebooks show explicit retrieval step with clear table output
- [ ] Benchmarking skips already-benchmarked trials using stable keys
- [ ] Warnings logged when sources disagree on "best" trial
- [ ] All existing tests pass
- [ ] Documentation updated with new workflow

## Current State Analysis

### Existing Behavior

**Current flow in `01_orchestrate_training_colab.ipynb`:**

```python
# Step 1: Find best trials (implicit, mixed sources)
best_trials = find_best_trials_for_backbones(
    backbone_values=backbone_values,
    hpo_studies=hpo_studies,  # Optuna study objects
    hpo_config=hpo_config,
    data_config=data_config,
    root_dir=ROOT_DIR,
    environment=environment,
)

# Step 2: Benchmark (immediately after, no idempotency check)
benchmark_results = benchmark_best_trials(
    best_trials=best_trials,
    ...
)
```

**Current `find_best_trials_for_backbones()` logic:**
1. Try Optuna study (if `hpo_studies` provided)
2. Fallback to disk search (metrics.json files)
3. No MLflow querying
4. No variant handling
5. No idempotency check

**Current `benchmark_best_trials()` logic:**
1. Loops through `best_trials` dict
2. No check if benchmark already exists
3. Always runs benchmark (wastes compute on reruns)

### Pain Points

- **P1**: Source priority is implicit (study > disk, no MLflow)
- **P2**: No handling of multiple variants (v1, v2, v3)
- **P3**: No idempotency - reruns waste compute
- **P4**: Hard to debug which trial was selected and why
- **P5**: No warning when sources disagree

## High-Level Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Retrieve Best Trials (Explicit)                    │
├─────────────────────────────────────────────────────────────┤
│ retrieve_best_trials()                                      │
│   ├─ Source Priority: MLflow > Study > Disk                │
│   ├─ selection_scope: "overall" (default) or "per_variant"│
│   ├─ Returns: {backbone: best_trial_info}                  │
│   └─ Logs: source used, warnings if sources disagree       │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Filter Already-Benchmarked (Optional)              │
├─────────────────────────────────────────────────────────────┤
│ filter_missing_benchmarks()                                 │
│   ├─ Check MLflow: benchmark run with stable key           │
│   ├─ Check Disk: cached benchmark_{key}.json              │
│   └─ Returns: trials_to_benchmark (subset)                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Benchmark Only Missing Trials                       │
├─────────────────────────────────────────────────────────────┤
│ benchmark_best_trials()                                    │
│   ├─ Loops through trials_to_benchmark                     │
│   ├─ Creates stable benchmark_key for each                │
│   └─ Logs benchmark results to MLflow + disk               │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. `retrieve_best_trials()` Function

**Location:** `src/evaluation/selection/trial_retrieval.py` (new file)

**Signature:**
```python
def retrieve_best_trials(
    backbone_values: List[str],
    hpo_experiments: Dict[str, Dict[str, str]],  # backbone -> {name, id}
    benchmark_experiment: Optional[Dict[str, str]],  # {name, id}
    selection_config: Dict[str, Any],
    hpo_config: Dict[str, Any],
    data_config: Dict[str, Any],
    root_dir: Path,
    environment: str,
    selection_scope: str = "overall",  # "overall" or "per_variant"
    mlflow_client: Optional[MlflowClient] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve best trials per backbone with deterministic source priority.
    
    Source Priority (deterministic):
    1. MLflow (query runs + objective metric + required tags)
    2. Optuna study (study.best_trial)
    3. Disk (metrics.json fallback)
    
    If multiple sources disagree, logs warning and picks earlier source.
    
    Args:
        selection_scope: "overall" (best across all variants) or "per_variant" (best per variant)
    
    Returns:
        Dict mapping backbone -> best_trial_info dict
    """
```

**Source Priority Logic:**
1. **MLflow-first**: Query MLflow for best run per backbone (by objective metric)
   - Filter by required tags (`code.study_key_hash`, `code.trial_key_hash`, `code.backbone`)
   - Group by variant if `selection_scope="per_variant"`
   - Select best by objective metric value
2. **Study fallback**: If MLflow unavailable or no runs found, use Optuna study
   - Load study from disk (study.db)
   - Use `study.best_trial`
3. **Disk fallback**: If study unavailable, search disk for metrics.json
   - Find best by objective metric value

**Variant Handling:**
- `selection_scope="overall"` (default): Select best trial across all variants
- `selection_scope="per_variant"`: Select best trial per variant (returns multiple per backbone)

#### 2. `filter_missing_benchmarks()` Function

**Location:** `src/evaluation/benchmarking/orchestrator.py`

**Signature:**
```python
def filter_missing_benchmarks(
    best_trials: Dict[str, Dict[str, Any]],
    benchmark_experiment: Dict[str, str],
    benchmark_config: Dict[str, Any],
    data_config: Dict[str, Any],
    root_dir: Path,
    environment: str,
    mlflow_client: Optional[MlflowClient] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Filter out trials that already have benchmarks.
    
    Uses stable benchmark_key to check:
    - MLflow: existing benchmark run with matching key tag
    - Disk: cached benchmark_{key}.json
    
    Args:
        best_trials: Dict mapping backbone -> trial_info
    
    Returns:
        Dict mapping backbone -> trial_info (only missing benchmarks)
    """
```

**Stable Benchmark Key:**
```python
benchmark_key = f"{backbone}:{variant}:{run_id}:{trial_id}:{data_fingerprint}:{benchmark_config_hash}"
```

**Components:**
- `backbone`: Model backbone name
- `variant`: HPO variant (v1, v2, v3) or "default"
- `run_id`: MLflow run ID (if available)
- `trial_id`: Trial identifier (trial_key_hash or trial_name)
- `data_fingerprint`: Dataset fingerprint (data_config hash)
- `benchmark_config_hash`: Benchmark config hash (batch_sizes, iterations, etc.)

#### 3. Updated `benchmark_best_trials()` Function

**Location:** `src/evaluation/benchmarking/orchestrator.py`

**Changes:**
- Accept pre-retrieved `best_trials` dict
- Create stable `benchmark_key` for each trial
- Log `benchmark_key` as MLflow tag for future lookups
- Save benchmark results with key-based filename (optional)

### Responsibility Breakdown

| Component | Responsibility |
|-----------|---------------|
| `retrieve_best_trials()` | Deterministic best trial selection with source priority |
| `filter_missing_benchmarks()` | Idempotency check using stable keys |
| `benchmark_best_trials()` | Benchmark only provided trials, log stable keys |
| Notebooks | Explicit retrieval step with clear output table |

## Detailed Design

### 1. Source Priority Implementation

**MLflow Query Strategy:**

```python
# Pseudo-code for MLflow-first retrieval
def retrieve_from_mlflow(backbone, hpo_experiments, selection_config):
    client = MlflowClient()
    
    # Get HPO experiment for this backbone
    hpo_exp = hpo_experiments.get(backbone)
    if not hpo_exp:
        return None
    
    # Query all runs in HPO experiment
    runs = client.search_runs(
        experiment_ids=[hpo_exp["id"]],
        filter_string=f"tags.{backbone_tag} = '{backbone}'",
        max_results=1000,
    )
    
    # Filter by required tags and objective metric
    valid_runs = [
        r for r in runs
        if has_required_tags(r) and has_objective_metric(r)
    ]
    
    # Group by variant if per_variant mode
    if selection_scope == "per_variant":
        # Group by variant tag, select best per variant
        ...
    else:
        # Select overall best
        best_run = max(valid_runs, key=lambda r: r.data.metrics[objective_metric])
    
    return best_run
```

**Source Disagreement Handling:**

```python
# When multiple sources available, compare and warn
mlflow_best = retrieve_from_mlflow(...)
study_best = retrieve_from_study(...)

if mlflow_best and study_best:
    if mlflow_best.trial_id != study_best.trial_id:
        logger.warning(
            f"Sources disagree on best trial for {backbone}: "
            f"MLflow={mlflow_best.trial_id}, Study={study_best.trial_id}. "
            f"Using MLflow (higher priority)."
        )
    return mlflow_best  # Always prefer earlier source
```

### 2. Variant Handling

**Overall Mode (Default):**
```python
# Select best trial across all variants
best_trials = {
    "distilbert": {
        "trial_id": "trial-abc123",
        "variant": "v2",  # Best happened to be in v2
        "metric": 0.85,
        ...
    }
}
```

**Per-Variant Mode:**
```python
# Select best trial per variant
best_trials = {
    "distilbert": {
        "v1": {
            "trial_id": "trial-xyz789",
            "metric": 0.82,
            ...
        },
        "v2": {
            "trial_id": "trial-abc123",
            "metric": 0.85,
            ...
        }
    }
}
```

**Notebook Usage:**
```python
# Default: overall best
best_trials = retrieve_best_trials(
    backbone_values=["distilbert", "distilroberta"],
    selection_scope="overall",  # Default
    ...
)

# Optional: per-variant
best_trials_per_variant = retrieve_best_trials(
    backbone_values=["distilbert"],
    selection_scope="per_variant",
    ...
)
```

### 3. Stable Benchmark Key

**Key Components:**

```python
def build_benchmark_key(
    backbone: str,
    variant: Optional[str],
    run_id: Optional[str],
    trial_id: str,
    data_config: Dict[str, Any],
    benchmark_config: Dict[str, Any],
) -> str:
    """Build stable benchmark identity key."""
    
    # Data fingerprint (dataset version + config hash)
    data_fp = compute_data_fingerprint(data_config)
    
    # Benchmark config hash (batch_sizes, iterations, warmup, etc.)
    bench_fp = compute_config_hash(benchmark_config)
    
    # Variant (or "default" if not applicable)
    variant_str = variant or "default"
    
    # Run ID (or "unknown" if not available)
    run_id_str = run_id or "unknown"
    
    # Build key
    key = f"{backbone}:{variant_str}:{run_id_str}:{trial_id}:{data_fp}:{bench_fp}"
    
    return key
```

**Key Properties:**
- **Stable**: Same inputs → same key (deterministic)
- **Unique**: Different trial/config → different key
- **Readable**: Human-readable components for debugging

### 4. Idempotency Check

**MLflow Check:**
```python
def benchmark_exists_in_mlflow(
    benchmark_key: str,
    benchmark_experiment: Dict[str, str],
    mlflow_client: MlflowClient,
) -> bool:
    """Check if benchmark run exists in MLflow with matching key."""
    
    runs = mlflow_client.search_runs(
        experiment_ids=[benchmark_experiment["id"]],
        filter_string=f"tags.benchmark_key = '{benchmark_key}'",
        max_results=1,
    )
    
    return len(runs) > 0 and runs[0].info.status == "FINISHED"
```

**Disk Check:**
```python
def benchmark_exists_on_disk(
    benchmark_key: str,
    root_dir: Path,
    environment: str,
) -> Optional[Path]:
    """Check if benchmark file exists on disk."""
    
    # Look in outputs/benchmarking/{env}/cache/
    cache_dir = root_dir / "outputs" / "benchmarking" / environment / "cache"
    benchmark_file = cache_dir / f"benchmark_{benchmark_key}.json"
    
    if benchmark_file.exists():
        return benchmark_file
    
    return None
```

**Combined Check:**
```python
def benchmark_already_exists(
    benchmark_key: str,
    benchmark_experiment: Dict[str, str],
    root_dir: Path,
    environment: str,
    mlflow_client: Optional[MlflowClient] = None,
) -> bool:
    """Check if benchmark exists (MLflow or disk)."""
    
    # Check MLflow first (authoritative)
    if mlflow_client:
        if benchmark_exists_in_mlflow(benchmark_key, benchmark_experiment, mlflow_client):
            return True
    
    # Fallback to disk
    if benchmark_exists_on_disk(benchmark_key, root_dir, environment):
        return True
    
    return False
```

### 5. Notebook Integration

**Updated Notebook Flow:**

```python
# Step 1: Explicit Retrieval (NEW)
print("📊 Retrieving best trials per backbone...")
best_trials = retrieve_best_trials(
    backbone_values=backbone_values,
    hpo_experiments=hpo_experiments,
    benchmark_experiment=benchmark_experiment,
    selection_config=selection_config,
    hpo_config=hpo_config,
    data_config=data_config,
    root_dir=ROOT_DIR,
    environment=environment,
    selection_scope="overall",  # or "per_variant"
)

# Print retrieval results table
print("\n✓ Best Trials Retrieved:")
print(f"{'Backbone':<15} {'Trial ID':<20} {'Metric':<10} {'Source':<10}")
print("-" * 60)
for backbone, trial_info in best_trials.items():
    print(f"{backbone:<15} {trial_info['trial_id']:<20} "
          f"{trial_info['accuracy']:<10.4f} {trial_info['source']:<10}")

# Step 2: Filter Missing Benchmarks (NEW)
print("\n🔍 Checking for existing benchmarks...")
trials_to_benchmark = filter_missing_benchmarks(
    best_trials=best_trials,
    benchmark_experiment=benchmark_experiment,
    benchmark_config=benchmark_config,
    data_config=data_config,
    root_dir=ROOT_DIR,
    environment=environment,
)

skipped_count = len(best_trials) - len(trials_to_benchmark)
if skipped_count > 0:
    print(f"⏭️  Skipping {skipped_count} already-benchmarked trial(s)")

# Step 3: Benchmark Only Missing Trials
if trials_to_benchmark:
    print(f"\n🏃 Benchmarking {len(trials_to_benchmark)} trial(s)...")
    benchmark_results = benchmark_best_trials(
        best_trials=trials_to_benchmark,  # Only missing ones
        test_data_path=test_data_path,
        root_dir=ROOT_DIR,
        environment=environment,
        data_config=data_config,
        hpo_config=hpo_config,
        benchmark_config=benchmark_config,
        ...
    )
else:
    print("✓ All trials already benchmarked - nothing to do!")
```

## Module & File Structure

### New Files

- `src/evaluation/selection/trial_retrieval.py`
  - `retrieve_best_trials()` - Main retrieval function
  - `retrieve_from_mlflow()` - MLflow query logic
  - `retrieve_from_study()` - Optuna study logic
  - `retrieve_from_disk()` - Disk fallback logic
  - `compare_sources()` - Source disagreement detection

### Modified Files

- `src/evaluation/benchmarking/orchestrator.py`
  - `filter_missing_benchmarks()` - New idempotency check
  - `build_benchmark_key()` - Stable key generation
  - `benchmark_best_trials()` - Update to use pre-retrieved trials
  - `benchmark_already_exists()` - Existence check logic

- `src/evaluation/selection/trial_finder.py`
  - Deprecate `find_best_trials_for_backbones()` (replace with `retrieve_best_trials()`)
  - Keep for backward compatibility with deprecation warning

- `notebooks/01_orchestrate_training_colab.ipynb`
  - Add explicit retrieval step
  - Add filtering step
  - Update benchmarking step to use filtered trials

- `notebooks/01b_benchmark_hpo_results.ipynb` (if created)
  - Use same retrieval + filtering pattern

## Implementation Plan

### Phase 1: Core Retrieval Function

1. **Create `trial_retrieval.py`**
   - Implement `retrieve_best_trials()` with MLflow-first logic
   - Implement source priority (MLflow > study > disk)
   - Add `selection_scope` parameter
   - Add source disagreement warnings

2. **Add MLflow Query Logic**
   - Query HPO experiments for best runs
   - Filter by required tags
   - Select by objective metric
   - Handle variant grouping

3. **Add Study Fallback**
   - Load Optuna study from disk
   - Use `study.best_trial`
   - Extract trial info

4. **Add Disk Fallback**
   - Search for metrics.json files
   - Select best by objective metric
   - Extract trial info

### Phase 2: Idempotency

1. **Implement Stable Key Generation**
   - `build_benchmark_key()` function
   - Include all relevant components (backbone, variant, run_id, trial_id, data_fp, bench_fp)

2. **Implement Existence Checks**
   - MLflow check (query by tag)
   - Disk check (cache file lookup)

3. **Implement Filtering**
   - `filter_missing_benchmarks()` function
   - Return only trials without benchmarks

### Phase 3: Integration

1. **Update Benchmarking Function**
   - Accept pre-retrieved trials
   - Create stable keys for each trial
   - Log keys as MLflow tags
   - Save with key-based filenames (optional)

2. **Update Notebooks**
   - Add explicit retrieval step
   - Add filtering step
   - Update benchmarking step
   - Add clear output tables

3. **Add Tests**
   - Test source priority logic
   - Test variant handling (overall vs per-variant)
   - Test idempotency checks
   - Test source disagreement warnings

### Phase 4: Documentation & Cleanup

1. **Update Documentation**
   - Add workflow diagram
   - Document `selection_scope` parameter
   - Document stable key format
   - Add examples

2. **Deprecate Old Functions**
   - Add deprecation warnings to `find_best_trials_for_backbones()`
   - Keep for backward compatibility

3. **Update Tests**
   - Ensure all existing tests pass
   - Add new tests for retrieval logic

## Testing Strategy

### Unit Tests

- **Source Priority:**
  - Test MLflow-first selection
  - Test study fallback when MLflow unavailable
  - Test disk fallback when study unavailable
  - Test source disagreement warning

- **Variant Handling:**
  - Test overall mode (best across variants)
  - Test per-variant mode (best per variant)
  - Test variant detection from MLflow tags

- **Idempotency:**
  - Test stable key generation (deterministic)
  - Test MLflow existence check
  - Test disk existence check
  - Test filtering logic

### Integration Tests

- **End-to-End Workflow:**
  - Test retrieval → filtering → benchmarking flow
  - Test skipping already-benchmarked trials
  - Test notebook cell execution

- **MLflow Integration:**
  - Test MLflow querying with real experiments
  - Test tag-based filtering
  - Test benchmark key tagging

## Migration Plan

### Backward Compatibility

- Keep `find_best_trials_for_backbones()` with deprecation warning
- New code should use `retrieve_best_trials()`
- Old notebooks continue to work (with warning)

### Gradual Migration

1. **Phase 1**: Add new functions alongside old ones
2. **Phase 2**: Update notebooks to use new functions
3. **Phase 3**: Deprecate old functions
4. **Phase 4**: Remove old functions (after deprecation period)

## Success Metrics

- **Determinism**: Same inputs → same best trial selection (tested)
- **Efficiency**: Skipped benchmarks reduce compute time (measured)
- **Debuggability**: Explicit retrieval step shows selection reasoning (verified)
- **Flexibility**: Both overall and per-variant modes work (tested)

## Open Questions

1. **Benchmark Key Storage**: Should we store benchmark keys in a central index file, or rely on MLflow tags + disk cache?
   - **Decision**: Use MLflow tags (authoritative) + disk cache (fast lookup)

2. **Variant Detection**: How do we detect variants from MLflow tags?
   - **Decision**: Use `code.variant` tag if present, else infer from study_name pattern

3. **Data Fingerprint**: What components should be included in data fingerprint?
   - **Decision**: Dataset version + data_config hash (excludes local_path)

4. **Benchmark Config Hash**: Should we include all benchmark config, or only "significant" parameters?
   - **Decision**: Include all benchmark config (batch_sizes, iterations, warmup, max_length, device)

## References

- Current implementation: `src/evaluation/selection/trial_finder.py`
- Current benchmarking: `src/evaluation/benchmarking/orchestrator.py`
- MLflow querying: `src/evaluation/selection/mlflow_selection.py`

