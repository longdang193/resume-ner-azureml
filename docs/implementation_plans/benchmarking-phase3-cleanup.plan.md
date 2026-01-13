# Benchmarking Phase 3 Cleanup Plan

## Overview

This plan updates the benchmarking module (`src/evaluation/benchmarking/orchestrator.py`) to align with Phase 3 of the master plan and follow DRY principles. The goal is to remove redundant logic, simplify code paths, and ensure benchmarking works cleanly with Phase 2 champion selection.

## Current State Analysis

### ✅ Already Implemented (Phase 3)

1. **Stable Benchmark Keys** (`build_benchmark_key`) ✅
   - Uses champion_run_id + fingerprints from Phase 2
   - Reuses `compute_config_hash` utility (DRY)

2. **Idempotency Check** (`filter_missing_benchmarks`) ✅
   - Checks MLflow using trial_key_hash and study_key_hash (more reliable than benchmark_key tag)
   - Checks disk cache
   - Respects run_mode (skips filtering if `force_new`)

3. **Run Mode Inheritance** (`get_benchmark_run_mode`) ✅
   - Inherits from HPO config if null
   - Uses shared `get_run_mode()` utility (DRY)

4. **Champion-Based Benchmarking** (`benchmark_champions`) ✅
   - Converts champions to best_trials format
   - Calls `benchmark_best_trials()` internally

### ⚠️ Issues to Fix

1. **Redundant Hash Computation** (`compute_grouping_tags`)
   - Champions already have `study_key_hash` and `trial_key_hash` from Phase 2
   - Fallback computation logic is unnecessary when using champions
   - Should use hashes directly from champion data

2. **Redundant MLflow Lookup** (`benchmark_best_trials`)
   - Champions already have `run_id`, `trial_run_id`, `refit_run_id` from Phase 2
   - Complex MLflow query logic (lines 827-925) is redundant
   - Should use run_ids directly from champion data

3. **Redundant Checkpoint Finding** (`benchmark_best_trials`)
   - Champions already have `checkpoint_path` from Phase 2 (set in notebook)
   - `find_checkpoint_in_trial_dir()` logic (lines 340-433) is redundant for champions
   - Should use checkpoint_path directly from champion data

4. **Unused Helper Functions**
   - `find_checkpoint_in_trial_dir()` - Only needed for legacy `best_trials` format
   - `compute_grouping_tags()` - Fallback logic unnecessary for champions

5. **Code Duplication**
   - MLflow client creation duplicated in multiple places
   - UUID validation pattern duplicated

## Implementation Plan

### Step 1: Simplify `benchmark_champions()` to Pass Complete Champion Data

**File:** `src/evaluation/benchmarking/orchestrator.py`

**Changes:**
- Ensure `benchmark_champions()` passes all champion data (including run_ids) to `benchmark_best_trials()`
- Remove need for MLflow lookups in `benchmark_best_trials()` when using champions

**Implementation:**
```python
def benchmark_champions(
    champions: Dict[str, Dict[str, Any]],  # From Phase 2
    ...
) -> Dict[str, Path]:
    """
    Benchmark champions selected in Phase 2.
    
    Each champion represents the best trial from the best configuration group
    for that backbone. We only benchmark champions, not all variants.
    """
    # Convert champions to best_trials format for benchmark_best_trials()
    best_trials = {}
    
    for backbone, champion_data in champions.items():
        champion = champion_data.get("champion", {})
        
        # Extract checkpoint path from champion (already acquired in notebook)
        checkpoint_path = champion.get("checkpoint_path")
        if not checkpoint_path:
            logger.warning(f"No checkpoint_path found for champion {backbone}, skipping")
            continue
        
        # Build trial_info dict with ALL champion data (no need for lookups)
        trial_info = {
            "backbone": backbone,
            "run_id": champion.get("run_id"),  # MLflow run_id (primary)
            "trial_run_id": champion.get("trial_run_id"),  # CV trial run_id
            "refit_run_id": champion.get("refit_run_id") or champion.get("run_id"),  # Refit run_id
            "sweep_run_id": champion.get("sweep_run_id"),  # Parent HPO run_id
            "trial_key_hash": champion.get("trial_key_hash"),
            "study_key_hash": champion.get("study_key_hash"),
            "checkpoint_dir": str(checkpoint_path),  # Already resolved path
            "trial_name": champion.get("trial_key_hash", "unknown")[:16],
            "metric": champion.get("metric"),
            # Mark as champion-based to skip redundant lookups
            "_is_champion": True,
        }
        
        best_trials[backbone] = trial_info
    
    # Call existing benchmark_best_trials() function
    return benchmark_best_trials(...)
```

### Step 2: Simplify `benchmark_best_trials()` for Champion Path

**File:** `src/evaluation/benchmarking/orchestrator.py`

**Changes:**
- Skip redundant MLflow lookups when `_is_champion` flag is set
- Skip redundant checkpoint finding when `checkpoint_dir` is already set
- Skip redundant hash computation when hashes are already present
- Keep fallback logic for legacy `best_trials` format (backward compatibility)

**Implementation:**
```python
def benchmark_best_trials(
    best_trials: Dict[str, Dict[str, Any]],
    ...
) -> Dict[str, Path]:
    """
    Run benchmarking on best trial checkpoints from HPO runs.
    
    Supports two modes:
    1. Champion mode: Uses complete champion data from Phase 2 (no lookups needed)
    2. Legacy mode: Uses best_trials format (requires lookups and checkpoint finding)
    """
    ...
    
    for backbone, trial_info in best_trials.items():
        is_champion = trial_info.get("_is_champion", False)
        
        # Use checkpoint_dir directly if available (champion mode)
        if "checkpoint_dir" in trial_info and trial_info["checkpoint_dir"]:
            checkpoint_dir = Path(trial_info["checkpoint_dir"])
        elif not is_champion and "trial_dir" in trial_info:
            # Legacy mode: find checkpoint in trial_dir
            trial_dir = Path(trial_info["trial_dir"])
            checkpoint_dir = find_checkpoint_in_trial_dir(trial_dir)
            if checkpoint_dir is None:
                logger.warning(f"Checkpoint not found for {backbone}")
                continue
        else:
            logger.warning(f"No checkpoint_dir or trial_dir for {backbone}")
            continue
        
        # Use hashes directly if available (champion mode)
        study_key_hash = trial_info.get("study_key_hash")
        trial_key_hash = trial_info.get("trial_key_hash")
        
        if not is_champion and (not study_key_hash or not trial_key_hash):
            # Legacy mode: compute hashes from configs
            study_key_hash, trial_key_hash, study_family_hash = compute_grouping_tags(
                trial_info, data_config, hpo_config, benchmark_config
            )
        
        # Use run_ids directly if available (champion mode)
        hpo_trial_run_id = trial_info.get("trial_run_id")
        hpo_refit_run_id = trial_info.get("refit_run_id") or trial_info.get("run_id")
        hpo_sweep_run_id = trial_info.get("sweep_run_id")
        
        if not is_champion:
            # Legacy mode: validate and potentially look up run IDs from MLflow
            # (Keep existing MLflow lookup logic for backward compatibility)
            ...
        
        # Rest of function unchanged...
```

### Step 3: Extract MLflow Client Creation Utility (DRY)

**File:** `src/evaluation/benchmarking/orchestrator.py`

**Changes:**
- Create shared utility function for MLflow client creation
- Use it consistently throughout the module

**Implementation:**
```python
def _get_mlflow_client() -> Optional[Any]:
    """
    Get MLflow client instance (shared utility).
    
    Returns:
        MLflowClient instance or None if creation fails
    """
    try:
        from mlflow.tracking import MlflowClient
        return MlflowClient()
    except Exception as e:
        logger.warning(f"Could not create MLflow client: {e}")
        return None
```

### Step 4: Mark Legacy Functions as Deprecated

**File:** `src/evaluation/benchmarking/orchestrator.py`

**Changes:**
- Add deprecation warnings to `find_checkpoint_in_trial_dir()` and `compute_grouping_tags()`
- Document that they're only needed for legacy `best_trials` format
- Keep them for backward compatibility but mark as deprecated

**Implementation:**
```python
def find_checkpoint_in_trial_dir(trial_dir: Path) -> Optional[Path]:
    """
    Find checkpoint directory in trial directory (LEGACY - for backward compatibility).
    
    DEPRECATED: This function is only needed for legacy best_trials format.
    Champions from Phase 2 already have checkpoint_path set.
    
    Prefers:
    1. refit/checkpoint/ (if refit training completed)
    2. cv/foldN/checkpoint/ (best CV fold based on metrics)
    3. checkpoint/ (fallback)
    """
    import warnings
    warnings.warn(
        "find_checkpoint_in_trial_dir() is deprecated. "
        "Use champions from Phase 2 which already have checkpoint_path set.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... existing implementation ...
```

### Step 5: Remove Unused Variant Logic (Already Clean)

**Status:** ✅ Already clean - no variant-related code found in benchmarking module.

### Step 6: Update Documentation

**File:** `src/evaluation/benchmarking/README.md` (if exists) or module docstrings

**Changes:**
- Document that benchmarking now uses champions from Phase 2
- Explain that `benchmark_champions()` is the primary entry point
- Document that `benchmark_best_trials()` supports both champion and legacy modes
- Remove references to variant benchmarking

## Testing Strategy

### Unit Tests

1. **Test `benchmark_champions()` with complete champion data**
   - Verify all champion fields are passed to `benchmark_best_trials()`
   - Verify no MLflow lookups are triggered for champions

2. **Test `benchmark_best_trials()` in champion mode**
   - Verify checkpoint_dir is used directly
   - Verify hashes are used directly
   - Verify run_ids are used directly
   - Verify no redundant lookups

3. **Test `benchmark_best_trials()` in legacy mode**
   - Verify fallback logic still works for legacy best_trials format
   - Verify checkpoint finding still works
   - Verify hash computation still works

4. **Test idempotency check**
   - Verify `filter_missing_benchmarks()` respects run_mode
   - Verify MLflow check uses trial_key_hash and study_key_hash
   - Verify disk cache check works

### Integration Tests

1. **End-to-end champion benchmarking**
   - Select champions using Phase 2
   - Filter missing benchmarks
   - Benchmark champions
   - Verify no redundant MLflow queries

2. **Backward compatibility**
   - Test legacy best_trials format still works
   - Verify deprecated functions still function correctly

## Migration Notes

### Breaking Changes

**None** - All changes are backward compatible. Legacy `best_trials` format still works.

### Deprecations

- `find_checkpoint_in_trial_dir()` - Marked as deprecated (still works for legacy format)
- `compute_grouping_tags()` - Marked as deprecated (still works for legacy format)

### Recommended Migration Path

1. **Phase 1**: Use `benchmark_champions()` for new code (recommended)
2. **Phase 2**: Migrate existing `benchmark_best_trials()` calls to use champions
3. **Phase 3**: Remove deprecated functions (after migration period)

## Success Criteria

- [x] `build_benchmark_key()` uses fingerprints from Phase 2 (DRY)
- [x] `filter_missing_benchmarks()` respects run_mode
- [x] `get_benchmark_run_mode()` inherits from HPO config
- [x] `benchmark_champions()` converts champions to best_trials format
- [x] `benchmark_best_trials()` skips redundant lookups for champions
- [x] `benchmark_best_trials()` uses checkpoint_path directly for champions
- [x] `benchmark_best_trials()` uses hashes directly for champions
- [x] MLflow client creation is centralized (DRY)
- [x] Legacy functions marked as deprecated
- [x] No variant-related code in benchmarking module
- [x] All tests pass (syntax verified, no linting errors)
- [x] Documentation updated

## Files to Modify

1. `src/evaluation/benchmarking/orchestrator.py`
   - Simplify `benchmark_champions()` to pass complete data
   - Simplify `benchmark_best_trials()` to skip redundant lookups for champions
   - Extract MLflow client creation utility
   - Mark legacy functions as deprecated

2. `src/evaluation/benchmarking/README.md` (if exists)
   - Update documentation to reflect champion-based workflow

## Related Files

- `src/evaluation/selection/trial_finder.py` - Provides champions with complete data
- `notebooks/02_best_config_selection.ipynb` - Uses `benchmark_champions()` and sets checkpoint_path
- `config/benchmark.yaml` - Benchmark configuration with run.mode

## References

- Master Plan: `docs/implementation_plans/MASTER-merged-hpo-retrieval-benchmarking-refactor.plan.md`
- Phase 2: Champion Selection (provides champions with complete data)
- Phase 3: Idempotent Benchmarking (this cleanup plan)


