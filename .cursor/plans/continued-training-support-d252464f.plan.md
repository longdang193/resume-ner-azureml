<!-- d252464f-f506-47cb-8e88-f0338641ab44 2045067b-811d-4f6b-916c-7e373267c762 -->
# Selective Checkpoint Saving and MLflow Logging

## Overview

### Purpose

This plan introduces selective checkpoint management for HPO to reduce storage costs while maintaining full MLflow tracking. Currently, all trial checkpoints are saved locally (~30 GB for 100 trials). This plan changes behavior to save only best trial checkpoints locally (~300 MB), and adds MLflow logging of the final best trial checkpoint after HPO completes. All metrics and parameters continue to be logged to MLflow for all trials, ensuring complete tracking without storage bloat.

### Scope

**In scope**

- Modify checkpoint saving to only save best trial checkpoints locally during HPO
- Add MLflow checkpoint logging for final best trial after HPO completes
- Add configuration flags to control checkpoint saving behavior
- Preserve all existing MLflow metrics and parameter logging (no changes)
- Handle k-fold CV checkpoint selection (use last fold's checkpoint)
- Error handling that doesn't fail HPO if checkpoint operations fail

**Out of scope**

- Changing MLflow metrics/parameters logging (keep as-is for all trials)
- Logging intermediate best trial checkpoints during HPO
- Checkpoint cleanup or deletion logic for non-best trials
- MLflow model registry integration
- Checkpoint compression or optimization
- Other pipeline stages (benchmarking, conversion, continued training) - future phases

### Guiding Principles

- Single Responsibility Principle (SRP): Checkpoint logic isolated in appropriate modules
- Clean Code & modular design: Extend existing methods, don't duplicate logic
- Config-driven behavior: Feature controlled via YAML config flags
- Backward compatibility: Existing MLflow tracking behavior unchanged
- Storage efficiency: Only save/log what's needed
- Testability: Error handling allows HPO to continue if checkpoint operations fail

## Goals & Success Criteria

### Goals

- G1: Save only best trial checkpoints locally during HPO (reduce from ~30 GB to ~300 MB)
- G2: Log final best trial checkpoint to MLflow after HPO completes
- G3: Preserve all MLflow metrics and parameters logging for all trials (no changes)
- G4: Make checkpoint saving configurable and non-breaking

### Success Criteria

- [ ] Only best trial checkpoints saved locally during HPO
- [ ] Best trial checkpoint logged to MLflow parent run after HPO completes
- [ ] Artifact path is `best_trial_checkpoint` in MLflow
- [ ] All trial metrics and parameters still logged to MLflow (regression test)
- [ ] Works with k-fold CV (uses last fold's checkpoint)
- [ ] Works with single training (no CV)
- [ ] HPO continues successfully if checkpoint operations fail
- [ ] Config flags control behavior (can disable/enable)

## Current State Analysis

### Existing Behavior

**Checkpoint Saving During HPO**:

- Currently ALL trial checkpoints are saved locally
- Checkpoints saved to `outputs/hpo/{backbone}/trial_{number}_{run_id}/checkpoint/`
- With k-fold CV: Each fold saves checkpoint to `trial_{number}_{run_id}_fold{idx}/checkpoint/`
- Storage: ~30 GB for 100 trials (20 trials × 5 folds × ~300 MB each)
- All checkpoints preserved locally for benchmarking and analysis

**MLflow Logging - Current State**:

**HPO Stage (Parent Run)**:

- Parameters: hyperparameters (learning_rate, batch_size, dropout, weight_decay), backbone, max_trials, study_name, objective_metric, objective_goal, checkpoint_enabled, checkpoint_path, checkpoint_storage_type, resumed_from_checkpoint
- Metrics: n_trials, n_completed_trials, best_{objective_metric}, best hyperparameters
- Tags: azureml.runType: "sweep", best_trial_run_id, best_trial_number
- Artifacts: none (checkpoints not logged)

**HPO Trial Runs (Child Runs)**:

- Parameters: All hyperparameters for the trial, trial_number, fold_idx (if k-fold CV)
- Metrics: macro-f1, loss, macro-f1-span, per-fold metrics (if k-fold CV)
- Tags: mlflow.parentRunId, trial_number, fold_idx
- Artifacts: none

**Final Training Stage**:

- Parameters: learning_rate, batch_size, dropout, weight_decay, epochs, backbone
- Metrics: macro-f1, loss, macro-f1-span, per-entity metrics
- Tags: run_name (backbone_runid)
- Artifacts: none (checkpoints not logged)

**After HPO**:

- `MLflowSweepTracker.log_final_metrics()` called (line 1063 in `local_sweeps.py`)
- Logs final metrics and best trial information
- Does not log checkpoint artifacts

### Pain Points / Limitations

- L1: All trial checkpoints saved locally (storage inefficient - ~30 GB for 100 trials)
- L2: No checkpoint available in MLflow for model deployment or sharing
- L3: Cannot download best trial checkpoint from MLflow UI
- L4: If local checkpoints lost, cannot recover from MLflow
- L5: Manual process needed to identify and use best trial checkpoint

### Architectural / SRP Issues

- None identified - checkpoint saving fits in training module, logging fits in tracker
- Current design already separates concerns (tracking vs training)

## High-Level Design

### Architecture Overview

```
HPO Orchestration (local_sweeps.py)
|
v
Training Execution (trainer.py)
|  -> Check if trial is best (NEW)
|  -> Save checkpoint only if best (MODIFIED)
|
v
HPO Completes
|
v
MLflowSweepTracker.log_final_metrics()
|  -> Logs metrics (existing - unchanged)
|  -> Logs best trial checkpoint (NEW)
|
v
MLflow Parent Run
  -> Artifact: best_trial_checkpoint/
```

### Responsibility Breakdown

| Layer | Responsibility |

|-------|---------------|

| HPO Orchestration | Pass checkpoint info and best trial status to training |

| Training | Save checkpoint only if trial is best (modified behavior) |

| MLflow Tracker | Log best trial checkpoint to MLflow after HPO |

| Configuration | Control feature via YAML flags |

## Module & File Structure

### New Files to Create

None - extending existing functionality

### Files to Modify

- `config/hpo/prod.yaml`
  - Add `checkpoint.save_only_best` configuration flag
  - Add `mlflow.log_best_checkpoint` configuration flag

- `src/training/trainer.py`
  - Modify `save_checkpoint()` to accept `is_best_trial` parameter
  - Modify `train_model()` to determine if trial is best and pass to `save_checkpoint()`
  - Add logic to check if trial is best (requires study context)

- `src/orchestration/jobs/hpo/local_sweeps.py`
  - Pass best trial information to training execution
  - Update `log_final_metrics()` call to pass checkpoint parameters

- `src/orchestration/jobs/tracking/mlflow_tracker.py`
  - Extend `log_final_metrics()` method signature (lines 145-181)
  - Add `_log_best_trial_checkpoint()` private method

### Files Explicitly Not Touched

- `src/training/orchestrator.py` - No changes needed
- `src/orchestration/jobs/hpo/study_extractor.py` - No changes needed
- `src/platform_adapters/logging_adapter.py` - MLflow metrics/params logging unchanged

## Detailed Design per Component

### Component: Training Checkpoint Saving

**Responsibility (SRP)**

- Save model checkpoints during training
- Only save if trial is best (when configured)

**Inputs**

- `model`: Trained model
- `tokenizer`: Tokenizer instance
- `output_dir`: Directory to save checkpoint
- `is_best_trial`: Boolean indicating if this trial is the best so far (NEW)
- `save_only_best`: Config flag to enable selective saving (NEW)

**Outputs**

- Checkpoint saved to disk (only if best trial or save_only_best is false)
- Logging messages indicating save status

**Public API**

```python
def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    is_best_trial: bool = False,
    save_only_best: bool = False,
) -> None:
    """
    Save model and tokenizer checkpoint.
    
    Args:
        model: Trained model (may be wrapped in DDP).
        tokenizer: Tokenizer instance.
        output_dir: Directory to save checkpoint.
        is_best_trial: Whether this trial is the best so far.
        save_only_best: If True, only save when is_best_trial is True.
    """
```

**Implementation Notes**

- Check `save_only_best` flag first
- If `save_only_best=True` and `is_best_trial=False`: Skip saving, log message
- If `save_only_best=False`: Save always (backward compatible)
- If `save_only_best=True` and `is_best_trial=True`: Save checkpoint
- Error handling: Log warnings but don't raise exceptions

### Component: HPO Best Trial Detection

**Responsibility (SRP)**

- Determine if a trial is the best so far during HPO execution

**Implementation Notes**

- After trial completes, compare trial value with current best
- Use Optuna study to get current best trial
- Pass `is_best_trial` flag to training checkpoint saving
- Handle k-fold CV: Trial is best if average metric is best

**Key Code Location**: `local_sweeps.py` around trial execution (line 500-550)

### Component: MLflowSweepTracker

**Responsibility (SRP)**

- Track HPO sweep metrics and artifacts in MLflow
- Log best trial checkpoint after HPO completes

**Inputs**

- `study`: Completed Optuna study with best trial identified
- `hpo_output_dir`: Path to HPO output directory containing trial checkpoints
- `backbone`: Model backbone name (e.g., "distilbert")
- `run_id`: HPO run identifier for directory resolution
- `fold_splits`: Optional list of fold splits (if k-fold CV used)
- `hpo_config`: HPO configuration dict containing MLflow settings

**Outputs**

- Best trial checkpoint logged to MLflow parent run as artifact
- Logging messages indicating success or failure

**Public API**

```python
def log_final_metrics(
    self,
    study: Any,
    objective_metric: str,
    parent_run_id: str,
    run_name: Optional[str] = None,
    should_resume: bool = False,
    hpo_output_dir: Optional[Path] = None,
    backbone: Optional[str] = None,
    run_id: Optional[str] = None,
    fold_splits: Optional[List] = None,
    hpo_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log final metrics and best trial checkpoint to parent run.
    
    Args:
        study: Completed Optuna study.
        objective_metric: Name of the objective metric.
        parent_run_id: ID of the parent MLflow run.
        run_name: Optional run name (for resumed runs).
        should_resume: Whether this is a resumed run.
        hpo_output_dir: Path to HPO output directory.
        backbone: Model backbone name.
        run_id: HPO run identifier.
        fold_splits: Optional fold splits for k-fold CV.
        hpo_config: HPO configuration dictionary.
    """
```

**Implementation Notes**

- Checkpoint directory resolution:
  - Single training: `{hpo_output_dir}/{backbone}/trial_{number}_{run_id}/checkpoint/`
  - K-fold CV: `{hpo_output_dir}/{backbone}/trial_{number}_{run_id}_fold{last_idx}/checkpoint/`
- Error handling: Log warnings but don't raise exceptions (HPO should continue)
- MLflow artifact path: `best_trial_checkpoint` (consistent naming)
- Config check: Only log if `hpo_config.mlflow.log_best_checkpoint == true`
- Note: Best trial checkpoint should exist locally (it was saved when trial became best)

## Configuration & Controls

### Configuration Sources

- YAML: `config/hpo/prod.yaml` (or other HPO config files)

### Example Configuration

```yaml
checkpoint:
  enabled: true
  study_name: "hpo_{backbone}_prod"
  storage_path: "{study_name}/study.db"
  auto_resume: true
  # NEW: Only save checkpoints for best trials locally
  save_only_best: true

mlflow:
  # Log best trial checkpoint to MLflow after HPO completes
  # Set to false to disable MLflow checkpoint logging entirely
  log_best_checkpoint: true
```

### Validation Rules

- `save_only_best`: Boolean, optional (default: `false` for backward compatibility)
- `log_best_checkpoint`: Boolean, optional (default: `true` if `mlflow` section exists)
- If `checkpoint` section missing: Use defaults (save all checkpoints)
- If `mlflow` section missing: Feature disabled (backward compatible)
- If `save_only_best` is `false`: Save all checkpoints (current behavior)
- If `log_best_checkpoint` is `false`: Skip MLflow checkpoint logging

## Implementation Steps

1. Add configuration flags to `config/hpo/prod.yaml`
2. Modify `save_checkpoint()` in `trainer.py` to accept `is_best_trial` and `save_only_best` parameters
3. Add best trial detection logic in `local_sweeps.py` during trial execution
4. Pass best trial status to training execution
5. Extend `log_final_metrics()` method signature in `mlflow_tracker.py`
6. Implement `_log_best_trial_checkpoint()` method in `mlflow_tracker.py`
7. Update `log_final_metrics()` call in `local_sweeps.py` to pass checkpoint parameters
8. Test with single training (no CV)
9. Test with k-fold CV enabled
10. Verify error handling (checkpoint operations fail scenario)
11. Verify config flags (enable/disable)
12. Regression test: Verify all metrics/parameters still logged to MLflow

## Testing Strategy

### Unit Tests

- `save_checkpoint()` with `save_only_best=True` and `is_best_trial=False` (should skip)
- `save_checkpoint()` with `save_only_best=True` and `is_best_trial=True` (should save)
- `save_checkpoint()` with `save_only_best=False` (should always save - backward compat)
- `_log_best_trial_checkpoint()` with mock MLflow client
- Checkpoint directory resolution for single training
- Checkpoint directory resolution for k-fold CV
- Config flag parsing (enabled/disabled)

### Integration Tests

- End-to-end HPO with `save_only_best=True`
- Verify only best trial checkpoints saved locally
- Verify best trial checkpoint logged to MLflow after HPO completes
- Verify checkpoint appears in MLflow parent run artifacts
- Verify artifact path is `best_trial_checkpoint`
- Verify all trial metrics and parameters still logged to MLflow (regression)
- Verify only final best trial logged (not intermediate)

### Edge Cases

- Checkpoint directory not found (should log warning, continue)
- MLflow unavailable (should log warning, continue HPO)
- K-fold CV with missing fold checkpoint
- Empty study (no completed trials)
- Multiple backbones (each saves/logs its best trial)
- Best trial checkpoint missing (unlikely but possible)
- Trial becomes best but checkpoint save fails (should not fail HPO)

### Performance / Load Tests

- Checkpoint save time (only for best trials - faster overall)
- Checkpoint upload time (~300 MB per backbone to MLflow)
- MLflow API call latency
- Impact on HPO completion time (should be minimal)
- Storage reduction verification (~30 GB to ~300 MB)

## Backward Compatibility & Migration

**What remains compatible**

- All existing MLflow metrics and parameter logging (unchanged)
- HPO execution flow unchanged
- Config files without new flags (defaults to current behavior: save all checkpoints)
- MLflow tracking behavior unchanged (metrics/params for all trials)

**Deprecated behavior**

None - this is a new feature with backward-compatible defaults

**Migration steps**

- No migration needed - feature is opt-in via config
- Existing HPO runs continue with current behavior (save all checkpoints)
- New HPO runs with `save_only_best=true` will save only best trials
- New HPO runs with `mlflow.log_best_checkpoint=true` will log to MLflow

## Documentation Updates

### New Documentation

None required - feature is self-explanatory via config

### Updated Documentation

- `config/hpo/prod.yaml` - Add inline comments for new config sections
- Consider adding note in README about selective checkpoint saving and MLflow logging

## Rollout & Validation Checklist

- [ ] Feature behind configuration flags (`checkpoint.save_only_best`, `mlflow.log_best_checkpoint`)
- [ ] Unit tests added for `save_checkpoint()` with selective saving
- [ ] Unit tests added for `_log_best_trial_checkpoint()`
- [ ] Integration test for end-to-end selective saving and MLflow logging
- [ ] CI passing with new tests
- [ ] Error handling verified (checkpoint operations fail scenario)
- [ ] Config flags verified (enable/disable)
- [ ] K-fold CV checkpoint selection verified
- [ ] Single training checkpoint selection verified
- [ ] MLflow artifact path verified (`best_trial_checkpoint`)
- [ ] Regression test: All metrics/parameters still logged to MLflow for all trials
- [ ] Storage reduction verified (only best trial checkpoints saved locally)
- [ ] Best trial checkpoint existence verified before MLflow logging