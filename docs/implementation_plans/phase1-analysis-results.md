# Phase 1: Pre-Implementation Analysis Results

## 1. resolve_output_path_v2() Audit

**Findings:**
- Function defined in `orchestration/paths.py` (line 662)
- Exported from `orchestration/__init__.py` (lines 20, 87)
- **No actual imports found** - only used in docstring examples within paths.py itself
- **Decision**: Can be removed from paths/ module. Add thin wrapper in `orchestration/paths.py` facade only if needed for backward compatibility (likely not needed).

## 2. Token Expansion Logic Audit

**Found 43 locations with hash slicing (`[:8]`):**

### Files requiring updates:
1. `orchestration/naming_centralized.py` - 6 locations (lines 262, 263, 282, 283, 295, 296, 462-467)
2. `orchestration/paths.py` - 4 locations (lines 774, 788, 826, 843-844)
3. `orchestration/jobs/tracking/naming/policy.py` - 6 locations (lines 179, 180, 191, 192, 452, 463)
4. `orchestration/jobs/tracking/naming/run_names.py` - 1 location (line 34)
5. `orchestration/jobs/hpo/local/refit/executor.py` - 2 locations (lines 145, 626)
6. `orchestration/jobs/hpo/local_sweeps.py` - 2 locations (lines 262, 478)
7. `orchestration/jobs/selection/cache.py` - 3 locations (lines 130, 131, 248) - *Note: These are for cache keys, may not need token expansion*
8. `orchestration/jobs/selection/trial_finder.py` - 2 locations (lines 201, 203)
9. `orchestration/jobs/selection/study_summary.py` - 1 location (line 190)
10. `orchestration/jobs/selection/artifact_acquisition.py` - 6 locations (lines 125, 126, 146, 148, 620, 638, 648) - *Note: Some are for run_id display, may not need token expansion*
11. `orchestration/jobs/hpo/local/cv/orchestrator.py` - 3 locations (lines 173, 210, 234)

**Action**: Consolidate all token expansion into `naming/context_tokens.py::build_token_values()`. Some locations (cache keys, run_id display) may remain as-is if they're not part of the naming context system.

## 3. MLflow Config Dependencies Analysis

**Dependencies of `orchestration/jobs/tracking/config/loader.py`:**
- `shared.yaml_utils.load_yaml` - YAML file loading
- `shared.logging_utils.get_logger` - Logging

**Functions used by MLflow naming:**
- `get_naming_config()` - Used by `tags.py` and `run_names.py`
- `load_mlflow_config()` - Used by `run_names.py`

**Decision: Option A (Move to naming/mlflow/)**
- Config loader is lightweight (just YAML loading + validation)
- No heavy dependencies on orchestration internals
- Can be moved to `naming/mlflow/config.py` easily
- Will move entire `orchestration/jobs/tracking/naming/` to `naming/mlflow/`

## 4. ExperimentConfig Usage Verification

**Current usage:**
- `get_stage_config()` in `orchestration/naming.py` accepts `ExperimentConfig`
- Implementation is simple: `experiment_config.stages.get(stage, {}) or {}`
- Only 1 caller found: exported from `orchestration/__init__.py` but no direct usage found

**Action**: 
- Change signature to `get_stage_config(experiment_cfg: dict, stage: str) -> dict`
- Implementation: `experiment_cfg.get("stages", {}).get(stage, {}) or {}`
- No ExperimentConfig dependency needed

## Summary

All Phase 1 analysis complete. Ready to proceed with Phase 2: Create Core Module.

