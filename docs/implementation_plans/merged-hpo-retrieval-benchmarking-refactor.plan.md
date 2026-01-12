# Merged Plan: HPO Run Mode + Variants → Deterministic Retrieval → Idempotent Benchmarking

## Overview

### Purpose

This plan merges two related refactoring efforts into a unified, step-by-step implementation:

1. **Unify Run Mode Logic**: Single source of truth for `run.mode` extraction across all stages
2. **Refactor Benchmarking**: Deterministic best trial retrieval + idempotent benchmarking

**Key Principle**: Follow DRY by reusing and generalizing existing code rather than duplicating.

### Why Merge?

- **HPO variants** are the foundation for everything else
- **Retrieval** needs variant-aware logic from HPO
- **Benchmarking** uses both HPO variants and retrieval results
- **Run mode** controls behavior across all three stages
- **Shared utilities** reduce duplication (DRY)

### Scope

**In scope:**
- Phase 1: HPO run mode + variant generation (foundation)
- Phase 2: Deterministic best trial retrieval (uses HPO variants)
- Phase 3: Idempotent benchmarking (uses both HPO + retrieval)
- Extract shared utilities following DRY principles
- Reuse existing code where possible

**Out of scope:**
- Changing HPO execution logic
- Changing benchmark metrics/scoring
- Changing MLflow tracking structure (only reading)

## Goals & Success Criteria

### Goals

- **G1**: Single source of truth for run mode extraction (no duplication)
- **G2**: HPO variants (v1, v2, v3) with run.mode control
- **G3**: Deterministic best trial retrieval (MLflow > Study > Disk)
- **G4**: Idempotent benchmarking with stable keys
- **G5**: Reuse existing code (DRY) - no unnecessary duplication

### Success Criteria

- [ ] `run_mode.py` utility replaces all 4+ duplicate extractions
- [ ] HPO creates variants (v1, v2, v3) based on `run.mode`
- [ ] `retrieve_best_trials()` with MLflow-first priority
- [ ] Benchmarking skips already-benchmarked trials
- [ ] All existing tests pass
- [ ] No code duplication (shared utilities used)

## DRY Analysis: Reusable Modules

### Existing Code to Reuse

| Component | Location | How to Reuse |
|-----------|----------|--------------|
| **Variant Computation** | `src/infrastructure/config/training.py::_compute_next_variant()` | Generalize to shared `variants.py` module |
| **Variant Scanning** | `src/infrastructure/config/training.py::_find_existing_variant()` | Generalize for HPO variants |
| **Run Mode Extraction** | Duplicated in 4+ places | Extract to `run_mode.py` utility |
| **MLflow Querying** | `src/evaluation/selection/mlflow_selection.py` | Extract patterns to `mlflow/queries.py` |
| **Best Trial Finding** | `src/evaluation/selection/trial_finder.py` | Enhance existing, don't duplicate |
| **Fingerprint/Hash** | `src/infrastructure/fingerprints/` | Reuse existing utilities |

### Strategy: Enhance, Don't Duplicate

- ✅ **Generalize** existing variant logic for HPO
- ✅ **Extract** run mode to shared utility
- ✅ **Enhance** existing `trial_finder.py` with MLflow-first
- ✅ **Reuse** fingerprint utilities for benchmark keys

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: HPO Foundation (Run Mode + Variants)              │
├─────────────────────────────────────────────────────────────┤
│ 1.1: Extract run_mode.py utility (replaces 4+ duplicates)  │
│ 1.2: Generalize variants.py (reuse from training.py)       │
│ 1.3: Add run.mode to HPO configs                           │
│ 1.4: Implement HPO variant generation (uses variants.py)   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Deterministic Retrieval (Uses HPO Variants)       │
├─────────────────────────────────────────────────────────────┤
│ 2.1: Extract MLflow query patterns (reuse from mlflow_     │
│      selection.py)                                          │
│ 2.2: Enhance trial_finder.py (add MLflow-first priority)   │
│ 2.3: Add selection_scope parameter (overall vs per_variant)│
│ 2.4: Update notebooks (explicit retrieval step)             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Idempotent Benchmarking (Uses Both)               │
├─────────────────────────────────────────────────────────────┤
│ 3.1: Build stable benchmark keys (reuse fingerprints/)     │
│ 3.2: Add idempotency check (MLflow + disk)                 │
│ 3.3: Add run mode inheritance (uses run_mode.py)           │
│ 3.4: Add variant completeness check (uses variants.py)     │
│ 3.5: Update benchmarking function                          │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: HPO Foundation (Run Mode + Variants)

### Step 1.1: Create Unified Run Mode Utility

**File:** `src/infrastructure/config/run_mode.py` (NEW)

**Purpose:** Single source of truth for run mode extraction (replaces 4+ duplicates)

**Implementation:**
```python
from typing import Literal, Any, Dict

RunMode = Literal["reuse_if_exists", "force_new", "resume_if_incomplete"]

def get_run_mode(config: Dict[str, Any], default: RunMode = "reuse_if_exists") -> RunMode:
    """
    Extract run.mode from configuration with consistent defaults.
    
    Used across all stages:
    - HPO: Controls whether to create new study or reuse existing
    - Final Training: Controls variant creation and checkpoint reuse
    - Best Model Selection: Controls cache reuse
    - Benchmarking: Inherits from HPO
    
    Args:
        config: Configuration dictionary (e.g., from YAML)
        default: Default mode if not specified (default: "reuse_if_exists")
    
    Returns:
        Run mode string: "reuse_if_exists", "force_new", or "resume_if_incomplete"
    """
    return config.get("run", {}).get("mode", default)

def is_force_new(config: Dict[str, Any]) -> bool:
    """Check if run.mode is force_new."""
    return get_run_mode(config) == "force_new"

def is_reuse_if_exists(config: Dict[str, Any]) -> bool:
    """Check if run.mode is reuse_if_exists."""
    return get_run_mode(config) == "reuse_if_exists"
```

**Refactor existing code:**
- `src/evaluation/selection/cache.py:97` → Use `get_run_mode()`
- `src/selection/cache.py:97` → Use `get_run_mode()`
- `src/training/execution/executor.py:190` → Use `get_run_mode()`
- `src/infrastructure/config/training.py:142` → Use `get_run_mode()`

**Tests:** Unit tests for all helper functions

---

### Step 1.2: Generalize Variant Logic (DRY)

**File:** `src/infrastructure/config/variants.py` (NEW - extracted from `training.py`)

**Purpose:** Shared variant computation for both `final_training` and `hpo`

**Implementation:**
```python
from pathlib import Path
from typing import Optional, List
from common.shared.platform_detection import detect_platform
from infrastructure.paths import build_output_path

def compute_next_variant(
    root_dir: Path,
    config_dir: Path,
    process_type: str,  # "final_training" or "hpo"
    model: str,
    spec_fp: Optional[str] = None,  # Required for final_training
    exec_fp: Optional[str] = None,  # Required for final_training
    base_name: Optional[str] = None,  # For HPO: "hpo_distilbert"
) -> int:
    """
    Compute next available variant number for any process type.
    
    Generalizes existing _compute_next_variant() from training.py
    to support both final_training and hpo.
    
    Args:
        process_type: "final_training" or "hpo"
        model: Model backbone name
        spec_fp: Specification fingerprint (final_training only)
        exec_fp: Execution fingerprint (final_training only)
        base_name: Base study name (hpo only, e.g., "hpo_distilbert")
    
    Returns:
        Next available variant number (starts at 1 if none exist)
    """
    existing = find_existing_variants(
        root_dir, config_dir, process_type, model, spec_fp, exec_fp, base_name
    )
    if not existing:
        return 1
    return max(existing) + 1

def find_existing_variants(
    root_dir: Path,
    config_dir: Path,
    process_type: str,
    model: str,
    spec_fp: Optional[str] = None,
    exec_fp: Optional[str] = None,
    base_name: Optional[str] = None,
) -> List[int]:
    """
    Find all existing variant numbers for a process type.
    
    Generalizes existing _find_existing_variant() from training.py.
    """
    if process_type == "final_training":
        # Reuse existing logic from training.py
        from infrastructure.metadata.training import find_by_spec_and_env
        environment = detect_platform()
        entries = find_by_spec_and_env(root_dir, spec_fp, environment, "final_training")
        if entries:
            variants = [e.get("variant", 1) for e in entries if e.get("exec_fp") == exec_fp]
            return variants if variants else []
        # Fallback: scan filesystem
        return _scan_final_training_variants(root_dir, config_dir, spec_fp, exec_fp, model)
    
    elif process_type == "hpo":
        # New logic: scan HPO output directories
        return _scan_hpo_variants(root_dir, config_dir, model, base_name)
    
    return []

def _scan_hpo_variants(
    root_dir: Path,
    config_dir: Path,
    model: str,
    base_name: str,
) -> List[int]:
    """
    Scan HPO output directories for existing variants.
    
    Looks for study folders matching:
    - {base_name} (variant 1, implicit)
    - {base_name}_v1, {base_name}_v2, etc.
    """
    environment = detect_platform()
    model_name = model.split("-")[0] if "-" in model else model
    
    # Build HPO output path
    hpo_output_dir = build_output_path(
        root_dir=root_dir,
        config_dir=config_dir,
        process_type="hpo",
        model=model_name,
        environment=environment,
    )
    
    if not hpo_output_dir.exists():
        return []
    
    variants = []
    for item in hpo_output_dir.iterdir():
        if not item.is_dir():
            continue
        
        folder_name = item.name
        
        # Check for base name (variant 1, implicit)
        if folder_name == base_name:
            variants.append(1)
        # Check for explicit variant suffix (_v1, _v2, etc.)
        elif folder_name.startswith(f"{base_name}_v"):
            try:
                variant_num = int(folder_name.split("_v")[-1])
                variants.append(variant_num)
            except ValueError:
                pass
    
    return sorted(set(variants))

def _scan_final_training_variants(
    root_dir: Path,
    config_dir: Path,
    spec_fp: str,
    exec_fp: str,
    model: str,
) -> List[int]:
    """Reuse existing filesystem scanning logic from training.py."""
    # Copy logic from _find_existing_variant() fallback
    # ... (existing implementation)
```

**Update existing code:**
- `src/infrastructure/config/training.py` → Import from `variants.py`
- Keep backward compatibility (re-export functions)

**Tests:** Unit tests for both process types

---

### Step 1.3: Add Run Mode to HPO Configs

**Files:**
- `config/hpo/smoke.yaml`
- `config/hpo/prod.yaml`

**Changes:**
```yaml
# Run mode configuration (unified behavior control)
run:
  # Run mode determines overall behavior:
  # - reuse_if_exists: Reuse existing study if found (default, respects auto_resume)
  # - force_new: Always create a new study with next variant (v1, v2, v3...)
  mode: force_new  # or reuse_if_exists

checkpoint:
  enabled: true
  # study_name: null  # null = auto-generate base as "hpo_{backbone}" (default)
  #                    # When run.mode=force_new, code will compute next variant (v1, v2, v3...)
  storage_path: "{study_name}/study.db"
  auto_resume: true
  save_only_best: true
```

**Documentation:** Add comments explaining variant behavior

---

### Step 1.4: Implement HPO Variant Generation

**File:** `src/training/hpo/utils/helpers.py`

**Changes:**
```python
from infrastructure.config.run_mode import get_run_mode
from infrastructure.config.variants import compute_next_variant, find_existing_variants

def create_study_name(
    backbone: str,
    run_id: str,
    should_resume: bool,
    checkpoint_config: Optional[Dict[str, Any]] = None,
    hpo_config: Optional[Dict[str, Any]] = None,
    run_mode: Optional[str] = None,
    root_dir: Optional[Path] = None,
    config_dir: Optional[Path] = None,
) -> str:
    """
    Create Optuna study name with variant support (like final training).
    
    When run_mode == "force_new", computes next variant number (v1, v2, v3...).
    When run_mode == "reuse_if_exists", uses base name for resumability.
    
    Uses shared variants.py module (DRY).
    """
    checkpoint_config = checkpoint_config or {}
    hpo_config = hpo_config or {}
    checkpoint_enabled = checkpoint_config.get("enabled", False)
    
    # Get run_mode from config if not provided
    if run_mode is None:
        combined_config = {**hpo_config, **checkpoint_config}
        run_mode = get_run_mode(combined_config)
    
    # Check for custom study_name in checkpoint config first, then HPO config
    study_name_template = checkpoint_config.get("study_name") or hpo_config.get("study_name")
    
    if study_name_template:
        study_name = study_name_template.replace("{backbone}", backbone)
        # If force_new and we have root_dir/config_dir, compute variant
        if run_mode == "force_new" and root_dir and config_dir:
            variant = compute_next_variant(
                root_dir=root_dir,
                config_dir=config_dir,
                process_type="hpo",
                model=backbone,
                base_name=study_name,
            )
            return f"{study_name}_v{variant}" if variant > 1 else study_name
        # If reuse_if_exists, use base name
        return study_name
    
    # Default behavior when no custom study_name is provided
    base_name = f"hpo_{backbone}"
    
    if run_mode == "force_new":
        # Compute next variant when force_new
        if root_dir and config_dir:
            variant = compute_next_variant(
                root_dir=root_dir,
                config_dir=config_dir,
                process_type="hpo",
                model=backbone,
                base_name=base_name,
            )
            return f"{base_name}_v{variant}" if variant > 1 else base_name
        else:
            # Fallback: use run_id if root_dir/config_dir not available
            return f"{base_name}_{run_id}"
    elif checkpoint_enabled or should_resume:
        # Use consistent name for resumability (no variant suffix)
        return base_name
    else:
        # Use unique name for fresh start (only when checkpointing is disabled)
        return f"{base_name}_{run_id}"

def find_study_variants(
    output_dir: Path,
    backbone: str,
) -> List[str]:
    """
    Find all study variants for a given backbone.
    
    Uses shared variants.py module (DRY).
    
    Scans output directory for study folders matching pattern:
    - hpo_{backbone} (variant 1, implicit)
    - hpo_{backbone}_v1, hpo_{backbone}_v2, etc.
    
    Returns:
        List of variant names (study folder names)
    """
    base_name = f"hpo_{backbone}"
    variants = []
    
    if not output_dir.exists():
        return variants
    
    for item in output_dir.iterdir():
        if not item.is_dir():
            continue
        
        folder_name = item.name
        if folder_name == base_name:
            variants.append(base_name)
        elif folder_name.startswith(f"{base_name}_v"):
            variants.append(folder_name)
    
    return sorted(variants)
```

**Update HPO study creation:**
- `src/training/hpo/core/study.py` → Extract `run_mode` using `get_run_mode()`
- Pass `root_dir`, `config_dir`, `run_mode` to `create_study_name()`

**Tests:** Test variant generation (v1, v2, v3) with `force_new`

---

## Phase 2: Deterministic Retrieval (Uses HPO Variants)

### Step 2.1: Extract MLflow Query Patterns (DRY)

**File:** `src/infrastructure/tracking/mlflow/queries.py` (NEW)

**Purpose:** Extract common MLflow query patterns from `mlflow_selection.py`

**Implementation:**
```python
from mlflow.tracking import MlflowClient
from typing import List, Optional, Dict, Any, Tuple

def query_runs_by_tags(
    client: MlflowClient,
    experiment_ids: List[str],
    required_tags: Dict[str, str],
    filter_string: str = "",
    max_results: int = 1000,
) -> List[Any]:
    """
    Query MLflow runs filtered by required tags.
    
    Reuses pattern from mlflow_selection.py.
    
    Args:
        client: MLflow client
        experiment_ids: List of experiment IDs to query
        required_tags: Dict of tag_key -> tag_value to filter by
        filter_string: Additional MLflow filter string
        max_results: Maximum number of results
    
    Returns:
        List of runs matching criteria
    """
    all_runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        max_results=max_results,
    )
    
    # Filter for finished runs
    finished_runs = [r for r in all_runs if r.info.status == "FINISHED"]
    
    # Filter by required tags
    valid_runs = []
    for run in finished_runs:
        has_all_tags = all(
            run.data.tags.get(tag_key) == tag_value
            for tag_key, tag_value in required_tags.items()
        )
        if has_all_tags:
            valid_runs.append(run)
    
    return valid_runs

def find_best_run_by_metric(
    runs: List[Any],
    metric_name: str,
    maximize: bool = True,
) -> Optional[Any]:
    """
    Select best run by metric value.
    
    Args:
        runs: List of MLflow runs
        metric_name: Name of metric to optimize
        maximize: True to maximize, False to minimize
    
    Returns:
        Best run or None if no runs have metric
    """
    runs_with_metric = [
        r for r in runs
        if metric_name in r.data.metrics
    ]
    
    if not runs_with_metric:
        return None
    
    if maximize:
        return max(runs_with_metric, key=lambda r: r.data.metrics[metric_name])
    else:
        return min(runs_with_metric, key=lambda r: r.data.metrics[metric_name])

def group_runs_by_variant(
    runs: List[Any],
    variant_tag: str = "code.variant",
) -> Dict[str, List[Any]]:
    """
    Group runs by variant tag.
    
    Args:
        runs: List of MLflow runs
        variant_tag: Tag key for variant (default: "code.variant")
    
    Returns:
        Dict mapping variant -> list of runs
    """
    grouped = {}
    for run in runs:
        variant = run.data.tags.get(variant_tag, "default")
        if variant not in grouped:
            grouped[variant] = []
        grouped[variant].append(run)
    return grouped
```

**Update existing code:**
- `src/evaluation/selection/mlflow_selection.py` → Use `queries.py` utilities
- Keep backward compatibility

**Tests:** Unit tests for query utilities

---

### Step 2.2: Enhance Trial Finder (MLflow-First)

**File:** `src/evaluation/selection/trial_finder.py`

**Changes:** Enhance existing `find_best_trials_for_backbones()` with MLflow-first priority

**Implementation:**
```python
from infrastructure.tracking.mlflow.queries import (
    query_runs_by_tags,
    find_best_run_by_metric,
    group_runs_by_variant,
)
from infrastructure.config.run_mode import get_run_mode

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
    
    This is an enhanced version of find_best_trials_for_backbones().
    """
    if mlflow_client is None:
        from mlflow.tracking import MlflowClient
        mlflow_client = MlflowClient()
    
    objective_metric = selection_config.get("objective", {}).get("metric", "macro-f1")
    best_trials = {}
    
    for backbone in backbone_values:
        backbone_name = backbone.split("-")[0] if "-" in backbone else backbone
        
        # Try sources in priority order
        mlflow_best = None
        study_best = None
        disk_best = None
        
        # Source 1: MLflow (highest priority)
        if backbone in hpo_experiments:
            mlflow_best = _retrieve_from_mlflow(
                backbone, backbone_name, hpo_experiments[backbone],
                objective_metric, selection_scope, mlflow_client
            )
        
        # Source 2: Optuna study (fallback)
        study_best = _retrieve_from_study(
            backbone, backbone_name, hpo_config, data_config,
            root_dir, environment, objective_metric
        )
        
        # Source 3: Disk (last resort)
        if not mlflow_best and not study_best:
            disk_best = _retrieve_from_disk(
                backbone, backbone_name, root_dir, environment, objective_metric
            )
        
        # Select best source (with warnings if disagree)
        best_trial_info = _select_best_source(
            backbone, mlflow_best, study_best, disk_best, objective_metric
        )
        
        if best_trial_info:
            best_trials[backbone] = best_trial_info
    
    return best_trials

def _retrieve_from_mlflow(
    backbone: str,
    backbone_name: str,
    hpo_experiment: Dict[str, str],
    objective_metric: str,
    selection_scope: str,
    client: MlflowClient,
) -> Optional[Dict[str, Any]]:
    """Retrieve best trial from MLflow (highest priority)."""
    from infrastructure.naming.mlflow.tags_registry import TagsRegistry
    
    tags_config = TagsRegistry.load_default()
    study_key_tag = tags_config.key("grouping", "study_key_hash")
    trial_key_tag = tags_config.key("grouping", "trial_key_hash")
    backbone_tag = tags_config.key("process", "backbone")
    variant_tag = tags_config.key("process", "variant")
    stage_tag = tags_config.key("process", "stage")
    
    # Query HPO runs
    required_tags = {
        backbone_tag: backbone_name,
        stage_tag: "hpo",  # Only trial runs, not refit
    }
    
    runs = query_runs_by_tags(
        client=client,
        experiment_ids=[hpo_experiment["id"]],
        required_tags=required_tags,
        max_results=1000,
    )
    
    # Filter by grouping tags and objective metric
    valid_runs = []
    for run in runs:
        has_grouping_tags = (
            study_key_tag in run.data.tags and
            trial_key_tag in run.data.tags
        )
        has_metric = objective_metric in run.data.metrics
        
        if has_grouping_tags and has_metric:
            valid_runs.append(run)
    
    if not valid_runs:
        return None
    
    # Handle selection scope
    if selection_scope == "per_variant":
        # Group by variant, select best per variant
        grouped = group_runs_by_variant(valid_runs, variant_tag)
        # Return dict of variant -> best trial (for now, return overall best)
        # TODO: Implement per-variant return structure
        pass
    
    # Select overall best
    best_run = find_best_run_by_metric(valid_runs, objective_metric, maximize=True)
    
    if not best_run:
        return None
    
    # Convert MLflow run to trial_info dict
    return {
        "backbone": backbone_name,
        "trial_id": best_run.data.tags.get(trial_key_tag, "unknown"),
        "study_key_hash": best_run.data.tags.get(study_key_tag),
        "trial_key_hash": best_run.data.tags.get(trial_key_tag),
        "run_id": best_run.info.run_id,
        "accuracy": best_run.data.metrics.get(objective_metric),
        "source": "mlflow",
        "variant": best_run.data.tags.get(variant_tag, "default"),
    }

def _retrieve_from_study(
    backbone: str,
    backbone_name: str,
    hpo_config: Dict[str, Any],
    data_config: Dict[str, Any],
    root_dir: Path,
    environment: str,
    objective_metric: str,
) -> Optional[Dict[str, Any]]:
    """Retrieve best trial from Optuna study (fallback)."""
    # Reuse existing find_best_trial_from_study() logic
    # ... (existing implementation from trial_finder.py)
    # Add "source": "study" to return dict
    pass

def _retrieve_from_disk(
    backbone: str,
    backbone_name: str,
    root_dir: Path,
    environment: str,
    objective_metric: str,
) -> Optional[Dict[str, Any]]:
    """Retrieve best trial from disk (last resort)."""
    # Reuse existing find_best_trial_in_study_folder() logic
    # ... (existing implementation from trial_finder.py)
    # Add "source": "disk" to return dict
    pass

def _select_best_source(
    backbone: str,
    mlflow_best: Optional[Dict[str, Any]],
    study_best: Optional[Dict[str, Any]],
    disk_best: Optional[Dict[str, Any]],
    objective_metric: str,
) -> Optional[Dict[str, Any]]:
    """Select best source with disagreement warnings."""
    from common.shared.logging_utils import get_logger
    logger = get_logger(__name__)
    
    # Priority: MLflow > Study > Disk
    if mlflow_best:
        # Check if other sources disagree
        if study_best and mlflow_best.get("trial_id") != study_best.get("trial_id"):
            logger.warning(
                f"Sources disagree on best trial for {backbone}: "
                f"MLflow={mlflow_best.get('trial_id')}, Study={study_best.get('trial_id')}. "
                f"Using MLflow (higher priority)."
            )
        return mlflow_best
    
    if study_best:
        return study_best
    
    return disk_best

# Keep existing function for backward compatibility
def find_best_trials_for_backbones(
    backbone_values: list[str],
    hpo_studies: Optional[Dict[str, Any]],
    hpo_config: Dict[str, Any],
    data_config: Dict[str, Any],
    root_dir: Path,
    environment: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Find best trials for multiple backbones (backward compatibility).
    
    DEPRECATED: Use retrieve_best_trials() instead.
    This function now calls retrieve_best_trials() internally.
    """
    import warnings
    warnings.warn(
        "find_best_trials_for_backbones() is deprecated. Use retrieve_best_trials() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Call new function with defaults
    return retrieve_best_trials(
        backbone_values=backbone_values,
        hpo_experiments={},  # Empty - will fallback to study/disk
        benchmark_experiment=None,
        selection_config={"objective": {"metric": hpo_config.get("objective", {}).get("metric", "macro-f1")}},
        hpo_config=hpo_config,
        data_config=data_config,
        root_dir=root_dir,
        environment=environment,
        selection_scope="overall",
    )
```

**Tests:** Test MLflow-first priority, source fallbacks, disagreement warnings

---

### Step 2.3: Update Notebooks (Explicit Retrieval)

**File:** `notebooks/01_orchestrate_training_colab.ipynb`

**Changes:** Add explicit retrieval step before benchmarking

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
print(f"{'Backbone':<15} {'Trial ID':<20} {'Metric':<10} {'Source':<10} {'Variant':<10}")
print("-" * 75)
for backbone, trial_info in best_trials.items():
    print(f"{backbone:<15} {trial_info.get('trial_id', 'unknown'):<20} "
          f"{trial_info.get('accuracy', 0):<10.4f} {trial_info.get('source', 'unknown'):<10} "
          f"{trial_info.get('variant', 'default'):<10}")
```

---

## Phase 3: Idempotent Benchmarking (Uses Both)

### Step 3.1: Build Stable Benchmark Keys (Reuse Fingerprints)

**File:** `src/evaluation/benchmarking/orchestrator.py`

**Implementation:**
```python
from infrastructure.fingerprints import compute_config_hash
import hashlib
import json

def build_benchmark_key(
    backbone: str,
    variant: Optional[str],
    run_id: Optional[str],
    trial_id: str,
    data_config: Dict[str, Any],
    benchmark_config: Dict[str, Any],
) -> str:
    """
    Build stable benchmark identity key.
    
    Reuses existing fingerprint utilities (DRY).
    
    Key format: {backbone}:{variant}:{run_id}:{trial_id}:{data_fp}:{bench_fp}
    """
    # Data fingerprint (dataset version + config hash, excludes local_path)
    data_fp = _compute_data_fingerprint(data_config)
    
    # Benchmark config hash (reuse existing utility)
    bench_fp = compute_config_hash(benchmark_config)
    
    # Variant (or "default" if not applicable)
    variant_str = variant or "default"
    
    # Run ID (or "unknown" if not available)
    run_id_str = run_id or "unknown"
    
    # Build key
    key = f"{backbone}:{variant_str}:{run_id_str}:{trial_id}:{data_fp}:{bench_fp}"
    
    return key

def _compute_data_fingerprint(data_config: Dict[str, Any]) -> str:
    """Compute data fingerprint (version + config hash, excludes local_path)."""
    # Exclude local_path (environment-specific)
    config_copy = {k: v for k, v in data_config.items() if k != "local_path"}
    
    # Normalize and hash
    normalized = json.dumps(config_copy, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
```

---

### Step 3.2: Add Idempotency Check

**File:** `src/evaluation/benchmarking/orchestrator.py`

**Implementation:**
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
    """
    if mlflow_client is None:
        from mlflow.tracking import MlflowClient
        mlflow_client = MlflowClient()
    
    trials_to_benchmark = {}
    
    for backbone, trial_info in best_trials.items():
        # Build stable key
        benchmark_key = build_benchmark_key(
            backbone=backbone,
            variant=trial_info.get("variant"),
            run_id=trial_info.get("run_id"),
            trial_id=trial_info.get("trial_id", trial_info.get("trial_name", "unknown")),
            data_config=data_config,
            benchmark_config=benchmark_config,
        )
        
        # Check if benchmark exists
        if benchmark_already_exists(
            benchmark_key, benchmark_experiment, root_dir, environment, mlflow_client
        ):
            logger.info(f"Skipping {backbone} - benchmark already exists (key: {benchmark_key[:32]}...)")
            continue
        
        trials_to_benchmark[backbone] = trial_info
    
    return trials_to_benchmark

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
        if _benchmark_exists_in_mlflow(benchmark_key, benchmark_experiment, mlflow_client):
            return True
    
    # Fallback to disk
    if _benchmark_exists_on_disk(benchmark_key, root_dir, environment):
        return True
    
    return False

def _benchmark_exists_in_mlflow(
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

def _benchmark_exists_on_disk(
    benchmark_key: str,
    root_dir: Path,
    environment: str,
) -> Optional[Path]:
    """Check if benchmark file exists on disk."""
    cache_dir = root_dir / "outputs" / "benchmarking" / environment / "cache"
    benchmark_file = cache_dir / f"benchmark_{benchmark_key}.json"
    
    return benchmark_file if benchmark_file.exists() else None
```

---

### Step 3.3: Add Run Mode Inheritance

**File:** `src/evaluation/benchmarking/orchestrator.py`

**Implementation:**
```python
from infrastructure.config.run_mode import get_run_mode

def get_benchmark_run_mode(
    benchmark_config: Dict[str, Any],
    hpo_config: Dict[str, Any],
) -> str:
    """
    Get benchmark run mode (inherits from HPO if null).
    
    Uses shared run_mode.py utility (DRY).
    """
    # Get run mode from benchmark config (null = inherit from HPO)
    benchmark_run_mode = benchmark_config.get("run", {}).get("mode")
    
    if benchmark_run_mode is None:
        # Inherit from HPO config
        hpo_run_mode = get_run_mode(hpo_config, default="reuse_if_exists")
        return hpo_run_mode
    
    return get_run_mode(benchmark_config)
```

**Update config:** `config/benchmark.yaml`
```yaml
# Run mode configuration (inherits from HPO if null)
run:
  # Run mode determines overall behavior:
  # - null: Inherit from HPO config (default behavior)
  # - reuse_if_exists: Reuse existing benchmark results if found
  # - force_new: Always create new benchmark run (ignores existing)
  mode: null  # null = inherit from HPO config
```

---

### Step 3.4: Add Variant Completeness Check

**File:** `src/evaluation/benchmarking/orchestrator.py`

**Implementation:**
```python
from infrastructure.config.variants import find_existing_variants
from infrastructure.config.run_mode import get_run_mode
from training.hpo.utils.helpers import find_study_variants

def ensure_all_variants_benchmarked(
    root_dir: Path,
    config_dir: Path,
    backbone: str,
    hpo_config: Dict[str, Any],
    benchmark_config: Dict[str, Any],
    data_config: Dict[str, Any],
    benchmark_experiment: Dict[str, str],
    mlflow_client: Optional[MlflowClient] = None,
) -> List[str]:
    """
    Check all HPO variants and benchmark best trial from each missing variant.
    
    Uses shared variants.py and run_mode.py utilities (DRY).
    """
    from common.shared.platform_detection import detect_platform
    from infrastructure.paths import build_output_path
    
    # Get benchmark strategy
    benchmark_all = benchmark_config.get("benchmark_all_variants", True)
    strategy = benchmark_config.get("benchmark_strategy", "best_per_variant")
    
    if not benchmark_all or strategy == "latest_only":
        return []
    
    # Find all study variants for this backbone
    environment = detect_platform()
    hpo_output_dir = build_output_path(
        root_dir=root_dir,
        config_dir=config_dir,
        process_type="hpo",
        model=backbone,
        environment=environment,
    )
    
    # Scan for all variants (uses shared find_study_variants)
    variants = find_study_variants(hpo_output_dir, backbone)
    
    if not variants:
        return []
    
    # Get run mode (inherited from HPO)
    hpo_run_mode = get_run_mode(hpo_config, default="reuse_if_exists")
    benchmark_run_mode = get_benchmark_run_mode(benchmark_config, hpo_config)
    
    benchmarked_variants = []
    
    for variant_name in variants:
        # Find best trial in this variant
        best_trial = _find_best_trial_in_variant(
            root_dir, config_dir, backbone, variant_name, hpo_config, data_config
        )
        
        if not best_trial:
            continue
        
        # Check if this best trial has been benchmarked
        benchmark_exists = benchmark_already_exists(
            build_benchmark_key(...),  # Build key for this trial
            benchmark_experiment,
            root_dir,
            environment,
            mlflow_client,
        )
        
        if not benchmark_exists:
            # Benchmark missing variant's best trial
            if benchmark_run_mode == "force_new" or not benchmark_exists:
                logger.info(f"Benchmarking best trial from variant: {variant_name}")
                # Call benchmark_best_trials() for this variant
                benchmarked_variants.append(variant_name)
        else:
            benchmarked_variants.append(variant_name)
    
    return benchmarked_variants
```

---

### Step 3.5: Update Benchmarking Function

**File:** `src/evaluation/benchmarking/orchestrator.py`

**Changes:** Update `benchmark_best_trials()` to:
- Accept pre-retrieved trials
- Create stable keys for each trial
- Log `benchmark_key` as MLflow tag
- Respect run mode for idempotency

```python
def benchmark_best_trials(
    best_trials: Dict[str, Dict[str, Any]],
    test_data_path: Path,
    root_dir: Path,
    environment: str,
    data_config: dict,
    hpo_config: dict,
    benchmark_config: Optional[dict] = None,
    # ... other params
) -> Dict[str, Path]:
    """
    Run benchmarking on best trial checkpoints from HPO runs.
    
    Enhanced with:
    - Stable benchmark keys
    - MLflow tag logging
    - Idempotency support
    """
    # ... existing implementation ...
    
    for backbone, trial_info in best_trials.items():
        # Build stable key
        benchmark_key = build_benchmark_key(
            backbone=backbone,
            variant=trial_info.get("variant"),
            run_id=trial_info.get("run_id"),
            trial_id=trial_info.get("trial_id", trial_info.get("trial_name", "unknown")),
            data_config=data_config,
            benchmark_config=benchmark_config or {},
        )
        
        # ... existing benchmarking logic ...
        
        # Log benchmark_key as MLflow tag (for future lookups)
        if benchmark_tracker:
            benchmark_tracker.log_tag("benchmark_key", benchmark_key)
        
        # ... rest of implementation ...
```

---

### Step 3.6: Update Notebooks (Complete Flow)

**File:** `notebooks/01_orchestrate_training_colab.ipynb`

**Changes:** Complete 3-step flow

```python
# Step 1: Explicit Retrieval
best_trials = retrieve_best_trials(...)

# Step 2: Filter Missing Benchmarks
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
    benchmark_results = benchmark_best_trials(
        best_trials=trials_to_benchmark,  # Only missing ones
        ...
    )
else:
    print("✓ All trials already benchmarked - nothing to do!")
```

---

## Implementation Plan Summary

### Phase 1: HPO Foundation (Week 1)
- ✅ Step 1.1: Create `run_mode.py` utility
- ✅ Step 1.2: Generalize `variants.py` module
- ✅ Step 1.3: Add run.mode to HPO configs
- ✅ Step 1.4: Implement HPO variant generation

### Phase 2: Deterministic Retrieval (Week 2)
- ✅ Step 2.1: Extract MLflow query patterns
- ✅ Step 2.2: Enhance trial_finder.py
- ✅ Step 2.3: Update notebooks

### Phase 3: Idempotent Benchmarking (Week 3)
- ✅ Step 3.1: Build stable benchmark keys
- ✅ Step 3.2: Add idempotency check
- ✅ Step 3.3: Add run mode inheritance
- ✅ Step 3.4: Add variant completeness check
- ✅ Step 3.5: Update benchmarking function
- ✅ Step 3.6: Update notebooks

## Testing Strategy

### Unit Tests
- `run_mode.py`: Test extraction, helpers
- `variants.py`: Test computation for both process types
- `mlflow/queries.py`: Test query patterns
- `trial_finder.py`: Test MLflow-first priority, fallbacks
- `benchmarking/orchestrator.py`: Test idempotency, keys

### Integration Tests
- HPO variant creation (v1, v2, v3)
- Retrieval with MLflow-first priority
- Benchmarking idempotency
- Variant completeness check
- End-to-end notebook flow

## Migration & Backward Compatibility

### Backward Compatibility
- Keep `find_best_trials_for_backbones()` with deprecation warning
- Old notebooks continue to work
- Existing configs work (defaults unchanged)

### Gradual Migration
1. Phase 1: Add new utilities (no breaking changes)
2. Phase 2: Enhance existing functions (backward compatible)
3. Phase 3: Add new features (optional)
4. Deprecate old functions gradually

## Success Metrics

- **DRY Compliance**: No code duplication (shared utilities used)
- **Determinism**: Same inputs → same best trial selection
- **Efficiency**: Skipped benchmarks reduce compute time
- **Debuggability**: Explicit retrieval step shows reasoning
- **Flexibility**: Both overall and per-variant modes work

## References

- Existing variant logic: `src/infrastructure/config/training.py`
- Existing run mode extraction: 4+ locations (to be unified)
- Existing MLflow querying: `src/evaluation/selection/mlflow_selection.py`
- Existing trial finding: `src/evaluation/selection/trial_finder.py`
- Existing fingerprints: `src/infrastructure/fingerprints/`

