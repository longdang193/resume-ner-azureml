# Unify Run Mode Logic - Single Source of Truth

<!-- Implementation plan for creating a unified run.mode utility used across all stages (HPO, final training, best model selection) -->

## Overview

### Purpose

This plan introduces a **single source of truth for run mode logic** to eliminate code duplication and ensure consistent behavior across all stages:
- **HPO (Hyperparameter Optimization)**: Controls whether to create new study or reuse existing (with variant support)
- **Benchmarking**: Inherits run.mode from HPO, ensures all variants' best trials are benchmarked before model selection
- **Final Training**: Controls variant creation and checkpoint reuse
- **Best Model Selection**: Controls cache reuse

**Why it matters**: Currently, `run.mode` extraction is duplicated in 4+ locations with inconsistent defaults and patterns. This creates maintenance burden and potential bugs when behavior diverges.

### Scope

**In scope**
- Create unified `run_mode.py` utility module
- Refactor all existing `run.mode` extraction to use the utility
- Add `run.mode` support to HPO configs (currently missing) with variant support (v1, v2, v3...)
- Add `run.mode` support to benchmarking configs (inherits from HPO when null)
- Add variant completeness check for benchmarking (benchmark best trial from each variant before model selection)
- Update config files to use consistent `run.mode` structure
- Ensure backward compatibility (defaults remain the same)

**Out of scope**
- Changing the actual behavior of `run.mode` values (only unifying extraction)
- Adding new run mode values (can be done later if needed)
- Refactoring how each stage *uses* run mode (only unifying how it's *extracted*)

### Guiding Principles

- **DRY (Don't Repeat Yourself)**: Single function for run mode extraction
- **Backward Compatibility**: Default behavior unchanged
- **Type Safety**: Use type hints for better IDE support
- **Consistent Defaults**: Same default (`reuse_if_exists`) everywhere
- **Config-Driven**: All behavior controlled via YAML configs

## Goals & Success Criteria

### Goals

- **G1**: Single source of truth for run mode extraction
- **G2**: Consistent `run.mode` structure across all config files (HPO, benchmarking, final training, selection)
- **G3**: HPO configs support `run.mode` with variant support (currently missing)
- **G4**: Benchmarking inherits run.mode from HPO and ensures all variants are benchmarked
- **G5**: Zero code duplication for run mode logic

### Success Criteria

- [ ] All stages use `get_run_mode()` utility function
- [ ] No duplicate `config.get("run", {}).get("mode", ...)` patterns remain
- [ ] HPO configs have `run.mode` section with variant support (v1, v2, v3...)
- [ ] Benchmarking config inherits run.mode from HPO (when null)
- [ ] Variant completeness check benchmarks best trial from each variant before model selection
- [ ] All existing tests pass
- [ ] Type hints added for better IDE support
- [ ] Documentation updated

## Current State Analysis

### Existing Behavior

**Current run mode extraction (duplicated in 4+ places):**

1. **Best Model Selection** (`src/evaluation/selection/cache.py:97`):
   ```python
   run_mode = selection_config.get("run", {}).get("mode", "reuse_if_exists")
   if run_mode == "force_new":
       return None  # Skip cache
   ```

2. **Final Training - Executor** (`src/training/execution/executor.py:190-191`):
   ```python
   run_mode = final_training_yaml.get("run", {}).get("mode", "reuse_if_exists")
   if run_mode == "reuse_if_exists":
       # Check if checkpoint exists and reuse
   ```

3. **Final Training - Config Loader** (`src/infrastructure/config/training.py:142-143`):
   ```python
   run_mode = final_training_config.get("run", {}).get("mode", "reuse_if_exists")
   # Used for variant resolution
   ```

4. **Selection (duplicate)** (`src/selection/cache.py:97`):
   ```python
   run_mode = selection_config.get("run", {}).get("mode", "reuse_if_exists")
   if run_mode == "force_new":
       return None  # Skip cache
   ```

**Current config files:**

- ✅ `config/best_model_selection.yaml`: Has `run.mode` (simple: `force_new` vs `reuse_if_exists`)
- ✅ `config/final_training.yaml`: Has `run.mode` (complex: includes `resume_if_incomplete`)
- ❌ `config/hpo/smoke.yaml`: **Missing** `run.mode` (uses `study_name` to control behavior)
- ❌ `config/hpo/prod.yaml`: **Missing** `run.mode` (uses `study_name` to control behavior)
- ❌ `config/benchmark.yaml`: **Missing** `run.mode` (no inheritance or variant completeness check)

**HPO study name logic** (`src/training/hpo/utils/helpers.py:85-128`):
- Currently uses `study_name` from config to control new vs existing study
- If `study_name` is provided, uses it directly
- If not provided, defaults to `f"hpo_{backbone}"` (for resume) or `f"hpo_{backbone}_{run_id}"` (for new)

### Pain Points / Limitations

- **L1**: Code duplication - same extraction pattern in 4+ files
- **L2**: Inconsistent defaults - all use `"reuse_if_exists"` but hardcoded separately
- **L3**: HPO doesn't have `run.mode` - must manually change `study_name` to force new study
- **L4**: No type hints - IDE can't help with valid values
- **L5**: Hard to change default behavior - must update multiple files

### Architectural / SRP Issues

- **Mixed responsibilities**: Run mode extraction mixed with business logic
- **Tight coupling**: Each stage reimplements the same extraction logic
- **Hard-coded defaults**: Default value scattered across codebase

## High-Level Design

### Architecture Overview

```
Config Files (YAML)
├── config/best_model_selection.yaml (has run.mode)
├── config/final_training.yaml (has run.mode)
├── config/hpo/smoke.yaml (needs run.mode)
├── config/hpo/prod.yaml (needs run.mode)
└── config/benchmark.yaml (needs run.mode, inherits from HPO)
         │
         v
Unified Utility: infrastructure.config.run_mode
├── get_run_mode(config, default) -> RunMode
├── is_force_new(config) -> bool
└── is_reuse_if_exists(config) -> bool
         │
         v
Stage-Specific Usage
├── HPO: Controls study_name generation (with variants v1, v2, v3...)
├── Benchmarking: Inherits from HPO, benchmarks best trial from each variant
├── Final Training: Controls variant resolution
└── Best Model Selection: Controls cache reuse
```

### Responsibility Breakdown

| Component | Responsibility |
|-----------|---------------|
| `run_mode.py` | Extract and validate run mode from config (single source of truth) |
| HPO Stage | Use `run.mode` to control study_name generation with variants (v1, v2, v3...) |
| Benchmarking Stage | Inherit `run.mode` from HPO, benchmark best trial from each variant |
| Final Training Stage | Use `run.mode` for variant resolution (existing) |
| Best Model Selection | Use `run.mode` for cache control (existing) |

## Module & File Structure

### New Files to Create

- `src/infrastructure/config/run_mode.py` — Unified run mode utility
  - `get_run_mode(config, default)` — Extract run mode with consistent defaults
  - `is_force_new(config)` — Check if mode is force_new
  - `is_reuse_if_exists(config)` — Check if mode is reuse_if_exists
  - Type definitions: `RunMode` literal type

### Files to Modify

- `src/infrastructure/config/__init__.py`
  - Export `get_run_mode`, `is_force_new`, `is_reuse_if_exists` from `run_mode.py`

- `src/evaluation/selection/cache.py`
  - Replace inline extraction with `get_run_mode(selection_config)`

- `src/selection/cache.py` (duplicate)
  - Replace inline extraction with `get_run_mode(selection_config)`

- `src/training/execution/executor.py`
  - Replace inline extraction with `get_run_mode(final_training_yaml)`

- `src/infrastructure/config/training.py`
  - Replace inline extraction with `get_run_mode(final_training_config)`

- `src/training/hpo/utils/helpers.py`
  - Add `_compute_next_study_variant()` helper (similar to final training's variant logic)
  - Add `find_study_variants()` helper to scan for existing variants
  - Modify `create_study_name()` to accept `run_mode`, `root_dir`, `config_dir` parameters
  - When `run_mode == "force_new"`, compute next variant (v1, v2, v3...) instead of appending run_id

- `src/training/hpo/core/study.py`
  - Extract `run_mode` from HPO config using `get_run_mode()`
  - Pass `root_dir`, `config_dir`, and `run_mode` to `create_study_name()`

- `src/evaluation/benchmarking/orchestrator.py`
  - Add run mode inheritance from HPO config
  - Add `ensure_all_variants_benchmarked()` function
  - Check all variants and benchmark best trial from each missing variant

- `config/benchmark.yaml`
  - Add `run.mode` section (null = inherit from HPO)
  - Add `benchmark_all_variants` and `benchmark_strategy` options

- `config/hpo/smoke.yaml`
  - Add `run.mode` section (matching final_training.yaml pattern)
  - Change `study_name` from hardcoded to optional (null = auto-generate)

- `config/hpo/prod.yaml`
  - Add `run.mode` section (matching final_training.yaml pattern)

### Files Explicitly Not Touched

- `config/best_model_selection.yaml` — Already has correct structure
- `config/final_training.yaml` — Already has correct structure (may add comments for consistency)

## Detailed Design per Component

### Component: `run_mode.py`

**Responsibility (SRP)**
- Single source of truth for extracting and validating run mode from configuration

**Inputs**
- `config: Dict[str, Any]` — Configuration dictionary (from YAML)
- `default: RunMode = "reuse_if_exists"` — Default mode if not specified

**Outputs**
- `RunMode` — Literal type: `"reuse_if_exists" | "force_new" | "resume_if_incomplete"`

**Public API**

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
    
    Args:
        config: Configuration dictionary (e.g., from YAML)
        default: Default mode if not specified (default: "reuse_if_exists")
    
    Returns:
        Run mode string: "reuse_if_exists", "force_new", or "resume_if_incomplete"
    
    Example:
        >>> config = {"run": {"mode": "force_new"}}
        >>> get_run_mode(config)
        'force_new'
        
        >>> config = {}  # No run.mode specified
        >>> get_run_mode(config)
        'reuse_if_exists'
    """
    return config.get("run", {}).get("mode", default)

def is_force_new(config: Dict[str, Any]) -> bool:
    """Check if run.mode is force_new."""
    return get_run_mode(config) == "force_new"

def is_reuse_if_exists(config: Dict[str, Any]) -> bool:
    """Check if run.mode is reuse_if_exists."""
    return get_run_mode(config) == "reuse_if_exists"
```

**Implementation Notes**
- Simple wrapper around dict access - no complex logic
- Type hints use `Literal` for better IDE autocomplete
- Default value matches current behavior across all stages
- No validation needed (YAML loader handles type checking)

### Component: HPO Study Name Generation with Variants

**Responsibility (SRP)**
- Generate study name based on run mode and config, using variant numbers (like final training)

**Changes to `create_study_name()`**

```python
def _compute_next_study_variant(
    root_dir: Path,
    config_dir: Path,
    backbone: str,
    base_name: str
) -> int:
    """
    Compute next available study variant number.
    
    Scans existing study folders/names to find highest variant number,
    then returns next available (starts at 1 if none exist).
    
    Similar to final training's _compute_next_variant().
    
    Args:
        root_dir: Project root directory
        config_dir: Config directory
        backbone: Model backbone name
        base_name: Base study name (e.g., "hpo_distilbert")
    
    Returns:
        Next available variant number (1, 2, 3, ...)
    """
    from common.shared.platform_detection import detect_platform
    from infrastructure.paths import build_output_path
    
    environment = detect_platform()
    backbone_name = backbone.split("-")[0] if "-" in backbone else backbone
    
    # Scan output directory for existing studies
    output_base = build_output_path(
        root_dir=root_dir,
        config_dir=config_dir,
        process_type="hpo",
        model=backbone_name,
        environment=environment
    )
    
    if not output_base.exists():
        return 1
    
    # Find existing variants by scanning study folders
    variants = []
    for item in output_base.iterdir():
        if not item.is_dir():
            continue
        
        folder_name = item.name
        
        # Try to extract variant number
        if folder_name == base_name:
            variants.append(1)
        elif folder_name.startswith(f"{base_name}_v"):
            try:
                variant_num = int(folder_name.split("_v")[-1])
                variants.append(variant_num)
            except ValueError:
                pass
    
    if not variants:
        return 1
    
    return max(variants) + 1

def find_study_variants(
    output_dir: Path,
    backbone: str
) -> List[str]:
    """
    Find all study variants for a given backbone.
    
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

def create_study_name(
    backbone: str,
    run_id: str,
    should_resume: bool,
    checkpoint_config: Optional[Dict[str, Any]] = None,
    hpo_config: Optional[Dict[str, Any]] = None,
    run_mode: Optional[str] = None,
    root_dir: Optional[Path] = None,  # NEW: For variant computation
    config_dir: Optional[Path] = None,  # NEW: For variant computation
) -> str:
    """
    Create Optuna study name with variant support (like final training).
    
    When run_mode == "force_new", computes next variant number (v1, v2, v3...).
    When run_mode == "reuse_if_exists", uses base name for resumability.
    
    Args:
        backbone: Model backbone name
        run_id: Unique run ID (not used for variants, kept for backward compat)
        should_resume: Whether resuming from checkpoint
        checkpoint_config: Optional checkpoint configuration
        hpo_config: Optional HPO configuration
        run_mode: Optional run mode (extracted from config if not provided)
        root_dir: Project root directory (required for variant computation)
        config_dir: Config directory (required for variant computation)
    
    Returns:
        Study name string (e.g., "hpo_distilbert_v1", "hpo_distilbert_v2")
    """
    checkpoint_config = checkpoint_config or {}
    hpo_config = hpo_config or {}
    checkpoint_enabled = checkpoint_config.get("enabled", False)
    
    # Get run_mode from config if not provided
    if run_mode is None:
        from infrastructure.config.run_mode import get_run_mode
        combined_config = {**hpo_config, **checkpoint_config}
        run_mode = get_run_mode(combined_config)
    
    # Check for custom study_name in checkpoint config first, then HPO config
    study_name_template = checkpoint_config.get("study_name") or hpo_config.get("study_name")
    
    if study_name_template:
        study_name = study_name_template.replace("{backbone}", backbone)
        # If force_new and we have root_dir/config_dir, compute variant
        if run_mode == "force_new" and root_dir and config_dir:
            variant = _compute_next_study_variant(
                root_dir, config_dir, backbone, study_name
            )
            return f"{study_name}_v{variant}" if variant > 1 else study_name
        # If reuse_if_exists, use base name
        return study_name
    
    # Default behavior when no custom study_name is provided
    base_name = f"hpo_{backbone}"
    
    if run_mode == "force_new":
        # Compute next variant when force_new
        if root_dir and config_dir:
            variant = _compute_next_study_variant(
                root_dir, config_dir, backbone, base_name
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
```

**Implementation Notes**
- Backward compatible: `run_mode`, `root_dir`, `config_dir` are optional
- When `force_new` and `root_dir`/`config_dir` provided: computes next variant (v1, v2, v3...)
- When `force_new` but no `root_dir`/`config_dir`: falls back to run_id appending (backward compat)
- When `reuse_if_exists`: uses base name without variant suffix
- Variant computation scans filesystem similar to final training's variant logic

### Component: Benchmarking Integration with HPO Variants

**Responsibility (SRP)**
- Inherit run.mode from HPO config (when null/default)
- Check all HPO variants for missing benchmarks before model selection
- Automatically benchmark best trial from each missing variant to ensure completeness

**Configuration (benchmark.yaml):**

```yaml
# Run mode configuration (inherits from HPO if null)
run:
  # Run mode determines overall behavior:
  # - null: Inherit from HPO config (default behavior)
  # - reuse_if_exists: Reuse existing benchmark results if found
  # - force_new: Always create new benchmark run (ignores existing)
  mode: null  # null = inherit from HPO config

benchmarking:
  # Batch sizes to test during benchmarking
  batch_sizes: [1]  # [1, 8, 16]
  
  # Number of iterations per batch size for statistical significance
  iterations: 10
  
  # Number of warmup iterations before measurement
  warmup_iterations: 10
  
  # Maximum sequence length for benchmarking
  max_length: 512
  
  # Device preference (null = auto-detect, "cuda", or "cpu")
  device: null
  
  # Test data source (relative to config dir or absolute path)
  test_data: null

  # Variant completeness check (before model selection)
  # Benchmark best trial from each variant to ensure fair comparison
  benchmark_all_variants: true  # true = benchmark best from each variant, false = only latest
  benchmark_strategy: "best_per_variant"  # "best_per_variant" or "latest_only"

# Output configuration
output:
  filename: "benchmark.json"
```

**Implementation Logic:**

1. **Run mode inheritance:**
   ```python
   # In benchmarking orchestrator or before model selection
   from infrastructure.config.run_mode import get_run_mode
   
   # Get run mode from HPO config
   hpo_run_mode = get_run_mode(hpo_config, default="reuse_if_exists")
   
   # Get run mode from benchmark config (inherits from HPO if null)
   benchmark_run_mode = get_run_mode(
       benchmark_config, 
       default=hpo_run_mode  # Inherit from HPO
   )
   ```

2. **Variant completeness check (before model selection):**
   ```python
   def ensure_all_variants_benchmarked(
       root_dir: Path,
       config_dir: Path,
       backbone: str,
       hpo_config: Dict[str, Any],
       benchmark_config: Dict[str, Any],
   ) -> List[str]:
       """
       Check all HPO variants and benchmark best trial from each missing variant.
       
       Scans outputs/hpo/{env}/{backbone}/ for all study variants.
       For each variant, finds best trial and checks if benchmark exists.
       Benchmarks missing variants before model selection.
       
       Args:
           root_dir: Project root directory
           config_dir: Config directory
           backbone: Model backbone name
           hpo_config: HPO configuration
           benchmark_config: Benchmark configuration
       
       Returns:
           List of variant names that were benchmarked (newly or existing)
       """
       from common.shared.platform_detection import detect_platform
       from infrastructure.paths import build_output_path
       from training.hpo.utils.helpers import find_study_variants
       from infrastructure.config.run_mode import get_run_mode
       
       environment = detect_platform()
       
       # Get benchmark strategy
       benchmark_all = benchmark_config.get("benchmark_all_variants", True)
       strategy = benchmark_config.get("benchmark_strategy", "best_per_variant")
       
       if not benchmark_all or strategy == "latest_only":
           # Only benchmark latest variant (current behavior)
           return []
       
       # Find all study variants for this backbone
       hpo_output_dir = build_output_path(
           root_dir=root_dir,
           config_dir=config_dir,
           process_type="hpo",
           model=backbone,
           environment=environment
       )
       
       # Scan for all variants (hpo_distilbert, hpo_distilbert_v1, hpo_distilbert_v2, etc.)
       variants = find_study_variants(hpo_output_dir, backbone)
       
       if not variants:
           return []
       
       # Get run mode (inherited from HPO)
       hpo_run_mode = get_run_mode(hpo_config, default="reuse_if_exists")
       benchmark_run_mode = get_run_mode(
           benchmark_config,
           default=hpo_run_mode
       )
       
       benchmarked_variants = []
       
       for variant_name in variants:
           # Find best trial in this variant
           best_trial = find_best_trial_in_variant(
               root_dir, config_dir, backbone, variant_name
           )
           
           if not best_trial:
               continue
           
           # Check if this best trial has been benchmarked
           benchmark_exists = check_benchmark_exists(
               root_dir, config_dir, backbone, variant_name, best_trial
           )
           
           if not benchmark_exists:
               # Benchmark missing variant's best trial
               if benchmark_run_mode == "force_new" or not benchmark_exists:
                   logger.info(f"Benchmarking best trial from variant: {variant_name}")
                   benchmark_trial(
                       root_dir, config_dir, backbone, variant_name, best_trial,
                       hpo_config, benchmark_config
                   )
                   benchmarked_variants.append(variant_name)
           else:
               benchmarked_variants.append(variant_name)
       
       return benchmarked_variants
   ```

3. **Integration point (before model selection):**
   ```python
   # In model selection orchestrator, before find_best_model_from_mlflow()
   
   # Ensure all HPO variants' best trials are benchmarked
   benchmark_config = load_yaml(config_dir / "benchmark.yaml")
   benchmark_all = benchmark_config.get("benchmark_all_variants", True)
   
   if benchmark_all:
       for backbone in backbones:
           ensure_all_variants_benchmarked(
               root_dir, config_dir, backbone,
               hpo_config, benchmark_config
           )
   
   # Now proceed with model selection (all variants have benchmarks)
   best_model = find_best_model_from_mlflow(...)
   ```

**Implementation Notes:**
- Runs **before model selection** (not after each HPO variant)
- Only benchmarks best trial from each missing variant (efficient)
- Inherits run.mode from HPO automatically
- Can be disabled with `benchmark_all_variants: false` or `benchmark_strategy: "latest_only"`
- Ensures fair comparison across all variants

## Configuration & Controls

### Configuration Sources

All run mode configuration comes from YAML files:
- `config/best_model_selection.yaml`
- `config/final_training.yaml`
- `config/hpo/smoke.yaml` (to be added)
- `config/hpo/prod.yaml` (to be added)
- `config/benchmark.yaml` (to be added, inherits from HPO)

### Example Configuration

**Unified structure (all configs should follow this pattern):**

```yaml
# Run mode configuration (unified behavior control)
run:
  # Run mode determines overall behavior:
  # - reuse_if_exists: Reuse existing if found (default)
  # - force_new: Always create new, ignore existing
  # - resume_if_incomplete: Resume if incomplete, otherwise new (final_training only)
  mode: force_new
```

**HPO-specific (smoke.yaml):**

```yaml
# Run mode configuration (unified behavior control)
run:
  # Run mode determines overall behavior:
  # - reuse_if_exists: Reuse existing study if found (default, respects auto_resume)
  # - force_new: Always create a new study with next variant (v1, v2, v3...)
  mode: force_new

checkpoint:
  enabled: true
  # study_name: null  # null = auto-generate base as "hpo_{backbone}" (default)
  #                    # When run.mode=force_new, code will compute next variant (v1, v2, v3...)
  storage_path: "{study_name}/study.db"
  auto_resume: true
  save_only_best: true
```

**Benchmarking-specific (benchmark.yaml):**

```yaml
# Run mode configuration (inherits from HPO if null)
run:
  # Run mode determines overall behavior:
  # - null: Inherit from HPO config (default behavior)
  # - reuse_if_exists: Reuse existing benchmark results if found
  # - force_new: Always create new benchmark run (ignores existing)
  mode: null  # null = inherit from HPO config

benchmarking:
  # ... existing benchmarking config ...
  
  # Variant completeness check (before model selection)
  # Benchmark best trial from each variant to ensure fair comparison
  benchmark_all_variants: true  # true = benchmark best from each variant, false = only latest
  benchmark_strategy: "best_per_variant"  # "best_per_variant" or "latest_only"
```

### Validation Rules

- **Required**: `run.mode` is optional (defaults to `"reuse_if_exists"`)
- **Type**: Must be string
- **Allowed values**: `"reuse_if_exists"`, `"force_new"`, `"resume_if_incomplete"` (final_training only)
- **Defaults**: `"reuse_if_exists"` if not specified

## Implementation Steps

### Phase 1: Create Utility Module

1. **Create `src/infrastructure/config/run_mode.py`**
   - Implement `get_run_mode()`, `is_force_new()`, `is_reuse_if_exists()`
   - Add type hints with `RunMode` literal type
   - Add docstrings with examples

2. **Update `src/infrastructure/config/__init__.py`**
   - Export `get_run_mode`, `is_force_new`, `is_reuse_if_exists`
   - Add to `__all__` if present

3. **Add unit tests**
   - Test `get_run_mode()` with various config structures
   - Test default behavior
   - Test helper functions

### Phase 2: Refactor Existing Code

4. **Refactor Best Model Selection**
   - Update `src/evaluation/selection/cache.py`
   - Update `src/selection/cache.py` (duplicate)
   - Verify tests still pass

5. **Refactor Final Training**
   - Update `src/training/execution/executor.py`
   - Update `src/infrastructure/config/training.py`
   - Verify tests still pass

### Phase 3: Add HPO Support with Variants

6. **Add variant computation helper**
   - Create `_compute_next_study_variant()` in `src/training/hpo/utils/helpers.py`
   - Similar to final training's `_compute_next_variant()` but for HPO studies
   - Scans `outputs/hpo/{env}/{backbone}/` for existing study folders
   - Extracts variant numbers from study names (e.g., `hpo_distilbert_v2` → variant 2)
   - Returns next available variant number

7. **Add variant scanning helper**
   - Create `find_study_variants()` in `src/training/hpo/utils/helpers.py`
   - Scans output directory for all study variants
   - Returns list of variant names (for benchmarking completeness check)

8. **Update HPO study name generation**
   - Modify `src/training/hpo/utils/helpers.py::create_study_name()`
   - Add `root_dir` and `config_dir` parameters for variant computation
   - Add `run_mode` parameter support
   - Update logic to compute variants when `force_new` (instead of appending run_id)
   - When `force_new`: compute next variant → `hpo_{backbone}_v{N}`
   - When `reuse_if_exists`: use base name → `hpo_{backbone}`

9. **Update HPO study creation**
   - Modify `src/training/hpo/core/study.py`
   - Extract `run_mode` from config using `get_run_mode()`
   - Pass `root_dir` and `config_dir` to `create_study_name()`
   - Pass `run_mode` to `create_study_name()`

10. **Update HPO config files**
    - Add `run.mode` section to `config/hpo/smoke.yaml`
    - Add `run.mode` section to `config/hpo/prod.yaml`
    - Make `study_name` optional (can be null)

### Phase 4: Add Benchmarking Integration

11. **Add run mode inheritance**
    - Update `src/evaluation/benchmarking/orchestrator.py`
    - Extract run mode from benchmark config
    - If null, inherit from HPO config using `get_run_mode()`

12. **Add variant completeness check**
    - Create `ensure_all_variants_benchmarked()` function in `src/evaluation/benchmarking/orchestrator.py`
    - Scans all HPO variants for missing benchmarks
    - For each variant, finds best trial and checks if benchmarked
    - Benchmarks best trial from each missing variant before model selection
    - Respects `benchmark_all_variants` and `benchmark_strategy` config options

13. **Integrate before model selection**
    - Update model selection orchestrator
    - Call `ensure_all_variants_benchmarked()` before `find_best_model_from_mlflow()`
    - Only if `benchmark_all_variants: true` in config

14. **Update benchmarking config**
    - Add `run.mode` section to `config/benchmark.yaml` (null = inherit)
    - Add `benchmark_all_variants: true` option
    - Add `benchmark_strategy: "best_per_variant"` option

### Phase 5: Testing & Validation

15. **Run all tests**
    - Unit tests for `run_mode.py`
    - Integration tests for each stage
    - Test variant completeness check
    - Verify backward compatibility

16. **Manual testing**
    - Test HPO with `run.mode: force_new` (should create variants v1, v2, v3)
    - Test HPO with `run.mode: reuse_if_exists` (should reuse existing)
    - Test benchmarking inheritance (null inherits from HPO)
    - Test variant completeness check (benchmarks best trial from missing variants)
    - Test model selection with multiple variants (all should be considered)
    - Test final training with both modes
    - Test best model selection with both modes

### Phase 6: Documentation

17. **Update documentation**
    - Add `run.mode` explanation to relevant config files
    - Update HPO documentation to mention `run.mode` and variants
    - Update benchmarking documentation to mention variant completeness check
    - Add examples to docstrings

## Testing Strategy

### Unit Tests

**File**: `tests/infrastructure/config/test_run_mode.py`

```python
def test_get_run_mode_with_explicit_mode():
    """Test get_run_mode with explicit mode in config."""
    config = {"run": {"mode": "force_new"}}
    assert get_run_mode(config) == "force_new"

def test_get_run_mode_with_default():
    """Test get_run_mode with no run.mode specified."""
    config = {}
    assert get_run_mode(config) == "reuse_if_exists"

def test_get_run_mode_with_custom_default():
    """Test get_run_mode with custom default."""
    config = {}
    assert get_run_mode(config, default="force_new") == "force_new"

def test_is_force_new():
    """Test is_force_new helper."""
    assert is_force_new({"run": {"mode": "force_new"}}) == True
    assert is_force_new({"run": {"mode": "reuse_if_exists"}}) == False

def test_is_reuse_if_exists():
    """Test is_reuse_if_exists helper."""
    assert is_reuse_if_exists({"run": {"mode": "reuse_if_exists"}}) == True
    assert is_reuse_if_exists({"run": {"mode": "force_new"}}) == False
```

### Integration Tests

- **HPO**: Test that `run.mode: force_new` creates new study with next variant (v1, v2, v3)
- **HPO**: Test that `run.mode: reuse_if_exists` reuses existing study
- **Benchmarking**: Test run mode inheritance from HPO (null inherits)
- **Benchmarking**: Test variant completeness check (benchmarks best trial from each missing variant)
- **Benchmarking**: Test `benchmark_all_variants: false` (only benchmarks latest)
- **Model Selection**: Test that all variants are considered when selecting best model
- **Final Training**: Test variant resolution with both modes
- **Best Model Selection**: Test cache behavior with both modes

### Edge Cases

- Config with `run: {}` (empty dict) → should use default
- Config with `run: null` → should use default
- Config with invalid mode value → YAML loader will catch (type error)
- Missing config entirely → should use default

## Backward Compatibility & Migration

### What Remains Compatible

- ✅ All existing configs continue to work (default behavior unchanged)
- ✅ All existing code paths preserved
- ✅ Default value (`"reuse_if_exists"`) matches current behavior

### Deprecated Behavior

- None - this is purely additive/refactoring

### Migration Steps

**For HPO configs (smoke.yaml, prod.yaml):**

1. Add `run.mode` section:
   ```yaml
   run:
     mode: force_new  # or reuse_if_exists
   ```

2. Optionally make `study_name` null (will auto-generate):
   ```yaml
   checkpoint:
     study_name: null  # Optional: auto-generate as "hpo_{backbone}"
   ```

**No code changes needed** - existing code will automatically use the utility once refactored.

## Documentation Updates

### New Documentation

- `docs/RUN_MODE_CONFIGURATION.md` (optional) - Explains run.mode across all stages

### Updated Documentation

- `config/hpo/smoke.yaml` - Add comments explaining `run.mode` and variants
- `config/hpo/prod.yaml` - Add comments explaining `run.mode` and variants
- `config/benchmark.yaml` - Add comments explaining `run.mode` inheritance and variant completeness
- Inline docstrings in `run_mode.py`

## Rollout & Validation Checklist

- [ ] **Phase 1**: Create `run_mode.py` utility with tests
- [ ] **Phase 2**: Refactor existing code (best model selection, final training)
- [ ] **Phase 3**: Add HPO support with variants (study name generation + configs)
- [ ] **Phase 4**: Add benchmarking integration (inheritance + variant completeness)
- [ ] **Phase 5**: Run all tests (unit + integration)
- [ ] **Phase 6**: Manual testing of each stage
- [ ] **Phase 7**: Update documentation
- [ ] **Phase 8**: Code review
- [ ] **Phase 9**: Merge and deploy

## Appendix

### Current Run Mode Usage Locations

1. `src/evaluation/selection/cache.py:97` - Best model selection cache
2. `src/selection/cache.py:97` - Selection cache (duplicate)
3. `src/training/execution/executor.py:190-191` - Final training executor
4. `src/infrastructure/config/training.py:142-143` - Final training config loader

### Config Files Status

| Config File | Has `run.mode`? | Status |
|------------|----------------|--------|
| `config/best_model_selection.yaml` | ✅ Yes | Simple structure |
| `config/final_training.yaml` | ✅ Yes | Complex (includes `resume_if_incomplete`) |
| `config/hpo/smoke.yaml` | ❌ No | Needs to be added (with variant support) |
| `config/hpo/prod.yaml` | ❌ No | Needs to be added (with variant support) |
| `config/benchmark.yaml` | ❌ No | Needs to be added (inherits from HPO) |

### Example: Before vs After

**Before (duplicated):**
```python
# In cache.py
run_mode = selection_config.get("run", {}).get("mode", "reuse_if_exists")

# In executor.py  
run_mode = final_training_yaml.get("run", {}).get("mode", "reuse_if_exists")

# In training.py
run_mode = final_training_config.get("run", {}).get("mode", "reuse_if_exists")
```

**After (unified):**
```python
from infrastructure.config.run_mode import get_run_mode

# In cache.py
run_mode = get_run_mode(selection_config)

# In executor.py
run_mode = get_run_mode(final_training_yaml)

# In training.py
run_mode = get_run_mode(final_training_config)
```

### HPO Study Name Examples

**Before (manual study_name change):**
```yaml
checkpoint:
  study_name: "hpo_{backbone}_smoke_test_path_testing_23"  # Must change manually
```

**After (run.mode controls behavior with variants):**
```yaml
run:
  mode: force_new  # Automatically computes next variant (v1, v2, v3...)

checkpoint:
  study_name: null  # Auto-generates as "hpo_{backbone}_v{N}" when force_new
                    # Example: "hpo_distilbert_v1", "hpo_distilbert_v2", etc.
```

**Example study names:**
- First run with `force_new`: `hpo_distilbert_v1` (or just `hpo_distilbert` if variant 1)
- Second run with `force_new`: `hpo_distilbert_v2`
- Third run with `force_new`: `hpo_distilbert_v3`
- With `reuse_if_exists`: Always uses `hpo_distilbert` (base name, no variant)

### Benchmarking Variant Completeness Example

**Scenario:**
```
HPO creates:
- hpo_distilbert_v1 → best trial: trial_5 (F1: 0.85)
- hpo_distilbert_v2 → best trial: trial_3 (F1: 0.87)
- hpo_distilbert_v3 → best trial: trial_2 (F1: 0.86)

Before model selection (with benchmark_all_variants: true):
→ Detects v1, v2, v3 variants
→ Finds best trial in each variant
→ Checks if benchmarks exist for each best trial
→ Benchmarks missing ones:
  - v1's best trial (trial_5) ✓
  - v2's best trial (trial_3) ✓
  - v3's best trial (trial_2) ✓

Model selection:
→ Can now compare: v1's best vs v2's best vs v3's best
→ Selects overall best across all variants (v2's trial_3)
```

