# Notebook Updates Needed for Centralized Paths

This document outlines the manual updates needed in `notebooks/01_orchestrate_training_colab.ipynb` to use the centralized path configuration with dual file strategy.

## Summary

The centralized path system is implemented and ready to use. The following notebook cells need to be updated to use the new path resolution functions instead of hardcoded paths.

## Update 1: Best Configuration Cache Saving

**Location**: Cell that saves `best_configuration` (around line 2058)

**Current Code:**
```python
save_json(BEST_CONFIG_CACHE_FILE, best_configuration)

print(f"Best configuration selected:")
print(f"  Backbone: {best_configuration.get('backbone')}")
print(f"  Trial: {best_configuration.get('trial_name')}")
print(f"  Best {hpo_config['objective']['metric']}: {best_configuration.get('selection_criteria', {}).get('best_value'):.4f}")

# Show selection reasoning (if available)
selection_criteria = best_configuration.get('selection_criteria', {})
if 'reason' in selection_criteria:
    print(f"  Selection reason: {selection_criteria['reason']}")
if 'accuracy_diff_from_best' in selection_criteria:
    print(f"  Accuracy difference from best: {selection_criteria['accuracy_diff_from_best']:.4f}")

# Show all candidates (if available)
if 'all_candidates' in selection_criteria:
    print(f"\nAll candidates considered:")
    for c in selection_criteria['all_candidates']:
        marker = "✓" if c['backbone'] == best_configuration.get('backbone') else " "
        print(f"  {marker} {c['backbone']}: acc={c['accuracy']:.4f}, speed={c['speed_score']:.2f}x")

print(f"\nSaved to: {BEST_CONFIG_CACHE_FILE}")
```

**Updated Code:**
```python
from orchestration.paths import (
    resolve_output_path,
    save_cache_with_dual_strategy,
)
from datetime import datetime

# Use centralized path resolution
BEST_CONFIG_CACHE_DIR = resolve_output_path(
    ROOT_DIR,
    CONFIG_DIR,
    "cache",
    subcategory="best_configurations"
)

# Generate timestamp and identifiers
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backbone = best_configuration.get('backbone', 'unknown')
trial_name = best_configuration.get('trial_name', 'unknown')

# Save using dual file strategy
timestamped_file, latest_file, index_file = save_cache_with_dual_strategy(
    root_dir=ROOT_DIR,
    config_dir=CONFIG_DIR,
    cache_type="best_configurations",
    data=best_configuration,
    backbone=backbone,
    identifier=trial_name,
    timestamp=timestamp,
    additional_metadata={
        "experiment_name": experiment_config.name if 'experiment_config' in locals() else "unknown",
        "hpo_study_name": hpo_config.get('study_name', 'unknown') if 'hpo_config' in locals() else "unknown",
    }
)

# Also save to legacy location for backward compatibility
LEGACY_CACHE_FILE = ROOT_DIR / "notebooks" / "best_configuration_cache.json"
save_json(LEGACY_CACHE_FILE, best_configuration)

print(f"Best configuration selected:")
print(f"  Backbone: {backbone}")
print(f"  Trial: {trial_name}")
print(f"  Best {hpo_config['objective']['metric']}: {best_configuration.get('selection_criteria', {}).get('best_value'):.4f}")

# Show selection reasoning (if available)
selection_criteria = best_configuration.get('selection_criteria', {})
if 'reason' in selection_criteria:
    print(f"  Selection reason: {selection_criteria['reason']}")
if 'accuracy_diff_from_best' in selection_criteria:
    print(f"  Accuracy difference from best: {selection_criteria['accuracy_diff_from_best']:.4f}")

# Show all candidates (if available)
if 'all_candidates' in selection_criteria:
    print(f"\nAll candidates considered:")
    for c in selection_criteria['all_candidates']:
        marker = "✓" if c['backbone'] == backbone else " "
        print(f"  {marker} {c['backbone']}: acc={c['accuracy']:.4f}, speed={c['speed_score']:.2f}x")

print(f"\n✓ Saved timestamped cache: {timestamped_file}")
print(f"✓ Updated latest cache: {latest_file}")
print(f"✓ Updated index: {index_file}")
print(f"✓ Saved legacy cache (backward compatibility): {LEGACY_CACHE_FILE}")
print(f"\n  Cache directory: {BEST_CONFIG_CACHE_DIR}")
```

## Update 2: Best Configuration Cache Loading

**Location**: Cell that loads `best_configuration` (around line 2144)

**Current Code:**
```python
best_configuration = load_json(BEST_CONFIG_CACHE_FILE, default=None)

if best_configuration is None:
    raise FileNotFoundError(
        f"Best configuration cache not found: {BEST_CONFIG_CACHE_FILE}\n"
        f"Please run Step P1-3.6: Best Configuration Selection first."
    )
```

**Updated Code:**
```python
from orchestration.paths import load_cache_file

# Try loading from centralized cache first
best_configuration = load_cache_file(
    ROOT_DIR, CONFIG_DIR, "best_configurations", use_latest=True
)

# Fallback to legacy location
if best_configuration is None:
    LEGACY_CACHE_FILE = ROOT_DIR / "notebooks" / "best_configuration_cache.json"
    best_configuration = load_json(LEGACY_CACHE_FILE, default=None)

if best_configuration is None:
    raise FileNotFoundError(
        f"Best configuration cache not found.\n"
        f"Please run Step P1-3.6: Best Configuration Selection first.\n"
        f"Cache directory: {resolve_output_path(ROOT_DIR, CONFIG_DIR, 'cache', subcategory='best_configurations')}"
    )
```

## Update 3: Final Training Cache Saving

**Location**: Cell that saves final training cache (around line 2394)

**Current Code:**
```python
# Save cache file with actual paths
save_json(FINAL_TRAINING_CACHE_FILE, {
    "output_dir": str(final_output_dir),
    "backbone": final_training_config["backbone"],
    "config": final_training_config,
})
```

**Updated Code:**
```python
from orchestration.paths import (
    resolve_output_path,
    save_cache_with_dual_strategy,
)
from datetime import datetime

# Prepare cache data
final_training_cache_data = {
    "output_dir": str(final_output_dir),
    "backbone": final_training_config["backbone"],
    "run_id": final_training_run_id,
    "config": final_training_config,
    "metrics": metrics,  # Include metrics if available
}

# Save using dual file strategy
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backbone = final_training_config["backbone"].replace('-', '_').replace('/', '_')
run_id = final_training_run_id.replace('-', '_')

timestamped_file, latest_file, index_file = save_cache_with_dual_strategy(
    root_dir=ROOT_DIR,
    config_dir=CONFIG_DIR,
    cache_type="final_training",
    data=final_training_cache_data,
    backbone=backbone,
    identifier=run_id,
    timestamp=timestamp,
    additional_metadata={
        "checkpoint_path": str(checkpoint_source) if checkpoint_source else None,
    }
)

# Also save to legacy location for backward compatibility
LEGACY_CACHE_FILE = ROOT_DIR / "notebooks" / "final_training_cache.json"
save_json(LEGACY_CACHE_FILE, final_training_cache_data)

print(f"✓ Saved timestamped final training cache: {timestamped_file}")
print(f"✓ Updated latest cache: {latest_file}")
print(f"✓ Updated index: {index_file}")
print(f"✓ Saved legacy cache: {LEGACY_CACHE_FILE}")
```

## Update 4: Final Training Cache Loading (Conversion Step)

**Location**: Cell that loads final training cache for conversion (around line 2770)

**Current Code:**
```python
training_cache = load_json(FINAL_TRAINING_CACHE_FILE, default=None)

if training_cache is None:
    # Try to restore from Google Drive (if available)
    if restore_from_drive("final_training_cache.json", FINAL_TRAINING_CACHE_FILE, is_directory=False):
        training_cache = load_json(FINAL_TRAINING_CACHE_FILE, default=None)
    else:
        raise FileNotFoundError(
            f"Final training cache not found locally or in backup.\n"
            f"Please run Step P1-3.7: Final Training first."
        )
```

**Updated Code:**
```python
from orchestration.paths import load_cache_file

# Try loading from centralized cache first
training_cache = load_cache_file(
    ROOT_DIR, CONFIG_DIR, "final_training", use_latest=True
)

# Fallback to legacy location
if training_cache is None:
    LEGACY_CACHE_FILE = ROOT_DIR / "notebooks" / "final_training_cache.json"
    training_cache = load_json(LEGACY_CACHE_FILE, default=None)

# Try to restore from Google Drive if still not found
if training_cache is None:
    if restore_from_drive("final_training_cache.json", LEGACY_CACHE_FILE, is_directory=False):
        training_cache = load_json(LEGACY_CACHE_FILE, default=None)

if training_cache is None:
    raise FileNotFoundError(
        f"Final training cache not found locally or in backup.\n"
        f"Please run Step P1-3.7: Final Training first."
    )
```

## Update 5: Continued Training Cache Loading

**Location**: Cell in continued training section (around line 2493)

**Current Code:**
```python
previous_cache_path = ROOT_DIR / continued_training_config.get(
    "previous_training_cache", 
    "notebooks/best_configuration_cache.json"
)

previous_training = load_json(previous_cache_path, default=None)
```

**Updated Code:**
```python
from orchestration.paths import load_cache_file

# Try loading from centralized cache first
previous_training = load_cache_file(
    ROOT_DIR, CONFIG_DIR, "final_training", use_latest=True
)

# Fallback to legacy location
if previous_training is None:
    previous_cache_path = ROOT_DIR / continued_training_config.get(
        "previous_training_cache", 
        "notebooks/final_training_cache.json"
    )
    previous_training = load_json(previous_cache_path, default=None)
```

## Update 6: Remove Hardcoded Path Definitions

**Location**: Cells that define `BEST_CONFIG_CACHE_FILE` and `FINAL_TRAINING_CACHE_FILE`

**Remove or Update:**
```python
# OLD - Can be removed or kept for backward compatibility
BEST_CONFIG_CACHE_FILE = ROOT_DIR / "notebooks" / "best_configuration_cache.json"
FINAL_TRAINING_CACHE_FILE = ROOT_DIR / "notebooks" / "final_training_cache.json"
```

These are no longer needed if using centralized paths, but can be kept for backward compatibility.

## Testing the Updates

After making these updates:

1. Run the best configuration selection cell - should create timestamped, latest, and index files
2. Run the final training cell - should create final training cache with dual strategy
3. Run continued training - should load from new cache system
4. Verify files are created in `outputs/cache/best_configurations/` and `outputs/cache/final_training/`
5. Verify legacy files are still created in `notebooks/` for backward compatibility

## Benefits After Update

- ✅ Historical cache files preserved (never overwritten)
- ✅ Easy access via latest pointer files
- ✅ Index files for browsing all configurations
- ✅ Centralized path management
- ✅ Backward compatibility maintained

