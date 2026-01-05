# Centralized Path Configuration Guide

This guide explains the centralized directory configuration system and dual file strategy for cache files.

## Overview

The centralized path configuration system provides:
- **Single source of truth** for all directory paths
- **Dual file strategy** for cache files (timestamped history + latest pointer)
- **Flexible path resolution** with pattern support
- **Backward compatibility** with existing hardcoded paths

## Configuration File

All paths are defined in `config/paths.yaml`:

```yaml
base:
  outputs: "outputs"
  notebooks: "notebooks"

outputs:
  hpo: "hpo"
  final_training: "final_training"
  cache: "cache"

cache:
  best_configurations: "best_configurations"
  final_training: "final_training"

patterns:
  final_training: "{backbone}_{run_id}"
  best_config_file: "best_config_{backbone}_{trial}_{timestamp}.json"

cache_strategies:
  best_configurations:
    strategy: "dual"
    timestamped: { enabled: true }
    latest: { enabled: true }
    index: { enabled: true }
```

## Path Resolution API

### Basic Path Resolution

```python
from orchestration.paths import resolve_output_path

# Resolve simple output path
hpo_dir = resolve_output_path(ROOT_DIR, CONFIG_DIR, "hpo")
# -> outputs/hpo

# Resolve cache subdirectory
cache_dir = resolve_output_path(
    ROOT_DIR, CONFIG_DIR, "cache", subcategory="best_configurations"
)
# -> outputs/cache/best_configurations

# Resolve path with pattern
training_dir = resolve_output_path(
    ROOT_DIR, CONFIG_DIR, "final_training",
    backbone="distilbert", run_id="20251227_220407"
)
# -> outputs/final_training/distilbert_20251227_220407
```

### Cache File Paths

```python
from orchestration.paths import get_cache_file_path

# Get latest cache file path
latest_file = get_cache_file_path(
    ROOT_DIR, CONFIG_DIR, "best_configurations", file_type="latest"
)

# Get index file path
index_file = get_cache_file_path(
    ROOT_DIR, CONFIG_DIR, "best_configurations", file_type="index"
)
```

## Dual File Strategy

### Overview

The dual file strategy creates three types of files:

1. **Timestamped files**: `best_config_distilbert_trial_2_20251227_220407.json`
   - Never overwritten
   - Preserves full history
   - Contains complete data + metadata

2. **Latest pointer**: `latest_best_configuration.json`
   - Always updated to point to most recent
   - Easy access for loading
   - Contains same data + reference to timestamped file

3. **Index file**: `index.json`
   - Summary of all entries
   - Limited to N most recent entries
   - Quick browsing and searching

### Saving Cache with Dual Strategy

```python
from orchestration.paths import save_cache_with_dual_strategy
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

timestamped_file, latest_file, index_file = save_cache_with_dual_strategy(
    root_dir=ROOT_DIR,
    config_dir=CONFIG_DIR,
    cache_type="best_configurations",
    data=best_configuration,
    backbone="distilbert",
    identifier="trial_2",
    timestamp=timestamp,
    additional_metadata={
        "experiment_name": "resume_ner_baseline",
    }
)

print(f"Saved: {timestamped_file}")
print(f"Latest: {latest_file}")
print(f"Index: {index_file}")
```

### Loading Cache Files

```python
from orchestration.paths import load_cache_file

# Load latest (default)
config = load_cache_file(
    ROOT_DIR, CONFIG_DIR, "best_configurations", use_latest=True
)

# Load by specific timestamp
config = load_cache_file(
    ROOT_DIR, CONFIG_DIR, "best_configurations",
    use_latest=False,
    specific_timestamp="20251227_220407"
)

# Load by identifier (trial name or run_id)
config = load_cache_file(
    ROOT_DIR, CONFIG_DIR, "best_configurations",
    use_latest=False,
    specific_identifier="trial_2"
)
```

## Directory Structure

```
outputs/
└── cache/
    ├── best_configurations/
    │   ├── best_config_distilbert_trial_2_20251227_220407.json
    │   ├── best_config_distilbert_trial_5_20251228_143022.json
    │   ├── latest_best_configuration.json
    │   └── index.json
    │
    └── final_training/
        ├── final_training_distilbert_20251227_220407_20251227_220500.json
        ├── latest_final_training_cache.json
        └── final_training_index.json
```

## Cache Types

### Best Configurations Cache

Stores best HPO trial configurations:

```python
save_cache_with_dual_strategy(
    root_dir=ROOT_DIR,
    config_dir=CONFIG_DIR,
    cache_type="best_configurations",
    data=best_configuration,
    backbone="distilbert",
    identifier="trial_2",  # trial name
    timestamp=timestamp,
)
```

### Final Training Cache

Stores final training run information:

```python
save_cache_with_dual_strategy(
    root_dir=ROOT_DIR,
    config_dir=CONFIG_DIR,
    cache_type="final_training",
    data={
        "output_dir": str(final_output_dir),
        "backbone": "distilbert",
        "run_id": "20251227_220407",
        "metrics": metrics,
    },
    backbone="distilbert",
    identifier="20251227_220407",  # run_id
    timestamp=timestamp,
)
```

## Pattern Replacement

Path patterns support placeholder replacement:

- `{backbone}` - Replaced with backbone name
- `{run_id}` - Replaced with run ID
- `{trial}` - Replaced with trial name
- `{timestamp}` - Replaced with timestamp

Example:
```yaml
patterns:
  final_training: "{backbone}_{run_id}"
```

Resolves to: `outputs/final_training/distilbert_20251227_220407`

## Configuration Strategies

### Dual Strategy (Default)

Creates timestamped files, latest pointer, and index:

```yaml
cache_strategies:
  best_configurations:
    strategy: "dual"
    timestamped:
      enabled: true
      keep_all: true
    latest:
      enabled: true
      include_timestamped_ref: true
    index:
      enabled: true
      max_entries: 20
```

### Latest Only

Only maintains latest pointer (no history):

```yaml
cache_strategies:
  best_configurations:
    strategy: "latest_only"
    timestamped:
      enabled: false
    latest:
      enabled: true
```

### Timestamped Only

Only creates timestamped files (no latest pointer):

```yaml
cache_strategies:
  best_configurations:
    strategy: "timestamped_only"
    timestamped:
      enabled: true
    latest:
      enabled: false
```

## Backward Compatibility

The system maintains backward compatibility:

1. **Legacy files still created**: Old cache files in `notebooks/` are still created
2. **Default paths**: If `config/paths.yaml` doesn't exist, sensible defaults are used
3. **Fallback loading**: Loading functions check legacy locations if new cache not found

## Examples

### Example 1: Save Best Configuration

```python
from orchestration.paths import save_cache_with_dual_strategy
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

timestamped_file, latest_file, index_file = save_cache_with_dual_strategy(
    root_dir=ROOT_DIR,
    config_dir=CONFIG_DIR,
    cache_type="best_configurations",
    data=best_configuration,
    backbone=best_configuration.get('backbone'),
    identifier=best_configuration.get('trial_name'),
    timestamp=timestamp,
)

# Also save legacy file
LEGACY_FILE = ROOT_DIR / "notebooks" / "best_configuration_cache.json"
save_json(LEGACY_FILE, best_configuration)
```

### Example 2: Load for Continued Training

```python
from orchestration.paths import load_cache_file

# Load latest final training cache
training_cache = load_cache_file(
    ROOT_DIR, CONFIG_DIR, "final_training", use_latest=True
)

if training_cache:
    checkpoint_dir = Path(training_cache["output_dir"]) / "checkpoint"
    print(f"Using checkpoint: {checkpoint_dir}")
```

### Example 3: Browse Historical Configurations

```python
from orchestration.paths import get_cache_file_path
from shared.json_cache import load_json

# Load index to browse all configurations
index_file = get_cache_file_path(
    ROOT_DIR, CONFIG_DIR, "best_configurations", file_type="index"
)
index_data = load_json(index_file, default={"entries": []})

print("Available configurations:")
for entry in index_data.get("entries", []):
    print(f"  {entry['timestamp']}: {entry['backbone']} - {entry.get('best_value', 'N/A')}")
```

## Troubleshooting

### Config File Not Found

If `config/paths.yaml` doesn't exist, the system uses defaults. Create the file to customize paths.

### Cache Files Not Found

The loading functions check multiple locations in priority order:
1. Specific timestamp/identifier (if provided)
2. Latest pointer file
3. Index file (most recent entry)
4. Legacy location

### Pattern Not Resolving

Ensure placeholders in patterns match the kwargs provided:
- Pattern: `"{backbone}_{run_id}"`
- kwargs: `backbone="distilbert", run_id="20251227_220407"`

## Migration Guide

### From Hardcoded Paths

**Before:**
```python
BEST_CONFIG_CACHE_FILE = ROOT_DIR / "notebooks" / "best_configuration_cache.json"
save_json(BEST_CONFIG_CACHE_FILE, best_configuration)
```

**After:**
```python
from orchestration.paths import save_cache_with_dual_strategy

timestamped_file, latest_file, index_file = save_cache_with_dual_strategy(
    ROOT_DIR, CONFIG_DIR, "best_configurations",
    best_configuration, backbone, identifier, timestamp
)
```

### From Single Cache File

**Before:**
```python
cache_file = ROOT_DIR / "notebooks" / "final_training_cache.json"
cache_data = load_json(cache_file)
```

**After:**
```python
from orchestration.paths import load_cache_file

cache_data = load_cache_file(
    ROOT_DIR, CONFIG_DIR, "final_training", use_latest=True
)
```

## API Reference

See `src/orchestration/paths.py` for complete API documentation:

- `load_paths_config()` - Load paths configuration
- `resolve_output_path()` - Resolve output paths
- `get_cache_file_path()` - Get cache file paths
- `get_timestamped_cache_filename()` - Generate timestamped filenames
- `get_cache_strategy_config()` - Get cache strategy configuration
- `save_cache_with_dual_strategy()` - Save cache with dual file strategy
- `load_cache_file()` - Load cache files with flexible options

