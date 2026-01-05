<!-- e8104c15-91ae-4447-90ad-297e58cdd0eb 56ba7c69-85fb-4021-8796-98a644970bd3 -->
# Auto-Increment MLflow Run Names (Reserve/Commit System)

## Overview

Implement atomic counter-based auto-increment for MLflow run names for HPO and Benchmarking processes using a reserve/commit pattern. Counter keys use stable identity (`run_key_hash`) instead of display names to prevent mixing unrelated studies. Allocation is idempotent and never reuses numbers (allows gaps for robustness).

## Key Design Principles

1. **Stable Identity Keys**: Counter keys use `run_key_hash` (not `base_name`) to ensure numbering never mixes unrelated studies
2. **Reserve/Commit Pattern**: Two-phase allocation prevents duplicates on crashes
3. **Never Reuse Numbers**: Always increment from max, allow gaps (robustness over perfect sequencing)
4. **Unified Format**: `base.{n}` format across all processes
5. **Backward Compatible**: Opt-in via config, defaults to current behavior

## Implementation Steps

### 1. Add Configuration to mlflow.yaml

**File**: `config/mlflow.yaml`

Add to `naming.run_name` section:

```yaml
run_name:
  max_length: 100
  shorten_fingerprints: true
  
  # Auto-increment configuration
  auto_increment:
    enabled: false  # Global toggle (default: false for backward compatibility)
    processes:
      hpo: true  # Enable auto-increment for HPO run names
      benchmarking: true  # Enable auto-increment for benchmarking run names
    format: "{base}.{version}"  # Unified format: base.{n}
    stale_reservation_minutes: 30  # Cleanup threshold for stale reservations
```

### 2. Create Atomic Counter Store Functions (Reserve/Commit)

**File**: `src/orchestration/jobs/tracking/mlflow_index.py` (extend existing file)

**Add functions**:

- `get_run_name_counter_path(root_dir: Path, config_dir: Optional[Path] = None) -> Path`
  - Returns path to counter file: `outputs/cache/run_name_counter.json`
  - Uses same cache directory as `mlflow_index.json`

- `reserve_run_name_version(counter_key: str, run_id: str, root_dir: Path, config_dir: Optional[Path] = None) -> int`
  - **Reserve phase**: Acquire lock → read allocations → find max committed version for counter_key → increment → add entry with status="reserved" → atomic write → release lock
  - Returns reserved version number (starts at 1 if key doesn't exist)
  - Uses existing atomic write pattern (temp file + rename)
  - Key format: `"{project}:{process_type}:{run_key_hash}:{env}"` (uses stable identity hash)
  - Example key: `"resume-ner:hpo:abc123def456...:local"`
  - Never reuses numbers: Always increments from max committed version, allows gaps

- `commit_run_name_version(counter_key: str, run_id: str, version: int, root_dir: Path, config_dir: Optional[Path] = None) -> None`
  - **Commit phase**: Acquire lock → find matching reserved entry by (counter_key, run_id, version) → update status="committed" + set committed_at timestamp → atomic write → release lock
  - Called after MLflow run is successfully created
  - If entry not found (crash recovery), log warning but don't fail (idempotent)

- `cleanup_stale_reservations(root_dir: Path, config_dir: Optional[Path] = None, stale_minutes: Optional[int] = None) -> int`
  - **Cleanup phase**: Find "reserved" entries older than stale_minutes → mark as "expired" (or remove)
  - Called on startup/cleanup to handle crashed processes
  - Returns count of cleaned entries
  - Uses config value for stale_minutes if not provided

**Counter file structure**:

```json
{
  "allocations": [
    {
      "counter_key": "resume-ner:hpo:abc123def456...:local",
      "version": 23,
      "run_id": "run_abc123...",
      "status": "committed",
      "reserved_at": "2025-01-01T12:00:00Z",
      "committed_at": "2025-01-01T12:00:05Z"
    },
    {
      "counter_key": "resume-ner:hpo:abc123def456...:local",
      "version": 24,
      "run_id": "run_xyz789...",
      "status": "reserved",
      "reserved_at": "2025-01-01T12:10:00Z",
      "committed_at": null
    },
    {
      "counter_key": "resume-ner:hpo:abc123def456...:local",
      "version": 25,
      "run_id": "run_crashed...",
      "status": "expired",
      "reserved_at": "2025-01-01T11:00:00Z",
      "committed_at": null
    }
  ]
}
```

**Key implementation details**:

- Use existing `_acquire_lock()` and `_release_lock()` functions
- Use atomic write pattern: write to `.tmp` file, then rename
- When finding max version: only count "committed" entries (ignore "reserved" and "expired")
- Lock timeout: 10 seconds (reuse existing timeout)

### 3. Add Counter Key Generation Utilities

**File**: `src/orchestration/jobs/tracking/mlflow_naming.py`

**Add functions**:

- `_extract_base_name_from_study_name(study_name: str) -> str`
  - Extract base name from HPO study_name (remove version suffix if present)
  - Input: `"hpo_distilbert_smoke_test_3.23"` → Output: `"hpo_distilbert_smoke_test_3"`
  - Pattern: Remove trailing `.{digits}` or `_{digits}` if present
  - **Note**: Only used for display name generation, NOT for counter key

- `_strip_env_prefix(run_name: str, env: str) -> str`
  - Remove environment prefix for consistent base name extraction
  - Input: `"local_hpo_distilbert_smoke_test_3"`, env=`"local"` → Output: `"hpo_distilbert_smoke_test_3"`
  - **Note**: Only used for display name generation, NOT for counter key

- `_build_counter_key(project: str, process_type: str, run_key_hash: str, env: str) -> str`
  - Build counter key using **stable identity hash** (not base_name)
  - Format: `"{project}:{process_type}:{run_key_hash}:{env}"`
  - Example: `"resume-ner:hpo:abc123def456...:local"`
  - **Key improvement**: Uses `run_key_hash` (stable identity) instead of `base_name` (display name)
  - This ensures numbering never mixes unrelated studies that share similar display names

### 4. Update Config Loader

**File**: `src/orchestration/jobs/tracking/mlflow_config_loader.py`

**Add function**:

- `get_auto_increment_config(config_dir: Optional[Path] = None, process_type: Optional[str] = None) -> Dict[str, Any]`
  - Load auto-increment config from mlflow.yaml
  - If process_type provided, return enabled status for that process
  - Returns: `{"enabled": bool, "processes": {...}, "format": str, "stale_reservation_minutes": int}`
  - Defaults: `enabled=False`, `processes={}`, `format="{base}.{version}"`, `stale_reservation_minutes=30`

### 5. Update build_mlflow_run_name for HPO

**File**: `src/orchestration/jobs/tracking/mlflow_naming.py`

**Modify HPO parent run logic**:

- Build `run_key` from context: `"hpo:{model}:{trial_id}"` (where trial_id is study_name)
- Compute `run_key_hash` from `run_key` using `build_mlflow_run_key_hash()`
- Extract base name from study_name (for display only, remove existing version if present)
- Build counter key using **stable identity**: `project:hpo:{run_key_hash}:{env}`
- If auto-increment enabled: 
  - Call `reserve_run_name_version()` to reserve a version (returns version number)
  - Format: `"{env}_{base_name}.{version}"` (e.g., `"local_hpo_distilbert_smoke_test_3.24"`)
  - **Note**: Version is reserved but not committed yet (commit happens after MLflow run creation)
- If disabled: use study_name as-is (current behavior)

**Key changes in `build_mlflow_run_name()`**:

```python
if context.process_type == "hpo":
    env_prefix = f"{context.environment}_" if context.environment else ""
    if context.trial_id and context.trial_id.startswith("hpo_"):
        # Parent run: extract base for display, use run_key_hash for counter key
        base_name = _extract_base_name_from_study_name(context.trial_id)
        base_without_env = _strip_env_prefix(base_name, context.environment)
        
        # Check if auto-increment enabled
        auto_inc_config = get_auto_increment_config(config_dir, "hpo")
        if auto_inc_config.get("enabled") and auto_inc_config.get("processes", {}).get("hpo"):
            # Build run_key and hash for stable identity
            run_key = build_mlflow_run_key(context)
            run_key_hash = build_mlflow_run_key_hash(run_key)
            
            # Build counter key using stable identity (not base_name)
            counter_key = _build_counter_key(
                naming_config.get("project_name", "resume-ner"),
                "hpo",
                run_key_hash,  # Use hash, not base_name
                context.environment
            )
            # Reserve version (commit happens after run creation)
            # Note: run_id not available yet, use temporary placeholder
            temp_run_id = f"pending_{datetime.now().isoformat()}"
            version = reserve_run_name_version(counter_key, temp_run_id, root_dir, config_dir)
            return f"{env_prefix}{base_without_env}.{version}"
        else:
            # Disabled: use study_name as-is (current behavior)
            return f"{env_prefix}{context.trial_id}"
```

**Important**: After MLflow run is created, caller must commit the reservation (see step 8).

### 6. Update build_mlflow_run_name for Benchmarking

**File**: `src/orchestration/jobs/tracking/mlflow_naming.py`

**Modify benchmarking logic**:

- Build `run_key` from context: `"benchmark:{model}:{trial_id}"`
- Compute `run_key_hash` from `run_key` using `build_mlflow_run_key_hash()`
- Build base name for display: `f"benchmark_{context.model}"`
- Build counter key using **stable identity**: `project:benchmarking:{run_key_hash}:{env}`
- If auto-increment enabled: 
  - Call `reserve_run_name_version()` to reserve a version
  - Format: `"{base_name}.{version}"` (e.g., `"benchmark_distilbert.1"`)
  - **Note**: Version is reserved but not committed yet (commit happens after MLflow run creation)
- If disabled: use current format with trial_id

**Key changes**:

```python
elif context.process_type == "benchmarking":
    base_name = f"benchmark_{context.model}"
    
    # Check if auto-increment enabled
    auto_inc_config = get_auto_increment_config(config_dir, "benchmarking")
    if auto_inc_config.get("enabled") and auto_inc_config.get("processes", {}).get("benchmarking"):
        # Build run_key and hash for stable identity
        run_key = build_mlflow_run_key(context)
        run_key_hash = build_mlflow_run_key_hash(run_key)
        
        # Build counter key using stable identity (not base_name)
        counter_key = _build_counter_key(
            naming_config.get("project_name", "resume-ner"),
            "benchmarking",
            run_key_hash,  # Use hash, not base_name
            context.environment
        )
        # Reserve version (commit happens after run creation)
        temp_run_id = f"pending_{datetime.now().isoformat()}"
        version = reserve_run_name_version(counter_key, temp_run_id, root_dir, config_dir)
        return f"{base_name}.{version}"
    else:
        # Disabled: use current format
        trial_short = context.trial_id[:20] if context.trial_id and len(context.trial_id) > 20 else (context.trial_id or "unknown")
        return f"{base_name}_{trial_short}"
```

**Important**: After MLflow run is created, caller must commit the reservation (see step 8).

### 7. Pass root_dir to build_mlflow_run_name

**File**: `src/orchestration/jobs/tracking/mlflow_naming.py`

**Update function signature**:

- Add optional `root_dir: Optional[Path] = None` parameter
- Infer from `output_dir` if not provided: `output_dir.parent.parent` (if `output_dir` exists)
- Fallback: `Path.cwd()` if `output_dir` is None

**Update call sites**:

- `src/orchestration/jobs/hpo/local_sweeps.py`: Pass `output_dir.parent.parent` as `root_dir`
- `src/orchestration/jobs/tracking/mlflow_tracker.py`: Infer from `output_dir` parameter
- Other call sites: Infer from `output_dir` or use `Path.cwd()` as fallback

### 8. Add Commit Logic After Run Creation

**File**: `src/orchestration/jobs/tracking/mlflow_tracker.py`

**Update `start_sweep_run()` and `start_benchmark_run()`**:

- After MLflow run is successfully created (inside context manager, before return):
  - Store `counter_key` and `version` in RunHandle (extend RunHandle dataclass) OR return separately
  - Call `commit_run_name_version(counter_key, run_id, version, root_dir, config_dir)`
  - Handle errors gracefully (log warning, don't fail run creation)
  - If commit fails, reservation will be cleaned up by `cleanup_stale_reservations()` later

**Alternative approach** (if RunHandle extension is complex):

- Return tuple `(RunHandle, counter_key, version)` from `build_mlflow_run_name()` or store in context
- Caller commits after run creation

**File**: `src/orchestration/jobs/hpo/local_sweeps.py`

**Update HPO parent run creation**:

- After `tracker.start_sweep_run()` returns RunHandle:
  - Extract `counter_key` and `version` from context (store during reservation)
  - Call `commit_run_name_version(counter_key, run_id, version, root_dir, config_dir)` with actual run_id
  - Handle errors gracefully (log warning, don't fail)

**Note**: Need to track `counter_key` and `version` between reservation and commit. Options:

- Store in NamingContext (add optional fields)
- Store in module-level dict keyed by temp_run_id
- Return from `build_mlflow_run_name()` and pass through to tracker

### 9. Add Cleanup on Startup

**File**: `src/orchestration/jobs/hpo/local_sweeps.py`

**Add cleanup at start of HPO sweep**:

- Call `cleanup_stale_reservations(root_dir, config_dir)` at the start of `run_local_sweep()`
- This handles crashed processes that left "reserved" entries
- Log count of cleaned entries
- Use config value for stale_minutes

**File**: `src/orchestration/jobs/tracking/mlflow_tracker.py` (if benchmarking has startup)

**Add cleanup for benchmarking** (if applicable):

- Similar cleanup call at start of benchmarking process

## Files to Modify

1. `config/mlflow.yaml` - Add auto_increment configuration
2. `src/orchestration/jobs/tracking/mlflow_index.py` - Add counter store functions (reserve, commit, cleanup)
3. `src/orchestration/jobs/tracking/mlflow_naming.py` - Add utilities and update `build_mlflow_run_name()`
4. `src/orchestration/jobs/tracking/mlflow_config_loader.py` - Add config loader
5. `src/orchestration/jobs/tracking/mlflow_tracker.py` - Add commit logic after run creation
6. `src/orchestration/jobs/hpo/local_sweeps.py` - Update calls, add cleanup, add commit logic

## Testing Considerations

- Test with empty counter file (first run starts at 1)
- Test with existing committed allocations (increment works correctly)
- Test concurrent runs (file locking prevents race conditions)
- Test with disabled auto-increment (backward compatibility)
- Test base name extraction (handles various study_name formats)
- Test env prefix stripping (local_, colab_, kaggle_)
- Test counter key uniqueness (different run_key_hash don't conflict, even with same base_name)
- Test reserve/commit flow (reservation → run creation → commit)
- Test crash recovery (reserved but never committed → cleanup marks as expired → next run increments correctly)
- Test stale cleanup (old reserved entries are marked expired)

## Edge Cases

- First run: Counter key doesn't exist → reserve version 1
- Missing config: Disabled by default, graceful fallback
- Lock failures: Fall back to base_name (no increment, log warning)
- Invalid base name: Use as-is if extraction fails
- Missing root_dir: Infer from output_dir or use Path.cwd()
- Counter file corruption: Recreate with empty allocations list (log warning)
- **Reserved but never committed**: Handled by `cleanup_stale_reservations()` (marks as expired after stale_minutes)
- **Crash after reserve, before commit**: Cleanup will mark as expired, next run will increment from max committed version
- **Duplicate reservation attempt**: Lock prevents concurrent reservations, always increments from max committed
- **Commit for non-existent reservation**: Log warning, don't fail (idempotent)
- **Same run_key_hash, different base_name**: Counter key is unique, numbering is separate (correct behavior)
- **Different run_key_hash, same base_name**: Counter keys are different, numbering is separate (correct behavior)

## Documentation Notes

- Emphasize: Run names are UI-only, never used for retrieval
- Identity is via tags: `code.run_key_hash`, `code.study_key_hash`
- Version numbers are for readability, not functional requirements
- Counter keys use stable identity (run_key_hash) to prevent mixing unrelated studies
- Reserve/commit pattern ensures no duplicates even on crashes
- Numbers are never reused (allows gaps for robustness)

### To-dos

- [x] Add auto_increment configuration section to config/mlflow.yaml (enabled, processes, format)
- [x] Add get_auto_increment_config() function to mlflow_config_loader.py
- [x] Add get_run_name_counter_path() function to mlflow_index.py to get counter file path
- [x] Add reserve_run_name_version() and commit_run_name_version() functions with atomic increment logic (reserve/commit pattern)
- [x] Add _extract_base_name_from_study_name() utility to extract base name (remove version suffix)
- [x] Add _strip_env_prefix() utility to remove env prefix from run names
- [x] Add build_counter_key() utility to build counter key string (project:process:run_key_hash:env)
- [x] Update build_mlflow_run_name() signature to accept optional root_dir and output_dir parameters
- [x] Update build_mlflow_run_name() HPO logic to use auto-increment counter when enabled
- [x] Update build_mlflow_run_name() benchmarking logic to use auto-increment counter when enabled
- [x] Update build_mlflow_run_name() call sites to work with new signature (backward compatible, optional parameters)
- [x] Verify HPO study_name base extraction works correctly in local_sweeps.py (commit logic already implemented)