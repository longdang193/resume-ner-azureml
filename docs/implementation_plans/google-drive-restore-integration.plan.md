# Google Drive Restore Integration for Colab

## Overview

### Purpose

This plan extends the Google Drive backup system to include automatic restore functionality across the entire ML pipeline. When running on Google Colab, if any local files are missing (e.g., after session disconnect), the system will automatically restore them from Google Drive backups before attempting to use them. This ensures zero data loss and seamless continuation of interrupted processes at any pipeline stage - HPO, benchmarking, training, conversion, or any intermediate step.

**Key Benefits:**
- Automatic restore from Drive when local files missing (anywhere in pipeline)
- Seamless resume from any interruption point
- Works for all outputs: checkpoints, benchmarks, cache files, models
- Zero manual intervention required
- Transparent to user - works automatically
- Pipeline can continue effortlessly from any step

### Scope

**In scope:**
- General restore helper function usable across entire pipeline
- Integrate Drive restore into HPO checkpoint system
- Restore for all pipeline outputs: checkpoints, benchmarks, cache files, models
- Automatic restore before any file access (proactive restore)
- Automatic backup after all operations complete
- Restore for: HPO checkpoints, trial checkpoints, benchmark results, training checkpoints, conversion outputs, cache files
- Seamless resume from any pipeline stage interruption

**Out of scope:**
- Real-time sync during operations (backup happens after completion)
- Conflict resolution (Drive always wins if local missing)
- Backup compression or deduplication
- Other cloud storage providers
- Restoring files outside outputs/ directory

### Guiding Principles

- **Mirror local development workflow** - work with local files normally
- **Check local first** - if local file exists, use it (same as local dev)
- **If local missing** → check Drive backup
  - If Drive has it → restore to local (resume scenario)
  - If Drive doesn't have it → proceed normally (create new, same as local dev)
- **At session end** → backup to Drive (save progress before disconnect)
- Fail gracefully (continue if restore fails)
- Transparent operation (user doesn't need to know)
- Backward compatible (works without Drive)
- Universal pattern (same restore approach everywhere)

## Goals & Success Criteria

### Goals

- G1: General restore helper works seamlessly across entire pipeline
- G2: All pipeline outputs can be restored from Drive automatically
- G3: HPO checkpoint system automatically restores from Drive if local missing
- G4: Benchmarking can use Drive backup if local results missing
- G5: Model conversion automatically restores final training checkpoint from Drive
- G6: Automatic backup of all outputs after operations complete
- G7: Seamless resume from any interruption point in pipeline
- G8: Zero data loss for interrupted processes at any stage

### Success Criteria

- [ ] General restore helper (`ensure_restored_from_drive`) implemented
- [ ] HPO can resume from Drive backup after session disconnect
- [ ] Benchmarking can use Drive backup if local results missing
- [ ] Model conversion can use Drive backup of training checkpoint
- [ ] All outputs automatically backed up after operations
- [ ] Restore happens transparently before existence checks
- [ ] System gracefully handles missing Drive backups
- [ ] All restore operations logged for observability
- [ ] Pipeline can resume from any interruption point
- [ ] Restore works seamlessly across all pipeline stages

## Current State Analysis

### Existing Behavior

**HPO Checkpoint System:**
- `setup_checkpoint_storage()` checks if local `study.db` exists
- If exists → resume, if not → start fresh
- No Drive restore integration
- Checkpoints saved locally only

**Benchmarking:**
- Benchmark results saved to `benchmark.json` in trial directories
- No restore from Drive if missing
- Benchmarking re-runs if results missing

**Model Conversion:**
- Loads checkpoint path from training cache
- Checks if local checkpoint directory exists
- Raises error if missing (no restore attempt)
- No Drive restore integration

**Backup System:**
- Drive backup functions exist in notebook
- Manual backup calls only (conversion step)
- No automatic backup after HPO/training/benchmarking

### Pain Points / Limitations

- L1: HPO progress lost if Colab disconnects (no Drive restore)
- L2: Benchmarking re-runs unnecessarily if results lost (no restore)
- L3: Model conversion fails if training checkpoint lost (no restore)
- L4: No automatic backup of checkpoints during/after operations
- L5: Manual intervention required to restore from Drive
- L6: No proactive restore before operations
- L7: Cannot resume from any pipeline stage interruption

## High-Level Design

### Architecture Overview

```
Notebook (Colab)
│
├─> General Restore Helper (Cell 14)
│   └─> ensure_restored_from_drive() - Universal restore function
│
├─> HPO Execution
│   ├─> setup_checkpoint_storage()
│   │   └─> restore_from_drive() if local missing
│   └─> Run HPO
│       └─> backup_to_drive() after completion
│
├─> Benchmarking
│   ├─> ensure_restored_from_drive(benchmark.json)
│   └─> backup_to_drive() after completion
│
├─> Final Training
│   └─> backup_to_drive() after training completes
│
└─> Model Conversion
    ├─> ensure_restored_from_drive(training cache)
    └─> ensure_restored_from_drive(checkpoint)
```

### Restore Flow (Mirrors Local Development)

```
Operation Starts (Same as Local Dev)
│
├─> Check if local file exists
│   │
│   ├─> YES → Use local file (same as local dev)
│   │
│   └─> NO → Check Drive backup (Colab-specific)
│       │
│       ├─> Drive EXISTS → Restore from Drive to local (resume)
│       │   └─> Use restored file (now works like local dev)
│       │
│       └─> Drive NOT EXISTS → Proceed normally (create new, same as local dev)
│
Work Normally (Same as Local Dev)
│
└─> Work with local files (mirror local development workflow)

Session Ends (Colab-specific)
│
└─> Backup to Drive (save progress before disconnect)
```

### Universal Restore Pattern (Mirrors Local Dev)

```python
# At operation start - mirror local development workflow
# 1. Check local first (same as local dev)
if local_path.exists():
    # Use local file (same as local dev)
    use_file(local_path)
else:
    # 2. Local missing - check Drive backup (Colab-specific)
    if ensure_restored_from_drive(local_path):
        # Restored from Drive - now use it (same as local dev)
        use_file(local_path)
    else:
        # 3. Drive doesn't have it - proceed normally (same as local dev)
        create_file(local_path)  # Normal operation

# Work normally with local files (mirror local development)
work_with_local_files(local_path)

# At session end - backup to Drive (Colab-specific)
if BACKUP_ENABLED:
    backup_to_drive(local_path)  # Save progress before disconnect
```

**Key Principle:**
- Workflow mirrors local development (check local → use it)
- Only difference: if local missing, check Drive before creating new
- After work completes → backup to Drive (save progress)

## Module & File Structure

### Files to Modify

- `src/orchestration/jobs/hpo/hpo_helpers.py`
  - Add `restore_from_drive` parameter to `setup_checkpoint_storage()`
  - Integrate restore before existence check
  
- `src/orchestration/jobs/hpo/local_sweeps.py`
  - Add `restore_from_drive` parameter to `run_local_hpo_sweep()`
  - Pass restore function to `setup_checkpoint_storage()`
  
- `notebooks/01_orchestrate_training_colab.ipynb`
  - Cell 14: Add `ensure_restored_from_drive()` helper function
  - Cell 43: Add restore function for HPO checkpoints + backup after HPO
  - Cell 46: Add restore for benchmark results + backup after benchmarking
  - Cell 47: Add restore before accessing benchmark files
  - Cell 61: Add backup after final training completes
  - Cell 68: Add restore for training cache
  - Cell 69: Add restore for training checkpoint

### Files Explicitly Not Touched

- `src/orchestration/jobs/hpo/checkpoint_manager.py` (no changes needed)
- `src/model_conversion/convert_to_onnx.py` (restore handled in notebook)

## Detailed Design per Component

### Component: General Restore Helper (Notebook Cell 14)

**Responsibility (SRP)**
- Provide universal restore function for any pipeline output
- Can be used before any file access operation

**Public API**

```python
def ensure_restored_from_drive(local_path: Path, is_directory: bool = False) -> bool:
    """
    Ensure file/directory exists locally, restoring from Drive if missing.
    Universal helper for seamless restore across entire pipeline.
    
    Flow:
    1. If local exists → return True (skip Drive check)
    2. If local missing → check Drive backup
    3. If Drive has it → restore to local and return True
    4. If Drive doesn't have it → return False (proceed to create new)
    
    Args:
        local_path: Local path to check and potentially restore
        is_directory: True if checking/restoring a directory, False for a file
    
    Returns:
        True if file exists (local or restored), False if Drive backup doesn't exist
    
    Usage: 
        Call before any file access. If returns False, proceed to create new file.
        After creating new file, backup to Drive if required.
    """
    # If local already exists, no need to check Drive
    if local_path.exists():
        return True
    
    # If backup not enabled, can't restore
    if not BACKUP_ENABLED:
        return False
    
    # Check Drive backup and restore if exists
    return restore_from_drive(local_path, is_directory=is_directory)
```

**Implementation Notes**
- Wrapper around `restore_if_missing()` for clarity
- Can be called before any file operation
- Returns True if file exists (local or restored), False otherwise
- Enables seamless restore pattern across entire pipeline

### Component: HPO Checkpoint Restore (`hpo_helpers.py`)

**Responsibility (SRP)**
- Integrate Drive restore into checkpoint setup
- Restore checkpoint before checking existence

**Public API Extension**

```python
def setup_checkpoint_storage(
    output_dir: Path,
    checkpoint_config: Optional[Dict[str, Any]],
    backbone: str,
    study_name: Optional[str] = None,
    restore_from_drive: Optional[Callable[[Path], bool]] = None,  # New
) -> Tuple[Optional[Path], Optional[str], bool]:
    """
    Set up checkpoint storage with optional Drive restore.
    
    If local checkpoint missing and restore_from_drive provided,
    attempts restore before checking existence.
    """
```

**Implementation Notes**
- Check local first (mirror local dev workflow)
- If local missing → check Drive backup
- If Drive has backup → restore to local (resume scenario)
- If Drive doesn't have backup → proceed to create new (normal flow, same as local dev)
- Restore happens before `storage_path.exists()` check
- Log restore operations for observability
- After creating new checkpoint → backup to Drive at session end
- Maintain backward compatibility (restore_from_drive optional)

### Component: HPO Execution Integration (`local_sweeps.py`)

**Responsibility (SRP)**
- Accept restore function and pass to checkpoint setup
- Enable Drive restore for HPO checkpoints

**Public API Extension**

```python
def run_local_hpo_sweep(
    # ... existing parameters ...
    restore_from_drive: Optional[Callable[[Path], bool]] = None,  # New
) -> Any:
    """
    Run HPO with optional Drive restore support.
    """
```

**Implementation Notes**
- Pass restore function to `setup_checkpoint_storage()`
- No changes to core HPO logic
- Restore happens transparently

### Component: Notebook HPO Execution (Cell 43)

**Responsibility (SRP)**
- Create restore function for HPO checkpoints
- Pass restore function to HPO system
- Backup checkpoint after HPO completes

**Implementation**

```python
# Create restore function for each backbone
restore_fn = None
if checkpoint_config.get("enabled", False) and BACKUP_ENABLED:
    storage_path_template = checkpoint_config.get("storage_path", "{backbone}/study.db")
    storage_path_str = storage_path_template.replace("{backbone}", backbone)
    expected_checkpoint = backbone_output_dir / storage_path_str
    
    def make_restore_fn(checkpoint_path):
        def restore_fn_inner():
            return restore_if_missing(checkpoint_path, is_directory=False)
        return restore_fn_inner
    
    restore_fn = make_restore_fn(expected_checkpoint)

# Pass to HPO system
study = run_local_hpo_sweep(
    # ... existing params ...
    restore_from_drive=restore_fn,
)

# Backup after completion
if checkpoint_config.get("enabled", False) and BACKUP_ENABLED:
    checkpoint_path = backbone_output_dir / storage_path_str
    if checkpoint_path.exists():
        backup_to_drive(checkpoint_path, is_directory=False)
        print(f"✓ Backed up HPO checkpoint to Drive")
```

### Component: Benchmarking Restore (Cell 46-47)

**Responsibility (SRP)**
- Restore benchmark results from Drive if missing
- Enable benchmarking step to skip if results already exist in Drive

**Implementation Notes**
- Check local first (mirror local dev workflow)
- If local missing → check Drive backup
- If Drive has benchmark.json → restore to local and use it (resume)
- If Drive doesn't have it → proceed to run benchmarking normally (create new, same as local dev)
- After benchmarking completes → backup results to Drive at session end
- Optional: Skip benchmarking if results already exist (optimization)

**Example Usage:**

```python
# Before benchmarking - check Drive backup first
benchmark_file = trial_dir / BENCHMARK_FILENAME
if ensure_restored_from_drive(benchmark_file, is_directory=False):
    print(f"✓ Restored benchmark results from Drive")
    # Skip benchmarking, use restored results
else:
    # Drive backup doesn't exist - run benchmarking to create new
    print(f"Drive backup not found, running benchmarking...")
    run_benchmarking_local(...)
    # Backup new results to Drive after creation
    if BACKUP_ENABLED:
        backup_to_drive(benchmark_file, is_directory=False)
```

### Component: Model Conversion Checkpoint Restore (Cell 68-69)

**Responsibility (SRP)**
- Restore training cache from Drive if missing
- Restore training checkpoint from Drive if missing
- Enable model conversion to use Drive backups

**Implementation Notes**
- Check local first (mirror local dev workflow)
- If local missing → check Drive backup
- If Drive has cache file → restore to local (resume)
- If Drive doesn't have cache → proceed to load/create normally (same as local dev)
- If Drive has checkpoint → restore to local (resume)
- If Drive doesn't have checkpoint → proceed to load/create normally (same as local dev)
- Try restoring entire output directory if checkpoint dir missing
- After creating new files → backup to Drive at session end
- Clear error messages if restore fails

### Component: Automatic Backup After Operations

**Responsibility (SRP)**
- Backup checkpoints automatically after operations complete
- No manual intervention required

**Backup Points**
- After HPO completes: backup `study.db`
- After benchmarking completes: backup `benchmark.json`
- After final training completes: backup checkpoint directory and metrics
- After model conversion: backup ONNX model (already implemented)

## Configuration & Controls

### Configuration Sources

- Uses existing Drive config from `config/paths.yaml`
- Uses `BACKUP_ENABLED` flag from notebook environment detection
- No new configuration needed

### Validation Rules

- Restore only attempts if `BACKUP_ENABLED=True`
- Check Drive backup FIRST if local file missing
- If Drive has backup → restore to local
- If Drive doesn't have backup → proceed to create new normally
- After creating new → backup to Drive (if required)
- Graceful fallback if restore fails

### Key Flow Summary (Mirrors Local Development)

**For any file/directory access:**
1. **Check if local file exists** (same as local dev)
   - YES → Use local file (same as local dev)
   - NO → Continue to step 2

2. **Check Drive backup** (Colab-specific, only if local missing)
   - EXISTS → Restore from Drive to local (resume scenario)
   - NOT EXISTS → Proceed to step 3 (normal flow, same as local dev)

3. **Create new file normally** (standard operation, same as local dev)

4. **Work with local files** (mirror local development workflow)

5. **At session end** → Backup to Drive (Colab-specific, save progress before disconnect)

**This ensures:**
- Workflow mirrors local development (check local first, use it if exists)
- Only difference: if local missing, check Drive before creating new
- If Drive has backup → restore and use it (resume scenario)
- If Drive doesn't have backup → create new normally (same as local dev)
- At session end → backup to Drive (save progress)
- No data loss - Drive backup checked when local missing

## Implementation Steps

1. **Add general restore helper to notebook Cell 14**
   - Add `ensure_restored_from_drive()` function
   - Universal helper for seamless restore anywhere in pipeline
   - Wrapper around `restore_if_missing()` for clarity

2. **Extend `hpo_helpers.py`**
   - Add `restore_from_drive` parameter to `setup_checkpoint_storage()`
   - Check local first (mirror local dev workflow)
   - If local missing → check Drive backup
   - If Drive has backup → restore to local (resume)
   - If Drive doesn't have backup → proceed to create new (normal flow, same as local dev)
   - After creating new checkpoint → backup to Drive at session end
   - Add logging for restore operations

3. **Extend `local_sweeps.py`**
   - Add `restore_from_drive` parameter to `run_local_hpo_sweep()`
   - Pass restore function to `setup_checkpoint_storage()`

4. **Update notebook Cell 43 (HPO execution)**
   - Create restore function for HPO checkpoints
   - Pass restore function to `run_local_hpo_sweep()`
   - Flow: Check local → if missing check Drive → restore if exists → create new if not (mirror local dev)
   - Add backup after HPO completes (backup to Drive at session end)

5. **Update notebook Cell 46 (benchmarking)**
   - Check local first (mirror local dev workflow)
   - If local missing → check Drive backup using `ensure_restored_from_drive()`
   - If Drive has benchmark.json → restore to local and use it (resume)
   - If Drive doesn't have it → run benchmarking normally (create new, same as local dev)
   - Add backup after benchmarking completes (backup to Drive at session end)

6. **Update notebook Cell 47 (benchmark verification)**
   - Use `ensure_restored_from_drive()` before accessing benchmark files
   - Restore from Drive if missing

7. **Update notebook Cell 61 (after final training)**
   - Check local first (mirror local dev workflow)
   - If local missing → check Drive backup using `ensure_restored_from_drive()` for checkpoint
   - If Drive has checkpoint → restore to local (resume scenario)
   - If Drive doesn't have it → training already created it locally (normal flow, same as local dev)
   - Add automatic backup of checkpoint directory to Drive (at session end)
   - Add backup of metrics file to Drive (at session end)

8. **Update notebook Cell 68 (conversion cache loading)**
   - Check local first (mirror local dev workflow)
   - If local missing → check Drive backup using `ensure_restored_from_drive()` for training cache
   - If Drive has cache → restore to local (resume)
   - If Drive doesn't have cache → proceed to load/create normally (same as local dev)
   - Improve error handling

9. **Update notebook Cell 69 (conversion checkpoint loading)**
   - Check local first (mirror local dev workflow)
   - If local missing → check Drive backup using `ensure_restored_from_drive()` for training checkpoint
   - If Drive has checkpoint → restore to local (resume)
   - If Drive doesn't have checkpoint → proceed to load/create normally (same as local dev)
   - Try restoring entire output directory if checkpoint dir missing
   - Improve error messages

10. **Add restore to other pipeline steps**
    - Best configuration selection: use `ensure_restored_from_drive()` for cache files
    - Trial checkpoint access: restore before use
    - Any file access in outputs/: use `ensure_restored_from_drive()` pattern

11. **Test and verify**
    - Test HPO resume from Drive backup
    - Test benchmarking restore from Drive
    - Test model conversion with Drive backup
    - Test resume from any pipeline stage interruption
    - Test graceful fallback when Drive backup missing
    - Verify automatic backups work for all outputs

## Testing Strategy

### Unit Tests

- `setup_checkpoint_storage()`: Test restore integration
- `ensure_restored_from_drive()`: Test universal restore helper
- Restore function creation in notebook
- Error handling when restore fails

### Integration Tests

- End-to-end HPO resume from Drive backup
- Benchmarking restore from Drive backup
- Model conversion using Drive backup
- Resume from any pipeline stage interruption
- Automatic backup after all operations
- Multiple restore scenarios across pipeline
- Sequential restore (restore dependency chain)
- Resume after interruption at different stages

### Edge Cases

- Drive backup exists but restore fails
- Partial checkpoint restore (some files missing)
- Drive quota exceeded
- Network timeouts during restore
- Corrupted Drive backups
- Local file exists but Drive backup is newer (current: use local)
- Multiple restore attempts in sequence
- Restore dependency chains (cache → checkpoint)

## Backward Compatibility & Migration

**Compatible:**
- All changes are additive (optional parameters)
- System works without Drive (restore functions are optional)
- Existing behavior preserved when restore not provided

**Migration:**
- No breaking changes
- Existing code continues to work
- Restore functionality opt-in via restore function

## Documentation Updates

### Updated Documentation

- Notebook Cell 14: Document `ensure_restored_from_drive()` helper
- Notebook Cell 43: Document restore functionality for HPO
- Notebook Cell 46-47: Document benchmarking restore
- Notebook Cell 68-69: Document checkpoint restore
- Update HPO checkpoint documentation to mention Drive restore
- Document universal restore pattern for pipeline

## Rollout & Validation Checklist

- [ ] General restore helper (`ensure_restored_from_drive`) implemented
- [ ] HPO restore integration implemented
- [ ] Benchmarking restore implemented
- [ ] Model conversion restore implemented
- [ ] Automatic backup after all operations
- [ ] Restore functions passed correctly
- [ ] Error handling tested
- [ ] Logging verified
- [ ] Backward compatibility confirmed
- [ ] Documentation updated
- [ ] Resume from any pipeline stage verified

