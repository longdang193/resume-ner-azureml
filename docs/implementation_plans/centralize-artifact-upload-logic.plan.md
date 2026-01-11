<!-- Centralize Artifact Upload Logic -->
# Centralize Artifact Upload Logic - Implementation Plan

## Overview

This plan centralizes artifact upload logic across all stages (HPO, benchmarking, training, and conversion) to ensure DRY principles and single source of truth. Currently, each stage has its own artifact uploading implementation with duplicated logic, inconsistent error handling, and varying retry strategies.

**Current State**: 
- ✅ Basic utilities exist (`log_artifact_safe`, `log_artifacts_safe`, `upload_checkpoint_archive`)
- ❌ Each stage implements its own artifact upload logic
- ❌ Duplicated run_id detection logic
- ❌ Inconsistent config checking
- ❌ Different error handling patterns

**Target State**:
- ✅ Unified `ArtifactUploader` class for all artifact operations
- ✅ Stage-specific helper functions for common patterns
- ✅ Consistent retry logic and error handling
- ✅ Single source of truth for artifact upload behavior
- ✅ Stage-aware config checking

**Prerequisites**: 
- Existing `infrastructure.tracking.mlflow.artifacts` module
- MLflow tracking infrastructure in place
- Stage-specific tracking configs

**Migration Strategy**: Incremental migration with backward compatibility. Each stage will be migrated independently to minimize risk.

## Current State Analysis

### Existing Centralized Utilities ✅

1. **`infrastructure.tracking.mlflow.artifacts`** module:
   - `log_artifact_safe()` - Single file upload with retry logic
   - `log_artifacts_safe()` - Directory upload with retry logic  
   - `upload_checkpoint_archive()` - Checkpoint archive upload with manifest

2. **`infrastructure.tracking.mlflow.artifacts.manager`**:
   - `create_checkpoint_archive()` - Archive creation
   - `should_skip_file()` - File filtering logic

### Duplication Issues ❌

1. **Training (`training/core/trainer.py`)**:
   - Lines 525-580: Inline artifact logging logic
   - Duplicates run_id detection (active_run vs MLFLOW_RUN_ID)
   - Custom error handling and logging
   - Manual config checking
   - **Impact**: ~55 lines of duplicated logic

2. **Conversion (`deployment/conversion/execution.py`)**:
   - Uses `log_artifact_safe()` correctly ✅
   - But has custom MLflow run management
   - Inconsistent with other stages
   - **Impact**: Minor, but inconsistent patterns

3. **HPO/Sweep (`infrastructure/tracking/mlflow/trackers/sweep_tracker.py`)**:
   - Uses `upload_checkpoint_archive()` correctly ✅
   - Good example of using centralized utilities
   - **Impact**: Minimal, serves as reference

4. **Training Tracker (`infrastructure/tracking/mlflow/trackers/training_tracker.py`)**:
   - Has `log_training_artifacts()` method (lines 289-335)
   - Duplicates config checking logic
   - Could be simplified
   - **Impact**: ~45 lines of duplicated logic

5. **Benchmarking Tracker (`infrastructure/tracking/mlflow/trackers/benchmark_tracker.py`)**:
   - Uses centralized utilities ✅
   - **Impact**: Minimal

## Target Architecture

### New Components

```
src/infrastructure/tracking/mlflow/artifacts/
├── __init__.py                    # Re-exports (existing)
├── artifacts.py                   # Basic utilities (existing)
├── manager.py                     # Archive management (existing)
├── uploader.py                    # 🆕 Unified ArtifactUploader class
└── stage_helpers.py               # 🆕 Stage-specific helper functions
```

### Core Classes and Functions

1. **`ArtifactUploader`** class:
   - Unified interface for all artifact uploads
   - Stage-aware config checking
   - Consistent run_id handling
   - Built-in retry logic via existing utilities

2. **Stage Helper Functions**:
   - `upload_training_artifacts()` - Training-specific uploads
   - `upload_conversion_artifacts()` - Conversion-specific uploads
   - `upload_benchmark_artifacts()` - Benchmark-specific uploads
   - `upload_hpo_artifacts()` - HPO-specific uploads

3. **Utility Functions**:
   - `get_mlflow_run_id()` - Unified run_id detection

## Implementation Phases

### Phase 1: Create Core Infrastructure ⏳ NOT STARTED

**Goal**: Create `ArtifactUploader` class and utility functions

#### Task 1.1: Create `get_mlflow_run_id()` utility

- [ ] **File**: `src/infrastructure/tracking/mlflow/utils.py`
- [ ] **Action**: Add function to detect run_id from active run or environment
- [ ] **Code**:
  ```python
  def get_mlflow_run_id() -> Optional[str]:
      """Get MLflow run ID from active run or environment variable."""
      try:
          import mlflow
          active_run = mlflow.active_run()
          if active_run:
              return active_run.info.run_id
      except Exception:
          pass
      return os.environ.get("MLFLOW_RUN_ID")
  ```
- [ ] **Tests**: Unit test for both active run and env var scenarios
- [ ] **Dependencies**: None

#### Task 1.2: Create `ArtifactUploader` class

- [ ] **File**: `src/infrastructure/tracking/mlflow/artifacts/uploader.py`
- [ ] **Action**: Create unified uploader class
- [ ] **Key Features**:
  - Constructor accepts `run_id`, `stage`, `config_dir`
  - Lazy loading of tracking config
  - Methods: `upload_checkpoint()`, `upload_file()`, `upload_checkpoint_archive()`
  - Stage-aware config checking
- [ ] **Code Structure**:
  ```python
  class ArtifactUploader:
      def __init__(self, run_id=None, stage=None, config_dir=None)
      def _get_tracking_config(self) -> Dict[str, Any]
      def upload_checkpoint(self, checkpoint_dir, artifact_path="checkpoint", skip_if_disabled=True) -> bool
      def upload_file(self, file_path, artifact_path=None) -> bool
      def upload_checkpoint_archive(self, archive_path, manifest=None, artifact_path="best_trial_checkpoint.tar.gz") -> bool
  ```
- [ ] **Tests**: Unit tests for all methods, config checking, error handling
- [ ] **Dependencies**: Task 1.1

#### Task 1.3: Create stage helper functions

- [ ] **File**: `src/infrastructure/tracking/mlflow/artifacts/stage_helpers.py`
- [ ] **Action**: Create convenience functions for each stage
- [ ] **Functions**:
  - `upload_training_artifacts(checkpoint_dir, metrics_json_path=None, run_id=None, config_dir=None) -> Dict[str, bool]`
  - `upload_conversion_artifacts(onnx_path, run_id=None, config_dir=None) -> bool`
  - `upload_benchmark_artifacts(...) -> Dict[str, bool]`
  - `upload_hpo_artifacts(...) -> bool`
- [ ] **Tests**: Integration tests for each helper function
- [ ] **Dependencies**: Task 1.2

#### Task 1.4: Update `__init__.py` exports

- [ ] **File**: `src/infrastructure/tracking/mlflow/artifacts/__init__.py`
- [ ] **Action**: Export new classes and functions
- [ ] **Add to `__all__`**:
  - `ArtifactUploader`
  - `upload_training_artifacts`
  - `upload_conversion_artifacts`
  - `upload_benchmark_artifacts`
  - `upload_hpo_artifacts`
- [ ] **Tests**: Verify imports work correctly
- [ ] **Dependencies**: Tasks 1.2, 1.3

**Phase 1 Completion Criteria**:
- ✅ `ArtifactUploader` class created and tested
- ✅ Stage helper functions created and tested
- ✅ All exports working
- ✅ Documentation added

---

### Phase 2: Migrate Training Module ⏳ NOT STARTED

**Goal**: Replace inline artifact logging in `trainer.py` with `ArtifactUploader`

#### Task 2.1: Refactor `trainer.py` artifact logging

- [ ] **File**: `src/training/core/trainer.py`
- [ ] **Action**: Replace lines 525-580 with `ArtifactUploader` usage
- [ ] **Before**: ~55 lines of inline logic
- [ ] **After**: ~15 lines using `ArtifactUploader`
- [ ] **Code Pattern**:
  ```python
  from infrastructure.tracking.mlflow.artifacts.uploader import ArtifactUploader
  from infrastructure.tracking.mlflow.utils import get_mlflow_run_id
  
  skip_artifact_logging = os.environ.get("MLFLOW_SKIP_ARTIFACT_LOGGING", "false").lower() == "true"
  
  if skip_artifact_logging:
      print("  [Training] Skipping artifact logging...", file=sys.stderr, flush=True)
  else:
      target_run_id = get_mlflow_run_id()
      if target_run_id:
          uploader = ArtifactUploader(
              run_id=target_run_id,
              stage="training",
              config_dir=config_dir,  # If available
          )
          
          checkpoint_dir = output_dir / "checkpoint"
          if checkpoint_dir.exists():
              uploader.upload_checkpoint(checkpoint_dir, artifact_path="checkpoint")
          
          metrics_json = output_dir / "metrics.json"
          if metrics_json.exists():
              uploader.upload_file(metrics_json, artifact_path="metrics.json")
      else:
          print("  [Training] ⚠ No MLflow run available - cannot log artifacts", file=sys.stderr, flush=True)
  ```
- [ ] **Tests**: 
  - Verify existing tests still pass
  - Add tests for new uploader integration
  - Test skip_artifact_logging behavior
- [ ] **Dependencies**: Phase 1 complete

#### Task 2.2: Update `training_tracker.py`

- [ ] **File**: `src/infrastructure/tracking/mlflow/trackers/training_tracker.py`
- [ ] **Action**: Refactor `log_training_artifacts()` to use `ArtifactUploader`
- [ ] **Before**: Lines 289-335 with custom config checking
- [ ] **After**: Simplified using `ArtifactUploader`
- [ ] **Code Pattern**:
  ```python
  def log_training_artifacts(self, checkpoint_dir, metrics_json_path=None):
      uploader = ArtifactUploader(
          run_id=None,  # Use active run
          stage="training",
          config_dir=self._infer_config_dir(checkpoint_dir),
      )
      
      uploader.upload_checkpoint(checkpoint_dir)
      if metrics_json_path:
          uploader.upload_file(metrics_json_path, "metrics.json")
  ```
- [ ] **Tests**: Verify tracker tests still pass
- [ ] **Dependencies**: Phase 1 complete

**Phase 2 Completion Criteria**:
- ✅ `trainer.py` uses `ArtifactUploader`
- ✅ `training_tracker.py` uses `ArtifactUploader`
- ✅ All training tests pass
- ✅ No regression in artifact upload behavior

---

### Phase 3: Migrate Conversion Module ⏳ NOT STARTED

**Goal**: Standardize conversion artifact uploads

#### Task 3.1: Refactor `conversion/execution.py`

- [ ] **File**: `src/deployment/conversion/execution.py`
- [ ] **Action**: Use `upload_conversion_artifacts()` helper
- [ ] **Before**: Lines 216-229 with direct `log_artifact_safe()` call
- [ ] **After**: Use stage helper function
- [ ] **Code Pattern**:
  ```python
  from infrastructure.tracking.mlflow.artifacts.stage_helpers import upload_conversion_artifacts
  
  # In main() function, after conversion success:
  if started_run_directly and onnx_path and onnx_path.exists():
      success = upload_conversion_artifacts(
          onnx_path=onnx_path,
          run_id=use_run_id,
          config_dir=config_dir,
      )
      if success:
          _log.info(f"Logged ONNX model to MLflow: {onnx_path}")
      else:
          _log.warning(f"Failed to log ONNX model artifact to MLflow. Model available at: {onnx_path}")
  ```
- [ ] **Tests**: Verify conversion tests still pass
- [ ] **Dependencies**: Phase 1 complete

**Phase 3 Completion Criteria**:
- ✅ Conversion uses stage helper
- ✅ All conversion tests pass
- ✅ Consistent with other stages

---

### Phase 4: Migrate Benchmarking Module ⏳ NOT STARTED

**Goal**: Ensure benchmarking uses unified uploader (may already be good)

#### Task 4.1: Audit benchmarking artifact uploads

- [ ] **File**: `src/infrastructure/tracking/mlflow/trackers/benchmark_tracker.py`
- [ ] **Action**: Review current implementation
- [ ] **Check**: Does it use centralized utilities?
- [ ] **If not**: Refactor to use `ArtifactUploader`
- [ ] **Tests**: Verify benchmark tests pass
- [ ] **Dependencies**: Phase 1 complete

**Phase 4 Completion Criteria**:
- ✅ Benchmarking uses unified uploader (or confirmed already using it)
- ✅ All benchmark tests pass

---

### Phase 5: Migrate HPO Module ⏳ NOT STARTED

**Goal**: Ensure HPO uses unified uploader (already good, but verify consistency)

#### Task 5.1: Audit HPO artifact uploads

- [ ] **File**: `src/infrastructure/tracking/mlflow/trackers/sweep_tracker.py`
- [ ] **Action**: Review current implementation
- [ ] **Check**: Already uses `upload_checkpoint_archive()` ✅
- [ ] **Enhancement**: Consider using `ArtifactUploader` for consistency
- [ ] **Optional**: Refactor if it improves maintainability
- [ ] **Tests**: Verify HPO tests pass
- [ ] **Dependencies**: Phase 1 complete

**Phase 5 Completion Criteria**:
- ✅ HPO confirmed using unified utilities
- ✅ All HPO tests pass
- ✅ Consistent with other stages

---

### Phase 6: Cleanup and Documentation ⏳ NOT STARTED

**Goal**: Remove duplicate code and document new patterns

#### Task 6.1: Remove duplicate code

- [ ] **Action**: Remove any remaining inline artifact upload logic
- [ ] **Files to check**:
  - `src/training/core/trainer.py` - Should be using `ArtifactUploader`
  - `src/infrastructure/tracking/mlflow/trackers/training_tracker.py` - Should be using `ArtifactUploader`
  - Any other files with duplicate logic
- [ ] **Tests**: Verify no functionality lost

#### Task 6.2: Update documentation

- [ ] **File**: `src/infrastructure/tracking/mlflow/artifacts/README.md` (create if needed)
- [ ] **Content**:
  - Overview of `ArtifactUploader` class
  - Usage examples for each stage
  - Migration guide for future code
  - Best practices
- [ ] **Update**: Any existing docs that reference old patterns

#### Task 6.3: Add deprecation warnings (if needed)

- [ ] **Action**: If any old functions are still used, add deprecation warnings
- [ ] **Timeline**: Remove deprecated functions after 1-2 releases

**Phase 6 Completion Criteria**:
- ✅ All duplicate code removed
- ✅ Documentation complete
- ✅ No deprecated patterns in use

---

## Testing Strategy

### Unit Tests

1. **`ArtifactUploader` class**:
   - Test constructor with various parameters
   - Test config loading (lazy loading)
   - Test each upload method
   - Test error handling
   - Test stage-specific config checking

2. **Stage helper functions**:
   - Test each helper function
   - Test with/without optional parameters
   - Test error scenarios

3. **`get_mlflow_run_id()` utility**:
   - Test active run detection
   - Test environment variable fallback
   - Test error handling

### Integration Tests

1. **Training artifact uploads**:
   - Test checkpoint upload
   - Test metrics.json upload
   - Test skip_artifact_logging behavior
   - Test with/without active run

2. **Conversion artifact uploads**:
   - Test ONNX model upload
   - Test with/without run_id

3. **HPO artifact uploads**:
   - Test checkpoint archive upload
   - Test manifest upload

### Regression Tests

1. **Verify existing functionality**:
   - All existing tests should still pass
   - No change in artifact upload behavior
   - Same retry logic and error handling

2. **End-to-end tests**:
   - Full training pipeline with artifact uploads
   - Full conversion pipeline with artifact uploads
   - Full HPO pipeline with artifact uploads

## Migration Checklist

### Pre-Migration

- [ ] Review all artifact upload code locations
- [ ] Document current behavior for each stage
- [ ] Create test cases for regression testing
- [ ] Backup current implementation

### During Migration

- [ ] Phase 1: Create core infrastructure
- [ ] Phase 2: Migrate training module
- [ ] Phase 3: Migrate conversion module
- [ ] Phase 4: Migrate benchmarking module
- [ ] Phase 5: Migrate HPO module
- [ ] Phase 6: Cleanup and documentation

### Post-Migration

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Code review completed
- [ ] Performance verified (no regression)
- [ ] Monitor production for issues

## Risk Assessment

### Low Risk ✅

- **Phase 1**: Creating new infrastructure (additive, no breaking changes)
- **Phase 4**: Benchmarking (already using centralized utilities)
- **Phase 5**: HPO (already using centralized utilities)

### Medium Risk ⚠️

- **Phase 2**: Training module (high usage, but well-tested)
- **Phase 3**: Conversion module (less usage, but critical path)

### Mitigation Strategies

1. **Incremental migration**: One phase at a time
2. **Backward compatibility**: Keep old code until migration complete
3. **Comprehensive testing**: Unit + integration + regression tests
4. **Feature flags**: Optional feature flag to switch between old/new code
5. **Monitoring**: Watch for artifact upload failures after migration

## Success Criteria

### Functional

- ✅ All artifact uploads work correctly
- ✅ No regression in functionality
- ✅ Consistent behavior across all stages
- ✅ Proper error handling and retry logic

### Code Quality

- ✅ Single source of truth for artifact uploads
- ✅ DRY principles followed
- ✅ No duplicate code
- ✅ Clear, maintainable code

### Performance

- ✅ No performance regression
- ✅ Same or better retry behavior
- ✅ Efficient config loading

## Timeline Estimate

- **Phase 1**: 2-3 days (core infrastructure)
- **Phase 2**: 2-3 days (training migration)
- **Phase 3**: 1-2 days (conversion migration)
- **Phase 4**: 1 day (benchmarking audit)
- **Phase 5**: 1 day (HPO audit)
- **Phase 6**: 1-2 days (cleanup and docs)

**Total**: ~8-12 days

## Dependencies

### Internal

- `infrastructure.tracking.mlflow.artifacts` module (existing)
- `infrastructure.tracking.mlflow.config_loader` (for stage configs)
- MLflow tracking infrastructure

### External

- MLflow library
- Azure ML MLflow integration (for Azure ML backend)

## Notes

- This is a refactoring effort, not a feature addition
- Focus on maintaining backward compatibility
- Incremental migration reduces risk
- Each phase can be done independently
- Consider feature flag for gradual rollout if needed

## Future Enhancements

1. **Artifact upload metrics**: Track upload success/failure rates
2. **Upload progress tracking**: Show progress for large uploads
3. **Parallel uploads**: Upload multiple artifacts in parallel
4. **Upload validation**: Verify uploaded artifacts
5. **Artifact compression**: Automatic compression for large artifacts

