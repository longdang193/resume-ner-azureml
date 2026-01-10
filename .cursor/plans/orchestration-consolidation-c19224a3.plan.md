<!-- c19224a3-5784-411b-b85b-6c47d05817f2 398e79e0-549f-4d11-bfdf-509555a60221 -->
# Orchestration Module Consolidation Plan

## Overview

Consolidate files under `src/orchestration/` into appropriate existing modules, eliminating duplication and improving cohesion while preserving backward compatibility through facades.

## ⚠️ Important Corrections (Review Findings)

**CRITICAL:** Several files are already consolidated and should be **DELETED** (not moved):

1. **`normalize.py`** - Already exists in `src/core/normalize.py` (identical) - DELETE duplicate
2. **`tokens.py`** - Already exists in `src/core/tokens.py` (identical) - DELETE duplicate  
3. **`constants.py`** - Already exists in `src/constants/orchestration.py` (identical) - DELETE duplicate
4. **`data_assets.py`** - Already exists in `src/azureml/data_assets.py` (identical) - DELETE duplicate

**Why `core/` is correct for normalize/tokens:**
- These are fundamental utilities used by both `paths` and `naming` modules
- They have no circular dependencies (as stated in `core/__init__.py`)
- Most codebase already imports from `core.normalize` and `core.tokens`
- Only 1 import found using `orchestration.normalize` (test file)

**Shared Module:** No changes needed - correctly contains runtime utilities (logging, file utils, MLflow setup, etc.)

## Current State Analysis

### Files Already Consolidated (Facades Only)

- `paths.py` - Facade to `paths` module
- `naming.py` - Facade to `naming` module  
- `naming_centralized.py` - Facade to `naming` and `paths` modules
- `mlflow_utils.py` - Facade to `tracking.mlflow.setup`
- `path_resolution.py` - Facade to `paths` and `hpo.utils.paths`
- `__init__.py` - Facade re-exporting from new modules

### Files with Duplicates in Target Modules

1. **config_loader.py** → `src/config/loader.py` (duplicate `ExperimentConfig`, `load_experiment_config`)
2. **final_training_config.py** → `src/config/training.py` (duplicate `load_final_training_config`)
3. **conversion_config.py** → `src/config/conversion.py` (duplicate `load_conversion_config`, imports from orchestration)
4. **environment.py** → `src/config/environment.py` (duplicate `EnvironmentConfig`, `build_environment_config`)
5. **config_compat.py** → `src/config/validation.py` (duplicate `validate_paths_and_naming_compatible`)
6. **fingerprints.py** → `src/fingerprints/` (duplicate fingerprint functions)
7. **metadata_manager.py** → `src/metadata/training.py` (duplicate metadata management)
8. **index_manager.py** → `src/metadata/index.py` (duplicate index management)
9. **data_assets.py** → `src/azureml/data_assets.py` (ALREADY EXISTS - identical, just delete duplicate)
10. **drive_backup.py** → `src/storage/drive.py` (duplicate `DriveBackupStore`)
11. **benchmark_utils.py** → `src/benchmarking/utils.py` (duplicate `run_benchmarking`)

### Files Already Consolidated (Delete Duplicates)

1. **normalize.py** → Already exists in `src/core/normalize.py` (IDENTICAL) - DELETE duplicate
2. **tokens.py** → Already exists in `src/core/tokens.py` (IDENTICAL) - DELETE duplicate
3. **constants.py** → Already exists in `src/constants/orchestration.py` (IDENTICAL) - DELETE duplicate
4. **data_assets.py** → Already exists in `src/azureml/data_assets.py` (IDENTICAL) - DELETE duplicate

**Note:** `normalize.py` and `tokens.py` are correctly placed in `core/` (not `naming/`) because:
- They're fundamental utilities used by both `paths` and `naming` modules
- They have no circular dependencies (as stated in `core/__init__.py`)
- Most codebase already imports from `core.normalize` and `core.tokens`

## Consolidation Strategy

### Phase 1: Move Files to Target Modules

#### 1.1 Config Module Consolidation

- **config_loader.py** → Delete, update imports to `config.loader`
- **final_training_config.py** → Delete, update imports to `config.training`
- **conversion_config.py** → Delete, update imports to `config.conversion` (fix circular import)
- **environment.py** → Delete, update imports to `config.environment`
- **config_compat.py** → Delete, update imports to `config.validation`

#### 1.2 Fingerprints Module Consolidation

- **fingerprints.py** → Delete, update imports to `fingerprints` module

#### 1.3 Metadata Module Consolidation

- **metadata_manager.py** → Delete, update imports to `metadata.training`
- **index_manager.py** → Delete, update imports to `metadata.index`

#### 1.4 Azure ML Module Consolidation

- **data_assets.py** → DELETE (already exists in `src/azureml/data_assets.py`, identical)

#### 1.5 Storage Module Consolidation

- **drive_backup.py** → Delete, update imports to `storage.drive`

#### 1.6 Benchmarking Module Consolidation

- **benchmark_utils.py** → Delete, update imports to `benchmarking.utils`

#### 1.7 Core Module Consolidation (Already Done)

- **normalize.py** → DELETE (already exists in `src/core/normalize.py`, identical)
- **tokens.py** → DELETE (already exists in `src/core/tokens.py`, identical)
- **constants.py** → DELETE (already exists in `src/constants/orchestration.py`, identical)

**Note:** These files are correctly placed in `core/` because they're fundamental utilities used by multiple modules (`paths`, `naming`, `config`) with no circular dependencies.

### Phase 2: Update Imports

#### 2.1 Fix Circular Imports

- `src/config/conversion.py` imports `orchestration.config_loader.ExperimentConfig` → Change to `config.loader.ExperimentConfig`
- `src/config/conversion.py` imports `orchestration.fingerprints.compute_conv_fp` → Change to `fingerprints.compute_conv_fp`

#### 2.2 Update All Import References

- Search and replace `from orchestration.config_loader` → `from config.loader`
- Search and replace `from orchestration.final_training_config` → `from config.training`
- Search and replace `from orchestration.conversion_config` → `from config.conversion`
- Search and replace `from orchestration.environment` → `from config.environment`
- Search and replace `from orchestration.config_compat` → `from config.validation`
- Search and replace `from orchestration.fingerprints` → `from fingerprints`
- Search and replace `from orchestration.metadata_manager` → `from metadata.training`
- Search and replace `from orchestration.index_manager` → `from metadata.index`
- Search and replace `from orchestration.data_assets` → `from azureml.data_assets`
- Search and replace `from orchestration.drive_backup` → `from storage.drive`
- Search and replace `from orchestration.benchmark_utils` → `from benchmarking.utils`
- Search and replace `from orchestration.normalize` → `from core.normalize` (or use facade)
- Search and replace `from orchestration.tokens` → `from core.tokens` (or use facade)
- Search and replace `from orchestration.constants` → `from constants` (already done in `__init__.py`)

### Phase 3: Create/Update Facades

#### 3.1 Update orchestration/**init**.py

- Remove exports for moved modules
- Add deprecation warnings for moved modules
- Keep only orchestration-specific exports

#### 3.2 Create Facade Files (if needed for backward compatibility)

- `orchestration/config_loader.py` → Facade to `config.loader`
- `orchestration/final_training_config.py` → Facade to `config.training`
- `orchestration/conversion_config.py` → Facade to `config.conversion`
- `orchestration/environment.py` → Facade to `config.environment`
- `orchestration/config_compat.py` → Facade to `config.validation`
- `orchestration/fingerprints.py` → Facade to `fingerprints`
- `orchestration/metadata_manager.py` → Facade to `metadata.training`
- `orchestration/index_manager.py` → Facade to `metadata.index`
- `orchestration/data_assets.py` → Facade to `azureml.data_assets`
- `orchestration/drive_backup.py` → Facade to `storage.drive`
- `orchestration/benchmark_utils.py` → Facade to `benchmarking.utils`
- `orchestration/normalize.py` → Facade to `core.normalize`
- `orchestration/tokens.py` → Facade to `core.tokens` (also re-export `extract_placeholders` from `core.placeholders`)

### Phase 4: Verify and Test

#### 4.1 Verify No Duplication

- ✅ Verified: `orchestration/normalize.py` == `core/normalize.py` (identical)
- ✅ Verified: `orchestration/tokens.py` == `core/tokens.py` (identical, except `extract_placeholders` which is in `core/placeholders.py`)
- ✅ Verified: `orchestration/constants.py` == `constants/orchestration.py` (identical)
- ✅ Verified: `orchestration/data_assets.py` == `azureml/data_assets.py` (identical)
- Compare implementations in orchestration vs target modules for remaining files
- Ensure single source of truth
- Merge any unique functionality

#### 4.2 Update Tests

- Update test imports to use new module paths
- Verify tests still pass

#### 4.3 Update Documentation

- Update any documentation referencing old paths

## Key Decisions

1. **Facade Strategy**: Create facades for all moved modules to maintain backward compatibility during transition period
2. **Token/Normalize Location**: Keep in `core/` (already correct) - these are fundamental utilities used by both `paths` and `naming` modules with no circular dependencies
3. **Constants**: Already consolidated in `constants/orchestration.py` - just delete duplicate
4. **Data Assets**: Already consolidated in `azureml/data_assets.py` - just delete duplicate
5. **Shared Module**: No changes needed - correctly contains runtime utilities (logging, file utils, MLflow setup, etc.)

## Files to Delete After Consolidation

**Already Consolidated (just delete duplicates):**
- `src/orchestration/normalize.py` (exists in `core/`)
- `src/orchestration/tokens.py` (exists in `core/`)
- `src/orchestration/constants.py` (exists in `constants/`)
- `src/orchestration/data_assets.py` (exists in `azureml/`)

**Need Consolidation (delete after moving/updating imports):**
- `src/orchestration/config_loader.py`
- `src/orchestration/final_training_config.py`
- `src/orchestration/conversion_config.py`
- `src/orchestration/environment.py`
- `src/orchestration/config_compat.py`
- `src/orchestration/fingerprints.py`
- `src/orchestration/metadata_manager.py`
- `src/orchestration/index_manager.py`
- `src/orchestration/drive_backup.py`
- `src/orchestration/benchmark_utils.py`

## Files Already Consolidated (No Action Needed)

- `src/core/normalize.py` ✓
- `src/core/tokens.py` ✓
- `src/core/placeholders.py` ✓ (contains `extract_placeholders` also in `orchestration/tokens.py`)
- `src/constants/orchestration.py` ✓
- `src/azureml/data_assets.py` ✓

## Expected Outcome

- All orchestration files consolidated into appropriate modules
- Single source of truth for each piece of functionality
- Backward compatibility maintained through facades
- Clear module boundaries and improved cohesion
- Reduced codebase size and complexity