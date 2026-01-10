<!-- 39335df2-9852-429b-bd12-cdbc66bf84ed f03d6ff5-ed9b-443a-87c6-8bde6a4a377f -->
# Reorganize src/ into Feature-Level Folders - Detailed Implementation Plan

## Overview

This plan reorganizes the `src/` directory into feature-level folders following ML workflow patterns:

- `infrastructure/` - ML-specific infrastructure (config, paths, naming, tracking, storage, etc.)
- `common/` - Generic shared utilities
- `data/` - Data handling (shared across training, evaluation, deployment)
- `testing/` - Testing framework (cross-cutting concern)
- `training/` - Training pipeline (to be done after infrastructure)
- `evaluation/` - Model evaluation (to be done after infrastructure)
- `deployment/` - Model deployment (to be done after infrastructure)

**Migration Strategy**: Start with `infrastructure/` as foundation, keep `orchestration/` as compatibility layer during migration, remove after 1-2 releases.

## Target Structure

```
src/
├── core/                    # Foundation (no changes - already correct)
│   ├── tokens.py
│   ├── normalize.py
│   └── placeholders.py
│
├── infrastructure/          # ML-specific infrastructure
│   ├── config/            # Configuration management
│   ├── paths/             # Path management (already exists)
│   ├── naming/            # Naming conventions (already exists)
│   ├── tracking/          # MLflow tracking
│   ├── storage/           # Storage abstractions
│   ├── fingerprints/      # Fingerprinting
│   ├── metadata/          # Metadata management
│   └── platform/          # Platform adapters
│       ├── azureml/      # Azure ML integration (moved from top-level)
│       └── adapters/      # Platform adapters (moved from top-level)
│
├── common/                 # Generic shared utilities
│   ├── shared/           # Shared utilities
│   └── constants/        # Constants
│
├── data/                  # Data handling (shared across features)
│   ├── loaders/          # Data loaders
│   ├── processing/       # Data processing
│   ├── datasets/         # Dataset definitions
│   └── assets/           # Data asset management
│
├── testing/               # Testing framework (cross-cutting concern)
│   ├── aggregators/      # Test result aggregation
│   ├── comparators/      # Comparison utilities
│   ├── fixtures/         # Test fixtures
│   ├── orchestrators/    # Test orchestration
│   ├── services/         # Testing services
│   ├── setup/            # Test setup
│   └── validators/       # Validation utilities
│
├── training/               # Training pipeline (Phase 2 - future)
│   ├── core/             # Core training logic
│   ├── hpo/              # Hyperparameter optimization
│   └── execution/        # Training execution
│
├── evaluation/            # Evaluation features (Phase 2 - future)
│   ├── benchmarking/     # Benchmarking
│   └── selection/        # Model selection
│
├── deployment/            # Deployment features (Phase 2 - future)
│   ├── conversion/       # Model conversion
│   └── api/              # Inference API
│
└── orchestration/         # Compatibility layer (deprecated, remove after 1-2 releases)
    └── [facades and compatibility shims]
```

## Important Decisions

1. **Start with infrastructure/** - Foundation for all other features
2. **Keep orchestration/ during migration** - Compatibility layer, remove after 1-2 releases
3. **azureml/ → infrastructure/platform/azureml/** - ML-specific infrastructure
4. **orchestration/ → gradually deprecated** - Split jobs into features, remove after migration

## Phase 1: Pre-Implementation Analysis

- [ ] **Audit orchestration/ dependencies**
  - [ ] List all modules in `orchestration/` and their purposes
  - [ ] Map `orchestration/jobs/` to target features (training/evaluation/deployment)
  - [ ] Document all external imports of `orchestration.*`
  - [ ] Identify which modules are pure infrastructure vs feature code

- [ ] **Audit infrastructure modules**
  - [ ] Verify `paths/` and `naming/` are already well-structured
  - [ ] Check `config/` dependencies and structure
  - [ ] Review `tracking/`, `storage/`, `fingerprints/`, `metadata/` structure
  - [ ] Document cross-dependencies between infrastructure modules

- [ ] **Audit platform adapters**
  - [ ] Review `azureml/` structure and dependencies
  - [ ] Review `platform_adapters/` structure
  - [ ] Identify ML-specific vs generic platform code
  - [ ] Document dependencies on infrastructure modules

- [ ] **Audit common/ modules**
  - [ ] Review `shared/` - identify truly generic utilities
  - [ ] Review `constants/` - verify domain-agnostic
  - [ ] Identify any ML-specific code that should stay in infrastructure/

- [ ] **Audit data/ modules**
  - [ ] Review `training/data.py` and `training/data_combiner.py` - identify shared vs training-specific
  - [ ] Review `benchmarking/data_loader.py` - identify shared vs evaluation-specific
  - [ ] Review `azureml/data_assets.py` - identify infrastructure vs data concerns
  - [ ] Document data dependencies across training, evaluation, and testing
  - [ ] Plan data module structure (loaders, processing, datasets, assets)

- [ ] **Audit testing/ modules**
  - [ ] Review `testing/` structure - verify it's a substantial cross-cutting module
  - [ ] Document testing dependencies on data/, training/, and infrastructure/
  - [ ] Identify any testing utilities that should be in common/ vs testing/

- [ ] **Create dependency graph**
  - [ ] Map all import relationships
  - [ ] Identify circular dependencies
  - [ ] Plan import order to avoid cycles

## Phase 2: Create Infrastructure Module Structure

- [ ] **Create src/infrastructure/ directory**
  - [ ] Create `src/infrastructure/` directory
  - [ ] Create `src/infrastructure/__init__.py`
  - [ ] Add module docstring explaining infrastructure purpose

- [ ] **Move config/ to infrastructure/config/**
  - [ ] Create `infrastructure/config/` directory
  - [ ] Move `src/config/` → `infrastructure/config/`
  - [ ] Update all internal imports within config/
  - [ ] Update `infrastructure/config/__init__.py` exports
  - [ ] Create compatibility shim in `orchestration/config.py`:
    ```python
    """Compatibility shim - use 'from infrastructure.config import ...'"""
    import warnings
    warnings.warn("orchestration.config is deprecated, use infrastructure.config", DeprecationWarning)
    from infrastructure.config import *
    ```

- [ ] **Verify paths/ and naming/ are already correct**
  - [ ] Confirm `paths/` and `naming/` are already well-structured
  - [ ] If needed, move to `infrastructure/paths/` and `infrastructure/naming/`
  - [ ] Update imports if moved
  - [ ] Create compatibility shims in `orchestration/` if moved

- [ ] **Move tracking/ to infrastructure/tracking/**
  - [ ] Create `infrastructure/tracking/` directory
  - [ ] Move `src/tracking/` → `infrastructure/tracking/`
  - [ ] Update all internal imports within tracking/
  - [ ] Update `infrastructure/tracking/__init__.py` exports
  - [ ] Create compatibility shim in `orchestration/tracking.py`

- [ ] **Move storage/ to infrastructure/storage/**
  - [ ] Create `infrastructure/storage/` directory
  - [ ] Move `src/storage/` → `infrastructure/storage/`
  - [ ] Update all internal imports within storage/
  - [ ] Update `infrastructure/storage/__init__.py` exports
  - [ ] Create compatibility shim in `orchestration/storage.py`

- [ ] **Move fingerprints/ to infrastructure/fingerprints/**
  - [ ] Create `infrastructure/fingerprints/` directory
  - [ ] Move `src/fingerprints/` → `infrastructure/fingerprints/`
  - [ ] Update all internal imports within fingerprints/
  - [ ] Update `infrastructure/fingerprints/__init__.py` exports
  - [ ] Create compatibility shim in `orchestration/fingerprints.py`

- [ ] **Move metadata/ to infrastructure/metadata/**
  - [ ] Create `infrastructure/metadata/` directory
  - [ ] Move `src/metadata/` → `infrastructure/metadata/`
  - [ ] Update all internal imports within metadata/
  - [ ] Update `infrastructure/metadata/__init__.py` exports
  - [ ] Create compatibility shim in `orchestration/metadata.py`

## Phase 3: Create Infrastructure Platform Module

- [ ] **Create infrastructure/platform/ directory**
  - [ ] Create `infrastructure/platform/` directory
  - [ ] Create `infrastructure/platform/__init__.py`
  - [ ] Add module docstring explaining platform adapters

- [ ] **Move azureml/ to infrastructure/platform/azureml/**
  - [ ] Create `infrastructure/platform/azureml/` directory
  - [ ] Move `src/azureml/` → `infrastructure/platform/azureml/`
  - [ ] Update all internal imports within azureml/
  - [ ] Update `infrastructure/platform/azureml/__init__.py` exports
  - [ ] Create compatibility shim in `orchestration/azureml.py`:
    ```python
    """Compatibility shim - use 'from infrastructure.platform.azureml import ...'"""
    import warnings
    warnings.warn("orchestration.azureml is deprecated, use infrastructure.platform.azureml", DeprecationWarning)
    from infrastructure.platform.azureml import *
    ```

- [ ] **Move platform_adapters/ to infrastructure/platform/adapters/**
  - [ ] Create `infrastructure/platform/adapters/` directory
  - [ ] Move `src/platform_adapters/` → `infrastructure/platform/adapters/`
  - [ ] Update all internal imports within platform_adapters/
  - [ ] Update `infrastructure/platform/adapters/__init__.py` exports
  - [ ] Create compatibility shim in `orchestration/platform_adapters.py`

- [ ] **Update infrastructure/platform/init.py**
  - [ ] Export azureml and adapters submodules
  - [ ] Provide convenience imports

## Phase 4: Create Common Module Structure

- [ ] **Create src/common/ directory**
  - [ ] Create `src/common/` directory
  - [ ] Create `src/common/__init__.py`
  - [ ] Add module docstring explaining common utilities

- [ ] **Move shared/ to common/shared/**
  - [ ] Create `common/shared/` directory
  - [ ] Move `src/shared/` → `common/shared/`
  - [ ] Review contents - move any ML-specific code to infrastructure/ if needed
  - [ ] Update all internal imports within shared/
  - [ ] Update `common/shared/__init__.py` exports
  - [ ] Create compatibility shim in `orchestration/shared.py`

- [ ] **Move constants/ to common/constants/**
  - [ ] Create `common/constants/` directory
  - [ ] Move `src/constants/` → `common/constants/`
  - [ ] Verify all constants are domain-agnostic
  - [ ] Update all internal imports within constants/
  - [ ] Update `common/constants/__init__.py` exports
  - [ ] Create compatibility shim in `orchestration/constants.py`

- [ ] **Keep testing/ as top-level module**
  - [ ] Verify `testing/` is already at top level (no move needed)
  - [ ] Review structure - ensure it's organized as cross-cutting concern
  - [ ] Update imports to use `testing.*` (no changes if already correct)
  - [ ] Update `testing/__init__.py` exports if needed
  - [ ] Create compatibility shim in `orchestration/testing.py` if needed

## Phase 5: Create Data Module Structure

- [ ] **Create src/data/ directory**
  - [ ] Create `src/data/` directory
  - [ ] Create `src/data/__init__.py`
  - [ ] Add module docstring explaining data module purpose

- [ ] **Consolidate data handling into data/ module**
  - [ ] Create `data/loaders/` directory
  - [ ] Move `training/data.py` → `data/datasets/` (if dataset definitions)
  - [ ] Move `training/data_combiner.py` → `data/processing/` (if processing logic)
  - [ ] Move `benchmarking/data_loader.py` → `data/loaders/` (if shared loader)
  - [ ] Review `azureml/data_assets.py` - move to `data/assets/` or keep in infrastructure
  - [ ] Update all internal imports within data/
  - [ ] Update `data/__init__.py` exports
  - [ ] Create compatibility shims:
    - [ ] `training/data.py` - shim to `data.datasets` or `data.loaders`
    - [ ] `benchmarking/data_loader.py` - shim to `data.loaders`

- [ ] **Update data module structure**
  - [ ] Organize by concern: loaders, processing, datasets, assets
  - [ ] Ensure clear separation between shared and feature-specific data code
  - [ ] Document data module dependencies on infrastructure/

## Phase 6: Update Infrastructure Module Exports

- [ ] **Create infrastructure/init.py**
  - [ ] Export all public APIs from submodules:
    - [ ] `from infrastructure.config import *`
    - [ ] `from infrastructure.paths import *`
    - [ ] `from infrastructure.naming import *`
    - [ ] `from infrastructure.tracking import *`
    - [ ] `from infrastructure.storage import *`
    - [ ] `from infrastructure.fingerprints import *`
    - [ ] `from infrastructure.metadata import *`
    - [ ] `from infrastructure.platform import *`
  - [ ] Maintain backward compatibility signatures
  - [ ] Add deprecation timeline documentation

- [ ] **Create common/init.py**
  - [ ] Export all public APIs from submodules:
    - [ ] `from common.shared import *`
    - [ ] `from common.constants import *`
  - [ ] Maintain backward compatibility signatures

- [ ] **Create data/__init__.py**
  - [ ] Export all public APIs from submodules:
    - [ ] `from data.loaders import *`
    - [ ] `from data.processing import *`
    - [ ] `from data.datasets import *`
    - [ ] `from data.assets import *` (if applicable)
  - [ ] Maintain backward compatibility signatures

- [ ] **Update testing/__init__.py**
  - [ ] Ensure proper exports from submodules
  - [ ] Maintain backward compatibility signatures

## Phase 7: Update Orchestration Compatibility Layer

- [ ] **Update orchestration/init.py**
  - [ ] Change imports to use new infrastructure/ and common/ modules
  - [ ] Re-export for backward compatibility
  - [ ] Add deprecation warnings:
    ```python
    import warnings
    warnings.warn(
        "orchestration module is deprecated. "
        "Use 'infrastructure' or 'common' modules instead. "
        "This will be removed in 2 releases.",
        DeprecationWarning,
        stacklevel=2
    )
    ```

- [ ] **Create orchestration compatibility shims**
  - [ ] `orchestration/config.py` - shim to `infrastructure.config`
  - [ ] `orchestration/tracking.py` - shim to `infrastructure.tracking`
  - [ ] `orchestration/storage.py` - shim to `infrastructure.storage`
  - [ ] `orchestration/fingerprints.py` - shim to `infrastructure.fingerprints`
  - [ ] `orchestration/metadata.py` - shim to `infrastructure.metadata`
  - [ ] `orchestration/azureml.py` - shim to `infrastructure.platform.azureml`
  - [ ] `orchestration/platform_adapters.py` - shim to `infrastructure.platform.adapters`
  - [ ] `orchestration/shared.py` - shim to `common.shared`
  - [ ] `orchestration/constants.py` - shim to `common.constants`
  - [ ] `orchestration/testing.py` - shim to `testing` (if needed)
  - [ ] `orchestration/data_assets.py` - shim to `data.assets` or `infrastructure.platform.azureml.data_assets`

- [ ] **Update orchestration/jobs/ imports**
  - [ ] Update all files in `orchestration/jobs/` to use new infrastructure/ imports
  - [ ] Keep public API stable for external callers
  - [ ] Add deprecation warnings to job execution functions

## Phase 8: Update Internal Imports

- [ ] **Update imports in feature modules (training/, hpo/, benchmarking/, etc.)**
  - [ ] Update `src/training/` imports to use `infrastructure.*`, `common.*`, and `data.*`
  - [ ] Update `src/hpo/` imports
  - [ ] Update `src/benchmarking/` imports to use `data.*`
  - [ ] Update `src/selection/` imports
  - [ ] Update `src/conversion/` imports
  - [ ] Update `src/api/` imports
  - [ ] Update `src/training_exec/` imports
  - [ ] Update `src/testing/` imports to use `data.*` instead of `training.data`

- [ ] **Update imports in tests/**
  - [ ] Update test imports to use new module paths
  - [ ] Keep tests working with both old and new imports during transition
  - [ ] Update test fixtures if needed

- [ ] **Update imports in notebooks**
  - [ ] Keep notebooks using `orchestration.*` imports (stable API)
  - [ ] No changes needed to notebooks during Phase 1

- [ ] **Update src/init.py**
  - [ ] Update exports to use new module structure
  - [ ] Maintain backward compatibility
  - [ ] Add deprecation warnings for old imports

## Phase 9: Testing and Verification

- [ ] **Run existing tests**
  - [ ] Run all infrastructure tests
  - [ ] Run all feature tests (training, hpo, benchmarking, etc.)
  - [ ] Run all integration tests
  - [ ] Fix any test failures

- [ ] **Test backward compatibility**
  - [ ] Test that `orchestration.*` imports still work
  - [ ] Test that deprecation warnings are shown
  - [ ] Test that notebooks work without changes
  - [ ] Verify no breaking changes for external users

- [ ] **Verify no circular dependencies**
  - [ ] Run dependency checker
  - [ ] Verify `core/` has no dependencies on infrastructure/, common/, data/, or testing/
  - [ ] Verify `infrastructure/` depends only on `core/` and `common/`
  - [ ] Verify `common/` has no dependencies on `infrastructure/`, `data/`, or `testing/`
  - [ ] Verify `data/` depends only on `core/`, `common/`, and `infrastructure/`
  - [ ] Verify `testing/` depends on `core/`, `common/`, `infrastructure/`, and `data/`

- [ ] **Test import performance**
  - [ ] Verify no significant import time regressions
  - [ ] Test config caching still works
  - [ ] Test path resolution performance

- [ ] **Verify module isolation**
  - [ ] Test each infrastructure module independently
  - [ ] Test each common module independently
  - [ ] Test data module independently
  - [ ] Test testing module independently
  - [ ] Verify SRP (Single Responsibility Principle) is maintained

## Phase 10: Documentation and Cleanup

- [ ] **Update documentation**
  - [ ] Document new import patterns in README
  - [ ] Document deprecation timeline for `orchestration.*` imports
  - [ ] Update architecture diagrams
  - [ ] Document migration path for internal code

- [ ] **Code cleanup**
  - [ ] Remove unused imports
  - [ ] Fix any linter warnings
  - [ ] Ensure consistent code style
  - [ ] Update type hints if needed

- [ ] **Final verification**
  - [ ] Run full test suite
  - [ ] Verify no regressions
  - [ ] Check code coverage
  - [ ] Verify all compatibility shims work

## Phase 11: Future Phases (After Infrastructure Migration)

- [ ] **Phase 2: Training Module** (future)
  - [ ] Consolidate `training/`, `training_exec/`, `hpo/` into `training/`
  - [ ] Create `training/core/`, `training/hpo/`, `training/execution/`
  - [ ] Note: Data handling moved to top-level `data/` module

- [ ] **Phase 3: Evaluation Module** (future)
  - [ ] Move `benchmarking/` and `selection/` to `evaluation/`
  - [ ] Create `evaluation/benchmarking/` and `evaluation/selection/`

- [ ] **Phase 4: Deployment Module** (future)
  - [ ] Move `conversion/` and `api/` to `deployment/`
  - [ ] Create `deployment/conversion/` and `deployment/api/`

- [ ] **Phase 5: Remove Orchestration** (after 1-2 releases)
  - [ ] Remove all compatibility shims
  - [ ] Remove `orchestration/` directory
  - [ ] Update all remaining imports
  - [ ] Update notebooks to use new imports

## Quick Reference: File Mapping

### Infrastructure Module

- `src/config/` → `infrastructure/config/`
- `src/paths/` → `infrastructure/paths/` (if moved)
- `src/naming/` → `infrastructure/naming/` (if moved)
- `src/tracking/` → `infrastructure/tracking/`
- `src/storage/` → `infrastructure/storage/`
- `src/fingerprints/` → `infrastructure/fingerprints/`
- `src/metadata/` → `infrastructure/metadata/`
- `src/azureml/` → `infrastructure/platform/azureml/`
- `src/platform_adapters/` → `infrastructure/platform/adapters/`

### Common Module

- `src/shared/` → `common/shared/`
- `src/constants/` → `common/constants/`

### Data Module

- `src/training/data.py` → `data/datasets/` or `data/loaders/` (after review)
- `src/training/data_combiner.py` → `data/processing/` (after review)
- `src/benchmarking/data_loader.py` → `data/loaders/` (after review)
- `src/azureml/data_assets.py` → `data/assets/` or keep in `infrastructure/platform/azureml/` (after review)

### Testing Module

- `src/testing/` → No changes (already top-level, keep as is)

### Core Module

- `src/core/` → No changes (already correct)

## Notes

- Start with infrastructure/ as foundation for all other features
- Keep orchestration/ as compatibility layer during migration
- Remove orchestration/ after 1-2 releases (breaking change)
- All compatibility shims should emit deprecation warnings
- Maintain backward compatibility for notebooks and external users
- Test thoroughly after each phase before proceeding
- **Testing/ is a top-level module** - It's a substantial cross-cutting concern (22+ files, multiple subdirectories)
- **Data/ is a top-level module** - Shared across training, evaluation, and testing, not just training-specific

### To-dos

- [ ] Phase 1: Audit orchestration/ dependencies, infrastructure modules, platform adapters, common/ modules, data/ modules, and testing/ modules
- [ ] Phase 2: Create infrastructure/ module structure and move config/, tracking/, storage/, fingerprints/, metadata/
- [ ] Phase 3: Create infrastructure/platform/ and move azureml/ and platform_adapters/
- [ ] Phase 4: Create common/ module structure and move shared/, constants/ (testing/ stays top-level)
- [ ] Phase 5: Create data/ module structure and consolidate data handling from training/, benchmarking/, etc.
- [ ] Phase 6: Update infrastructure/, common/, data/, and testing/ __init__.py exports
- [ ] Phase 7: Update orchestration/ compatibility layer with shims and deprecation warnings
- [ ] Phase 8: Update internal imports in feature modules, tests, and src/__init__.py
- [ ] Phase 9: Run tests, verify backward compatibility, check for circular dependencies
- [ ] Phase 10: Update documentation, code cleanup, final verification