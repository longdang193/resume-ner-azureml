# Module Documentation Index

This index is the starting point for **module-level documentation** in `src/`.

If you're new to the repo, start with:
- [`src/training/README.md`](../../src/training/README.md) - training + HPO (local and AzureML)
- [`src/evaluation/README.md`](../../src/evaluation/README.md) - selection + benchmarking
- [`src/deployment/README.md`](../../src/deployment/README.md) - conversion + API serving
- [`src/infrastructure/README.md`](../../src/infrastructure/README.md) - config/paths/naming/tracking/AzureML runtime

## Core modules

- [`src/core/README.md`](../../src/core/README.md) - core text/token utilities (normalization, placeholders)
- [`src/common/README.md`](../../src/common/README.md) - shared utilities (logging, hashing, shared helpers)
- [`src/data/README.md`](../../src/data/README.md) - dataset format + data loading/processing

## Training (HPO + final training)

- [`src/training/README.md`](../../src/training/README.md) - main entrypoints and workflows
- [`src/training/hpo/README.md`](../../src/training/hpo/README.md) - Optuna sweeps + sweep execution backends
- [`src/training/execution/README.md`](../../src/training/execution/README.md) - local execution + AzureML job builders
- [`src/training/core/README.md`](../../src/training/core/README.md) - trainer internals (models, checkpointing)

## Evaluation (selection + benchmarking)

- [`src/evaluation/README.md`](../../src/evaluation/README.md) - evaluation overview
- [`src/evaluation/selection/README.md`](../../src/evaluation/selection/README.md) - best-trial selection + discovery utilities
- [`src/evaluation/benchmarking/README.md`](../../src/evaluation/benchmarking/README.md) - benchmarking workflow

## Deployment (conversion + API serving)

- [`src/deployment/README.md`](../../src/deployment/README.md) - deployment overview
- [`src/deployment/conversion/README.md`](../../src/deployment/conversion/README.md) - PyTorch → ONNX conversion workflow
- [`src/deployment/api/README.md`](../../src/deployment/api/README.md) - FastAPI serving
- [`src/deployment/api/tools/README.md`](../../src/deployment/api/tools/README.md) - API utilities and tooling

## Infrastructure (config, paths, MLflow, AzureML runtime)

- [`src/infrastructure/README.md`](../../src/infrastructure/README.md) - infrastructure overview
- [`src/infrastructure/config/README.md`](../../src/infrastructure/config/README.md) - configuration loading/validation
- [`src/infrastructure/paths/README.md`](../../src/infrastructure/paths/README.md) - output path resolution and layout
- [`src/infrastructure/naming/README.md`](../../src/infrastructure/naming/README.md) - naming policies and run keys
- [`src/infrastructure/tracking/README.md`](../../src/infrastructure/tracking/README.md) - MLflow setup and tracking utilities
- [`src/infrastructure/platform/README.md`](../../src/infrastructure/platform/README.md) - platform adapters (local/AzureML/Colab)

## Testing utilities

- [`src/testing/README.md`](../../src/testing/README.md) - testing helpers and patterns

## Key supporting docs

- [`docs/docker_build.md`](../docker_build.md) - Docker build/run and troubleshooting
- [`docs/api_testing_prerequisites.md`](../api_testing_prerequisites.md) - prerequisites and setup for API testing
