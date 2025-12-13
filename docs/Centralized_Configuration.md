## Purpose

Centralize all non-secret configuration to ensure **reproducibility, safety, and auditability** across training, inference, and operations.

## Core Principles

* **Single source of truth:** All configuration lives in `config/` and is version-controlled.
* **Separation of concerns:** Data, model, HPO, environment, and monitoring configs evolve independently.
* **Declarative:** Configuration describes behavior; code must not mutate it.
* **Compose → validate → freeze:** Hydra composes configs; Pydantic validates and freezes them.

## Directory Structure (Updated)

```text
config/
├── data/                  # Multiple datasets (versioned)
│   └── resume_v1.yaml
├── model/                 # Multiple model architectures
│   ├── distilbert.yaml
│   └── deberta.yaml
├── hpo/                   # Multiple HPO strategies
│   ├── smoke.yaml
│   └── prod.yaml
├── env/                   # Execution environment
│   ├── local.yaml
│   └── azure.yaml
├── train.yaml             # Global training defaults
└── monitoring.yaml        # Metrics & alert thresholds

src/
├── train.py
├── inference.py
├── orchestration/
│   └── submit_jobs.py
└── schemas/
    ├── __init__.py
    ├── data_config.py
    ├── train_config.py
    ├── inference_config.py
    └── monitoring_config.py
```

## Config Responsibilities

| Config | Responsibility |
| ----------------- | ----------------------------------------- |
| `data/*.yaml` | Dataset contract (name, version, schema) |
| `model/*.yaml` | Model architecture–specific settings |
| `train.yaml` | Global training defaults |
| `hpo/*.yaml` | HPO search strategies (smoke, prod, etc.) |
| `env/*.yaml` | Execution environment (compute, logging) |
| `monitoring.yaml` | Metrics and alert thresholds |

## Hydra Usage

* Used only for **composition and selection**, not validation.
* Selects dataset, model, HPO profile, and environment.
* No secrets and no validation logic in Hydra configs.

Example:

```bash
python train.py data=resume_v1 model=distilbert hpo=smoke env=azure
```

## Pydantic Validation

* All composed configs are validated via schemas in `src/schemas/` before:
	* Azure ML job submission
	* inference or deployment steps
* Invalid configuration fails immediately.
* Config objects are treated as immutable after validation.

## Phase-Specific Consumption

| Phase | Configs Used |
| --------------- | --------------------------------------------------- |
| Training (P1) | `data/*`, `model/*`, `train.yaml`, `hpo/*`, `env/*` |
| Inference (P2) | `model/*`, `env/*` |
| Operations (P3) | `monitoring.yaml`, `env/*` |

## Lineage & Reproducibility

For every training run:

* Log fully resolved config files.
* Log config hashes.
* Attach config hashes as model registry tags.

This ensures every model artifact is fully traceable to its configuration.

## Environment Overrides

**Allowed overrides**

* Compute targets
* Logging sinks
* Resource limits

**Not allowed**

* Model architecture
* Training hyperparameters
* Dataset schema

## Secrets

Secrets are **never** stored in configuration files.

Use:

* Azure Key Vault
* GitHub Secrets

Configs may reference secret **names only**, never values.

## Forbidden Patterns

* Single, monolithic `config.yaml`
* Editing config files to switch experiments
* Runtime mutation of validated config
* Hardcoded values in notebooks
* Auto-generated config files
