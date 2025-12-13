## Step P1-1: Set up the Project (Repo + README)

### Context

Before any infrastructure or training runs, the project must define clear boundaries between configuration, code, and orchestration to ensure reproducibility and reviewability.

### Action

1. Create a version-controlled repository.
2. Define a clear directory structure for:

* configs
* source code
* notebooks

3. Write a README describing:

* problem statement
* data source
* training workflow
* artifacts produced

### Rationale

* Establishes clear ownership of code, configuration, and orchestration.
* Makes the project reviewable and reproducible.

### Important

* All executable training logic must live in `src/`.
* Notebooks must not contain training code.
* The README must explicitly state that training happens in Azure ML jobs, not locally.

### Deliverable

* Version-controlled repository with a documented structure.
* README that clearly explains the end-to-end workflow.

## Step P1-2: Infrastructure Setup (Azure)

### Context

Training and inference depend on cloud resources that are stateful, slow-changing, and shared across experiments.

### Action

1. Provision or verify existence of:

* Azure ML Workspace
* Blob Storage account and containers
* GPU compute cluster (training + HPO)
* CPU compute cluster (conversion)

2. Configure identity, permissions, and access to storage.

### Rationale

* Infrastructure should be stable and reused across runs.
* Keeping infra separate avoids accidental recreation or drift.
* Allows the orchestration notebook to focus purely on ML logic.

### Important

* Infrastructure must not be created implicitly by the orchestration notebook.
* The notebook should validate infrastructure existence and fail fast if missing.
* Compute clusters should have auto-scale limits configured to control cost.

### Deliverable

* Verified Azure ML workspace with accessible storage and compute resources.

## Step P1-3: The Orchestration Notebook (The Control Center)

**Location:** `notebooks/01_orchestrate_training.ipynb`

### Context

A single control plane is required to orchestrate all Phase 1 activities without performing local computation.

### Action

1. Act as the single entry point for Phase 1.
2. Define, submit, and monitor Azure ML jobs only.
3. Never perform local training or data processing.

### Rationale

* Ensures parity between development and production environments.
* Avoids hidden local state and cost surprises.
* Makes execution behavior explicit and auditable.

### Important

* The notebook must be re-runnable end-to-end.
* No training logic or model code is allowed inside the notebook.
* Steps P1-3.1–P1-3.7 are orchestrated from this notebook only and executed remotely.

### Deliverable

* A runnable orchestration notebook that can reproduce Phase 1 end-to-end.

## Step P1-3.1: Load Centralized Configs

### Context

Configuration must define system behavior independently of code to enable reproducibility and auditability.

### Action

1. Load necessary configs.
2. Pass configurations verbatim to Azure ML jobs.

### Rationale

* Decouples behavior from implementation.
* Makes experiments declarative and reviewable.
* Enables exact experiment reproduction.

### Important

* Config files must not be mutated at runtime.
* Hashes of all configs must be logged with each job.

### Deliverable

* Immutable configuration inputs passed to all submitted jobs.

## Step P1-3.2: Data Ingestion & Versioning (Asset Layer)

### Context

Training data must be versioned and consumed consistently across runs to ensure reproducibility.

### Action

1. Upload the dataset to Blob Storage (idempotent).
2. Register or resolve an Azure ML Data Asset:

* `type: uri_folder`

3. Validate the asset:

* file presence
* sample readability
* token/label alignment

### Rationale

* Makes data immutable and versioned.
* Azure ML jobs consume assets rather than raw paths.
* Enables dataset-level lineage tracking.

### Important

* `uri_folder` must be used.
* Data asset versions are immutable; data changes require a new version (`v2`, `v3`, …).
* Jobs must reference the asset by **name + version**, never by path.

### Deliverable

* Registered Azure ML Data Asset with a fixed name and version.

## Step P1-3.3: Environment Definition

### Context

A stable execution environment is required to ensure consistent behavior across jobs and time.

### Action

1. Define a training environment (Docker image + Conda dependencies).
2. Build or resolve the environment in Azure ML.
3. Reuse the same environment for:

* dry run
* sweep
* final training

### Rationale

* The environment defines the execution contract.
* Prevents dependency drift.
* Ensures reproducibility across all runs.

### Important

* Environment definitions must be versioned.
* Jobs must reference the environment by name and version.
* Environments must be immutable once used in a production lineage.

### Deliverable

* Versioned Azure ML environment reusable across jobs.

## Step P1-3.4: The Dry Run

### Context

Before launching expensive sweeps, the full pipeline must be validated cheaply.

### Action

1. Submit a single Azure ML Command Job:

* 1 epoch
* minimal resources

2. Verify:

* data loading
* label alignment
* logging
* checkpoint creation

### Rationale

* Catches integration errors early.
* Prevents wasted GPU cost.
* Validates end-to-end pipeline integrity.

### Important

* If the dry run fails, the pipeline must stop.
* Dry-run artifacts are not production-eligible.

### Deliverable

* Successful dry-run job confirming pipeline correctness.

## Step P1-3.5: The Sweep (HPO)

### Context

Model architecture and hyperparameters must be selected systematically using validation data.

### Action

1. Submit an Azure ML Sweep Job.
2. Search over, for example:

* backbone (DeBERTa vs DistilBERT)
* learning rate
* batch size
* epochs
* dropout
* CRF on/off

3. Optimize for validation entity-level F1.

### Rationale

* Separates selection from production.
* Identifies optimal configuration efficiently.
* Avoids overfitting to training data.

### Important

* Sweep runs are for selection only.
* Sweep checkpoints must never be deployed.

### Deliverable

* Best-performing configuration identified via HPO.

## Step P1-3.6: Best Configuration Selection (Automated)

### Context

Manual selection introduces error and irreproducibility.

### Action

1. Programmatically select the best sweep run.
2. Extract:

* backbone
* hyperparameters
* dataset version
* git commit hash

### Rationale

* Enables automation and repeatability.
* Creates a clean link between experimentation and production.

### Important

* Selection logic must be deterministic.
* Selection criteria must be explicitly logged.

### Deliverable

* A fully specified best configuration for final training.

## Step P1-3.7: Final Training (Post-HPO, Single Run)

### Context

The final production model must be trained under stable, controlled conditions.

### Action

1. Launch a single Azure ML training job.
2. Use:

* best hyperparameters from HPO
* combined train + validation data
* fixed random seed

3. Train for the full epoch budget.

### Rationale

* HPO runs are noisy and constrained.
* Final training produces a stable, high-quality model.
* Defines the production lineage.

### Important

* Only this run produces a checkpoint eligible for conversion.
* Early stopping and sweeping are disabled.

### Deliverable

* Final PyTorch checkpoint ready for optimization.

## Step P1-4: Model Conversion & Optimization

### Context

Production inference requires optimized, framework-agnostic artifacts.

### Action

1. Launch a separate CPU job.
2. Load the final checkpoint.
3. Convert to ONNX.
4. Quantize to `int8`.
5. Run a smoke inference test.

### Rationale

* Conversion is deterministic and inexpensive.
* Separates optimization from training.
* Ensures deployment readiness.

### Important

* Fail the pipeline if conversion or smoke test fails.
* Training checkpoints must never be registered directly.

### Deliverable

* Validated, optimized ONNX model.

## Step P1-5: Model Registration (The Handover)

### Context

A formal handover is required between training and serving systems.

### Action

1. Register the final ONNX artifact in Azure ML Model Registry.
2. Attach metadata and tags:

* dataset version
* backbone
* metric
* `stage=prod`

### Rationale

* The registry enforces the production contract.
* Decouples training from serving.
* Enables safe and repeatable deployment.

### Important

* Only optimized ONNX models are registered.
* Registry models are immutable and versioned.

### Deliverable

* Production-registered ONNX model ready for Phase 2 inference.
