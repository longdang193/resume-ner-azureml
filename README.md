# Resume NER with Azure ML

A production-ready Named Entity Recognition (NER) system for extracting structured information from resumes using Azure Machine Learning. This project demonstrates end-to-end MLOps practices including hyperparameter optimization, model training, benchmarking, conversion to ONNX, and deployment as a FastAPI service.

## Project Overview

This project implements a complete ML pipeline for resume NER that:

- **Extracts entities** from resume text (e.g., names, skills, education, experience)
- **Trains transformer models** (DistilBERT, DeBERTa, etc.) using Azure ML
- **Optimizes hyperparameters** through systematic sweeps with MLflow tracking
- **Benchmarks models** to select the best configuration
- **Converts models** to ONNX format for efficient inference
- **Deploys as API** via FastAPI with Docker containerization

### Target Users

- **ML Engineers** building production NER systems
- **Data Scientists** exploring transformer-based entity extraction
- **DevOps Engineers** implementing MLOps pipelines on Azure

### Models & Optimization

- **Base Models**: DistilBERT, DeBERTa, BERT variants
- **Optimization**: Azure ML Hyperparameter Sweeps with Bayesian search
- **Metrics**: Macro-F1 (handles class imbalance in NER tags)
- **Deployment**: ONNX runtime for fast inference

---

## Pain Points & Solutions

### The Challenges

Building production ML systems for NER involves several common pain points that this project addresses:

#### 1. **Reproducibility Nightmare**

**Problems**:

- Different environments (local, Colab, Kaggle, Azure ML) produce inconsistent results
- Hash mismatches between runs make it impossible to track which model came from which configuration
- No single source of truth for experiment configurations leads to "works on my machine" scenarios

**How This Project Solves It**:

- **MLflow Tags as Single Source of Truth**: All hashes and configurations are stored as MLflow tags, ensuring consistency across environments
- **Centralized Hash Computation**: `src/infrastructure/tracking/mlflow/hash_utils.py` computes hashes identically everywhere, eliminating mismatches
- **Environment-Agnostic Code**: Same codebase works identically in local, Colab, Kaggle, and Azure ML environments
- **Full Traceability**: Every run stores its complete configuration as MLflow tags, enabling full reproducibility

**Impact**: Eliminates "works on my machine" issues. Any run can be reproduced by reading its MLflow tags, regardless of where it was executed.

#### 2. **Model Selection Complexity**

**Problems**:

- After running hundreds of HPO trials, finding the best model requires manual MLflow queries and guesswork
- No deterministic way to select champions per model backbone
- Benchmarking the same model multiple times wastes compute and time

**How This Project Solves It**:

- **Automated Champion Selection**: `notebooks/02_best_config_selection.ipynb` automatically discovers top HPO candidates from MLflow, benchmarks them, and selects champions based on macro-F1 score
- **Deterministic Selection**: Uses consistent hash-based grouping (`study_key_hash_v2`) to ensure the same champion is selected every time
- **Idempotent Benchmarking**: Benchmark runs are keyed by champion `run_id` + fingerprints, preventing redundant benchmarks
- **Structured Logging**: Clear logs show which models were considered, why champions were selected, and benchmark results

**Impact**: No more manual MLflow queries. Run the notebook once, get deterministic results every time. Saves hours of manual work and prevents wasted compute.

#### 3. **Artifact Management Chaos**

**Problems**:

- Checkpoints scattered across local disk, Google Drive, MLflow, and Azure ML storage
- No unified way to retrieve artifacts - each stage implements its own logic
- Refit runs create duplicate artifacts, leading to storage bloat and confusion

**How This Project Solves It**:

- **Unified Artifact Acquisition**: `src/evaluation/selection/artifact_unified/` provides a single system for all artifact retrieval
- **Refit-Aware Mapping**: Automatically maps trial runs to their refit runs, preventing duplicate downloads and storage bloat
- **Priority-Based Retrieval**: Configurable priority per artifact kind (checkpoint, ONNX model, etc.) - checks local → drive → MLflow in order
- **Artifact-Kind-Specific Validation**: Ensures artifact integrity before use, preventing corrupted checkpoint issues
- **Multi-Source Support**: Seamlessly retrieves from local disk, Google Drive, or MLflow based on availability

**Impact**: One system handles all artifact retrieval. No more scattered checkpoint management or duplicate storage. Saves storage costs and eliminates confusion.

#### 4. **Configuration Drift**

**Problems**:

- Training configs differ between HPO, final training, and benchmarking stages
- Environment-specific settings (local vs cloud) require manual changes
- No way to track which config produced which model

**How This Project Solves It**:

- **Hierarchical YAML Configuration**: Global defaults in `config/train.yaml`, stage-specific overrides in subdirectories (`config/hpo/`, `config/data/`, etc.)
- **Environment-Aware Defaults**: Configs automatically adapt to local vs Azure ML vs Colab/Kaggle environments without manual changes
- **MLflow Integration**: Every run logs its full configuration as MLflow tags, providing complete traceability
- **Version Control**: All configs are versioned in git, providing audit trail of what changed when
- **Single Source of Truth**: `config/mlflow.yaml` centralizes experiment naming and tracking settings across all stages

**Impact**: Change configs in one place, see effects everywhere. Full audit trail of what changed when. No more configuration drift between stages.

---

## End-to-End Pipeline

| Stage                    | How it runs                                                                                                                                    | Key artifacts                                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **EDA**                  | `notebooks/eda.ipynb` analyzes sentence length, tag distribution, vocabulary, and class balance to guide preprocessing decisions.             | EDA insights, visualizations, preprocessing recommendations.                                    |
| **HPO**                  | `notebooks/01_orchestrate_training_colab.ipynb` submits AML sweeps, monitors runs, tracks results in MLflow.                                 | MLflow runs (one per trial), hyperparameter search results, best configurations.                |
| **Model Selection**      | `notebooks/02_best_config_selection.ipynb` benchmarks top candidates, selects champion model based on macro-F1.                              | Benchmark results, selected model configuration, artifact acquisition logs.                     |
| **Final Training**       | `src/orchestration/jobs/` submits training jobs using selected hyperparameters, logs checkpoints and metrics.                                | Trained model checkpoints, metrics.json, MLflow artifacts.                                       |
| **ONNX Conversion**      | `src/deployment/conversion/` converts PyTorch checkpoints to ONNX format for optimized inference.                                          | ONNX model files, conversion logs, tokenizer/config artifacts.                                  |
| **API Deployment**       | `src/api/` provides FastAPI server with Docker support; `notebooks/api_testing.ipynb` tests endpoints.                                        | Docker image, API endpoints, inference results.                                                  |
| **Infrastructure Setup** | `notebooks/00_setup_infrastructure.ipynb` provisions Azure ML workspace, compute clusters, and data assets.                                 | Azure ML workspace, compute resources, registered datasets.                                      |

Together these deliverables cover the optimize → benchmark → retrain → convert → deploy loop currently implemented in the repo.

---

## Project Structure

```text
resume-ner-azureml/
├── src/                          # Core source code
│   ├── api/                      # FastAPI application
│   ├── training/                 # Training logic and HPO
│   │   ├── hpo/                 # Hyperparameter optimization
│   │   └── execution/           # Training execution
│   ├── evaluation/              # Model evaluation and benchmarking
│   │   ├── benchmarking/        # Benchmark orchestrator
│   │   └── selection/           # Best model selection
│   ├── deployment/              # Deployment utilities
│   │   ├── conversion/          # ONNX conversion
│   │   └── api/                 # API deployment
│   ├── infrastructure/          # Infrastructure utilities
│   │   ├── tracking/            # MLflow tracking
│   │   ├── config/              # Configuration management
│   │   └── naming/              # Naming conventions
│   ├── orchestration/           # Pipeline orchestration
│   │   └── jobs/                # Azure ML job definitions
│   ├── data/                    # Data processing
│   ├── common/                  # Shared utilities
│   └── testing/                 # Testing utilities
├── notebooks/                   # Jupyter notebooks
│   ├── 00_setup_infrastructure.ipynb
│   ├── 01_orchestrate_training_colab.ipynb
│   ├── 02_best_config_selection.ipynb
│   ├── eda.ipynb
│   └── api_testing.ipynb
├── config/                      # Configuration files
│   ├── data/                    # Data configuration
│   ├── hpo/                     # HPO configuration
│   ├── train.yaml               # Training defaults
│   ├── benchmark.yaml           # Benchmarking config
│   ├── conversion.yaml          # ONNX conversion config
│   ├── mlflow.yaml              # MLflow tracking config
│   └── infrastructure.yaml      # Azure ML infrastructure
├── dataset/                     # Training datasets
├── dataset_tiny/                 # Small test datasets
├── outputs/                     # Model outputs and artifacts
├── tests/                       # Test suite
├── docs/                        # Documentation
├── tools/                       # Utility scripts
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
├── requirements.txt             # Python dependencies
├── config.env.example           # Azure configuration template
└── README.md                    # This file
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+ (matches Docker image)
- Azure subscription with Azure ML workspace
- Docker (for containerized deployment)
- Azure CLI (`az login`)

### Local Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd resume-ner-azureml

# Create a Python 3.10 virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest pytest-cov black ruff
```

Deliverable: reproducible local environment aligned with the Dockerfile.

### Azure ML Setup

1. **Create `config.env`** from `config.env.example`:

   ```bash
   cp config.env.example config.env
   ```

2. **Fill in Azure credentials** (see `config.env.example` for details)

3. **Authenticate**:

   ```bash
   az login
   ```

### Infrastructure Setup

Run the infrastructure setup notebook to provision Azure ML resources:

```bash
jupyter notebook notebooks/00_setup_infrastructure.ipynb
```

This creates:

- Azure ML workspace
- Compute clusters
- Data assets
- Container registry (if needed)

---

## Configuration

### Configuration Files Overview

| File                      | Purpose                                                                      |
| ------------------------- | ---------------------------------------------------------------------------- |
| `config/train.yaml`       | Training hyperparameters (epochs, batch size, learning rate, etc.)         |
| `config/hpo/*.yaml`       | Hyperparameter search spaces for sweeps                                     |
| `config/benchmark.yaml`   | Benchmarking configuration and evaluation metrics                            |
| `config/conversion.yaml`  | ONNX conversion settings (opset version, dynamic axes, etc.)                |
| `config/mlflow.yaml`      | MLflow experiment naming, tracking settings, run finder configuration       |
| `config/infrastructure.yaml` | Azure ML workspace, compute, and data asset names                          |
| `config/data/*.yaml`      | Data preprocessing, tokenization, and split configuration                    |

### Key Configuration Patterns

- **YAML-based**: All configurations use YAML for readability and version control
- **Hierarchical**: Global defaults in `train.yaml`, stage-specific overrides in subdirectories
- **Environment-aware**: Configs adapt to local vs. Azure ML vs. Colab/Kaggle environments
- **MLflow integration**: Unified experiment naming and tracking across all stages

---

## Running the Pipeline

The pipeline is designed to run seamlessly across multiple platforms: **local**, **Google Colab**, **Kaggle**, and **Azure ML**. The same notebooks and scripts work identically everywhere, thanks to environment-aware configuration.

### 1. Hyperparameter Optimization

**Notebook-driven HPO** (`notebooks/01_orchestrate_training_colab.ipynb`):

- Builds sweeps from `config/hpo/` configurations
- Can submit to Azure ML sweep jobs (when Azure ML is configured) or run locally/Colab/Kaggle
- Tracks results in MLflow (unified tracking across all platforms)
- Platform-agnostic execution - same code works everywhere

**Key features**:

- Bayesian search optimization
- Cross-validation support
- Automatic run tracking and artifact logging
- Best trial selection

### 2. Model Selection & Benchmarking

**Best model selection** (`notebooks/02_best_config_selection.ipynb`):

- Discovers top HPO candidates from MLflow (works with any MLflow backend)
- Benchmarks models on held-out test set
- Selects champion based on macro-F1 score
- Exports best configuration for final training

**Idempotent**: Rerunning benchmarks the same model uses cached results, preventing wasted compute.

### 3. Final Training Pipeline

**Orchestration scripts** (`src/orchestration/jobs/`):

- Submit training jobs using selected hyperparameters
- Can run on Azure ML compute (when configured) or local/Colab/Kaggle environments
- Log checkpoints and metrics to MLflow
- Support distributed training (multi-GPU/DDP) when available
- Generate ONNX models for deployment

### 4. ONNX Conversion

**Conversion pipeline** (`src/deployment/conversion/`):

- Converts PyTorch checkpoints to ONNX format
- Optimizes for inference performance
- Validates conversion correctness
- Logs conversion artifacts to MLflow

### 5. API Deployment

**FastAPI server** (`src/api/`):

- RESTful endpoints for resume NER inference
- Docker containerization support
- PDF extraction and OCR capabilities
- Health check and monitoring endpoints

**Testing** (`notebooks/api_testing.ipynb`):

- Tests API endpoints with sample resumes
- Validates inference results
- Performance benchmarking

All flows use configuration files (`config/*.yaml`) and adapt automatically to the execution environment. Azure ML-specific settings (when used) are loaded from `config.env`.

---

## Documentation

- `docs/DOCKER.md` – Docker build and deployment guide
- `docs/rules/CLEAN_CODE.md` – Code style and best practices
- `docs/rules/CLEAN_DOC.md` – Documentation standards
- `docs/qa/` – Quality assurance guides and test data descriptions
- `docs/implementation_plans/` – Detailed implementation plans and architecture

---

## Docker Usage

### Build the API Image

**With build arguments (model files in image):**

```bash
docker build \
  --build-arg ONNX_MODEL_PATH=path/to/model.onnx \
  --build-arg CHECKPOINT_PATH=path/to/checkpoint \
  -t resume-ner-api:latest .
```

**With volume mounts (model files from host):**

```bash
docker build -t resume-ner-api:latest .

docker run -d -p 8000:8000 \
  -v $(pwd)/path/to/model.onnx:/app/models/model.onnx \
  -v $(pwd)/path/to/checkpoint:/app/models/checkpoint \
  resume-ner-api:latest
```

### Docker Compose

```bash
docker-compose up -d
```

See `docs/DOCKER.md` for detailed Docker usage instructions.

---

## Evaluation Criteria Checklist

| Requirement           | Evidence in this repo                                                                                                                                                                                      |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Problem description** | Project Overview explains the resume NER problem, target users, models, optimization path, and deployment approach.                                                                                        |
| **EDA**               | `notebooks/eda.ipynb` performs sentence length analysis, tag distribution, class imbalance checks, vocabulary exploration, and sequence pattern analysis to guide preprocessing decisions.                  |
| **Model training**    | Multiple transformer models (DistilBERT, DeBERTa, BERT) are trained. Azure ML sweeps (defined in `config/hpo/`) tune hyperparameters such as learning rate, batch size, warmup steps, achieving systematic optimization. |
| **Notebook → script** | Core training logic lives in `src/training/`; notebooks orchestrate sweeps, benchmarking, and deployments. Pipeline orchestration in `src/orchestration/jobs/`.                                        |
| **Reproducibility**   | Dataset (`dataset/`) and model artifacts (`outputs/`) are versioned. README documents environment setup, configuration, and pipeline execution for both local and AML runs. MLflow tracks all experiments. |
| **Model deployment**  | `src/api/` provides FastAPI server; `notebooks/api_testing.ipynb` tests endpoints. Docker containerization enables cloud deployment. ONNX conversion optimizes inference.                                 |
| **Dependencies & env** | `requirements.txt`, `Dockerfile`, and virtual environment instructions describe installation/activation, satisfying production requirements.                                                              |
| **Containerization**  | `Dockerfile` builds the runtime image used locally and in Azure ML. `docker-compose.yml` simplifies local development.                                                                                   |

---

## Project Limitations & Future Work

| Area                        | Current state                                                              | Future direction                                                                                                |
| --------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Automated CI/CD**         | Manual notebook verification; pytest suite exists but CI pipeline not configured. | Add GitHub Actions to lint/test `src/` scripts, configs, and notebooks. Automated deployment pipelines.        |
| **Monitoring & drift**      | No production monitoring after deployment.                                 | Integrate Application Insights or scheduled batch scoring to track drift, latency, and accuracy metrics.        |
| **Deployment & registry lifecycle** | Single replica endpoint; models registered per run without promotion stages. | Add blue/green rollout logic, autoscaling policies, and MLflow/Azure ML registry stages (dev → staging → prod). |
| **Multi-model ensemble**    | Single model deployment.                                                   | Explore ensemble methods for improved accuracy.                                                                 |
| **Active learning**         | Static training dataset.                                                   | Implement active learning loop to improve model with user feedback.                                              |

---

## Acknowledgments

This project demonstrates production-ready MLOps practices for Named Entity Recognition using Azure Machine Learning, following industry best practices for model development, optimization, and deployment.

---

## License

[Add your license here]
