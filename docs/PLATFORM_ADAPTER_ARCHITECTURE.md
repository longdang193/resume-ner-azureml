# Platform Adapter Architecture

## Overview

This document describes the platform adapter architecture that separates Azure-specific concerns from platform-agnostic core logic, enabling the same code to run consistently on both Azure ML and local environments.

## Architecture

### Core Principle

**Platform-specific concerns** (environment variables, output paths, logging, MLflow context) are isolated in adapter classes, while **core logic** (training, model conversion, data processing) remains platform-agnostic.

### Structure

```
src/platform_adapters/
├── __init__.py              # Public API exports
├── adapters.py              # PlatformAdapter interface and implementations
├── outputs.py               # Output path resolution
├── logging_adapter.py       # Logging abstraction
├── mlflow_context.py        # MLflow context management
└── checkpoint_resolver.py   # Checkpoint path resolution
```

### Key Components

#### 1. PlatformAdapter Interface

The `PlatformAdapter` interface provides a unified way to access platform-specific functionality:

- `get_output_path_resolver()` - Resolve output directories
- `get_logging_adapter()` - Platform-specific logging
- `get_mlflow_context_manager()` - MLflow run context management
- `get_checkpoint_resolver()` - Checkpoint path resolution
- `is_platform_job()` - Detect platform-managed execution

#### 2. Implementations

**AzureMLAdapter**: Handles Azure ML-specific concerns
- Reads `AZURE_ML_OUTPUT_*` environment variables
- Uses Azure ML's automatic MLflow integration
- Handles Azure ML data asset references
- Creates placeholder files for output materialization

**LocalAdapter**: Handles local execution
- Uses simple file system paths
- Manages MLflow runs explicitly
- No special environment variable handling

#### 3. Auto-Detection

The `get_platform_adapter()` function automatically detects the execution environment:

```python
from platform_adapters import get_platform_adapter

# Automatically detects Azure ML vs local
adapter = get_platform_adapter(default_output_dir=Path("./outputs"))
```

## Usage Examples

### Training Script (`train.py`)

**Before** (Azure-specific code mixed in):
```python
# Azure-specific environment variable reading
output_dir = Path(
    os.getenv("AZURE_ML_OUTPUT_checkpoint")
    or os.getenv("AZURE_ML_OUTPUT_DIR", "./outputs")
)

# Azure-specific MLflow handling
is_azure_ml_job = any(key.startswith("AZURE_ML_") for key in os.environ.keys())
if is_azure_ml_job:
    # Azure ML path
    log_training_parameters(config)
    metrics = train_model(config, dataset, output_dir)
    log_metrics(output_dir, metrics)
else:
    # Local path
    with mlflow.start_run():
        log_training_parameters(config)
        metrics = train_model(config, dataset, output_dir)
        log_metrics(output_dir, metrics)
```

**After** (Platform-agnostic):
```python
from platform_adapters import get_platform_adapter

# Get platform adapter (auto-detects environment)
platform_adapter = get_platform_adapter(default_output_dir=Path("./outputs"))
output_resolver = platform_adapter.get_output_path_resolver()
logging_adapter = platform_adapter.get_logging_adapter()
mlflow_context = platform_adapter.get_mlflow_context_manager()

# Resolve output directory (works for both Azure and local)
output_dir = output_resolver.resolve_output_path("checkpoint", default=Path("./outputs"))
output_dir = output_resolver.ensure_output_directory(output_dir)

# Use platform-appropriate MLflow context
with mlflow_context.get_context():
    log_training_parameters(config, logging_adapter)
    metrics = train_model(config, dataset, output_dir)
    log_metrics(output_dir, metrics, logging_adapter)
```

### Conversion Script (`convert_to_onnx.py`)

**Before** (Azure-specific checkpoint resolution):
```python
def resolve_checkpoint_dir(checkpoint_path: str) -> Path:
    """Resolve an Azure ML mounted input folder..."""
    root = Path(checkpoint_path)
    # Azure ML-specific logic...
```

**After** (Platform-agnostic):
```python
from platform_adapters import get_platform_adapter

def resolve_checkpoint_dir(checkpoint_path: str) -> Path:
    """Resolve checkpoint path using platform adapter."""
    platform_adapter = get_platform_adapter()
    checkpoint_resolver = platform_adapter.get_checkpoint_resolver()
    return checkpoint_resolver.resolve_checkpoint_dir(checkpoint_path)
```

### Logging (`training/logging.py`)

**Before** (Azure-specific imports):
```python
try:
    from azureml.core import Run
    _azureml_run = Run.get_context()
    _azureml_available = True
except Exception:
    _azureml_run = None
    _azureml_available = False
```

**After** (Platform-agnostic):
```python
from platform_adapters.logging_adapter import LoggingAdapter

def log_metrics(
    output_dir: Path,
    metrics: Dict[str, float],
    logging_adapter: Optional[LoggingAdapter] = None,
) -> None:
    # Use provided adapter or create default one
    if logging_adapter is None:
        from platform_adapters import get_platform_adapter
        platform_adapter = get_platform_adapter()
        logging_adapter = platform_adapter.get_logging_adapter()
    
    logging_adapter.log_metrics(metrics)
```

## Benefits

1. **Testability**: Core logic can be unit tested without Azure ML dependencies
2. **Portability**: Same code runs on Azure ML, local machines, or other platforms
3. **Maintainability**: Platform-specific code is isolated and easier to update
4. **Clarity**: Clear separation of concerns makes the codebase easier to understand

## Migration Guide

### For New Code

1. Import platform adapter: `from platform_adapters import get_platform_adapter`
2. Get adapter instance: `adapter = get_platform_adapter()`
3. Use adapter methods instead of direct environment variable access
4. Pass adapters to functions that need platform-specific behavior

### For Existing Code

1. Replace direct `os.getenv("AZURE_ML_*")` calls with `output_resolver.resolve_output_path()`
2. Replace Azure ML detection logic with `platform_adapter.is_platform_job()`
3. Replace MLflow context management with `mlflow_context.get_context()`
4. Replace direct Azure ML logging with `logging_adapter.log_metrics()`

## Testing

The adapter pattern makes testing easier:

```python
# Test with local adapter
from platform_adapters import LocalAdapter

adapter = LocalAdapter(default_output_dir=Path("./test_outputs"))
output_resolver = adapter.get_output_path_resolver()
# Test output path resolution...

# Test with Azure adapter (mocked environment)
import os
os.environ["AZURE_ML_OUTPUT_checkpoint"] = "/mnt/checkpoint"
adapter = AzureMLAdapter()
output_resolver = adapter.get_output_path_resolver()
# Test Azure ML output path resolution...
```

## Future Extensions

The adapter pattern can be extended to support:
- Other cloud platforms (AWS SageMaker, GCP Vertex AI)
- Different logging backends
- Custom output storage systems
- Additional platform-specific features

