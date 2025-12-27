"""Platform-aware MLflow setup utility for cross-platform tracking."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Any

try:
    import mlflow
except ImportError:
    raise ImportError(
        "mlflow is required for experiment tracking. "
        "Install it with: pip install mlflow"
    )

from shared.platform_detection import detect_platform
from shared.yaml_utils import load_yaml


def _load_env_file(env_file_path: Path) -> dict:
    """
    Load environment variables from a .env file.

    Supports simple KEY="VALUE" or KEY=VALUE format.
    Comments (lines starting with #) are ignored.

    Args:
        env_file_path: Path to .env file

    Returns:
        Dictionary of key-value pairs
    """
    env_vars = {}
    if not env_file_path.exists():
        return env_vars

    try:
        with open(env_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Parse KEY="VALUE" or KEY=VALUE
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    env_vars[key] = value
    except Exception as e:
        print(f"  ⚠ Warning: Could not load {env_file_path}: {e}")

    return env_vars


def setup_mlflow_cross_platform(
    experiment_name: str,
    ml_client: Optional[Any] = None,
    fallback_to_local: bool = True,
) -> str:
    """
    Setup MLflow for cross-platform tracking.

    If ml_client provided, uses Azure ML workspace (unified tracking).
    Otherwise, falls back to platform-specific local tracking (SQLite backend).
    Also sets the MLflow experiment name.

    Args:
        experiment_name: MLflow experiment name (will be created if doesn't exist)
        ml_client: Optional Azure ML client for unified tracking. If provided,
                  must be an instance of azure.ai.ml.MLClient
        fallback_to_local: If True, fallback to local tracking when Azure ML fails

    Returns:
        Tracking URI string that was configured

    Raises:
        ImportError: If mlflow is not installed (with helpful error message)
        RuntimeError: If Azure ML required but unavailable and fallback disabled
    """
    # Try Azure ML first if ml_client provided
    if ml_client is not None:
        try:
            tracking_uri = _get_azure_ml_tracking_uri(ml_client)
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            print(f"✓ Using Azure ML workspace tracking")
            print(f"  Tracking URI: {tracking_uri}")
            return tracking_uri
        except Exception as e:
            if not fallback_to_local:
                raise RuntimeError(
                    f"Azure ML tracking failed and fallback disabled: {e}"
                ) from e
            print(f"⚠ Azure ML tracking failed: {e}")
            print("  Falling back to local tracking...")

    # Fallback to local tracking
    tracking_uri = _get_local_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"✓ Using local tracking: {tracking_uri}")
    return tracking_uri


def _get_azure_ml_tracking_uri(ml_client: Any) -> str:
    """
    Get Azure ML workspace tracking URI.

    Args:
        ml_client: Azure ML client instance (azure.ai.ml.MLClient)

    Returns:
        Azure ML workspace tracking URI string

    Raises:
        ImportError: If azureml.mlflow is not available
        RuntimeError: If workspace access fails
    """
    # Import azureml.mlflow to register the 'azureml' URI scheme
    try:
        import azureml.mlflow  # noqa: F401
    except ImportError:
        raise ImportError(
            "azureml.mlflow is required for Azure ML tracking. "
            "Install it with: pip install azureml-mlflow"
        )

    # Get workspace tracking URI
    try:
        workspace = ml_client.workspaces.get(name=ml_client.workspace_name)
        return workspace.mlflow_tracking_uri
    except Exception as e:
        raise RuntimeError(
            f"Failed to get Azure ML workspace tracking URI: {e}"
        ) from e


def _get_local_tracking_uri() -> str:
    """
    Get local tracking URI with platform-aware path resolution.

    Uses SQLite backend to address MLflow deprecation warning for file-based tracking.

    Returns:
        SQLite tracking URI string (e.g., "sqlite:///path/to/mlflow.db")
    """
    platform = detect_platform()

    if platform == "colab":
        # Check if Google Drive is mounted
        drive_path = Path("/content/drive/MyDrive")
        if drive_path.exists() and drive_path.is_dir():
            # Use Drive for persistence across sessions
            mlflow_db = drive_path / "resume-ner-mlflow" / "mlflow.db"
        else:
            # Fallback to /content/ if Drive not mounted
            mlflow_db = Path("/content") / "mlflow.db"
    elif platform == "kaggle":
        # Kaggle outputs in /kaggle/working/ are automatically persisted
        mlflow_db = Path("/kaggle/working") / "mlflow.db"
    else:
        # Local: use current directory
        mlflow_db = Path("./mlruns") / "mlflow.db"

    # Ensure parent directory exists
    mlflow_db.parent.mkdir(parents=True, exist_ok=True)

    # Convert to absolute path and return SQLite URI
    abs_path = mlflow_db.resolve()
    return f"sqlite:///{abs_path}"


def create_ml_client_from_config(
    config_dir: Path,
    mlflow_config: Optional[dict] = None,
) -> Optional[Any]:
    """
    Create Azure ML client from configuration files.

    Args:
        config_dir: Path to config directory (e.g., Path("config"))
        mlflow_config: Optional pre-loaded MLflow config dict. If None, loads from config/mlflow.yaml

    Returns:
        MLClient instance if Azure ML is enabled and credentials available, None otherwise

    Raises:
        ImportError: If azure-ai-ml is not installed
    """
    # Load MLflow config if not provided
    if mlflow_config is None:
        mlflow_config_path = config_dir / "mlflow.yaml"
        if not mlflow_config_path.exists():
            return None
        mlflow_config = load_yaml(mlflow_config_path)

    # Check if Azure ML is enabled
    azure_ml_config = mlflow_config.get("azure_ml", {})
    if not azure_ml_config.get("enabled", False):
        return None

    # Try to import Azure ML SDK
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
    except ImportError:
        raise ImportError(
            "azure-ai-ml and azure-identity are required for Azure ML tracking. "
            "Install with: pip install azure-ai-ml azure-identity"
        )

    # Get credentials from environment variables
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = azure_ml_config.get("workspace_name", "resume-ner-ws")

    # If environment variables not set, try loading from config.env file (in project root)
    if not subscription_id or not resource_group:
        project_root = config_dir.parent
        config_env_path = project_root / "config.env"
        if config_env_path.exists():
            print(
                f"  Environment variables not set, loading from {config_env_path}")
            env_vars = _load_env_file(config_env_path)
            subscription_id = subscription_id or env_vars.get(
                "AZURE_SUBSCRIPTION_ID")
            resource_group = resource_group or env_vars.get(
                "AZURE_RESOURCE_GROUP")
            if subscription_id and resource_group:
                # Set environment variables for this process
                os.environ["AZURE_SUBSCRIPTION_ID"] = subscription_id
                os.environ["AZURE_RESOURCE_GROUP"] = resource_group
                print(f"  ✓ Loaded credentials from config.env")
                print(f"    AZURE_SUBSCRIPTION_ID: {subscription_id[:8]}...")
                print(f"    AZURE_RESOURCE_GROUP: {resource_group}")
            else:
                print(
                    f"  ⚠ config.env exists but missing AZURE_SUBSCRIPTION_ID or AZURE_RESOURCE_GROUP")
        else:
            print(f"  ⚠ config.env not found at {config_env_path}")

    # If still not set, try loading from infrastructure.yaml
    # (infrastructure.yaml may contain ${AZURE_SUBSCRIPTION_ID} placeholders)
    if not subscription_id or not resource_group:
        infra_config_path = config_dir / "infrastructure.yaml"
        if infra_config_path.exists():
            infra_config = load_yaml(infra_config_path)
            azure_config = infra_config.get("azure", {})
            # Only use infrastructure.yaml values if they're not placeholders
            infra_sub_id = azure_config.get("subscription_id", "")
            infra_rg = azure_config.get("resource_group", "")
            if infra_sub_id and not infra_sub_id.startswith("${"):
                subscription_id = subscription_id or infra_sub_id
            if infra_rg and not infra_rg.startswith("${"):
                resource_group = resource_group or infra_rg

    # Check if we have required values
    if not subscription_id or not resource_group:
        print(
            "⚠ Azure ML enabled but credentials not found. Falling back to local tracking.")
        print("  Set AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP environment variables.")
        return None

    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        return ml_client
    except Exception as e:
        print(f"⚠ Failed to create Azure ML client: {e}")
        print("  Falling back to local tracking.")
        return None


def setup_mlflow_from_config(
    experiment_name: str,
    config_dir: Path,
    fallback_to_local: bool = True,
) -> str:
    """
    Setup MLflow from configuration files.

    Reads config/mlflow.yaml to determine whether to use Azure ML Workspace
    or local tracking. Automatically creates MLClient if Azure ML is enabled.

    Args:
        experiment_name: MLflow experiment name (will be created if doesn't exist)
        config_dir: Path to config directory (e.g., Path("config"))
        fallback_to_local: If True, fallback to local tracking when Azure ML fails

    Returns:
        Tracking URI string that was configured

    Raises:
        ImportError: If mlflow is not installed
        FileNotFoundError: If config/mlflow.yaml doesn't exist (only if Azure ML enabled)
    """
    # Load MLflow config
    mlflow_config_path = config_dir / "mlflow.yaml"
    if not mlflow_config_path.exists():
        # If config doesn't exist, use local tracking
        print("⚠ MLflow config not found, using local tracking")
        return setup_mlflow_cross_platform(
            experiment_name=experiment_name,
            ml_client=None,
            fallback_to_local=fallback_to_local,
        )

    mlflow_config = load_yaml(mlflow_config_path)

    # Try to create MLClient if Azure ML is enabled
    ml_client = None
    azure_ml_config = mlflow_config.get("azure_ml", {})
    if azure_ml_config.get("enabled", False):
        print(f"  Azure ML enabled in config, attempting to connect...")
        print(
            f"  Workspace: {azure_ml_config.get('workspace_name', 'unknown')}")
        # Always try to create ML client - it will load from config.env if env vars not set
        ml_client = create_ml_client_from_config(config_dir, mlflow_config)
        if ml_client is None:
            print(
                f"  ⚠ Warning: Failed to create Azure ML client, falling back to local tracking")
    else:
        print(f"  Azure ML disabled in config, using local tracking")

    # Setup MLflow with or without Azure ML
    return setup_mlflow_cross_platform(
        experiment_name=experiment_name,
        ml_client=ml_client,
        fallback_to_local=fallback_to_local,
    )
