from __future__ import annotations

"""
@meta
name: shared_mlflow_setup
type: utility
domain: shared
responsibility:
  - Set up MLflow tracking for cross-platform execution
  - Handle Azure ML and local MLflow URI registration
  - Configure MLflow tracking URI and experiment
inputs:
  - Experiment names
  - ML clients (for Azure ML)
outputs:
  - Configured MLflow tracking
tags:
  - utility
  - shared
  - mlflow
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Platform-aware MLflow setup utility for cross-platform tracking."""

import os
import sys
from pathlib import Path
from typing import Optional, Any

# Import azureml.mlflow early to register the 'azureml' URI scheme before MLflow initializes
# This must happen before mlflow is imported to ensure the scheme is registered
# We do this at module load time, but the package might be installed later, so we'll re-check at runtime
# Use importlib to ensure the import works even if there are path issues
_AZUREML_MLFLOW_AVAILABLE = False
_AZUREML_MLFLOW_IMPORT_ERROR = None

def _try_import_azureml_mlflow():
    """
    Try to import azureml.mlflow using multiple methods.
    
    Note: There's a namespace collision with our local src/azureml module.
    We always import from site-packages to avoid the collision.
    """
    global _AZUREML_MLFLOW_AVAILABLE, _AZUREML_MLFLOW_IMPORT_ERROR
    
    # Always try to import from site-packages first to avoid namespace collision
    # with our local src/azureml module
    try:
        import importlib.util
        import site
        
        # Check if our local azureml is shadowing
        local_azureml_backup = None
        if 'azureml' in sys.modules:
            azureml_module = sys.modules['azureml']
            if hasattr(azureml_module, '__file__') and azureml_module.__file__:
                if 'src/azureml' in azureml_module.__file__ or azureml_module.__file__.endswith('src/azureml/__init__.py'):
                    # Backup our local azureml
                    local_azureml_backup = azureml_module
                    # Remove it temporarily so we can import the installed one
                    del sys.modules['azureml']
        
        # Find the installed azureml package in site-packages
        installed_azureml_path = None
        for site_dir in site.getsitepackages():
            potential_path = os.path.join(site_dir, 'azureml', '__init__.py')
            if os.path.exists(potential_path):
                installed_azureml_path = potential_path
                break
        
        if installed_azureml_path:
            # Import the installed azureml package first
            spec = importlib.util.spec_from_file_location('azureml', installed_azureml_path)
            if spec and spec.loader:
                installed_azureml = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(installed_azureml)
                sys.modules['azureml'] = installed_azureml
                
                # Now import mlflow from the installed azureml
                import azureml.mlflow  # noqa: F401
                
                # Restore our local azureml if we backed it up, but keep mlflow accessible
                if local_azureml_backup:
                    # Store the mlflow module before restoring
                    mlflow_module = sys.modules['azureml.mlflow']
                    # Restore our local azureml
                    sys.modules['azureml'] = local_azureml_backup
                    # Add mlflow as an attribute to our local azureml so it's accessible
                    setattr(local_azureml_backup, 'mlflow', mlflow_module)
                    # Also ensure azureml.mlflow is still in sys.modules
                    sys.modules['azureml.mlflow'] = mlflow_module
                
                _AZUREML_MLFLOW_AVAILABLE = True
                _AZUREML_MLFLOW_IMPORT_ERROR = None
                return True
    except Exception as e:
        # Restore local azureml if we removed it
        if local_azureml_backup and 'azureml' not in sys.modules:
            sys.modules['azureml'] = local_azureml_backup
        pass  # Fall through to try other methods
    
    # Fallback: try normal import (works if local azureml isn't shadowing)
    try:
        import azureml.mlflow  # noqa: F401
        _AZUREML_MLFLOW_AVAILABLE = True
        _AZUREML_MLFLOW_IMPORT_ERROR = None
        return True
    except ImportError as e1:
        # Try using importlib.import_module
        try:
            import importlib
            importlib.import_module('azureml.mlflow')
            _AZUREML_MLFLOW_AVAILABLE = True
            _AZUREML_MLFLOW_IMPORT_ERROR = None
            return True
        except ImportError as e2:
            _AZUREML_MLFLOW_AVAILABLE = False
            _AZUREML_MLFLOW_IMPORT_ERROR = e1  # Use the first error
            return False

# Try importing at module load time
_try_import_azureml_mlflow()

def _check_azureml_mlflow_available():
    """
    Check if azureml.mlflow is available (re-check at runtime in case it was installed after module load).
    
    This function always re-checks, even if the initial import failed, to handle cases where
    the package was installed after the module was loaded (e.g., in a notebook kernel).
    """
    global _AZUREML_MLFLOW_AVAILABLE, _AZUREML_MLFLOW_IMPORT_ERROR
    # Always re-check to handle cases where package was installed after module load
    # This is important for notebook kernels where packages can be installed mid-session
    try:
        # Force a fresh import by clearing any cached import if needed
        import importlib
        if 'azureml.mlflow' in sys.modules:
            # If it was previously imported and failed, remove it to force re-import
            del sys.modules['azureml.mlflow']
        if 'azureml' in sys.modules and hasattr(sys.modules['azureml'], 'mlflow'):
            # Also clear the mlflow attribute if it exists
            delattr(sys.modules['azureml'], 'mlflow')
        
        import azureml.mlflow  # noqa: F401
        _AZUREML_MLFLOW_AVAILABLE = True
        _AZUREML_MLFLOW_IMPORT_ERROR = None
        return True
    except ImportError as e:
        _AZUREML_MLFLOW_AVAILABLE = False
        _AZUREML_MLFLOW_IMPORT_ERROR = e
        return False

try:
    import mlflow
except ImportError:
    raise ImportError(
        "mlflow is required for experiment tracking. "
        "Install it with: pip install mlflow"
    )

from .platform_detection import detect_platform
from .yaml_utils import load_yaml
from .logging_utils import get_logger

logger = get_logger(__name__)


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
        logger.warning(f"Could not load {env_file_path}: {e}")

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
    import warnings

    # Suppress MLflow Azure ML artifact store "uploading mode" deprecation warning
    # We're intentionally using mlflow.log_artifact() instead of Azure ML SDK
    warnings.filterwarnings(
        "ignore",
        message=".*uploading mode.*deprecated.*",
        category=DeprecationWarning,
        module="mlflow.*"
    )
    warnings.filterwarnings(
        "ignore",
        message=".*deprecated.*uploading mode.*",
        category=UserWarning,
        module="mlflow.*"
    )

    # Check if Azure ML tracking is already configured
    current_tracking_uri = mlflow.get_tracking_uri()
    is_azure_ml_already_set = current_tracking_uri and "azureml" in current_tracking_uri.lower()

    # If Azure ML is already set and we don't have ml_client, preserve it
    if is_azure_ml_already_set and ml_client is None:
        logger.debug(
            f"Preserving existing Azure ML tracking URI: {current_tracking_uri[:50]}...")
        mlflow.set_experiment(experiment_name)
        
        # Set Azure ML artifact upload timeout if not already set
        import os
        if "AZUREML_ARTIFACTS_DEFAULT_TIMEOUT" not in os.environ:
            os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "600"
            logger.debug("Set AZUREML_ARTIFACTS_DEFAULT_TIMEOUT=600 for artifact uploads")
        
        return current_tracking_uri

    # Try Azure ML first if ml_client provided
    if ml_client is not None:
        try:
            tracking_uri = _get_azure_ml_tracking_uri(ml_client)
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            
            # Set Azure ML artifact upload timeout (default 300s, increase to 600s for large artifacts)
            import os
            if "AZUREML_ARTIFACTS_DEFAULT_TIMEOUT" not in os.environ:
                os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "600"
                logger.debug("Set AZUREML_ARTIFACTS_DEFAULT_TIMEOUT=600 for artifact uploads")
            
            logger.info("Using Azure ML workspace tracking")
            logger.debug(f"Tracking URI: {tracking_uri}")
            return tracking_uri
        except Exception as e:
            if not fallback_to_local:
                raise RuntimeError(
                    f"Azure ML tracking failed and fallback disabled: {e}"
                ) from e
            logger.warning(f"Azure ML tracking failed: {e}")
            logger.info("Falling back to local tracking...")

    # Fallback to local tracking
    tracking_uri = _get_local_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"Using local tracking: {tracking_uri}")
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
    # Check if azureml.mlflow is available (lazy check at runtime)
    # This re-checks even if the initial module-level import failed,
    # to handle cases where the package was installed after the module was loaded
    if not _check_azureml_mlflow_available():
        import sys
        import subprocess
        # Check if package is actually installed
        package_installed = False
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', 'azureml-mlflow'],
                capture_output=True,
                text=True,
                timeout=5
            )
            package_installed = result.returncode == 0
        except Exception:
            pass
        
        error_msg = (
            f"azureml.mlflow is required for Azure ML tracking.\n"
            f"Install it with: pip install azureml-mlflow\n\n"
            f"Diagnostics:\n"
            f"  Python executable: {sys.executable}\n"
            f"  Python version: {sys.version}\n"
            f"  Package installed (pip show): {package_installed}\n"
        )
        if _AZUREML_MLFLOW_IMPORT_ERROR:
            error_msg += f"  Import error: {_AZUREML_MLFLOW_IMPORT_ERROR}\n"
        error_msg += (
            f"\nTroubleshooting:\n"
            f"  1. If package is NOT installed: {sys.executable} -m pip install azureml-mlflow\n"
            f"  2. If package IS installed: RESTART THE KERNEL/NOTEBOOK to reload modules\n"
            f"  3. Check that you're using the correct Python environment\n"
            f"  4. Try: import azureml.mlflow (to test if import works directly)"
        )
        raise ImportError(error_msg)
    
    # Import azureml.mlflow to register the 'azureml' URI scheme
    # This must happen before MLflow tries to use the Azure ML URI
    # The lazy check above should have already imported it, but import again to be sure
    import azureml.mlflow  # noqa: F401

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

    # Suppress OpenTelemetry warnings (common in Colab/Kaggle)
    os.environ.setdefault("OTEL_SDK_DISABLED", "false")
    os.environ.setdefault("OTEL_LOG_LEVEL", "ERROR")

    # Try to import Azure ML SDK
    try:
        # Suppress verbose OpenTelemetry logging
        import logging
        logging.getLogger("opentelemetry").setLevel(logging.ERROR)

        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
    except ImportError:
        logger.warning(
            "azure-ai-ml and azure-identity are required for Azure ML tracking. "
            "Install with: pip install azure-ai-ml azure-identity. "
            "Falling back to local tracking."
        )
        return None

    # Get credentials from environment variables
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = azure_ml_config.get("workspace_name", "resume-ner-ws")

    # Load Service Principal credentials (for Colab/Kaggle authentication)
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    tenant_id = os.getenv("AZURE_TENANT_ID")

    logger.warning(
        f"[DEBUG] Initial env check - "
        f"subscription_id: {bool(subscription_id)}, "
        f"resource_group: {bool(resource_group)}, "
        f"client_id: {bool(client_id)}, "
        f"client_secret: {bool(client_secret)}, "
        f"tenant_id: {bool(tenant_id)}"
    )

    # Detect platform early to determine if we need to load from config.env
    platform = detect_platform()
    logger.debug(f"Detected platform: {platform}")

    # Always try to load from config.env if we're in Colab/Kaggle or if credentials are missing
    # In Colab/Kaggle, config.env is the primary source of credentials
    project_root = config_dir.parent
    config_env_path = project_root / "config.env"

    has_env_creds = bool(subscription_id and resource_group)
    has_sp_creds = bool(client_id and client_secret and tenant_id)

    should_load_config_env = (
        platform in ("colab", "kaggle") or
        not has_env_creds or
        not has_sp_creds
    )

    if should_load_config_env:
        logger.info(
            f"Attempting to load credentials from config.env at: {config_env_path}")

    if should_load_config_env and config_env_path.exists():
        logger.info(
            f"Loading credentials from {config_env_path}")
        env_vars = _load_env_file(config_env_path)
        logger.debug(f"Loaded {len(env_vars)} variables from config.env")

        subscription_id = subscription_id or env_vars.get(
            "AZURE_SUBSCRIPTION_ID")
        resource_group = resource_group or env_vars.get(
            "AZURE_RESOURCE_GROUP")
        client_id = client_id or env_vars.get("AZURE_CLIENT_ID")
        client_secret = client_secret or env_vars.get(
            "AZURE_CLIENT_SECRET")
        tenant_id = tenant_id or env_vars.get("AZURE_TENANT_ID")

        # Log what was found (without exposing secrets)
        found_vars = []
        if env_vars.get("AZURE_SUBSCRIPTION_ID"):
            found_vars.append("AZURE_SUBSCRIPTION_ID")
        if env_vars.get("AZURE_RESOURCE_GROUP"):
            found_vars.append("AZURE_RESOURCE_GROUP")
        if env_vars.get("AZURE_CLIENT_ID"):
            found_vars.append("AZURE_CLIENT_ID")
        if env_vars.get("AZURE_CLIENT_SECRET"):
            found_vars.append("AZURE_CLIENT_SECRET")
        if env_vars.get("AZURE_TENANT_ID"):
            found_vars.append("AZURE_TENANT_ID")
        if found_vars:
            logger.debug(f"Found in config.env: {', '.join(found_vars)}")

        if subscription_id and resource_group:
            # Set environment variables for this process
            os.environ["AZURE_SUBSCRIPTION_ID"] = subscription_id
            os.environ["AZURE_RESOURCE_GROUP"] = resource_group
            logger.info(
                "Loaded subscription/resource group from config.env")
            logger.debug(
                f"AZURE_SUBSCRIPTION_ID: {subscription_id[:8]}...")
            logger.debug(f"AZURE_RESOURCE_GROUP: {resource_group}")

        if client_id and client_secret and tenant_id:
            # Set service principal credentials for authentication
            os.environ["AZURE_CLIENT_ID"] = client_id
            os.environ["AZURE_CLIENT_SECRET"] = client_secret
            os.environ["AZURE_TENANT_ID"] = tenant_id
            logger.info(
                "Loaded service principal credentials from config.env")
            logger.debug(f"AZURE_CLIENT_ID: {client_id[:8]}...")
            logger.debug(f"AZURE_TENANT_ID: {tenant_id[:8]}...")
        else:
            missing = []
            if not client_id:
                missing.append("AZURE_CLIENT_ID")
            if not client_secret:
                missing.append("AZURE_CLIENT_SECRET")
            if not tenant_id:
                missing.append("AZURE_TENANT_ID")
            logger.debug(
                f"Service principal credentials not fully configured in config.env. "
                f"Missing: {', '.join(missing)}. "
                "Will try DefaultAzureCredential (may not work in Colab/Kaggle).")
    elif should_load_config_env:
        # Only warn if we actually needed to load from config.env
        logger.warning(
            f"config.env not found at {config_env_path}. "
            f"Looking for: {config_env_path.absolute()}"
        )
        if platform in ("colab", "kaggle"):
            logger.warning(
                f"For {platform.upper()}, you must upload config.env to the project root. "
                f"Expected location: {config_env_path}"
            )

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
        logger.warning(
            "Azure ML enabled but subscription/resource group not found. Falling back to local tracking.")
        logger.info(
            "Set AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP environment variables or add to config.env")
        return None

    # Determine authentication method
    platform = detect_platform()
    logger.warning(f"[DEBUG] Platform detected: {platform}")

    # Re-check environment variables after potential loading from config.env
    # (they may have been set in os.environ but local variables not updated)
    client_id = client_id or os.getenv("AZURE_CLIENT_ID")
    client_secret = client_secret or os.getenv("AZURE_CLIENT_SECRET")
    tenant_id = tenant_id or os.getenv("AZURE_TENANT_ID")

    has_service_principal = bool(client_id and client_secret and tenant_id)
    logger.warning(
        f"[DEBUG] Service Principal check - "
        f"client_id present: {bool(client_id)}, "
        f"client_secret present: {bool(client_secret)}, "
        f"tenant_id present: {bool(tenant_id)}, "
        f"has_service_principal: {has_service_principal}"
    )

    if platform in ("colab", "kaggle") and not has_service_principal:
        logger.warning(
            f"Azure ML authentication requires Service Principal credentials in {platform.upper()} environments. "
            "Add to config.env: AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID. "
            "Falling back to local tracking."
        )
        logger.info(
            f"Looking for config.env at: {config_dir.parent / 'config.env'}"
        )
        logger.info(
            "To use Azure ML from Colab/Kaggle, create a Service Principal and add credentials to config.env:\n"
            "  AZURE_CLIENT_ID=<your-client-id>\n"
            "  AZURE_CLIENT_SECRET=<your-client-secret>\n"
            "  AZURE_TENANT_ID=<your-tenant-id>\n"
            "Make sure config.env is uploaded to Colab in the project root directory."
        )
        return None

    try:
        # Suppress verbose credential chain warnings
        import logging
        azure_logger = logging.getLogger("azure.identity")
        azure_logger.setLevel(logging.ERROR)

        # Use Service Principal if available (required for Colab/Kaggle)
        if has_service_principal:
            from azure.identity import ClientSecretCredential
            logger.info(
                "Using Service Principal authentication (from config.env)")
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret
            )
        else:
            # Use DefaultAzureCredential (works in local/Azure environments)
            logger.info(
                "Using DefaultAzureCredential (trying multiple auth methods)")
            credential = DefaultAzureCredential()

        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        logger.info(
            f"Successfully connected to Azure ML workspace: {workspace_name}")
        return ml_client
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Failed to create Azure ML client: {error_msg}")

        # Provide helpful guidance based on platform
        if platform in ("colab", "kaggle"):
            logger.info(
                "For Colab/Kaggle, you need Service Principal credentials in config.env:\n"
                "  AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID"
            )
        else:
            logger.info(
                "Ensure you're authenticated (Azure CLI: 'az login' or set Service Principal env vars)"
            )
        logger.info("Falling back to local tracking.")
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
        logger.warning("MLflow config not found, using local tracking")
        return setup_mlflow_cross_platform(
            experiment_name=experiment_name,
            ml_client=None,
            fallback_to_local=fallback_to_local,
        )

    mlflow_config = load_yaml(mlflow_config_path)

    # Try to create MLClient if Azure ML is enabled
    ml_client = None
    azure_ml_config = mlflow_config.get("azure_ml", {})
    platform = detect_platform()

    if azure_ml_config.get("enabled", False):
        logger.info("Azure ML enabled in config, attempting to connect...")
        logger.debug(
            f"Workspace: {azure_ml_config.get('workspace_name', 'unknown')}")
        # Try to create ML client - it will load from config.env if env vars not set
        # For Colab/Kaggle, requires Service Principal credentials
        ml_client = create_ml_client_from_config(config_dir, mlflow_config)
        if ml_client is None:
            logger.info("Falling back to local tracking")
    else:
        logger.info("Azure ML disabled in config, using local tracking")

    # Setup MLflow with or without Azure ML
    return setup_mlflow_cross_platform(
        experiment_name=experiment_name,
        ml_client=ml_client,
        fallback_to_local=fallback_to_local,
    )
