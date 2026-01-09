from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.core.exceptions import ResourceNotFoundError


def resolve_dataset_path(data_config: Dict[str, Any]) -> Path:
    """
    Resolve the local dataset folder from the data configuration.

    Handles seed-based dataset structures (e.g., dataset_tiny/seed0/) when:
    - The config contains a ``seed`` field
    - The resolved path contains ``dataset_tiny``

    Args:
        data_config: Data configuration dictionary containing a ``local_path`` field
            and optionally a ``seed`` field.

    Returns:
        Path to the dataset directory. Defaults to ``../dataset`` if ``local_path``
        is not specified in the config. If ``seed`` is specified and the path
        contains ``dataset_tiny``, appends ``seed{N}`` subdirectory.

    Raises:
        ValueError: If ``local_path`` is specified but is not a valid string.
    """
    local_path = data_config.get("local_path", "../dataset")
    if not isinstance(local_path, str):
        raise ValueError(
            f"data_config['local_path'] must be a string, got {type(local_path).__name__}"
        )

    dataset_path = Path(local_path)

    # Check if seed-based dataset structure (for dataset_tiny with seed subdirectories)
    seed = data_config.get("seed")
    if seed is not None and "dataset_tiny" in str(dataset_path):
        dataset_path = dataset_path / f"seed{seed}"

    return dataset_path


def register_data_asset(
    ml_client: MLClient,
    name: str,
    version: str,
    uri: str,
    description: str,
) -> Data:
    """
    Register or fetch an Azure ML ``Data`` asset of type ``uri_folder``.

    This helper is idempotent: if the asset already exists for the given
    ``name`` and ``version`` it is returned as-is; otherwise a new asset
    pointing to ``uri`` is created.

    Args:
        ml_client: Azure ML client used for asset operations.
        name: Logical name of the data asset.
        version: Version string for the data asset.
        uri: Backing URI (local path or datastore URI) for the folder.
        description: Human-readable description of the data asset.

    Returns:
        The resolved or newly-created ``Data`` asset.
    """
    try:
        return ml_client.data.get(name=name, version=version)
    except Exception:
        # Normalize local paths to absolute paths for consistency
        # Skip normalization for datastore URIs (azureml://) and HTTP(S) URLs
        normalized_uri = uri
        if not uri.startswith(("azureml://", "http://", "https://")):
            try:
                normalized_uri = str(Path(uri).resolve())
            except (ValueError, OSError):
                # Path can't be resolved, use as-is
                normalized_uri = uri

        data_asset = Data(
            name=name,
            version=version,
            description=description,
            path=normalized_uri,
            type=AssetTypes.URI_FOLDER,
        )
        return ml_client.data.create_or_update(data_asset)


def ensure_data_asset_uploaded(
    ml_client: MLClient,
    data_asset: Data,
    local_path: Path,
    description: str,
) -> Data:
    """
    Ensure the data asset is registered with the correct local path.

    Azure ML data assets are immutable with respect to their underlying
    URI. This helper:

    * First checks if the asset already exists in Azure ML.
    * If it exists and points to a datastore path (azureml://), returns it as-is.
    * If it exists and points to a local path, compares paths and returns it.
    * If it doesn't exist yet, attempts to create it with the local path.
    * Never attempts to update an existing asset's path (which would fail).

    Args:
        ml_client: Azure ML client used for asset operations.
        data_asset: Previously resolved data asset (may be newly created or existing).
        local_path: Local folder containing the dataset.
        description: Description to associate with the asset.

    Returns:
        The resulting ``Data`` asset (either newly created or existing).
    """
    local_path_resolved = local_path.resolve()
    local_path_str = str(local_path_resolved)

    # Always check if the asset already exists in Azure ML first
    try:
        existing_asset = ml_client.data.get(
            name=data_asset.name,
            version=data_asset.version
        )
        # Asset exists - check if it's already uploaded to datastore
        existing_path = existing_asset.path.rstrip("/")

        if existing_path.startswith("azureml://"):
            # Already uploaded to datastore, no action needed
            return existing_asset

        # Asset exists with a local path - check if it matches our expected path
        # Normalize both paths for comparison
        try:
            existing_resolved = Path(existing_path).resolve()
            path_matches = existing_resolved == local_path_resolved
        except (ValueError, OSError):
            # Path can't be resolved, compare as normalized strings
            existing_normalized = existing_path.replace("\\", "/").rstrip("/")
            local_normalized = local_path_str.replace("\\", "/").rstrip("/")
            path_matches = (
                existing_normalized == local_normalized or
                existing_path == local_path_str or
                existing_path == str(local_path) or
                existing_path == str(local_path_resolved)
            )

        # Asset exists - return it regardless of path match
        # Azure ML will resolve the path when jobs run
        return existing_asset

    except ResourceNotFoundError:
        # Asset doesn't exist yet - create it with the local path
        # Note: This registers the asset but doesn't upload files.
        # Upload happens automatically when Azure ML jobs access the data.
        try:
            new_asset = ml_client.data.create_or_update(
                Data(
                    name=data_asset.name,
                    version=data_asset.version,
                    description=description,
                    path=local_path_str,
                    type=AssetTypes.URI_FOLDER,
                )
            )
            return new_asset
        except Exception as e:
            # Creation might fail if asset was created between check and create
            # Try to fetch it one more time
            try:
                return ml_client.data.get(
                    name=data_asset.name,
                    version=data_asset.version
                )
            except Exception:
                # If we still can't get it, return the data_asset object we were given
                # This should not happen in normal operation
                return data_asset


def build_data_asset_reference(ml_client: MLClient, data_asset: Data) -> Dict[str, Any]:
    """
    Construct canonical references for a data asset.

    This provides both:

    * ``asset_uri``: ``azureml:<name>:<version>`` suitable for job inputs.
    * ``datastore_path``: fully-qualified datastore URI for direct path access.

    Args:
        ml_client: Azure ML client used to resolve the default datastore.
        data_asset: Data asset whose backing path should be normalised.

    Returns:
        Dictionary containing ``asset_uri`` and ``datastore_path`` keys.
    """
    default_datastore = ml_client.datastores.get_default()
    if "/paths/" in data_asset.path:
        relative = data_asset.path.split("/paths/", 1)[1].rstrip("/")
        datastore_path = f"azureml://datastores/{default_datastore.name}/paths/{relative}"
    else:
        datastore_path = data_asset.path.rstrip("/")

    return {
        "asset_uri": f"azureml:{data_asset.name}:{data_asset.version}",
        "datastore_path": datastore_path,
    }
