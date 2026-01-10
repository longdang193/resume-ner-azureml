from __future__ import annotations

"""
@meta
name: tracking_mlflow_compatibility
type: utility
domain: tracking
responsibility:
  - Provide Azure ML compatibility patches for MLflow
  - Fix compatibility issues between MLflow and azureml-mlflow
inputs:
  - Module imports (patch applied automatically)
outputs:
  - Patched MLflow artifact builders
tags:
  - utility
  - tracking
  - mlflow
  - compatibility
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Azure ML compatibility patches for MLflow.

This module provides a monkey-patch for azureml_artifacts_builder to handle
tracking_uri parameter gracefully. This fixes a compatibility issue between
MLflow 3.5.0 and azureml-mlflow 1.61.0.

The patch is automatically applied when this module is imported.
"""
import functools
import inspect
from common.shared.logging_utils import get_logger

logger = get_logger(__name__)

# Track if patch has been applied to prevent double-patching
_patch_applied = False

def apply_azureml_artifact_patch() -> None:
    """
    Apply monkey-patch to azureml_artifacts_builder to handle tracking_uri parameter.

    This patch fixes a compatibility issue where MLflow passes tracking_uri,
    but some versions of azureml-mlflow don't accept it despite the signature.

    The patch is idempotent - calling it multiple times is safe.
    """
    global _patch_applied

    # Check if already patched by looking for __wrapped__ attribute
    if _patch_applied:
        logger.debug("Azure ML artifact patch already applied, skipping")
        return

    try:
        import azureml.mlflow  # noqa: F401
    except ImportError:
        logger.debug("azureml.mlflow not available, skipping patch")
        return

    try:
        import mlflow.store.artifact.artifact_repository_registry as arr

        original_builder = arr._artifact_repository_registry._registry.get('azureml')
        if not original_builder:
            logger.debug("Azure ML artifact builder not found in registry")
            return

        # Check if already wrapped (prevent double-patching)
        if hasattr(original_builder, '__wrapped__'):
            logger.debug("Azure ML artifact builder already patched")
            _patch_applied = True
            return

        # Get the original function's signature
        sig = inspect.signature(original_builder)
        param_names = list(sig.parameters.keys())

        @functools.wraps(original_builder)
        def patched_azureml_builder(artifact_uri=None, tracking_uri=None, registry_uri=None):
            """Patched builder that handles tracking_uri parameter gracefully."""
            # Try calling with all parameters first
            try:
                return original_builder(
                    artifact_uri=artifact_uri,
                    tracking_uri=tracking_uri,
                    registry_uri=registry_uri
                )
            except TypeError as e:
                # If tracking_uri is not accepted, try without it
                if 'tracking_uri' in str(e) and 'unexpected keyword argument' in str(e):
                    # Call without tracking_uri (some versions don't accept it despite signature)
                    try:
                        return original_builder(
                            artifact_uri=artifact_uri,
                            registry_uri=registry_uri
                        )
                    except TypeError:
                        # Last resort: call with just artifact_uri
                        return original_builder(artifact_uri=artifact_uri)
                # Re-raise if it's a different error
                raise

        # Register the patched builder
        arr._artifact_repository_registry.register('azureml', patched_azureml_builder)
        _patch_applied = True
        logger.info("Applied Azure ML artifact compatibility patch")
    except Exception as e:
        logger.warning(f"Failed to apply Azure ML artifact patch: {e}", exc_info=True)

# Auto-apply patch on module import
apply_azureml_artifact_patch()

