from __future__ import annotations

"""
@meta
name: tracking_mlflow_finder
type: utility
domain: tracking
responsibility:
  - Find MLflow runs using priority-based retrieval strategies
  - Support strict and non-strict lookup modes
inputs:
  - Experiment names
  - Naming contexts
  - Run IDs and output directories
outputs:
  - Run lookup reports
tags:
  - utility
  - tracking
  - mlflow
  - finder
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""MLflow run finder with priority-based retrieval and strict mode."""
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

from infrastructure.naming.context import NamingContext
from infrastructure.tracking.mlflow.types import RunLookupReport
from infrastructure.tracking.mlflow.naming import build_mlflow_run_key, build_mlflow_run_key_hash
from infrastructure.tracking.mlflow.index import find_in_mlflow_index
from infrastructure.tracking.mlflow.config_loader import get_run_finder_config
from common.shared.logging_utils import get_logger

logger = get_logger(__name__)

def find_mlflow_run(
    experiment_name: str,
    context: Optional[NamingContext] = None,
    run_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
    run_key_hash: Optional[str] = None,
    strict: Optional[bool] = None,
    root_dir: Optional[Path] = None,
    config_dir: Optional[Path] = None,
) -> RunLookupReport:
    """
    Find MLflow run using priority order with strict validation.
    
    Priority order:
    1. Direct run_id (if provided, verify exists)
    2. run_id from metadata.json (if output_dir provided)
    3. Local index lookup by run_key_hash
    4. Tag search by code.run_key_hash (if context/run_key_hash provided)
    5. Tag search by code.stage + code.model + code.env + code.spec_fp + code.variant (constrained)
    6. Search by attributes.run_name (fallback, only if not strict)
    7. Most recent run with tag constraints (last resort, only if not strict)
    
    Args:
        experiment_name: MLflow experiment name.
        context: Optional NamingContext with full information.
        run_id: Direct run_id (highest priority if provided).
        output_dir: Path to output directory containing metadata.json.
        run_key_hash: Direct run_key_hash (if known without context).
        strict: If None, reads from config. If True, disable weak fallbacks (5-7). If False, allow weak fallbacks.
        root_dir: Project root directory (for index lookup, defaults to output_dir.parent.parent if output_dir provided).
        config_dir: Configuration directory (for config lookup).
    
    Returns:
        RunLookupReport with run_id, method, and metadata.
    
    Raises:
        ValueError: If strict=True and no run found via priority methods (1-4).
        RuntimeError: If MLflow client errors occur.
    """
    # Get config if not provided
    if strict is None:
        if config_dir:
            finder_config = get_run_finder_config(config_dir)
            strict = finder_config.get("strict", False)
        else:
            strict = False
    
    client = MlflowClient()
    
    # Get experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        experiment_id = experiment.experiment_id
    except Exception as e:
        raise RuntimeError(f"Could not get experiment '{experiment_name}': {e}") from e
    
    # Priority 1: Direct run_id
    if run_id:
        try:
            run = client.get_run(run_id)
            if run.info.experiment_id != experiment_id:
                logger.warning(
                    f"Run {run_id[:12]}... exists but in different experiment "
                    f"({run.info.experiment_id} != {experiment_id})"
                )
            else:
                logger.info(f"Found run via direct run_id: {run_id[:12]}...")
                return RunLookupReport(
                    run_id=run_id,
                    method="direct_run_id",
                    experiment_id=experiment_id,
                    run_name=run.info.run_name,
                )
        except Exception as e:
            logger.warning(f"Direct run_id {run_id[:12]}... not found: {e}")
    
    # Priority 2: run_id from metadata.json
    if output_dir:
        metadata_file = output_dir / "metadata.json"
        if metadata_file.exists():
            try:
                import json
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                mlflow_info = metadata.get("mlflow_info", {})
                metadata_run_id = mlflow_info.get("run_id")
                if metadata_run_id:
                    try:
                        run = client.get_run(metadata_run_id)
                        if run.info.experiment_id == experiment_id:
                            logger.info(
                                f"Found run via metadata.json: {metadata_run_id[:12]}..."
                            )
                            return RunLookupReport(
                                run_id=metadata_run_id,
                                method="metadata_json",
                                experiment_id=experiment_id,
                                run_name=run.info.run_name,
                            )
                    except Exception as e:
                        logger.debug(f"Run from metadata.json not found: {e}")
            except Exception as e:
                logger.debug(f"Could not read metadata.json: {e}")
    
    # Priority 3: Local index lookup
    if run_key_hash or context:
        if not run_key_hash and context:
            run_key = build_mlflow_run_key(context)
            run_key_hash = build_mlflow_run_key_hash(run_key)
        
        if run_key_hash:
            # Determine root_dir if not provided
            if not root_dir and output_dir:
                # Assume standard structure: outputs/<process>/<env>/<model>/<spec_exec>/<variant>
                root_dir = output_dir.parent.parent.parent.parent.parent
            
            if root_dir:
                index_result = find_in_mlflow_index(
                    root_dir=root_dir,
                    run_key_hash=run_key_hash,
                    experiment_name=experiment_name,
                )
                if index_result and index_result.get("run_id"):
                    index_run_id = index_result["run_id"]
                    try:
                        run = client.get_run(index_run_id)
                        if run.info.experiment_id == experiment_id:
                            logger.info(
                                f"Found run via local index: {index_run_id[:12]}..."
                            )
                            return RunLookupReport(
                                run_id=index_run_id,
                                method="local_index",
                                experiment_id=experiment_id,
                                run_name=run.info.run_name,
                            )
                    except Exception as e:
                        logger.debug(f"Run from index not found: {e}")
    
    # Priority 4: Tag search by run_key_hash
    if run_key_hash or context:
        if not run_key_hash and context:
            run_key = build_mlflow_run_key(context)
            run_key_hash = build_mlflow_run_key_hash(run_key)
        
        if run_key_hash:
            try:
                runs = client.search_runs(
                    experiment_ids=[experiment_id],
                    filter_string=f"tags.code.run_key_hash = '{run_key_hash}'",
                    max_results=1,
                )
                if runs:
                    run = runs[0]
                    logger.info(
                        f"Found run via tag search (run_key_hash): {run.info.run_id[:12]}..."
                    )
                    return RunLookupReport(
                        run_id=run.info.run_id,
                        method="tag_run_key_hash",
                        experiment_id=experiment_id,
                        run_name=run.info.run_name,
                    )
            except Exception as e:
                logger.debug(f"Tag search by run_key_hash failed: {e}")
    
    # Priority 5-7: Weak fallbacks (only if not strict)
    if not strict:
        # Priority 5: Tag search by stage + model + env + spec_fp + variant
        if context:
            filter_parts = []
            if hasattr(context, "stage") and context.stage:
                filter_parts.append(f"tags.code.stage = '{context.stage}'")
            if hasattr(context, "model") and context.model:
                filter_parts.append(f"tags.code.model = '{context.model}'")
            if hasattr(context, "environment") and context.environment:
                filter_parts.append(f"tags.code.env = '{context.environment}'")
            if hasattr(context, "spec_fp") and context.spec_fp:
                filter_parts.append(f"tags.code.spec_fp = '{context.spec_fp}'")
            if hasattr(context, "variant") and context.variant:
                filter_parts.append(f"tags.code.variant = '{context.variant}'")
            
            if filter_parts:
                try:
                    filter_string = " AND ".join(filter_parts)
                    runs = client.search_runs(
                        experiment_ids=[experiment_id],
                        filter_string=filter_string,
                        max_results=1,
                    )
                    if runs:
                        run = runs[0]
                        logger.info(
                            f"Found run via tag search (context): {run.info.run_id[:12]}..."
                        )
                        return RunLookupReport(
                            run_id=run.info.run_id,
                            method="tag_context",
                            experiment_id=experiment_id,
                            run_name=run.info.run_name,
                        )
                except Exception as e:
                    logger.debug(f"Tag search by context failed: {e}")
        
        # Priority 6: Search by run_name (if context provides it)
        if context and hasattr(context, "run_name") and context.run_name:
            try:
                # MLflow doesn't support direct run_name search, so we search by tag
                runs = client.search_runs(
                    experiment_ids=[experiment_id],
                    filter_string=f"tags.mlflow.runName = '{context.run_name}'",
                    max_results=1,
                )
                if runs:
                    run = runs[0]
                    logger.info(
                        f"Found run via run_name tag: {run.info.run_id[:12]}..."
                    )
                    return RunLookupReport(
                        run_id=run.info.run_id,
                        method="run_name_tag",
                        experiment_id=experiment_id,
                        run_name=run.info.run_name,
                    )
            except Exception as e:
                logger.debug(f"Search by run_name failed: {e}")
        
        # Priority 7: Most recent run (last resort)
        try:
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                max_results=1,
                order_by=["attributes.start_time DESC"],
            )
            if runs:
                run = runs[0]
                logger.warning(
                    f"Using most recent run as fallback: {run.info.run_id[:12]}..."
                )
                return RunLookupReport(
                    run_id=run.info.run_id,
                    method="most_recent_fallback",
                    experiment_id=experiment_id,
                    run_name=run.info.run_name,
                )
        except Exception as e:
            logger.debug(f"Most recent run search failed: {e}")
    
    # No run found
    if strict:
        raise ValueError(
            f"Could not find run in experiment '{experiment_name}' "
            f"using strict mode (tried priorities 1-4)"
        )
    else:
        raise ValueError(
            f"Could not find run in experiment '{experiment_name}' "
            f"(tried all priority methods)"
        )

