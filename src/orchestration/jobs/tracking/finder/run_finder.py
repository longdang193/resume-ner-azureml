"""MLflow run finder with priority-based retrieval and strict mode."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

from infrastructure.naming.context import NamingContext
from orchestration.jobs.tracking.mlflow_types import RunLookupReport
from orchestration.jobs.tracking.mlflow_naming import build_mlflow_run_key, build_mlflow_run_key_hash
from orchestration.jobs.tracking.mlflow_index import find_in_mlflow_index
from orchestration.jobs.tracking.config.loader import get_run_finder_config
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
        config_dir: Optional config directory for reading strict_mode_default.
    
    Returns:
        RunLookupReport with found status, run_id if found, strategy used, and any errors.
    """
    # Read strict mode default from config if not provided
    if strict is None:
        run_finder_config = get_run_finder_config(config_dir)
        strict = run_finder_config.get("strict_mode_default", True)
    report = RunLookupReport(found=False)
    
    if root_dir is None and output_dir:
        # Try to infer root_dir from output_dir
        # output_dir is typically: outputs/{process_type}/{env}/{model}/...
        # root_dir would be output_dir.parent.parent.parent.parent
        # But safer to use output_dir.parent.parent (outputs level)
        root_dir = output_dir.parent.parent if output_dir.parent.name == "outputs" else output_dir.parent.parent.parent
    
    if root_dir is None:
        root_dir = Path.cwd()
    
    client = MlflowClient()
    tracking_uri = mlflow.get_tracking_uri()
    
    # Strategy 1: Direct run_id
    if run_id:
        report.strategies_attempted.append("direct_run_id")
        try:
            run = client.get_run(run_id)
            report.found = True
            report.run_id = run_id
            report.strategy_used = "direct_run_id"
            logger.debug(f"Found run using direct run_id: {run_id[:12]}...")
            return report
        except Exception as e:
            report.error = f"Direct run_id lookup failed: {e}"
            logger.debug(f"Direct run_id lookup failed: {e}")
    
    # Strategy 2: metadata.json
    if output_dir:
        report.strategies_attempted.append("metadata_json")
        try:
            metadata_file = output_dir / "metadata.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    mlflow_info = metadata.get("mlflow", {})
                    stored_run_id = mlflow_info.get("run_id")
                    stored_tracking_uri = mlflow_info.get("tracking_uri")
                    
                    if stored_run_id:
                        # Verify tracking URI alignment
                        if stored_tracking_uri and stored_tracking_uri != tracking_uri:
                            logger.warning(
                                f"Tracking URI mismatch in metadata: "
                                f"stored={stored_tracking_uri[:50]}..., "
                                f"current={tracking_uri[:50]}..."
                            )
                        else:
                            # Verify run exists
                            try:
                                run = client.get_run(stored_run_id)
                                report.found = True
                                report.run_id = stored_run_id
                                report.strategy_used = "metadata_json"
                                logger.debug(f"Found run using metadata.json: {stored_run_id[:12]}...")
                                return report
                            except Exception as e:
                                report.error = f"Run from metadata not found in MLflow: {e}"
                                logger.debug(f"Run from metadata not found: {e}")
        except Exception as e:
            report.error = f"Metadata.json read failed: {e}"
            logger.debug(f"Metadata.json read failed: {e}")
    
    # Strategy 3: Local index
    if run_key_hash or context:
        report.strategies_attempted.append("local_index")
        try:
            if not run_key_hash and context:
                run_key = build_mlflow_run_key(context)
                run_key_hash = build_mlflow_run_key_hash(run_key)
            
            if run_key_hash:
                index_result = find_in_mlflow_index(
                    root_dir=root_dir,
                    run_key_hash=run_key_hash,
                    tracking_uri=tracking_uri,
                )
                
                if index_result:
                    stored_run_id = index_result.get("run_id")
                    if stored_run_id:
                        # Verify run exists
                        try:
                            run = client.get_run(stored_run_id)
                            report.found = True
                            report.run_id = stored_run_id
                            report.strategy_used = "local_index"
                            logger.debug(f"Found run using local index: {stored_run_id[:12]}...")
                            return report
                        except Exception as e:
                            report.error = f"Run from index not found in MLflow: {e}"
                            logger.debug(f"Run from index not found: {e}")
        except Exception as e:
            report.error = f"Local index lookup failed: {e}"
            logger.debug(f"Local index lookup failed: {e}")
    
    # Strategy 4: Tag search by run_key_hash
    if run_key_hash or context:
        report.strategies_attempted.append("tag_search_run_key_hash")
        try:
            if not run_key_hash and context:
                run_key = build_mlflow_run_key(context)
                run_key_hash = build_mlflow_run_key_hash(run_key)
            
            if run_key_hash:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    runs = client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string=f"tags.code.run_key_hash = '{run_key_hash}' AND (tags.code.interrupted != 'true' OR tags.code.interrupted IS NULL)",
                        max_results=1,
                        order_by=["start_time DESC"],
                    )
                    if runs:
                        found_run_id = runs[0].info.run_id
                        report.found = True
                        report.run_id = found_run_id
                        report.strategy_used = "tag_search_run_key_hash"
                        logger.debug(f"Found run using tag search (run_key_hash): {found_run_id[:12]}...")
                        return report
        except Exception as e:
            report.error = f"Tag search (run_key_hash) failed: {e}"
            logger.debug(f"Tag search (run_key_hash) failed: {e}")
    
    # Strategy 5: Constrained tag search (stage + model + env + spec_fp + variant)
    if context and not strict:
        report.strategies_attempted.append("tag_search_constrained")
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                filters = [
                    f"tags.code.stage = '{context.process_type}'",
                    f"tags.code.model = '{context.model}'",
                    f"tags.code.env = '{context.environment}'",
                ]
                if context.spec_fp:
                    filters.append(f"tags.code.spec_fp = '{context.spec_fp}'")
                if context.variant:
                    filters.append(f"tags.code.variant = '{context.variant}'")
                
                filter_string = " AND ".join(filters) + " AND (tags.code.interrupted != 'true' OR tags.code.interrupted IS NULL)"
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=filter_string,
                    max_results=1,
                    order_by=["start_time DESC"],
                )
                if runs:
                    found_run_id = runs[0].info.run_id
                    report.found = True
                    report.run_id = found_run_id
                    report.strategy_used = "tag_search_constrained"
                    logger.debug(f"Found run using constrained tag search: {found_run_id[:12]}...")
                    return report
        except Exception as e:
            report.error = f"Constrained tag search failed: {e}"
            logger.debug(f"Constrained tag search failed: {e}")
    
    # Strategy 6: Search by attributes.run_name (only if not strict)
    if not strict and context:
        report.strategies_attempted.append("attributes_run_name")
        try:
            from orchestration.jobs.tracking.mlflow_naming import build_mlflow_run_name
            run_name = build_mlflow_run_name(context)
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"attributes.run_name = '{run_name}' AND (tags.code.interrupted != 'true' OR tags.code.interrupted IS NULL)",
                    max_results=1,
                    order_by=["start_time DESC"],
                )
                if runs:
                    found_run_id = runs[0].info.run_id
                    report.found = True
                    report.run_id = found_run_id
                    report.strategy_used = "attributes_run_name"
                    logger.debug(f"Found run using attributes.run_name: {found_run_id[:12]}...")
                    return report
        except Exception as e:
            report.error = f"Attributes.run_name search failed: {e}"
            logger.debug(f"Attributes.run_name search failed: {e}")
    
    # Strategy 7: Most recent with tag constraints (only if not strict)
    if not strict and context:
        report.strategies_attempted.append("most_recent_constrained")
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                # Constrain by stage and model at minimum
                filter_string = (
                    f"tags.code.stage = '{context.process_type}' AND "
                    f"tags.code.model = '{context.model}' AND "
                    f"(tags.code.interrupted != 'true' OR tags.code.interrupted IS NULL)"
                )
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=filter_string,
                    max_results=1,
                    order_by=["start_time DESC"],
                )
                if runs:
                    found_run_id = runs[0].info.run_id
                    report.found = True
                    report.run_id = found_run_id
                    report.strategy_used = "most_recent_constrained"
                    logger.warning(
                        f"Found run using most recent constrained search (risky): {found_run_id[:12]}..."
                    )
                    return report
        except Exception as e:
            report.error = f"Most recent constrained search failed: {e}"
            logger.debug(f"Most recent constrained search failed: {e}")
    
    # All strategies exhausted
    if not report.found:
        if strict:
            report.error = (
                f"Run not found with strict mode enabled. "
                f"Attempted strategies: {', '.join(report.strategies_attempted)}. "
                f"Enable strict=False to allow weak fallbacks (risky)."
            )
        else:
            report.error = (
                f"Run not found after trying all strategies: "
                f"{', '.join(report.strategies_attempted)}"
            )
        logger.warning(report.error)
    
    return report


def find_run_by_trial_id(
    trial_id: str,
    experiment_name: Optional[str] = None,
    config_dir: Optional[Path] = None,
) -> RunLookupReport:
    """
    Find MLflow run by trial_id tag.
    
    This is a convenience function for finding HPO trial runs by their trial_id.
    
    Args:
        trial_id: Trial ID to search for (e.g., "trial_1_20251231_161745").
        experiment_name: Optional experiment name (if None, searches all experiments).
        config_dir: Optional config directory.
    
    Returns:
        RunLookupReport with found status, run_id if found, strategy used, and any errors.
    """
    report = RunLookupReport(found=False)
    report.strategies_attempted.append("trial_id_tag_search")
    
    try:
        client = MlflowClient()
        
        # Build filter string
        filter_string = f"tags.code.trial_id = '{trial_id}' AND (tags.code.interrupted != 'true' OR tags.code.interrupted IS NULL)"
        
        if experiment_name:
            # Search in specific experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                report.error = f"Experiment '{experiment_name}' not found"
                logger.warning(f"[Find Run by Trial ID] Experiment '{experiment_name}' not found")
                return report
            
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                max_results=1,
                order_by=["start_time DESC"],
            )
        else:
            # Search all experiments (slower but more flexible)
            experiments = client.search_experiments()
            runs = []
            for exp in experiments:
                try:
                    exp_runs = client.search_runs(
                        experiment_ids=[exp.experiment_id],
                        filter_string=filter_string,
                        max_results=1,
                        order_by=["start_time DESC"],
                    )
                    if exp_runs:
                        runs.extend(exp_runs)
                        break  # Found it, stop searching
                except Exception as e:
                    logger.debug(f"Error searching experiment {exp.name}: {e}")
                    continue
        
        if runs:
            found_run_id = runs[0].info.run_id
            report.found = True
            report.run_id = found_run_id
            report.strategy_used = "trial_id_tag_search"
            logger.info(
                f"[Find Run by Trial ID] Found run {found_run_id[:12]}... for trial_id={trial_id}"
            )
            return report
        else:
            report.error = f"No run found with trial_id='{trial_id}'"
            logger.debug(f"[Find Run by Trial ID] No run found with trial_id='{trial_id}'")
            return report
            
    except Exception as e:
        report.error = f"Trial ID search failed: {e}"
        logger.warning(f"[Find Run by Trial ID] Search failed: {e}", exc_info=True)
        return report

