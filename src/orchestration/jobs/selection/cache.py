"""Cache management for best model selection with validation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mlflow.tracking import MlflowClient

from orchestration.paths import get_cache_file_path, save_cache_with_dual_strategy
from shared.json_cache import load_json

from shared.logging_utils import get_logger

logger = get_logger(__name__)


def compute_selection_cache_key(
    experiment_name: str,
    selection_config: Dict[str, Any],
    tags_config: Dict[str, Any],
    benchmark_experiment_id: str,
    tracking_uri: Optional[str] = None,
) -> str:
    """
    Compute cache key (fingerprint) for best model selection.
    
    Includes all factors that affect selection result:
    - experiment_name
    - selection_config (weights, metrics, filters)
    - tags_config (tag keys)
    - benchmark_experiment_id
    - tracking_uri (optional, for workspace isolation)
    
    Args:
        experiment_name: Name of the experiment.
        selection_config: Selection configuration dict.
        tags_config: Tags configuration dict.
        benchmark_experiment_id: MLflow experiment ID for benchmark runs.
        tracking_uri: Optional MLflow tracking URI.
    
    Returns:
        16-character hex fingerprint.
    """
    cache_data = {
        "experiment_name": experiment_name,
        "selection_config": selection_config,
        "tags_config": tags_config,
        "benchmark_experiment_id": benchmark_experiment_id,
    }
    if tracking_uri:
        cache_data["tracking_uri"] = tracking_uri
    
    # Sort keys for deterministic JSON
    cache_json = json.dumps(cache_data, sort_keys=True, default=str)
    return hashlib.sha256(cache_json.encode('utf-8')).hexdigest()[:16]


def load_cached_best_model(
    root_dir: Path,
    config_dir: Path,
    experiment_name: str,
    selection_config: Dict[str, Any],
    tags_config: Dict[str, Any],
    benchmark_experiment_id: str,
    tracking_uri: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load cached best model selection if available and valid.
    
    Validates:
    1. Cache key matches current configs
    2. MLflow run still exists and is FINISHED
    3. Cache is not stale (optional: check benchmark run timestamps)
    
    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        experiment_name: Name of the experiment.
        selection_config: Selection configuration dict.
        tags_config: Tags configuration dict.
        benchmark_experiment_id: MLflow experiment ID for benchmark runs.
        tracking_uri: Optional MLflow tracking URI.
    
    Returns:
        Cached data dict if valid, None otherwise.
    """
    # Compute current cache key
    current_cache_key = compute_selection_cache_key(
        experiment_name, selection_config, tags_config,
        benchmark_experiment_id, tracking_uri
    )
    
    # Load latest cache
    cache_file = get_cache_file_path(
        root_dir, config_dir, "best_model_selection", file_type="latest"
    )
    
    print(f"🔍 Checking for cached best model selection...")
    print(f"  Cache file: {cache_file}")
    
    if not cache_file.exists():
        print(f"  ℹ Cache file does not exist (first run or cache was cleared)")
        logger.debug(f"Cache file does not exist: {cache_file}")
        return None
    
    try:
        cache_data = load_json(cache_file, default={})
        print(f"  ✓ Cache file found")
        
        # Validate cache key matches
        cached_key = cache_data.get("cache_key")
        if cached_key != current_cache_key:
            print(f"  ⚠ Cache key mismatch - config changed since cache was created")
            print(f"    Cached key: {cached_key[:8]}... (from {cache_data.get('timestamp', 'unknown')})")
            print(f"    Current key: {current_cache_key[:8]}...")
            print(f"    Reason: Selection config, tags config, experiment, or benchmark experiment changed")
            logger.debug(f"Cache key mismatch: cached={cached_key}, current={current_cache_key}")
            return None
        
        print(f"  ✓ Cache key matches")
        
        # Validate schema version
        if cache_data.get("schema_version") != 1:
            print(f"  ⚠ Cache schema version mismatch (cache format outdated)")
            print(f"    Cached version: {cache_data.get('schema_version')}")
            print(f"    Expected version: 1")
            logger.debug(f"Schema version: {cache_data.get('schema_version')}")
            return None
        
        print(f"  ✓ Schema version valid")
        
        best_model = cache_data.get("best_model")
        if not best_model:
            print(f"  ⚠ Cached data missing best_model field (corrupted cache)")
            return None
        
        # Validate MLflow run still exists and is valid
        run_id = best_model.get("run_id")
        if not run_id:
            print(f"  ⚠ Cached best model missing run_id (corrupted cache)")
            return None
        
        print(f"  ✓ Validating MLflow run: {run_id[:12]}...")
        
        try:
            client = MlflowClient()
            run = client.get_run(run_id)
            
            # Check run status
            if run.info.status != "FINISHED":
                print(f"  ⚠ MLflow run status is '{run.info.status}' (expected 'FINISHED')")
                print(f"    Run may have been deleted, failed, or is still running")
                logger.debug(f"Run status: {run.info.status}")
                return None
            
            print(f"  ✓ MLflow run exists and is FINISHED")
            
            # Optional: Check benchmark experiment hasn't changed
            # (could add benchmark_runs_digest comparison here)
            
            print(f"\n✓ Cache validation successful - reusing cached best model selection")
            print(f"  Cache timestamp: {cache_data.get('timestamp')}")
            print(f"  Run ID: {run_id[:12]}...")
            print(f"  Backbone: {best_model.get('backbone', 'unknown')}")
            logger.info(f"Using cached best model selection: run_id={run_id[:12]}...")
            return cache_data
            
        except Exception as e:
            print(f"  ⚠ Could not validate MLflow run: {e}")
            print(f"    Run may have been deleted or MLflow is unreachable")
            print(f"    Will query MLflow for fresh selection")
            logger.warning(f"Cache validation failed for run {run_id[:12]}...: {e}")
            return None
            
    except Exception as e:
        print(f"  ⚠ Error loading cache file: {e}")
        print(f"    Cache file may be corrupted or unreadable")
        logger.warning(f"Error loading cache: {e}")
        return None


def save_best_model_cache(
    root_dir: Path,
    config_dir: Path,
    best_model: Dict[str, Any],
    experiment_name: str,
    selection_config: Dict[str, Any],
    tags_config: Dict[str, Any],
    benchmark_experiment: Dict[str, str],
    hpo_experiments: Dict[str, Dict[str, str]],
    tracking_uri: Optional[str] = None,
    inputs_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path, Path]:
    """
    Save best model selection to cache using dual strategy.
    
    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        best_model: Best model dict from find_best_model_from_mlflow.
        experiment_name: Name of the experiment.
        selection_config: Selection configuration dict.
        tags_config: Tags configuration dict.
        benchmark_experiment: Dict with 'name' and 'id' of benchmark experiment.
        hpo_experiments: Dict mapping backbone -> experiment info (name, id).
        tracking_uri: Optional MLflow tracking URI.
        inputs_summary: Optional summary of inputs (n_benchmark_runs_considered, n_candidates).
    
    Returns:
        Tuple of (timestamped_file, latest_file, index_file) paths.
    """
    # Compute cache key
    cache_key = compute_selection_cache_key(
        experiment_name, selection_config, tags_config,
        benchmark_experiment["id"], tracking_uri
    )
    
    # Compute config hashes for reference
    selection_config_hash = hashlib.sha256(
        json.dumps(selection_config, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    tags_config_hash = hashlib.sha256(
        json.dumps(tags_config, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    
    # Prepare cache payload
    backbone = best_model.get("backbone", "unknown")
    run_id = best_model.get("run_id", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create identifier for cache file (experiment_name + cache_key prefix)
    identifier = f"{experiment_name}_{cache_key[:8]}"
    
    cache_data = {
        "schema_version": 1,
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "tracking_uri": tracking_uri,
        "benchmark_experiment": benchmark_experiment,
        "hpo_experiments": hpo_experiments,
        "selection_config_hash": selection_config_hash,
        "tags_config_hash": tags_config_hash,
        "cache_key": cache_key,
        "best_model": best_model,
        "inputs_summary": inputs_summary or {},
    }
    
    # Use dual strategy (timestamped + latest + index)
    timestamped_file, latest_file, index_file = save_cache_with_dual_strategy(
        root_dir=root_dir,
        config_dir=config_dir,
        cache_type="best_model_selection",
        data=cache_data,
        backbone=backbone,
        identifier=identifier,
        timestamp=timestamp,
    )
    
    logger.info(f"Saved best model selection cache: {latest_file}")
    return timestamped_file, latest_file, index_file

