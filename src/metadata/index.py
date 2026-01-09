"""Index file management for fast lookup by spec_fp, env, model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from shared.json_cache import load_json, save_json


def get_index_file_path(
    root_dir: Path,
    process_type: str,
    base_outputs: str = "outputs"
) -> Path:
    """
    Get path to index file for a process type.
    
    Args:
        root_dir: Project root directory.
        process_type: Process type (final_training, conversion, etc.).
        base_outputs: Base outputs directory name.
    
    Returns:
        Path to index file.
    """
    cache_dir = root_dir / base_outputs / "cache"
    return cache_dir / f"{process_type}_index.json"


def update_index(
    root_dir: Path,
    process_type: str,
    context: Any,  # NamingContext
    metadata: Dict[str, Any],
    max_entries: int = 100
) -> Path:
    """
    Update index file with new entry.
    
    Args:
        root_dir: Project root directory.
        process_type: Process type (final_training, conversion, etc.).
        context: NamingContext with fingerprint information.
        metadata: Metadata dictionary to index.
        max_entries: Maximum number of entries to keep in index.
    
    Returns:
        Path to index file.
    """
    index_file = get_index_file_path(root_dir, process_type)
    
    # Load existing index or create new
    index_data = load_json(index_file, default={
        "by_spec_fp": {},
        "by_env": {},
        "by_model": {},
        "entries": []
    })
    
    # Build entry
    entry = {
        "spec_fp": getattr(context, "spec_fp", None),
        "exec_fp": getattr(context, "exec_fp", None),
        "conv_fp": getattr(context, "conv_fp", None),
        "environment": context.environment,
        "model": context.model,
        "variant": getattr(context, "variant", 1),
        "trial_id": getattr(context, "trial_id", None),
        "parent_training_id": getattr(context, "parent_training_id", None),
        "path": str(metadata.get("_path", "")),
        "status": metadata.get("status", {}),
        "created_at": metadata.get("created_at", datetime.now().isoformat()),
        "last_updated": metadata.get("last_updated", datetime.now().isoformat()),
    }
    
    # Add to entries list
    index_data.setdefault("entries", []).append(entry)
    
    # Limit entries (keep most recent)
    if len(index_data["entries"]) > max_entries:
        index_data["entries"] = index_data["entries"][-max_entries:]
    
    # Update by_spec_fp index
    if entry["spec_fp"]:
        spec_fp = entry["spec_fp"]
        if spec_fp not in index_data["by_spec_fp"]:
            index_data["by_spec_fp"][spec_fp] = []
        index_data["by_spec_fp"][spec_fp].append(entry)
        # Limit per spec_fp
        if len(index_data["by_spec_fp"][spec_fp]) > 20:
            index_data["by_spec_fp"][spec_fp] = index_data["by_spec_fp"][spec_fp][-20:]
    
    # Update by_env index
    env = entry["environment"]
    if env not in index_data["by_env"]:
        index_data["by_env"][env] = []
    index_data["by_env"][env].append(entry)
    # Limit per env
    if len(index_data["by_env"][env]) > 50:
        index_data["by_env"][env] = index_data["by_env"][env][-50:]
    
    # Update by_model index
    model = entry["model"]
    if model not in index_data["by_model"]:
        index_data["by_model"][model] = []
    index_data["by_model"][model].append(entry)
    # Limit per model
    if len(index_data["by_model"][model]) > 50:
        index_data["by_model"][model] = index_data["by_model"][model][-50:]
    
    # Ensure directory exists
    index_file.parent.mkdir(parents=True, exist_ok=True)
    
    save_json(index_file, index_data)
    return index_file


def find_by_spec_fp(
    root_dir: Path,
    spec_fp: str,
    process_type: str = "final_training"
) -> List[Dict[str, Any]]:
    """
    Find all runs with matching spec_fp.
    
    Args:
        root_dir: Project root directory.
        spec_fp: Specification fingerprint to search for.
        process_type: Process type to search.
    
    Returns:
        List of index entries matching spec_fp.
    """
    index_file = get_index_file_path(root_dir, process_type)
    index_data = load_json(index_file, default={"by_spec_fp": {}})
    
    return index_data.get("by_spec_fp", {}).get(spec_fp, [])


def find_by_env(
    root_dir: Path,
    environment: str,
    process_type: str = "final_training"
) -> List[Dict[str, Any]]:
    """
    Find all runs in environment.
    
    Args:
        root_dir: Project root directory.
        environment: Environment to search (local, colab, kaggle, azure).
        process_type: Process type to search.
    
    Returns:
        List of index entries in environment.
    """
    index_file = get_index_file_path(root_dir, process_type)
    index_data = load_json(index_file, default={"by_env": {}})
    
    return index_data.get("by_env", {}).get(environment, [])


def find_by_model(
    root_dir: Path,
    model: str,
    process_type: str = "final_training"
) -> List[Dict[str, Any]]:
    """
    Find all runs for model.
    
    Args:
        root_dir: Project root directory.
        model: Model name to search for.
        process_type: Process type to search.
    
    Returns:
        List of index entries for model.
    """
    index_file = get_index_file_path(root_dir, process_type)
    index_data = load_json(index_file, default={"by_model": {}})
    
    return index_data.get("by_model", {}).get(model, [])


def find_by_spec_and_env(
    root_dir: Path,
    spec_fp: str,
    environment: str,
    process_type: str = "final_training"
) -> List[Dict[str, Any]]:
    """
    Find runs matching both spec_fp and environment.
    
    Args:
        root_dir: Project root directory.
        spec_fp: Specification fingerprint.
        environment: Environment to search.
        process_type: Process type to search.
    
    Returns:
        List of index entries matching both criteria.
    """
    by_spec = find_by_spec_fp(root_dir, spec_fp, process_type)
    return [entry for entry in by_spec if entry.get("environment") == environment]


def get_latest_entry(
    root_dir: Path,
    process_type: str = "final_training",
    spec_fp: Optional[str] = None,
    environment: Optional[str] = None,
    model: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get the most recent entry matching criteria.
    
    Args:
        root_dir: Project root directory.
        process_type: Process type to search.
        spec_fp: Optional spec_fp filter.
        environment: Optional environment filter.
        model: Optional model filter.
    
    Returns:
        Most recent entry matching criteria, or None.
    """
    index_file = get_index_file_path(root_dir, process_type)
    index_data = load_json(index_file, default={"entries": []})
    
    entries = index_data.get("entries", [])
    
    # Apply filters
    if spec_fp:
        entries = [e for e in entries if e.get("spec_fp") == spec_fp]
    if environment:
        entries = [e for e in entries if e.get("environment") == environment]
    if model:
        entries = [e for e in entries if e.get("model") == model]
    
    if not entries:
        return None
    
    # Sort by last_updated (most recent first)
    entries.sort(key=lambda e: e.get("last_updated", ""), reverse=True)
    return entries[0]


