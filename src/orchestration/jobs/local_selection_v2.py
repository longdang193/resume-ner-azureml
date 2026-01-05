"""Improved best configuration selection from local Optuna HPO studies.

This module provides config-aware study folder discovery and CV-based trial selection.
It replaces the previous approach with:
- Config pattern matching for study folders
- CV-only trial selection (refit used only as artifact)
- Deterministic fold ordering
- Safe version parsing
- Fast path via .active_study.json marker files

NOTE: The functions in `local_selection.py` now use these improved functions internally
when hpo_config is provided, maintaining backward compatibility while providing the
improved behavior. This module can be used directly for new code that needs the
improved functionality.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def parse_version_from_name(name: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse version from folder name like 'hpo_distilbert_smoke_test_3.69' or 'hpo_distilbert_smoke_test_3.69_1'.

    Returns:
        Tuple of (major, minor, suffix) or None if no version found.
        suffix is -1 if not present.
    """
    # Matches ..._3.69 or ..._3.69_1
    m = re.search(r'_(\d+)\.(\d+)(?:_(\d+))?$', name)
    if not m:
        return None
    major = int(m.group(1))
    minor = int(m.group(2))
    suffix = int(m.group(3)) if m.group(3) else -1
    return (major, minor, suffix)


def fold_index(name: str) -> int:
    """Extract numeric fold index from folder name (e.g., 'fold0' -> 0, 'fold10' -> 10)."""
    m = re.search(r'fold(\d+)', name)
    return int(m.group(1)) if m else 10**9


def find_study_folder_by_config(
    backbone_dir: Path,
    hpo_config: Dict[str, Any],
    backbone: str
) -> Optional[Path]:
    """
    Find study folder matching the study name pattern from HPO config.

    Args:
        backbone_dir: Backbone directory containing study folders
        hpo_config: HPO configuration dictionary
        backbone: Model backbone name

    Returns:
        Study folder with highest version matching the config pattern, or None if not found
    """
    if not backbone_dir.exists():
        return None

    # Check for .active_study.json first (fast path)
    active_study_file = backbone_dir / ".active_study.json"
    if active_study_file.exists():
        try:
            with open(active_study_file, "r") as f:
                active_info = json.load(f)
            study_path = Path(active_info.get("path", ""))

            # Sanity check: ensure path is inside backbone_dir (avoid stale pointers)
            try:
                study_path.relative_to(backbone_dir)
            except ValueError:
                study_path = None

            if study_path and study_path.exists() and study_path.is_dir():
                # Verify it still has trials
                has_trials = any(
                    item.is_dir() and item.name.startswith("trial_")
                    for item in study_path.iterdir()
                )
                if has_trials:
                    return study_path
        except Exception:
            # Fallback to scanning if .active_study.json is invalid
            pass

    # Extract study name base from config
    checkpoint_config = hpo_config.get("checkpoint", {})
    study_name_template = checkpoint_config.get(
        "study_name") or hpo_config.get("study_name")

    if not study_name_template:
        # No config pattern, fallback to scanning all hpo_* folders
        study_name_base = f"hpo_{backbone}"
    else:
        # Replace {backbone} placeholder
        study_name_base = study_name_template.replace("{backbone}", backbone)
        # Extract base pattern (everything before the last version number)
        # Use end-anchored pattern to avoid stripping too aggressively
        base_match = re.match(
            r'^(.*?)(?:_\d+\.\d+(?:_\d+)?)$', study_name_base)
        if base_match:
            study_name_base = base_match.group(1)

    # Find all study folders matching the base pattern
    matching_folders = []
    for item in backbone_dir.iterdir():
        if not item.is_dir() or item.name.startswith("trial_"):
            continue

        # Reduce false-positive prefix matches: exact match or starts with base + "_"
        if item.name == study_name_base or item.name.startswith(study_name_base + "_"):
            # Verify it contains trial directories (not just a regular folder)
            has_trials = any(
                subitem.is_dir() and subitem.name.startswith("trial_")
                for subitem in item.iterdir()
            )
            if has_trials:
                matching_folders.append(item)

    if not matching_folders:
        return None

    if len(matching_folders) == 1:
        return matching_folders[0]

    # Multiple matches: extract version numbers and pick the highest
    def get_version_key(folder_path: Path) -> Tuple[int, int, int, float]:
        """Get sort key: (major, minor, suffix, mtime) for deterministic sorting."""
        version = parse_version_from_name(folder_path.name)
        if version:
            # Use version tuple, with mtime as tiebreaker
            return (*version, folder_path.stat().st_mtime)
        else:
            # No version found, use mtime only (most recent)
            return (-1, -1, -1, folder_path.stat().st_mtime)

    # Sort by version (descending), fallback to modification time
    matching_folders.sort(key=get_version_key, reverse=True)
    return matching_folders[0]


def load_best_trial_from_study_folder(
    study_folder: Path,
    objective_metric: str = "macro-f1",
) -> Optional[Dict[str, Any]]:
    """
    Load best trial from a specific study folder.

    IMPORTANT: Selects best trial based on CV metrics ONLY (not refit).
    Refit is only used as the preferred checkpoint artifact after selection.

    Args:
        study_folder: Path to study folder (e.g., outputs/hpo/colab/distilbert/hpo_distilbert_smoke_test_3.69)
        objective_metric: Name of the objective metric

    Returns:
        Dictionary with best trial info, or None if not found
    """
    if not study_folder.exists():
        return None

    best_avg_metric = None
    best_trial_dir = None
    best_trial_name = None
    best_fold_metrics = None  # Store fold metrics for checkpoint selection

    # Step 1: Find best trial based on CV metrics ONLY
    for trial_dir in study_folder.iterdir():
        if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
            continue

        # Check for CV fold metrics
        cv_dir = trial_dir / "cv"
        if cv_dir.exists():
            # Collect fold metrics deterministically (sorted by numeric fold index)
            fold_pairs = []
            for fold_dir in sorted(cv_dir.iterdir(), key=lambda p: fold_index(p.name)):
                if not fold_dir.is_dir() or not fold_dir.name.startswith("fold"):
                    continue

                metrics_file = fold_dir / "metrics.json"
                if not metrics_file.exists():
                    continue

                try:
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)

                    # Guard against non-numeric metric values
                    raw = metrics.get(objective_metric)
                    if raw is None:
                        continue

                    try:
                        metric_value = float(raw)
                    except (TypeError, ValueError):
                        continue

                    fold_pairs.append((fold_dir, metric_value))
                except Exception:
                    continue

            if fold_pairs:
                # Compute average across folds
                avg_metric = sum(
                    metric for _, metric in fold_pairs) / len(fold_pairs)

                # Update best trial if this one is better
                if best_avg_metric is None or avg_metric > best_avg_metric:
                    best_avg_metric = avg_metric
                    best_trial_dir = trial_dir
                    best_trial_name = trial_dir.name
                    best_fold_metrics = fold_pairs

        # Optional fallback: non-CV trial (k=1 or legacy structure)
        elif not cv_dir.exists():
            metrics_file = trial_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)

                    raw = metrics.get(objective_metric)
                    if raw is not None:
                        try:
                            metric_value = float(raw)
                            # For non-CV, compare directly
                            if best_avg_metric is None or metric_value > best_avg_metric:
                                best_avg_metric = metric_value
                                best_trial_dir = trial_dir
                                best_trial_name = trial_dir.name
                                best_fold_metrics = None  # No fold structure
                        except (TypeError, ValueError):
                            continue
                except Exception:
                    continue

    if best_trial_dir is None:
        return None

    # Step 2: Select checkpoint (prefer refit, else best fold)
    checkpoint_dir = None
    checkpoint_type = None
    metrics_to_use = None

    # Prefer refit checkpoint if it exists (production artifact)
    refit_dir = best_trial_dir / "refit"
    refit_checkpoint = refit_dir / "checkpoint"
    if refit_checkpoint.exists():
        checkpoint_dir = refit_checkpoint
        checkpoint_type = "refit"
        # Try to load refit metrics (may not have objective_metric, that's OK)
        refit_metrics_file = refit_dir / "metrics.json"
        if refit_metrics_file.exists():
            try:
                with open(refit_metrics_file, "r") as f:
                    metrics_to_use = json.load(f)
            except Exception:
                pass
    elif best_fold_metrics is not None:
        # Fallback: use best fold checkpoint (fold with highest metric)
        best_fold_dir, _ = max(best_fold_metrics, key=lambda x: x[1])
        checkpoint_dir = best_fold_dir / "checkpoint"
        checkpoint_type = "fold"
        # Use metrics from best fold
        metrics_file = best_fold_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    metrics_to_use = json.load(f)
            except Exception:
                pass
    else:
        # Non-CV fallback: use trial root checkpoint
        checkpoint_dir = best_trial_dir / "checkpoint"
        checkpoint_type = "single"
        metrics_file = best_trial_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    metrics_to_use = json.load(f)
            except Exception:
                pass

    # If we couldn't load metrics from checkpoint location, use best fold metrics or trial metrics
    if metrics_to_use is None:
        if best_fold_metrics is not None:
            # Load from best fold
            best_fold_dir, _ = max(best_fold_metrics, key=lambda x: x[1])
            metrics_file = best_fold_dir / "metrics.json"
        else:
            # Non-CV: use trial root
            metrics_file = best_trial_dir / "metrics.json"

        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    metrics_to_use = json.load(f)
            except Exception:
                metrics_to_use = {}

    # Extract backbone from path (study_folder.parent.name)
    backbone = study_folder.parent.name

    return {
        "backbone": backbone,
        "trial_name": best_trial_name,
        "trial_dir": str(best_trial_dir),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else str(best_trial_dir / "checkpoint"),
        "checkpoint_type": checkpoint_type or "unknown",
        # CV average metric (or single metric for non-CV)
        "accuracy": best_avg_metric,
        "metrics": metrics_to_use or {},
    }


def write_active_study_marker(
    backbone_dir: Path,
    study_folder: Path,
    study_name: str,
    study_key_hash: Optional[str] = None
):
    """
    Write .active_study.json marker file for fast lookup.

    This makes finding the current study folder instant and unambiguous.

    Args:
        backbone_dir: Backbone directory where marker should be written
        study_folder: Path to the study folder
        study_name: Name of the study
        study_key_hash: Optional study key hash for tracking
    """
    active_study_file = backbone_dir / ".active_study.json"
    active_info = {
        "study_name": study_name,
        "path": str(study_folder),
        "study_key_hash": study_key_hash,
    }
    try:
        with open(active_study_file, "w") as f:
            json.dump(active_info, f, indent=2)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not write .active_study.json: {e}")


def _get_checkpoint_path_from_trial_dir(trial_dir: Path) -> Optional[Path]:
    """
    Get checkpoint path from trial directory.
    
    Prefers:
    1. refit/checkpoint/ (if refit training completed)
    2. cv/foldN/checkpoint/ (best CV fold based on metrics)
    3. checkpoint/ (fallback)
    
    Args:
        trial_dir: Path to trial directory
        
    Returns:
        Path to checkpoint directory, or None if not found
    """
    if not trial_dir.exists():
        return None
    
    # 1. Check for refit checkpoint
    refit_checkpoint = trial_dir / "refit" / "checkpoint"
    if refit_checkpoint.exists():
        return refit_checkpoint
    
    # 2. Check for CV fold checkpoints
    cv_dir = trial_dir / "cv"
    if cv_dir.exists():
        # Find all fold directories
        fold_dirs = []
        for item in cv_dir.iterdir():
            if item.is_dir():
                import re
                if re.match(r"fold_?\d+", item.name):
                    fold_dirs.append(item)
        
        if fold_dirs:
            # Try to find the best fold by looking for metrics.json
            best_fold = None
            best_score = None
            
            for fold_dir in fold_dirs:
                metrics_file = fold_dir / "metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file, "r") as f:
                            metrics = json.load(f)
                        # Look for macro-f1 or first numeric metric
                        score = metrics.get("macro-f1")
                        if score is None:
                            # Try to find first numeric value
                            for key, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    score = value
                                    break
                        
                        if score is not None and (best_score is None or score > best_score):
                            best_score = score
                            best_fold = fold_dir
                    except Exception:
                        continue
            
            # Use best fold if found, otherwise use first fold
            if best_fold:
                checkpoint = best_fold / "checkpoint"
                if checkpoint.exists():
                    return checkpoint
            
            # Fallback: try all folds in order
            for fold_dir in fold_dirs:
                checkpoint = fold_dir / "checkpoint"
                if checkpoint.exists():
                    return checkpoint
    
    # 3. Fallback: check root checkpoint
    root_checkpoint = trial_dir / "checkpoint"
    if root_checkpoint.exists():
        return root_checkpoint
    
    return None


def find_trial_checkpoint_by_hash(
    hpo_backbone_dir: Path,
    study_key_hash: str,
    trial_key_hash: str,
) -> Optional[Path]:
    """
    Find trial checkpoint by study_key_hash and trial_key_hash.
    
    Scans trial folders and reads trial_meta.json files to match by hash.
    This avoids Optuna DB dependencies and hash recomputation issues.
    
    Args:
        hpo_backbone_dir: Backbone directory containing study folders
        study_key_hash: Target study key hash (64 hex chars)
        trial_key_hash: Target trial key hash (64 hex chars)
        
    Returns:
        Path to checkpoint directory (prefers refit, else best CV fold), or None if not found
    """
    if not hpo_backbone_dir.exists():
        return None
    
    # Scan all study folders
    for study_folder in hpo_backbone_dir.iterdir():
        if not study_folder.is_dir() or study_folder.name.startswith("trial_"):
            continue
        
        # Scan all trial folders in this study
        for trial_dir in study_folder.iterdir():
            if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                continue
            
            # Read trial metadata
            trial_meta_path = trial_dir / "trial_meta.json"
            if not trial_meta_path.exists():
                continue
            
            try:
                with open(trial_meta_path, "r") as f:
                    meta = json.load(f)
                
                # Match by hashes
                if (meta.get("study_key_hash") == study_key_hash and 
                    meta.get("trial_key_hash") == trial_key_hash):
                    # Found match! Return checkpoint (prefer refit, else best CV fold)
                    return _get_checkpoint_path_from_trial_dir(trial_dir)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Error reading trial_meta.json from {trial_meta_path}: {e}")
                continue
    
    return None
