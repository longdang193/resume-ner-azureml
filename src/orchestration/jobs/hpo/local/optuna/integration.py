"""Optuna integration utilities for local HPO."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from common.shared.logging_utils import get_logger

logger = get_logger(__name__)

# Suppress Optuna's verbose output to reduce log clutter
logging.getLogger("optuna").setLevel(logging.WARNING)


def import_optuna():
    """
    Lazy import optuna and related modules.

    Returns:
        Tuple of (optuna, MedianPruner, RandomSampler, Trial).

    Raises:
        ImportError: If optuna is not installed.
    """
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import RandomSampler
        from optuna.trial import Trial
        return optuna, MedianPruner, RandomSampler, Trial
    except ImportError as e:
        raise ImportError(
            "optuna is required for local HPO execution. "
            "Install it with: pip install optuna"
        ) from e


def create_optuna_pruner(hpo_config: Dict[str, Any]) -> Optional[Any]:
    """
    Create Optuna pruner from HPO config early termination policy.

    Args:
        hpo_config: HPO configuration dictionary.

    Returns:
        Optuna pruner instance or None if no early termination configured.
    """
    if "early_termination" not in hpo_config:
        return None

    # Lazy import optuna
    _, MedianPruner, _, _ = import_optuna()

    et_cfg = hpo_config["early_termination"]
    policy = et_cfg.get("policy", "").lower()

    if policy == "bandit":
        # Optuna doesn't have exact bandit pruner, use MedianPruner as closest alternative
        # Bandit policy: stop if trial is worse than best by slack_factor
        # MedianPruner: stop if trial is worse than median
        return MedianPruner(
            n_startup_trials=et_cfg.get("delay_evaluation", 2),
            n_warmup_steps=et_cfg.get("evaluation_interval", 1),
        )
    elif policy == "median":
        return MedianPruner(
            n_startup_trials=et_cfg.get("delay_evaluation", 2),
            n_warmup_steps=et_cfg.get("evaluation_interval", 1),
        )
    else:
        return None

