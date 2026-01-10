"""Trial callback utilities for HPO.

Creates callbacks for displaying trial completion information.
"""

from __future__ import annotations

from typing import Any, Optional

import mlflow
from common.shared.logging_utils import get_logger
from hpo.core.optuna_integration import import_optuna as _import_optuna

logger = get_logger(__name__)


def create_trial_callback(
    objective_metric: str, parent_run_id: Optional[str] = None
):
    """
    Create a trial completion callback with parent run ID captured in closure.

    Args:
        objective_metric: Name of the objective metric being optimized.
        parent_run_id: Optional parent MLflow run ID.

    Returns:
        Callback function for Optuna study.optimize().
    """

    def trial_complete_callback(study: Any, trial: Any) -> None:
        """Callback to display consolidated metrics and parameters after trial completes."""
        optuna_module, _, _, _ = _import_optuna()

        if trial.state == optuna_module.trial.TrialState.COMPLETE:
            best_trial = study.best_trial
            is_best = trial.number == best_trial.number

            attrs = trial.user_attrs
            parts = [f"{objective_metric}={trial.value:.6f}"]

            if "macro_f1_span" in attrs:
                parts.append(f"span={attrs['macro_f1_span']:.6f}")
            if "loss" in attrs:
                parts.append(f"loss={attrs['loss']:.6f}")
            if "avg_entity_f1" in attrs:
                entity_count = attrs.get("entity_count", "?")
                parts.append(
                    f"entity_f1={attrs['avg_entity_f1']:.6f} ({entity_count} entities)"
                )

            param_parts = []
            for param_name, param_value in trial.params.items():
                if isinstance(param_value, float):
                    if param_name == "learning_rate":
                        param_parts.append(f"{param_name}={param_value:.2e}")
                    else:
                        param_parts.append(f"{param_name}={param_value:.6f}")
                else:
                    param_parts.append(f"{param_name}={param_value}")

            run_id_short = ""
            try:
                if parent_run_id:
                    client = mlflow.tracking.MlflowClient()
                    active_run = mlflow.active_run()
                    if active_run:
                        experiment_id = active_run.info.experiment_id
                        all_runs = client.search_runs(
                            experiment_ids=[experiment_id],
                            filter_string=(
                                f"tags.mlflow.parentRunId = '{parent_run_id}' "
                                f"AND tags.trial_number = '{trial.number}'"
                            ),
                            max_results=100,
                        )
                        if all_runs:
                            trial_run = None
                            for run in all_runs:
                                if not run.data.tags.get("fold_idx"):
                                    trial_run = run
                                    break
                            if not trial_run:
                                trial_run = all_runs[0]
                            run_id_short = (
                                f" (Run ID: {trial_run.info.run_id[:12]}...)"
                            )
            except Exception as e:
                logger.debug(f"Could not get run ID for trial {trial.number}: {e}")

            logger.info("")
            status = "[BEST]" if is_best else f"[Trial {trial.number}]"
            trial_name = f"trial_{trial.number}"
            logger.info(f"{status}: {trial_name}")
            logger.info(f"  Metrics: {' | '.join(parts)}")
            logger.info(f"  Params: {' | '.join(param_parts)}{run_id_short}")

    return trial_complete_callback

