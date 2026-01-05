"""Search space translation utilities for HPO configurations.

Provides unified translation between HPO config format and both Optuna and Azure ML formats.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class SearchSpaceTranslator:
    """Translates HPO search space configurations to different formats."""

    @staticmethod
    def to_optuna(
        hpo_config: Dict[str, Any],
        trial: Any,
        exclude_params: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Translate HPO config search space to Optuna trial suggestions.

        Args:
            hpo_config: HPO configuration dictionary with search_space.
            trial: Optuna trial object for suggesting values.
            exclude_params: Optional list of parameter names to exclude from search space.

        Returns:
            Dictionary of hyperparameter values for this trial.
        """
        try:
            import optuna
        except ImportError as e:
            raise ImportError(
                "optuna is required for Optuna search space translation. "
                "Install it with: pip install optuna"
            ) from e

        search_space = hpo_config["search_space"]
        params: Dict[str, Any] = {}
        exclude_set = set(exclude_params or [])

        for name, spec in search_space.items():
            # Skip excluded parameters (e.g., "backbone" when it's fixed per study)
            if name in exclude_set:
                continue

            p_type = spec["type"]
            if p_type == "choice":
                params[name] = trial.suggest_categorical(name, spec["values"])
            elif p_type == "uniform":
                params[name] = trial.suggest_float(
                    name, float(spec["min"]), float(spec["max"])
                )
            elif p_type == "loguniform":
                params[name] = trial.suggest_float(
                    name, float(spec["min"]), float(spec["max"]), log=True
                )
            else:
                raise ValueError(f"Unsupported search space type: {p_type}")

        return params

    @staticmethod
    def to_azure_ml(hpo_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate HPO config search space to Azure ML sweep primitives.

        Args:
            hpo_config: Configuration dictionary containing a search_space key.

        Returns:
            Dictionary mapping parameter names to Azure ML search distributions.
        """
        try:
            from azure.ai.ml.sweep import Choice, Uniform, LogUniform
        except ImportError as e:
            raise ImportError(
                "azure-ai-ml is required for Azure ML search space translation. "
                "Install it with: pip install azure-ai-ml"
            ) from e

        search_space: Dict[str, Any] = {}
        for name, spec in hpo_config["search_space"].items():
            p_type = spec["type"]
            if p_type == "choice":
                search_space[name] = Choice(values=spec["values"])
            elif p_type == "uniform":
                search_space[name] = Uniform(
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                )
            elif p_type == "loguniform":
                search_space[name] = LogUniform(
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                )
            else:
                raise ValueError(f"Unsupported search space type: {p_type}")

        return search_space


# Convenience functions for backward compatibility
def translate_search_space_to_optuna(
    hpo_config: Dict[str, Any],
    trial: Any,
    exclude_params: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Translate HPO config search space to Optuna trial suggestions.

    This is a convenience wrapper around SearchSpaceTranslator.to_optuna()
    for backward compatibility.

    Args:
        hpo_config: HPO configuration dictionary with search_space.
        trial: Optuna trial object for suggesting values.
        exclude_params: Optional list of parameter names to exclude from search space.

    Returns:
        Dictionary of hyperparameter values for this trial.
    """
    return SearchSpaceTranslator.to_optuna(hpo_config, trial, exclude_params)


def create_search_space(hpo_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate a config-defined search space into Azure ML sweep primitives.

    This is a convenience wrapper around SearchSpaceTranslator.to_azure_ml()
    for backward compatibility.

    Args:
        hpo_config: Configuration dictionary containing a search_space key.

    Returns:
        Dictionary mapping parameter names to Azure ML search distributions.
    """
    return SearchSpaceTranslator.to_azure_ml(hpo_config)
