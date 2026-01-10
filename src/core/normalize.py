"""Normalization helpers for names and filesystem-safe paths.

This module provides normalization utilities for the naming system foundation.
These functions are used to normalize values for display names and filesystem paths,
ensuring consistency and safety in naming conventions.

Note: These functions are distinct from text normalization used in training
(see training/data.py::normalize_text_for_tokenization() for tokenization-specific normalization).
"""

from typing import Any, Dict, List, Tuple


def normalize_for_name(value: Any, rules: Dict[str, Any] | None = None) -> Tuple[str, List[str]]:
    """
    Normalize a value for display/name usage.

    This function is part of the naming system foundation and is used to normalize
    values that will be displayed as names (e.g., in MLflow tags, experiment names).
    It applies replacement rules and optional lowercase conversion.

    Args:
        value: Value to normalize (will be converted to string)
        rules: Optional normalization rules dictionary with:
            - "replace": Dict[str, str] - character/string replacements
            - "lowercase": bool - whether to convert to lowercase

    Returns:
        Tuple of (normalized_value, warnings) where warnings is a list of
        normalization warnings encountered during processing.
    """
    if value is None:
        return "", []

    result = str(value)
    warnings: List[str] = []

    if not rules:
        return result, warnings

    replacements = rules.get("replace", {})
    if isinstance(replacements, dict):
        for old, new in replacements.items():
            result = result.replace(old, new)

    if rules.get("lowercase", False):
        result = result.lower()

    return result, warnings


def normalize_for_path(value: Any, rules: Dict[str, Any] | None = None) -> Tuple[str, List[str]]:
    """
    Normalize a value to be filesystem-safe.

    This function is part of the naming system foundation and is used to normalize
    values that will be used in filesystem paths (e.g., output directories, file names).
    It applies replacement rules, replaces forbidden characters, and optionally
    truncates to a maximum component length.

    Args:
        value: Value to normalize (will be converted to string)
        rules: Optional normalization rules dictionary with:
            - "replace": Dict[str, str] - character/string replacements
            - "forbidden_chars": List[str] - characters to replace with underscore
            - "lowercase": bool - whether to convert to lowercase
            - "max_component_length": int - maximum length before truncation

    Returns:
        Tuple of (normalized_value, warnings) where warnings is a list of
        normalization warnings encountered during processing.
    """
    if value is None:
        return "", []

    result = str(value)
    warnings: List[str] = []

    if rules:
        replacements = rules.get("replace", {})
        if isinstance(replacements, dict):
            for old, new in replacements.items():
                result = result.replace(old, new)

        forbidden_chars = rules.get("forbidden_chars")
        if isinstance(forbidden_chars, list):
            for ch in forbidden_chars:
                if ch in result:
                    result = result.replace(ch, "_")
                    warnings.append(f"Replaced forbidden char '{ch}'")

        if rules.get("lowercase", False):
            result = result.lower()

        max_component_length = rules.get("max_component_length")
        if isinstance(max_component_length, int) and max_component_length > 0:
            if len(result) > max_component_length:
                warnings.append(
                    f"Truncated to max_component_length={max_component_length}"
                )
                result = result[:max_component_length]
    return result, warnings

