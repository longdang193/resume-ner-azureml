"""Normalization helpers for names and filesystem-safe paths."""

from typing import Any, Dict, List, Tuple


def normalize_for_name(value: Any, rules: Dict[str, Any] | None = None) -> Tuple[str, List[str]]:
    """
    Normalize a value for display/name usage.

    Returns (normalized_value, warnings).
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

    Applies replace rules, then strips/replaces forbidden characters,
    and truncates to max_component_length if configured.

    Returns (normalized_value, warnings).
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


