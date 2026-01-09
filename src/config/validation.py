"""Cross-file compatibility checks for paths.yaml and naming.yaml."""

from typing import Any, Dict, List

from core.placeholders import extract_placeholders
from core.tokens import (
    is_token_allowed,
    is_token_known,
)


def validate_paths_and_naming_compatible(
    paths_cfg: Dict[str, Any],
    naming_cfg: Dict[str, Any],
    strict: bool = False,
) -> List[str]:
    """
    Validate that placeholders used in paths.yaml and naming.yaml are known and
    used within their allowed scopes.

    Args:
        paths_cfg: Loaded paths configuration.
        naming_cfg: Loaded naming configuration.
        strict: If True, raise ValueError on the first issue. If False, return
            a list of issue strings (empty if compatible).

    Returns:
        List of issue strings when strict is False.
    """
    issues: List[str] = []

    # Check path patterns
    for pattern_name, pattern in paths_cfg.get("patterns", {}).items():
        if not isinstance(pattern, str):
            continue
        for token in extract_placeholders(pattern):
            if not is_token_known(token):
                issues.append(
                    f"paths.patterns.{pattern_name} uses unknown placeholder '{{{token}}}'"
                )
            elif not is_token_allowed(token, "path"):
                issues.append(
                    f"paths.patterns.{pattern_name} uses placeholder '{{{token}}}' not allowed for path scope"
                )

    # Check naming patterns
    for run_name, entry in naming_cfg.get("run_names", {}).items():
        if not isinstance(entry, dict):
            continue
        pattern = entry.get("pattern")
        if not isinstance(pattern, str):
            continue
        for token in extract_placeholders(pattern):
            if not is_token_known(token):
                issues.append(
                    f"naming.run_names.{run_name} uses unknown placeholder '{{{token}}}'"
                )
            elif not is_token_allowed(token, "name"):
                issues.append(
                    f"naming.run_names.{run_name} uses placeholder '{{{token}}}' not allowed for name scope"
                )

    if strict and issues:
        raise ValueError("; ".join(issues))

    return issues


