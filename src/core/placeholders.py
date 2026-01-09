"""Placeholder extraction utility for parsing {placeholder} patterns."""

import re
from typing import Set


_PLACEHOLDER_RE = re.compile(r"{([^{}]+)}")


def extract_placeholders(pattern: str) -> Set[str]:
    """
    Extract placeholder names from a pattern string.
    
    Args:
        pattern: Pattern string containing placeholders like '{foo}_{bar}'
    
    Returns:
        Set of placeholder names found in the pattern
    
    Examples:
        >>> extract_placeholders("{model}_{stage}")
        {'model', 'stage'}
        >>> extract_placeholders("outputs/{storage_env}/{model}")
        {'storage_env', 'model'}
    """
    return set(_PLACEHOLDER_RE.findall(pattern))

