"""Legacy facade for tokens module.

This module provides backward compatibility by re-exporting from core.tokens and core.placeholders.
All imports from this module are deprecated.
"""

import warnings
from core.tokens import (
    TOKENS,
    Token,
    get_token,
    is_token_known,
    is_token_allowed,
    tokens_for_scope,
)
from core.placeholders import extract_placeholders

__all__ = [
    "TOKENS",
    "Token",
    "get_token",
    "is_token_known",
    "is_token_allowed",
    "tokens_for_scope",
    "extract_placeholders",
]

# Issue deprecation warning
warnings.warn(
    "Importing 'tokens' from 'orchestration' is deprecated. "
    "Please import from 'core.tokens' (and 'core.placeholders' for extract_placeholders) instead.",
    DeprecationWarning,
    stacklevel=2
)
