"""Shared token registry for naming and path patterns."""

from dataclasses import dataclass
from typing import Dict, Optional, Set, Iterable


@dataclass(frozen=True)
class Token:
    name: str
    scopes: Set[str]  # allowed scopes: "name", "path"


# Registry of known tokens and their scopes.
TOKENS: Dict[str, Token] = {
    # Common identity
    "env": Token("env", {"name"}),
    "storage_env": Token("storage_env", {"path"}),
    "model": Token("model", {"name", "path"}),
    # Fingerprints / hashes
    "spec_fp": Token("spec_fp", {"name", "path"}),
    "exec_fp": Token("exec_fp", {"name", "path"}),
    "spec_hash": Token("spec_hash", {"name"}),  # shortened spec_fp for names
    "exec_hash": Token("exec_hash", {"name"}),  # shortened exec_fp for names
    "variant": Token("variant", {"name", "path"}),
    # HPO/benchmark hashes (full + short forms)
    "study_hash": Token("study_hash", {"name"}),
    "trial_hash": Token("trial_hash", {"name"}),
    "study8": Token("study8", {"name", "path"}),
    "trial8": Token("trial8", {"name", "path"}),
    "bench8": Token("bench8", {"name", "path"}),
    "bench_hash": Token("bench_hash", {"name"}),
    # Trials / folds
    "trial_number": Token("trial_number", {"name"}),
    "fold_idx": Token("fold_idx", {"name"}),
    "trial_id": Token("trial_id", {"path"}),
    # Conversion / parent
    "parent_training_id": Token("parent_training_id", {"path"}),
    "conv_fp": Token("conv_fp", {"name", "path"}),
    "conv_hash": Token("conv_hash", {"name"}),
    # Cache-related identifiers
    "backbone": Token("backbone", {"path"}),
    "trial": Token("trial", {"path"}),
    "run_id": Token("run_id", {"path"}),
    "identifier": Token("identifier", {"path"}),
    "timestamp": Token("timestamp", {"path"}),
    # Short fingerprint helpers for lineage
    "spec8": Token("spec8", {"name", "path"}),
    "exec8": Token("exec8", {"name", "path"}),
    "conv8": Token("conv8", {"name", "path"}),
    # Naming-only helpers
    "semantic_suffix": Token("semantic_suffix", {"name"}),
    "version": Token("version", {"name"}),
}


def get_token(name: str) -> Optional[Token]:
    """Get a token by name."""
    return TOKENS.get(name)


def is_token_known(name: str) -> bool:
    """Check if a token name is known in the registry."""
    return name in TOKENS


def is_token_allowed(name: str, scope: str) -> bool:
    """Check if a token is allowed for a given scope (name or path)."""
    token = TOKENS.get(name)
    return token is not None and scope in token.scopes


def tokens_for_scope(scope: str) -> Iterable[str]:
    """Get all token names allowed for a given scope."""
    return (t.name for t in TOKENS.values() if scope in t.scopes)

