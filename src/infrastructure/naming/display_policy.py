from __future__ import annotations

"""
@meta
name: naming_display_policy
type: utility
domain: naming
responsibility:
  - Load naming policy from YAML with caching
  - Format display names using policy patterns
inputs:
  - Naming contexts
  - Configuration directories
outputs:
  - Formatted display names
tags:
  - utility
  - naming
  - policy
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Naming policy loading and display name formatting (with caching)."""
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from core.placeholders import extract_placeholders
from core.tokens import is_token_allowed, is_token_known
from .context import NamingContext
from .context_tokens import build_token_values
from common.shared.yaml_utils import load_yaml

logger = logging.getLogger(__name__)

# Cache for loaded policy: (config_dir, mtime) -> policy
_policy_cache: Dict[tuple, tuple] = {}  # (key, mtime) -> policy

def _get_policy_mtime(policy_path: Path) -> float:
    """Get modification time of policy file, or 0 if doesn't exist."""
    try:
        return policy_path.stat().st_mtime
    except (OSError, FileNotFoundError):
        return 0.0

def load_naming_policy(
    config_dir: Optional[Path] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Load and cache naming policy from config/naming.yaml with mtime-based caching.

    Args:
        config_dir: Path to config directory (defaults to current directory / "config").
        validate: If True, validate the policy schema.

    Returns:
        Naming policy dictionary, or empty dict if file not found.
    """
    if config_dir is None:
        config_dir = Path.cwd() / "config"

    policy_path = config_dir / "naming.yaml"
    mtime = _get_policy_mtime(policy_path)
    
    # Create cache key
    cache_key = str(config_dir)
    
    # Check cache
    if cache_key in _policy_cache:
        cached_mtime, cached_policy = _policy_cache[cache_key]
        if cached_mtime == mtime:
            return cached_policy
        # Cache invalid, remove it
        del _policy_cache[cache_key]

    # Load policy
    if not policy_path.exists():
        logger.warning(
            f"[Naming Policy] Policy file not found at {policy_path}, using empty policy")
        _policy_cache[cache_key] = (mtime, {})
        return {}

    try:
        policy = load_yaml(policy_path)
        if validate:
            validate_naming_policy(policy, policy_path)
        _policy_cache[cache_key] = (mtime, policy)
        return policy
    except Exception as e:
        logger.warning(
            f"[Naming Policy] Failed to load policy from {policy_path}: {e}", exc_info=True)
        _policy_cache[cache_key] = (mtime, {})
        return {}

def validate_naming_policy(policy: Dict[str, Any], policy_path: Optional[Path] = None) -> None:
    """
    Basic schema validation for naming.yaml.

    - v1 (schema_version missing or 1): minimal checks, mostly warnings.
    - v2 and above: can be tightened later; for now we ensure core sections exist.
    """
    location = f" ({policy_path})" if policy_path is not None else ""
    schema_version_raw = policy.get("schema_version", 1)

    try:
        schema_version = int(schema_version_raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"[naming.yaml] schema_version must be an integer, got {schema_version_raw!r}{location}"
        )

    if schema_version < 1:
        logger.warning(
            f"[naming.yaml] Unsupported schema_version={schema_version}{location}, "
            f"treating as v1 with minimal validation."
        )
        schema_version = 1

    run_names = policy.get("run_names")
    if run_names is None:
        # Only warn if there's no validate section either (truly incomplete config)
        # If validate section exists, it's acceptable for schema_version 1
        has_validate = "validate" in policy
        msg = f"[naming.yaml] Missing required 'run_names' section{location}"
        if schema_version >= 2:
            raise ValueError(msg)
        # Only warn if config is truly incomplete (no validate section)
        if not has_validate:
            logger.warning(msg)
        return

    if not isinstance(run_names, dict):
        raise ValueError(
            f"[naming.yaml] 'run_names' must be a mapping{location}")

    separators = policy.get("separators")
    if separators is not None and not isinstance(separators, dict):
        raise ValueError(
            f"[naming.yaml] 'separators' must be a mapping when present{location}")

    validate_sec = policy.get("validate")
    if validate_sec is not None and not isinstance(validate_sec, dict):
        raise ValueError(
            f"[naming.yaml] 'validate' must be a mapping when present{location}")

    # Validate placeholders in run name patterns
    for name, entry in run_names.items():
        if not isinstance(entry, dict):
            if schema_version >= 2:
                raise ValueError(
                    f"[naming.yaml] run_names.{name} must be a mapping{location}")
            logger.warning(
                f"[naming.yaml] run_names.{name} should be a mapping{location}")
            continue
        pattern = entry.get("pattern")
        if not pattern:
            continue
        if not isinstance(pattern, str):
            raise ValueError(
                f"[naming.yaml] run_names.{name}.pattern must be a string{location}")
        placeholders = extract_placeholders(pattern)
        for token in placeholders:
            if not is_token_known(token):
                if schema_version >= 2:
                    raise ValueError(
                        f"[naming.yaml] Unknown placeholder '{{{token}}}' in run_names.{name}{location}"
                    )
                logger.warning(
                    f"[naming.yaml] Unknown placeholder '{{{token}}}' in run_names.{name}{location}"
                )
                continue
            if not is_token_allowed(token, "name"):
                raise ValueError(
                    f"[naming.yaml] Placeholder '{{{token}}}' in run_names.{name} "
                    f"is not allowed for name scope{location}"
                )

def parse_parent_training_id(parent_id: str) -> Dict[str, str]:
    """
    Parse parent_training_id into spec_hash, exec_hash, and variant.

    Handles formats like:
    - "spec_81710c3324325ad0_exec_30fd84534691d188/v1"
    - "spec-81710c33-exec-30fd8453/v1"
    - "spec_abc123_exec_xyz789/v2"

    Args:
        parent_id: Parent training identifier string.

    Returns:
        Dictionary with keys: "spec_hash", "exec_hash", "variant"
        Uses "unknown" for missing values. Hashes are shortened to 8 chars.
    """
    if not parent_id:
        return {"spec_hash": "unknown", "exec_hash": "unknown", "variant": "1"}

    # Try pattern: spec_{hash}_exec_{hash}/v{variant} (with underscore or hyphen)
    pattern1 = r"spec[_-]([a-f0-9]+)[_-]exec[_-]([a-f0-9]+)/v(\d+)"
    match = re.search(pattern1, parent_id)
    if match:
        spec_hash_full = match.group(1)
        exec_hash_full = match.group(2)
        variant = match.group(3)
        return {
            "spec_hash": spec_hash_full[:8] if len(spec_hash_full) > 8 else spec_hash_full,
            "exec_hash": exec_hash_full[:8] if len(exec_hash_full) > 8 else exec_hash_full,
            "variant": variant
        }

    # Try pattern: spec_{hash}_exec_{hash} (no variant)
    pattern2 = r"spec[_-]([a-f0-9]+)[_-]exec[_-]([a-f0-9]+)"
    match = re.search(pattern2, parent_id)
    if match:
        spec_hash_full = match.group(1)
        exec_hash_full = match.group(2)
        return {
            "spec_hash": spec_hash_full[:8] if len(spec_hash_full) > 8 else spec_hash_full,
            "exec_hash": exec_hash_full[:8] if len(exec_hash_full) > 8 else exec_hash_full,
            "variant": "1"
        }

    # Fallback: log warning and return defaults
    logger.warning(
        f"[Naming Policy] Could not parse parent_training_id: {parent_id}, "
        f"using defaults"
    )
    return {"spec_hash": "unknown", "exec_hash": "unknown", "variant": "1"}

def normalize_value(value: str, rules: Optional[Dict[str, Any]] = None) -> str:
    """
    Apply normalization rules to a value.

    Args:
        value: Value to normalize.
        rules: Normalization rules dict with "replace" and "lowercase" keys.

    Returns:
        Normalized value.
    """
    if not value:
        return value

    if rules is None:
        return value

    result = value

    # Apply character replacements
    if "replace" in rules and isinstance(rules["replace"], dict):
        for old_char, new_char in rules["replace"].items():
            result = result.replace(old_char, new_char)

    # Apply lowercase if specified
    if rules.get("lowercase", False):
        result = result.lower()

    return result

def sanitize_semantic_suffix(study_name: str, max_length: int = 30, model: Optional[str] = None) -> str:
    """
    Sanitize HPO study semantic suffix.

    Args:
        study_name: Study name to sanitize.
        max_length: Maximum length for the suffix.
        model: Optional model name to strip from beginning if present (avoids duplication).

    Returns:
        Sanitized suffix (empty string if disabled or invalid).
    """
    if not study_name:
        return ""

    # Remove "hpo_" prefix if present
    label = study_name
    if label.startswith("hpo_"):
        label = label[len("hpo_"):]

    # Strip model name from beginning if present (avoids duplication in run name)
    if model and label.startswith(model):
        # Remove model name and any following separator (_, -, etc.)
        remaining = label[len(model):].lstrip("_-")
        if remaining:  # Only use if there's something left after stripping
            label = remaining

    # Replace spaces and slashes
    label = label.replace(" ", "").replace("/", "-")

    # Remove other problematic characters
    label = re.sub(r'[^\w\-]', '', label)

    # Truncate to max_length
    if len(label) > max_length:
        label = label[:max_length]

    # Add underscore prefix if non-empty
    if label:
        return f"_{label}"

    return ""

def _short(value: Optional[str], length: int = 8, default: str = "unknown") -> str:
    """Return a short hash of specified length or a default if missing."""
    if not value:
        return default
    return value[:length]

def extract_component(
    context: NamingContext,
    component_config: Dict[str, Any],
    policy: Dict[str, Any],
    process_type: str
) -> str:
    """
    Extract component value from context based on component config.

    Args:
        context: NamingContext with process data.
        component_config: Component configuration from policy.
        policy: Full naming policy.
        process_type: Process type (for special handling).

    Returns:
        Extracted and formatted component value.
    """
    source = component_config.get("source", "")
    default = component_config.get("default", "unknown")
    length = component_config.get("length", 8)
    format_str = component_config.get("format", "{value}")
    zero_pad = component_config.get("zero_pad", 0)

    # Get value from context
    value = None
    if source == "study_key_hash":
        value = getattr(context, "study_key_hash", None)
        if not value:
            # Fallback to env var
            value = os.environ.get("HPO_STUDY_KEY_HASH")
        if not value:
            logger.warning(
                f"[Naming Policy] study_key_hash not found in context or env var for process_type={process_type}. "
                f"Context has study_key_hash: {hasattr(context, 'study_key_hash')}, "
                f"value: {getattr(context, 'study_key_hash', None)}, "
                f"env var: {os.environ.get('HPO_STUDY_KEY_HASH', 'NOT SET')}"
            )
    elif source == "trial_key_hash":
        value = getattr(context, "trial_key_hash", None)
    elif source == "trial_number":
        value = getattr(context, "trial_number", None)
    elif source == "fold_idx":
        value = getattr(context, "fold_idx", None)
    elif source == "spec_fp":
        value = context.spec_fp
    elif source == "exec_fp":
        value = context.exec_fp
    elif source == "variant":
        value = context.variant
    elif source == "conv_fp":
        value = context.conv_fp
    elif source == "benchmark_config_hash":
        value = getattr(context, "benchmark_config_hash", None)
    elif source == "study_name":
        value = getattr(context, "study_name", None)

    # Handle special cases
    if source == "study_name":
        # Semantic suffix for hpo_sweep
        max_length = component_config.get("max_length", 30)
        enabled = component_config.get("enabled", True)
        if enabled and value:
            # Pass model name to avoid duplication (e.g., "distilbert" in study_name)
            model_name = context.model if hasattr(context, "model") else None
            return sanitize_semantic_suffix(value, max_length, model=model_name)
        return ""

    # Format numeric values
    if source == "trial_number" and value is not None:
        num_value = int(value)
        if zero_pad > 0:
            # Zero-pad the number, then format
            padded_num = str(num_value).zfill(zero_pad)
            formatted = format_str.format(number=padded_num)
        else:
            formatted = format_str.format(number=num_value)
        return formatted

    if source == "variant" and value is not None:
        return format_str.format(number=int(value))

    # Format fold_idx (just the number, no prefix)
    if source == "fold_idx" and value is not None:
        num_value = int(value)
        format_str = component_config.get("format", "{number}")
        return format_str.format(number=num_value)

    # Handle hash values (shorten to length)
    if value:
        short_value = _short(value, length, default)
        return short_value

    return default

def format_run_name(
    process_type: str,
    context: NamingContext,
    policy: Optional[Dict[str, Any]] = None,
    config_dir: Optional[Path] = None
) -> str:
    """
    Format run name using policy pattern.

    Does NOT add version suffix - that's handled by collision logic.

    Args:
        process_type: Process type (e.g., "hpo_trial", "hpo_sweep", "final_training").
        context: NamingContext with process data.
        policy: Optional naming policy dictionary (loaded if not provided).
        config_dir: Config directory (for loading policy and normalization rules).

    Returns:
        Formatted run name (without version suffix).
    """
    # Load policy if not provided
    if policy is None:
        if config_dir is None:
            config_dir = Path.cwd() / "config"
        policy = load_naming_policy(config_dir)

    if not policy or "run_names" not in policy:
        logger.warning(
            "[Naming Policy] Policy missing or incomplete, cannot format run name")
        return f"{context.process_type}_{context.model}_unknown"

    run_names_config = policy.get("run_names", {})
    if process_type not in run_names_config:
        logger.warning(
            f"[Naming Policy] No pattern for process_type: {process_type}")
        return f"{context.process_type}_{context.model}_unknown"

    process_config = run_names_config[process_type]
    pattern = process_config.get("pattern", "")
    components_config = process_config.get("components", {})

    # Get normalization rules
    normalize_rules = policy.get("normalize", {})

    # Extract environment and model (with normalization)
    env = context.storage_env if hasattr(
        context, "storage_env") else context.environment
    env = normalize_value(env, normalize_rules.get("env"))
    model = normalize_value(context.model, normalize_rules.get("model"))

    # Extract components
    component_values: Dict[str, Any] = {}

    # Special handling for conversion (parent_training_id parsing)
    if process_type == "conversion":
        if not context.parent_training_id:
            logger.warning(
                "[Naming Policy] Conversion process_type requires parent_training_id, "
                "but context.parent_training_id is None"
            )
        parsed = parse_parent_training_id(context.parent_training_id or "")
        for component_name, component_config in components_config.items():
            source = component_config.get("source", "")
            if source == "parent_training_id":
                # Map component name to parsed field
                if component_name == "spec_hash":
                    spec_hash = parsed.get("spec_hash", "unknown")
                    # Ensure it's exactly 8 chars (parse_parent_training_id should already shorten, but be safe)
                    component_values[component_name] = _short(
                        spec_hash, 8, "unknown")
                    if len(component_values[component_name]) != 8 and component_values[component_name] != "unknown":
                        logger.warning(
                            f"[Naming Policy] spec_hash length is {len(component_values[component_name])}, "
                            f"expected 8. Value: {component_values[component_name]}"
                        )
                        component_values[component_name] = component_values[component_name][:8]
                elif component_name == "exec_hash":
                    exec_hash = parsed.get("exec_hash", "unknown")
                    # Ensure it's exactly 8 chars (parse_parent_training_id should already shorten, but be safe)
                    component_values[component_name] = _short(
                        exec_hash, 8, "unknown")
                    if len(component_values[component_name]) != 8 and component_values[component_name] != "unknown":
                        logger.warning(
                            f"[Naming Policy] exec_hash length is {len(component_values[component_name])}, "
                            f"expected 8. Value: {component_values[component_name]}"
                        )
                        component_values[component_name] = component_values[component_name][:8]
                elif component_name == "variant":
                    variant_num = parsed.get("variant", "1")
                    format_str = component_config.get("format", "v{number}")
                    component_values[component_name] = format_str.format(
                        number=int(variant_num))
            else:
                # Regular extraction
                value = extract_component(
                    context, component_config, policy, process_type)
                component_values[component_name] = value
    else:
        # Regular extraction for all components
        for component_name, component_config in components_config.items():
            value = extract_component(
                context, component_config, policy, process_type)
            component_values[component_name] = value

    # Format the pattern
    # Replace {version} with empty string (version is added by collision logic)
    pattern = pattern.replace("{version}", "")

    # Build substitution dict
    substitutions = {
        "env": env,
        "model": model,
        **component_values
    }

    # Format the pattern
    try:
        name = pattern.format(**substitutions)
    except KeyError as e:
        logger.warning(
            f"[Naming Policy] Missing key in pattern substitution: {e}")
        # Fallback: replace missing keys with "unknown"
        for key in re.findall(r'\{(\w+)\}', pattern):
            if key not in substitutions:
                substitutions[key] = "unknown"
        name = pattern.format(**substitutions)

    return name

def validate_run_name(name: str, policy: Dict[str, Any]) -> None:
    """
    Validate run name against policy rules.

    Logs warnings for length violations. Raises ValueError for forbidden characters.

    Args:
        name: Run name to validate.
        policy: Naming policy dictionary.
        
    Raises:
        ValueError: If run name contains forbidden characters.
    """
    if not policy or "validate" not in policy:
        return

    validate_config = policy.get("validate", {})
    max_length = validate_config.get("max_length", 256)
    forbidden_chars = validate_config.get("forbidden_chars", [])
    warn_length = validate_config.get("warn_length", 150)

    # Check length
    if len(name) > max_length:
        error_msg = (
            f"Run name exceeds max length ({max_length}): "
            f"{name[:50]}... (length: {len(name)})"
        )
        logger.error(f"[Naming Policy] {error_msg}")
        raise ValueError(f"max_length: {error_msg}")
    elif len(name) > warn_length:
        logger.warning(
            f"[Naming Policy] Run name exceeds recommended length ({warn_length}): "
            f"{name[:50]}... (length: {len(name)})"
        )

    # Check forbidden characters
    found_forbidden = [char for char in forbidden_chars if char in name]
    if found_forbidden:
        error_msg = (
            f"Run name contains forbidden characters {found_forbidden}: "
            f"{name[:50]}..."
        )
        logger.error(f"[Naming Policy] {error_msg}")
        raise ValueError(f"forbidden: {error_msg}")

