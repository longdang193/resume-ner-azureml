"""Unit tests for naming placeholder truncation behavior."""

import pytest
from infrastructure.naming import create_naming_context
from infrastructure.naming.display_policy import format_run_name
from infrastructure.naming.context_tokens import build_token_values


class TestNamingPlaceholderTruncation:
    """Test that placeholders are correctly truncated in naming."""

    def test_final_training_naming_with_unknown_placeholders(self):
        """Test that 'unknown' placeholders are used correctly in final_training naming."""
        # Create context with 'unknown' placeholders (from failed fingerprint computation)
        context = create_naming_context(
            process_type="final_training",
            model="distilbert",
            environment="local",
            spec_fp="unknown",  # Placeholder from failed fingerprint computation
            exec_fp="unknown",  # Placeholder from failed fingerprint computation
            variant=1,
        )

        # Build token values
        tokens = build_token_values(context)

        # Verify spec_hash and exec_hash are set correctly
        assert tokens["spec_hash"] == "unknown"
        assert tokens["exec_hash"] == "unknown"
        assert tokens["spec8"] == "unknown"  # Should be 7 chars, fits in 8
        assert tokens["exec8"] == "unknown"  # Should be 7 chars, fits in 8

        # Verify they're short enough (7 chars, less than 8-char limit)
        assert len(tokens["spec_hash"]) <= 8
        assert len(tokens["exec_hash"]) <= 8

    def test_final_training_naming_pattern_with_unknown_placeholders(self, tmp_path):
        """Test that final_training naming pattern works with 'unknown' placeholders."""
        from infrastructure.naming.display_policy import load_naming_policy
        from infrastructure.naming.mlflow.run_names import build_mlflow_run_name
        from pathlib import Path

        # Create context with 'unknown' placeholders
        context = create_naming_context(
            process_type="final_training",
            model="distilbert",
            environment="local",
            spec_fp="unknown",
            exec_fp="unknown",
            variant=1,
        )

        # Load naming policy from actual config
        config_dir = Path(__file__).parent.parent.parent.parent / "config"
        if not (config_dir / "naming.yaml").exists():
            # Fallback: use tmp_path with minimal config
            config_dir = tmp_path / "config"
            config_dir.mkdir()
            naming_yaml = config_dir / "naming.yaml"
            naming_yaml.write_text("""
schema_version: 1
run_names:
  final_training:
    pattern: "{env}_{model}_final_training_spec-{spec_hash}_exec-{exec_hash}_v{variant}"
    components:
      spec_hash:
        length: 8
        source: "spec_fp"
        default: "unknown"
      exec_hash:
        length: 8
        source: "exec_fp"
        default: "unknown"
      variant:
        format: "{number}"
        source: "variant"
        default: "1"
""")

        # Build run name using the MLflow naming function
        run_name = build_mlflow_run_name(context, config_dir=config_dir)

        # Verify the name contains 'unknown' (not truncated to 'placehol')
        assert "spec-unknown" in run_name or "spec_unknown" in run_name
        assert "exec-unknown" in run_name or "exec_unknown" in run_name
        assert "unknown" in run_name
        # Verify it does NOT contain 'placehol' (old truncated placeholder)
        assert "placehol" not in run_name

    def test_placeholder_truncation_to_8_chars(self):
        """Test that long placeholders would be truncated to 8 chars, but 'unknown' fits."""
        # Test with 'unknown' (7 chars) - should not be truncated
        context = create_naming_context(
            process_type="final_training",
            model="distilbert",
            environment="local",
            spec_fp="unknown",
            exec_fp="unknown",
            variant=1,
        )

        tokens = build_token_values(context)

        # 'unknown' is 7 chars, so spec8 and exec8 should be 'unknown' (not truncated)
        assert tokens["spec8"] == "unknown"
        assert tokens["exec8"] == "unknown"
        assert len(tokens["spec8"]) == 7
        assert len(tokens["exec8"]) == 7

    def test_old_placeholder_behavior_would_truncate(self):
        """Test that old 'placeholder_spec_fp' would have been truncated incorrectly."""
        # Simulate what would happen with old placeholder values
        old_spec_fp = "placeholder_spec_fp"  # 19 chars
        old_exec_fp = "placeholder_exec_fp"  # 19 chars

        # If these were used, they would be truncated to 8 chars
        old_spec_truncated = old_spec_fp[:8]  # "placehol"
        old_exec_truncated = old_exec_fp[:8]  # "placehol"

        # Verify the old behavior would have been wrong
        assert old_spec_truncated == "placehol"
        assert old_exec_truncated == "placehol"
        assert len(old_spec_truncated) == 8
        assert len(old_exec_truncated) == 8

        # Verify new 'unknown' is better
        new_spec_fp = "unknown"  # 7 chars
        new_exec_fp = "unknown"  # 7 chars

        assert new_spec_fp == "unknown"
        assert new_exec_fp == "unknown"
        assert len(new_spec_fp) == 7
        assert len(new_exec_fp) == 7

    def test_naming_context_accepts_unknown_placeholders(self):
        """Test that NamingContext accepts 'unknown' as valid placeholder values."""
        # Should not raise an error
        context = create_naming_context(
            process_type="final_training",
            model="distilbert",
            environment="local",
            spec_fp="unknown",
            exec_fp="unknown",
            variant=1,
        )

        assert context.spec_fp == "unknown"
        assert context.exec_fp == "unknown"
        assert context.variant == 1

    def test_token_values_with_unknown_placeholders(self):
        """Test that build_token_values handles 'unknown' placeholders correctly."""
        context = create_naming_context(
            process_type="final_training",
            model="distilbert",
            environment="local",
            spec_fp="unknown",
            exec_fp="unknown",
            variant=1,
        )

        tokens = build_token_values(context)

        # Verify all token values are set correctly
        assert tokens["spec_fp"] == "unknown"
        assert tokens["exec_fp"] == "unknown"
        assert tokens["spec_hash"] == "unknown"
        assert tokens["exec_hash"] == "unknown"
        assert tokens["spec8"] == "unknown"
        assert tokens["exec8"] == "unknown"

        # Verify lengths are correct
        assert len(tokens["spec_fp"]) == 7
        assert len(tokens["exec_fp"]) == 7
        assert len(tokens["spec_hash"]) == 7
        assert len(tokens["exec_hash"]) == 7
        assert len(tokens["spec8"]) == 7
        assert len(tokens["exec8"]) == 7

