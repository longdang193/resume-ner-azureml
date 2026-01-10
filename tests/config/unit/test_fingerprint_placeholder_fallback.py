"""Unit tests for fingerprint placeholder fallback when import fails."""

import pytest
import warnings
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

from infrastructure.config.training import _compute_fingerprints


class TestFingerprintPlaceholderFallback:
    """Test that placeholder fallback works when fingerprint computation fails."""

    def test_compute_fingerprints_returns_placeholders_on_import_error(
        self, tmp_path
    ):
        """Test that _compute_fingerprints returns 'unknown' placeholders when import fails."""
        root_dir = tmp_path
        all_configs = {
            "model": {"backbone": "distilbert"},
            "data": {"dataset": "resume_ner"},
            "train": {"learning_rate": 2e-5},
            "env": {"platform": "local"},
        }
        seed = 42
        identity_config = {
            "include_code_fp": True,
            "include_precision_fp": True,
            "include_determinism_fp": False,
        }

        # Mock ImportError by patching the import statement
        original_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == "infrastructure.fingerprints" or name.startswith(
                "infrastructure.fingerprints."
            ):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                spec_fp, exec_fp = _compute_fingerprints(
                    root_dir=root_dir,
                    all_configs=all_configs,
                    seed=seed,
                    identity_config=identity_config,
                )

                # Verify placeholders are returned
                assert spec_fp == "unknown"
                assert exec_fp == "unknown"

                # Verify warning was issued
                assert len(w) >= 1
                warning_messages = [str(warning.message).lower() for warning in w]
                assert any("placeholder" in msg for msg in warning_messages)

    def test_compute_fingerprints_import_error_handling(
        self, tmp_path
    ):
        """Test that ImportError is caught and handled gracefully."""
        root_dir = tmp_path
        all_configs = {
            "model": {"backbone": "distilbert"},
            "data": {"dataset": "resume_ner"},
            "train": {"learning_rate": 2e-5},
            "env": {"platform": "local"},
        }
        seed = 42
        identity_config = {
            "include_code_fp": False,
            "include_precision_fp": False,
            "include_determinism_fp": False,
        }

        # Simulate ImportError by patching the import
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "infrastructure.fingerprints" or name.startswith(
                "infrastructure.fingerprints."
            ):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                spec_fp, exec_fp = _compute_fingerprints(
                    root_dir=root_dir,
                    all_configs=all_configs,
                    seed=seed,
                    identity_config=identity_config,
                )

                # Verify placeholders are returned
                assert spec_fp == "unknown"
                assert exec_fp == "unknown"

                # Verify warning was issued
                assert len(w) >= 1
                warning_messages = [str(warning.message).lower() for warning in w]
                assert any("placeholder" in msg for msg in warning_messages)

    def test_placeholder_values_are_short_enough_for_naming(
        self, tmp_path
    ):
        """Test that placeholder values ('unknown') are short enough for 8-char truncation."""
        root_dir = tmp_path
        all_configs = {
            "model": {"backbone": "distilbert"},
            "data": {"dataset": "resume_ner"},
            "train": {"learning_rate": 2e-5},
            "env": {"platform": "local"},
        }
        seed = 42
        identity_config = {
            "include_code_fp": False,
            "include_precision_fp": False,
            "include_determinism_fp": False,
        }

        # Mock ImportError
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            spec_fp, exec_fp = _compute_fingerprints(
                root_dir=root_dir,
                all_configs=all_configs,
                seed=seed,
                identity_config=identity_config,
            )

            # Verify placeholders are short (7 chars, fits in 8-char limit)
            assert len(spec_fp) <= 8
            assert len(exec_fp) <= 8
            assert spec_fp == "unknown"
            assert exec_fp == "unknown"

            # Verify truncation to 8 chars still makes sense
            spec_truncated = spec_fp[:8]
            exec_truncated = exec_fp[:8]
            assert spec_truncated == "unknown"
            assert exec_truncated == "unknown"

