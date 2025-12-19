"""Tests for ONNX model conversion functionality."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest
import torch


class TestParseArguments:
    """Tests for parse_arguments function."""

    @patch("sys.argv", ["convert_to_onnx.py", "--checkpoint-path", "/path/to/checkpoint", 
                        "--config-dir", "/path/to/config", "--backbone", "distilbert",
                        "--output-dir", "/path/to/output"])
    def test_parse_required_arguments(self):
        """Test parsing of required arguments."""
        from convert_to_onnx import parse_arguments
        
        args = parse_arguments()
        
        assert args.checkpoint_path == "/path/to/checkpoint"
        assert args.config_dir == "/path/to/config"
        assert args.backbone == "distilbert"
        assert args.output_dir == "/path/to/output"
        assert args.quantize_int8 is False
        assert args.run_smoke_test is False

    @patch("sys.argv", ["convert_to_onnx.py", "--checkpoint-path", "/path/to/checkpoint",
                        "--config-dir", "/path/to/config", "--backbone", "deberta",
                        "--output-dir", "/path/to/output", "--quantize-int8", "--run-smoke-test"])
    def test_parse_optional_flags(self):
        """Test parsing of optional flags."""
        from convert_to_onnx import parse_arguments
        
        args = parse_arguments()
        
        assert args.quantize_int8 is True
        assert args.run_smoke_test is True


class TestResolveCheckpointDir:
    """Tests for resolve_checkpoint_dir function."""

    @patch("convert_to_onnx.get_platform_adapter")
    def test_resolve_checkpoint_dir_success(self, mock_get_adapter):
        """Test successful checkpoint directory resolution."""
        from convert_to_onnx import resolve_checkpoint_dir
        
        mock_adapter = MagicMock()
        mock_resolver = MagicMock()
        mock_resolver.resolve_checkpoint_dir.return_value = Path("/resolved/checkpoint")
        mock_adapter.get_checkpoint_resolver.return_value = mock_resolver
        mock_get_adapter.return_value = mock_adapter
        
        result = resolve_checkpoint_dir("/input/checkpoint")
        
        assert result == Path("/resolved/checkpoint")
        mock_resolver.resolve_checkpoint_dir.assert_called_once_with("/input/checkpoint")

    @patch("convert_to_onnx.get_platform_adapter")
    def test_resolve_checkpoint_dir_failure(self, mock_get_adapter):
        """Test checkpoint resolution failure."""
        from convert_to_onnx import resolve_checkpoint_dir
        
        mock_adapter = MagicMock()
        mock_resolver = MagicMock()
        mock_resolver.resolve_checkpoint_dir.side_effect = FileNotFoundError("Checkpoint not found")
        mock_adapter.get_checkpoint_resolver.return_value = mock_resolver
        mock_get_adapter.return_value = mock_adapter
        
        with pytest.raises(FileNotFoundError):
            resolve_checkpoint_dir("/nonexistent/checkpoint")


class TestDynamicAxesFor:
    """Tests for _dynamic_axes_for function."""

    def test_dynamic_axes_2d_tensor(self):
        """Test dynamic axes for 2D tensors."""
        from convert_to_onnx import _dynamic_axes_for
        
        inputs = {
            "input_ids": torch.zeros(2, 10),
            "attention_mask": torch.zeros(2, 10),
        }
        
        axes = _dynamic_axes_for(inputs)
        
        assert axes["input_ids"] == {0: "batch", 1: "seq"}
        assert axes["attention_mask"] == {0: "batch", 1: "seq"}
        assert axes["logits"] == {0: "batch", 1: "seq"}

    def test_dynamic_axes_1d_tensor(self):
        """Test dynamic axes for 1D tensors."""
        from convert_to_onnx import _dynamic_axes_for
        
        inputs = {
            "input_ids": torch.zeros(2),
        }
        
        axes = _dynamic_axes_for(inputs)
        
        assert axes["input_ids"] == {0: "batch"}
        assert axes["logits"] == {0: "batch", 1: "seq"}

    def test_dynamic_axes_mixed_tensors(self):
        """Test dynamic axes for mixed tensor dimensions."""
        from convert_to_onnx import _dynamic_axes_for
        
        inputs = {
            "input_ids": torch.zeros(2, 10),
            "attention_mask": torch.zeros(2, 10),
            "token_type_ids": torch.zeros(2),
        }
        
        axes = _dynamic_axes_for(inputs)
        
        assert axes["input_ids"] == {0: "batch", 1: "seq"}
        assert axes["attention_mask"] == {0: "batch", 1: "seq"}
        assert axes["token_type_ids"] == {0: "batch"}
        assert axes["logits"] == {0: "batch", 1: "seq"}


class TestConvertToOnnx:
    """Tests for convert_to_onnx function."""

    @patch("convert_to_onnx.AutoTokenizer")
    @patch("convert_to_onnx.AutoModelForTokenClassification")
    @patch("torch.onnx.export")
    def test_convert_to_onnx_fp32(self, mock_onnx_export, mock_model_class, mock_tokenizer_class):
        """Test FP32 ONNX conversion without quantization."""
        from convert_to_onnx import convert_to_onnx
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 32),
            "attention_mask": torch.zeros(1, 32),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        checkpoint_dir = Path("/checkpoint")
        output_dir = Path("/output")
        
        result = convert_to_onnx(
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            quantize_int8=False,
        )
        
        assert result == output_dir / "model.onnx"
        mock_model.eval.assert_called_once()
        mock_onnx_export.assert_called_once()
        assert mock_onnx_export.call_args[1]["opset_version"] == 18

    @patch("convert_to_onnx.AutoTokenizer")
    @patch("convert_to_onnx.AutoModelForTokenClassification")
    @patch("torch.onnx.export")
    @patch("onnxruntime.quantization.quantize_dynamic")
    def test_convert_to_onnx_int8(self, mock_quantize, mock_onnx_export, mock_model_class, mock_tokenizer_class):
        """Test ONNX conversion with int8 quantization."""
        from convert_to_onnx import convert_to_onnx
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 32),
            "attention_mask": torch.zeros(1, 32),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        checkpoint_dir = Path("/checkpoint")
        output_dir = Path("/output")
        
        result = convert_to_onnx(
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            quantize_int8=True,
        )
        
        assert result == output_dir / "model_int8.onnx"
        mock_quantize.assert_called_once()

    @patch("convert_to_onnx.AutoTokenizer")
    @patch("convert_to_onnx.AutoModelForTokenClassification")
    @patch("torch.onnx.export")
    def test_convert_to_onnx_export_failure(self, mock_onnx_export, mock_model_class, mock_tokenizer_class):
        """Test ONNX conversion when export fails."""
        from convert_to_onnx import convert_to_onnx
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 32),
            "attention_mask": torch.zeros(1, 32),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_onnx_export.side_effect = RuntimeError("Export failed")
        
        checkpoint_dir = Path("/checkpoint")
        output_dir = Path("/output")
        
        with pytest.raises(RuntimeError):
            convert_to_onnx(
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                quantize_int8=False,
            )

    @patch("convert_to_onnx.AutoTokenizer")
    @patch("convert_to_onnx.AutoModelForTokenClassification")
    @patch("torch.onnx.export")
    @patch("onnxruntime.quantization.quantize_dynamic")
    def test_convert_to_onnx_quantization_fallback(self, mock_quantize, mock_onnx_export, mock_model_class, mock_tokenizer_class):
        """Test fallback to FP32 when quantization fails."""
        from convert_to_onnx import convert_to_onnx
        import warnings
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 32),
            "attention_mask": torch.zeros(1, 32),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_quantize.side_effect = ImportError("onnxruntime not available")
        
        checkpoint_dir = Path("/checkpoint")
        output_dir = Path("/output")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = convert_to_onnx(
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                quantize_int8=True,
            )
            
            assert result == output_dir / "model.onnx"
            assert len(w) > 0
            assert "quantization" in str(w[0].message).lower()


class TestRunSmokeTest:
    """Tests for run_smoke_test function."""

    @patch("convert_to_onnx.AutoTokenizer")
    def test_run_smoke_test_success(self, mock_tokenizer_class):
        """Test successful smoke test execution."""
        import sys
        from convert_to_onnx import run_smoke_test
        
        mock_tokenizer = MagicMock()
        # Tokenizer returns numpy arrays when return_tensors="np"
        mock_np_array = MagicMock()
        mock_np_array.astype = MagicMock(return_value=mock_np_array)
        mock_tokenizer.return_value = {
            "input_ids": mock_np_array,
            "attention_mask": mock_np_array,
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_np = MagicMock()
        mock_np.int64 = int
        mock_np.array = MagicMock(return_value=mock_np_array)
        
        mock_ort = MagicMock()
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        mock_session.run.return_value = [mock_np_array]
        mock_ort.InferenceSession = MagicMock(return_value=mock_session)
        
        onnx_path = Path("/model.onnx")
        checkpoint_dir = Path("/checkpoint")
        
        with patch.dict(sys.modules, {"numpy": mock_np, "onnxruntime": mock_ort}):
            run_smoke_test(onnx_path, checkpoint_dir)
        
        assert mock_ort.InferenceSession.called
        mock_session.run.assert_called_once()

    @patch("convert_to_onnx.AutoTokenizer")
    def test_run_smoke_test_import_error(self, mock_tokenizer_class):
        """Test smoke test when onnxruntime is not available."""
        import sys
        import builtins
        from convert_to_onnx import run_smoke_test
        import warnings
        
        # Save original __import__
        original_import = builtins.__import__
        
        # Create a custom import that raises ImportError for onnxruntime
        def mock_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("onnxruntime not available")
            # For all other imports, use the real import
            return original_import(name, *args, **kwargs)
        
        onnx_path = Path("/model.onnx")
        checkpoint_dir = Path("/checkpoint")
        
        try:
            # Patch __import__ to raise ImportError for onnxruntime
            builtins.__import__ = mock_import
            
            # Remove only onnxruntime from sys.modules so the import is attempted
            # Don't remove numpy to avoid reload warnings
            original_ort = sys.modules.pop("onnxruntime", None)
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                run_smoke_test(onnx_path, checkpoint_dir)
                
                # Find the warning about onnxruntime/numpy not being available
                # (may be mixed with numpy reload warnings)
                smoke_test_warnings = [
                    warning for warning in w
                    if "onnxruntime" in str(warning.message).lower()
                    or ("numpy" in str(warning.message).lower() and "not available" in str(warning.message).lower())
                    or "smoke test skipped" in str(warning.message).lower()
                ]
                
                assert len(smoke_test_warnings) > 0, f"No onnxruntime warning found. Warnings: {[str(w.message) for w in w]}"
                assert "onnxruntime" in str(smoke_test_warnings[0].message).lower() or "not available" in str(smoke_test_warnings[0].message).lower()
        finally:
            # Restore original import and modules
            builtins.__import__ = original_import
            if original_ort is not None:
                sys.modules["onnxruntime"] = original_ort

    @patch("convert_to_onnx.AutoTokenizer")
    def test_run_smoke_test_no_output(self, mock_tokenizer_class):
        """Test smoke test failure when no output is returned."""
        import sys
        from convert_to_onnx import run_smoke_test
        
        mock_tokenizer = MagicMock()
        # Tokenizer returns numpy arrays when return_tensors="np"
        mock_np_array = MagicMock()
        mock_np_array.astype = MagicMock(return_value=mock_np_array)
        mock_tokenizer.return_value = {
            "input_ids": mock_np_array,
            "attention_mask": mock_np_array,
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_np = MagicMock()
        mock_np.int64 = int
        mock_np.array = MagicMock(return_value=mock_np_array)
        
        mock_ort = MagicMock()
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        mock_session.run.return_value = []
        mock_ort.InferenceSession = MagicMock(return_value=mock_session)
        
        onnx_path = Path("/model.onnx")
        checkpoint_dir = Path("/checkpoint")
        
        with patch.dict(sys.modules, {"numpy": mock_np, "onnxruntime": mock_ort}):
            with pytest.raises(RuntimeError, match="no logits returned"):
                run_smoke_test(onnx_path, checkpoint_dir)


class TestMain:
    """Tests for main function."""

    @patch("convert_to_onnx.run_smoke_test")
    @patch("convert_to_onnx.convert_to_onnx")
    @patch("convert_to_onnx.resolve_checkpoint_dir")
    @patch("convert_to_onnx.get_platform_adapter")
    @patch("sys.argv", ["convert_to_onnx.py", "--checkpoint-path", "/checkpoint",
                        "--config-dir", "/config", "--backbone", "distilbert",
                        "--output-dir", "/output"])
    def test_main_success(self, mock_get_adapter, mock_resolve, mock_convert, mock_smoke):
        """Test successful main execution."""
        from convert_to_onnx import main
        
        mock_resolve.return_value = Path("/resolved/checkpoint")
        mock_convert.return_value = Path("/output/model.onnx")
        
        mock_adapter = MagicMock()
        mock_output_resolver = MagicMock()
        mock_output_resolver.resolve_output_path.return_value = Path("/output")
        mock_output_resolver.ensure_output_directory.return_value = Path("/output")
        mock_adapter.get_output_path_resolver.return_value = mock_output_resolver
        mock_get_adapter.return_value = mock_adapter
        
        with patch("convert_to_onnx.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            main()
        
        mock_convert.assert_called_once()
        mock_smoke.assert_not_called()

    @patch("convert_to_onnx.run_smoke_test")
    @patch("convert_to_onnx.convert_to_onnx")
    @patch("convert_to_onnx.resolve_checkpoint_dir")
    @patch("convert_to_onnx.get_platform_adapter")
    @patch("sys.argv", ["convert_to_onnx.py", "--checkpoint-path", "/checkpoint",
                        "--config-dir", "/config", "--backbone", "distilbert",
                        "--output-dir", "/output", "--run-smoke-test"])
    def test_main_with_smoke_test(self, mock_get_adapter, mock_resolve, mock_convert, mock_smoke):
        """Test main execution with smoke test."""
        from convert_to_onnx import main
        
        mock_resolve.return_value = Path("/resolved/checkpoint")
        mock_convert.return_value = Path("/output/model.onnx")
        
        mock_adapter = MagicMock()
        mock_output_resolver = MagicMock()
        mock_output_resolver.resolve_output_path.return_value = Path("/output")
        mock_output_resolver.ensure_output_directory.return_value = Path("/output")
        mock_adapter.get_output_path_resolver.return_value = mock_output_resolver
        mock_get_adapter.return_value = mock_adapter
        
        with patch("convert_to_onnx.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            main()
        
        mock_smoke.assert_called_once()

    @patch("sys.argv", ["convert_to_onnx.py", "--checkpoint-path", "/checkpoint",
                        "--config-dir", "/nonexistent", "--backbone", "distilbert",
                        "--output-dir", "/output"])
    def test_main_config_dir_not_found(self):
        """Test main execution when config directory doesn't exist."""
        from convert_to_onnx import main
        
        with patch("convert_to_onnx.Path") as mock_path_class:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = False
            mock_path_class.return_value = mock_path_instance
            
            with pytest.raises(FileNotFoundError, match="Config directory not found"):
                main()

