"""Component tests for conversion using config options."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from conversion.orchestration import execute_conversion
from config.conversion import load_conversion_config
from config.loader import ExperimentConfig


class TestConversionConfig:
    """Test conversion logic using config options."""

    @patch("conversion.orchestration.subprocess.Popen")
    @patch("conversion.orchestration.build_output_path")
    @patch("conversion.orchestration._find_onnx_model")
    def test_opset_version_passed_to_subprocess(
        self,
        mock_find_onnx,
        mock_build_path,
        mock_popen,
        tmp_path,
        resolved_conversion_config,
    ):
        """Test that opset_version from config is passed to conversion subprocess."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        output_dir = root_dir / "conversion" / "test"
        mock_build_path.return_value = output_dir
        mock_find_onnx.return_value = output_dir / "model.onnx"
        mock_process = Mock()
        mock_process.stdout = iter([])  # Empty iterator for stdout
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        # Test with custom opset_version
        conversion_config = resolved_conversion_config.copy()
        conversion_config["onnx"]["opset_version"] = 19
        
        with patch("conversion.orchestration.detect_platform") as mock_detect, \
             patch("conversion.orchestration.create_naming_context") as mock_create_context, \
             patch("conversion.orchestration.build_mlflow_run_name") as mock_build_name, \
             patch("conversion.orchestration.build_mlflow_tags") as mock_build_tags, \
             patch("conversion.orchestration.mlflow") as mock_mlflow, \
             patch("conversion.orchestration.MlflowClient") as mock_client_class, \
             patch("conversion.orchestration.update_mlflow_index"), \
             patch("conversion.orchestration.build_mlflow_run_key"), \
             patch("conversion.orchestration.build_mlflow_run_key_hash"), \
             patch("conversion.orchestration.load_conversion_config") as mock_load_config:
            
            # Setup mocks
            mock_detect.return_value = "local"
            mock_create_context.return_value = Mock()
            mock_build_name.return_value = "test_run_name"
            mock_build_tags.return_value = {}
            mock_mlflow.get_tracking_uri.return_value = None
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_experiment = Mock()
            mock_experiment.experiment_id = "test_exp_id"
            mock_client.get_experiment_by_name.return_value = mock_experiment
            mock_client.create_run.return_value = Mock(info=Mock(run_id="test_run_id"))
            mock_load_config.return_value = conversion_config
            
            # Create a mock parent training output dir
            parent_training_dir = tmp_path / "final_training" / "local" / "distilbert" / "spec_test_exec_test" / "v1"
            parent_training_dir.mkdir(parents=True)
            (parent_training_dir / "checkpoint").mkdir()
            
            try:
                execute_conversion(
                    root_dir=root_dir,
                    config_dir=config_dir,
                    parent_training_output_dir=parent_training_dir,
                    parent_spec_fp="test_spec_fp",
                    parent_exec_fp="test_exec_fp",
                    experiment_config=Mock(spec=ExperimentConfig),
                    conversion_experiment_name="test_conversion",
                    platform="local",
                )
            except Exception:
                pass  # May fail due to missing dependencies, but we check the args
            
            # Verify opset_version was passed to subprocess
            assert mock_popen.called
            call_args = mock_popen.call_args[0][0]  # First positional arg is the command list
            assert "--opset-version" in call_args
            opset_idx = call_args.index("--opset-version")
            assert call_args[opset_idx + 1] == "19"

    @patch("conversion.orchestration.subprocess.Popen")
    @patch("conversion.orchestration.build_output_path")
    @patch("conversion.orchestration._find_onnx_model")
    def test_quantization_int8_adds_flag(
        self,
        mock_find_onnx,
        mock_build_path,
        mock_popen,
        tmp_path,
        resolved_conversion_config,
    ):
        """Test that quantization=int8 adds --quantize-int8 flag."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        output_dir = root_dir / "conversion" / "test"
        mock_build_path.return_value = output_dir
        mock_find_onnx.return_value = output_dir / "model_int8.onnx"
        mock_process = Mock()
        mock_process.stdout = iter([])  # Empty iterator for stdout
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        # Test with quantization=int8
        conversion_config = resolved_conversion_config.copy()
        conversion_config["onnx"]["quantization"] = "int8"
        
        with patch("conversion.orchestration.detect_platform") as mock_detect, \
             patch("conversion.orchestration.create_naming_context") as mock_create_context, \
             patch("conversion.orchestration.build_mlflow_run_name") as mock_build_name, \
             patch("conversion.orchestration.build_mlflow_tags") as mock_build_tags, \
             patch("conversion.orchestration.mlflow") as mock_mlflow, \
             patch("conversion.orchestration.MlflowClient") as mock_client_class, \
             patch("conversion.orchestration.update_mlflow_index"), \
             patch("conversion.orchestration.build_mlflow_run_key"), \
             patch("conversion.orchestration.build_mlflow_run_key_hash"), \
             patch("conversion.orchestration.load_conversion_config") as mock_load_config:
            
            mock_detect.return_value = "local"
            mock_create_context.return_value = Mock()
            mock_build_name.return_value = "test_run_name"
            mock_build_tags.return_value = {}
            mock_mlflow.get_tracking_uri.return_value = None
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_experiment = Mock()
            mock_experiment.experiment_id = "test_exp_id"
            mock_client.get_experiment_by_name.return_value = mock_experiment
            mock_client.create_run.return_value = Mock(info=Mock(run_id="test_run_id"))
            mock_load_config.return_value = conversion_config
            
            # Create a mock parent training output dir
            parent_training_dir = tmp_path / "final_training" / "local" / "distilbert" / "spec_test_exec_test" / "v1"
            parent_training_dir.mkdir(parents=True)
            (parent_training_dir / "checkpoint").mkdir()
            
            try:
                execute_conversion(
                    root_dir=root_dir,
                    config_dir=config_dir,
                    parent_training_output_dir=parent_training_dir,
                    parent_spec_fp="test_spec_fp",
                    parent_exec_fp="test_exec_fp",
                    experiment_config=Mock(spec=ExperimentConfig),
                    conversion_experiment_name="test_conversion",
                    platform="local",
                )
            except Exception:
                pass
            
            # Verify --quantize-int8 flag was added
            assert mock_popen.called
            call_args = mock_popen.call_args[0][0]
            assert "--quantize-int8" in call_args

    @patch("conversion.orchestration.subprocess.Popen")
    @patch("conversion.orchestration.build_output_path")
    @patch("conversion.orchestration._find_onnx_model")
    def test_quantization_none_no_flag(
        self,
        mock_find_onnx,
        mock_build_path,
        mock_popen,
        tmp_path,
        resolved_conversion_config,
    ):
        """Test that quantization=none does not add quantization flag."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        output_dir = root_dir / "conversion" / "test"
        mock_build_path.return_value = output_dir
        mock_find_onnx.return_value = output_dir / "model.onnx"
        mock_process = Mock()
        mock_process.stdout = iter([])  # Empty iterator for stdout
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        # Test with quantization=none
        conversion_config = resolved_conversion_config.copy()
        conversion_config["onnx"]["quantization"] = "none"
        
        with patch("conversion.orchestration.detect_platform") as mock_detect, \
             patch("conversion.orchestration.create_naming_context") as mock_create_context, \
             patch("conversion.orchestration.build_mlflow_run_name") as mock_build_name, \
             patch("conversion.orchestration.build_mlflow_tags") as mock_build_tags, \
             patch("conversion.orchestration.mlflow") as mock_mlflow, \
             patch("conversion.orchestration.MlflowClient") as mock_client_class, \
             patch("conversion.orchestration.update_mlflow_index"), \
             patch("conversion.orchestration.build_mlflow_run_key"), \
             patch("conversion.orchestration.build_mlflow_run_key_hash"), \
             patch("conversion.orchestration.load_conversion_config") as mock_load_config:
            
            mock_detect.return_value = "local"
            mock_create_context.return_value = Mock()
            mock_build_name.return_value = "test_run_name"
            mock_build_tags.return_value = {}
            mock_mlflow.get_tracking_uri.return_value = None
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_experiment = Mock()
            mock_experiment.experiment_id = "test_exp_id"
            mock_client.get_experiment_by_name.return_value = mock_experiment
            mock_client.create_run.return_value = Mock(info=Mock(run_id="test_run_id"))
            mock_load_config.return_value = conversion_config
            
            # Create a mock parent training output dir
            parent_training_dir = tmp_path / "final_training" / "local" / "distilbert" / "spec_test_exec_test" / "v1"
            parent_training_dir.mkdir(parents=True)
            (parent_training_dir / "checkpoint").mkdir()
            
            try:
                execute_conversion(
                    root_dir=root_dir,
                    config_dir=config_dir,
                    parent_training_output_dir=parent_training_dir,
                    parent_spec_fp="test_spec_fp",
                    parent_exec_fp="test_exec_fp",
                    experiment_config=Mock(spec=ExperimentConfig),
                    conversion_experiment_name="test_conversion",
                    platform="local",
                )
            except Exception:
                pass
            
            # Verify --quantize-int8 flag was NOT added
            assert mock_popen.called
            call_args = mock_popen.call_args[0][0]
            assert "--quantize-int8" not in call_args

    @patch("conversion.orchestration.subprocess.Popen")
    @patch("conversion.orchestration.build_output_path")
    @patch("conversion.orchestration._find_onnx_model")
    def test_run_smoke_test_true_adds_flag(
        self,
        mock_find_onnx,
        mock_build_path,
        mock_popen,
        tmp_path,
        resolved_conversion_config,
    ):
        """Test that run_smoke_test=True adds --run-smoke-test flag."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        output_dir = root_dir / "conversion" / "test"
        mock_build_path.return_value = output_dir
        mock_find_onnx.return_value = output_dir / "model.onnx"
        mock_process = Mock()
        mock_process.stdout = iter([])  # Empty iterator for stdout
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        # Test with run_smoke_test=True
        conversion_config = resolved_conversion_config.copy()
        conversion_config["onnx"]["run_smoke_test"] = True
        
        with patch("conversion.orchestration.detect_platform") as mock_detect, \
             patch("conversion.orchestration.create_naming_context") as mock_create_context, \
             patch("conversion.orchestration.build_mlflow_run_name") as mock_build_name, \
             patch("conversion.orchestration.build_mlflow_tags") as mock_build_tags, \
             patch("conversion.orchestration.mlflow") as mock_mlflow, \
             patch("conversion.orchestration.MlflowClient") as mock_client_class, \
             patch("conversion.orchestration.update_mlflow_index"), \
             patch("conversion.orchestration.build_mlflow_run_key"), \
             patch("conversion.orchestration.build_mlflow_run_key_hash"), \
             patch("conversion.orchestration.load_conversion_config") as mock_load_config:
            
            mock_detect.return_value = "local"
            mock_create_context.return_value = Mock()
            mock_build_name.return_value = "test_run_name"
            mock_build_tags.return_value = {}
            mock_mlflow.get_tracking_uri.return_value = None
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_experiment = Mock()
            mock_experiment.experiment_id = "test_exp_id"
            mock_client.get_experiment_by_name.return_value = mock_experiment
            mock_client.create_run.return_value = Mock(info=Mock(run_id="test_run_id"))
            mock_load_config.return_value = conversion_config
            
            # Create a mock parent training output dir
            parent_training_dir = tmp_path / "final_training" / "local" / "distilbert" / "spec_test_exec_test" / "v1"
            parent_training_dir.mkdir(parents=True)
            (parent_training_dir / "checkpoint").mkdir()
            
            try:
                execute_conversion(
                    root_dir=root_dir,
                    config_dir=config_dir,
                    parent_training_output_dir=parent_training_dir,
                    parent_spec_fp="test_spec_fp",
                    parent_exec_fp="test_exec_fp",
                    experiment_config=Mock(spec=ExperimentConfig),
                    conversion_experiment_name="test_conversion",
                    platform="local",
                )
            except Exception:
                pass
            
            # Verify --run-smoke-test flag was added
            assert mock_popen.called
            call_args = mock_popen.call_args[0][0]
            assert "--run-smoke-test" in call_args

    @patch("conversion.orchestration.subprocess.Popen")
    @patch("conversion.orchestration.build_output_path")
    @patch("conversion.orchestration._find_onnx_model")
    def test_run_smoke_test_false_no_flag(
        self,
        mock_find_onnx,
        mock_build_path,
        mock_popen,
        tmp_path,
        resolved_conversion_config,
    ):
        """Test that run_smoke_test=False does not add smoke test flag."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        output_dir = root_dir / "conversion" / "test"
        mock_build_path.return_value = output_dir
        mock_find_onnx.return_value = output_dir / "model.onnx"
        mock_process = Mock()
        mock_process.stdout = iter([])  # Empty iterator for stdout
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        # Test with run_smoke_test=False
        conversion_config = resolved_conversion_config.copy()
        conversion_config["onnx"]["run_smoke_test"] = False
        
        with patch("conversion.orchestration.detect_platform") as mock_detect, \
             patch("conversion.orchestration.create_naming_context") as mock_create_context, \
             patch("conversion.orchestration.build_mlflow_run_name") as mock_build_name, \
             patch("conversion.orchestration.build_mlflow_tags") as mock_build_tags, \
             patch("conversion.orchestration.mlflow") as mock_mlflow, \
             patch("conversion.orchestration.MlflowClient") as mock_client_class, \
             patch("conversion.orchestration.update_mlflow_index"), \
             patch("conversion.orchestration.build_mlflow_run_key"), \
             patch("conversion.orchestration.build_mlflow_run_key_hash"), \
             patch("conversion.orchestration.load_conversion_config") as mock_load_config:
            
            mock_detect.return_value = "local"
            mock_create_context.return_value = Mock()
            mock_build_name.return_value = "test_run_name"
            mock_build_tags.return_value = {}
            mock_mlflow.get_tracking_uri.return_value = None
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_experiment = Mock()
            mock_experiment.experiment_id = "test_exp_id"
            mock_client.get_experiment_by_name.return_value = mock_experiment
            mock_client.create_run.return_value = Mock(info=Mock(run_id="test_run_id"))
            mock_load_config.return_value = conversion_config
            
            # Create a mock parent training output dir
            parent_training_dir = tmp_path / "final_training" / "local" / "distilbert" / "spec_test_exec_test" / "v1"
            parent_training_dir.mkdir(parents=True)
            (parent_training_dir / "checkpoint").mkdir()
            
            try:
                execute_conversion(
                    root_dir=root_dir,
                    config_dir=config_dir,
                    parent_training_output_dir=parent_training_dir,
                    parent_spec_fp="test_spec_fp",
                    parent_exec_fp="test_exec_fp",
                    experiment_config=Mock(spec=ExperimentConfig),
                    conversion_experiment_name="test_conversion",
                    platform="local",
                )
            except Exception:
                pass
            
            # Verify --run-smoke-test flag was NOT added
            assert mock_popen.called
            call_args = mock_popen.call_args[0][0]
            assert "--run-smoke-test" not in call_args

    @patch("conversion.orchestration.subprocess.Popen")
    @patch("conversion.orchestration.build_output_path")
    def test_filename_pattern_used_in_find_onnx_model(
        self,
        mock_build_path,
        mock_popen,
        tmp_path,
        resolved_conversion_config,
    ):
        """Test that filename_pattern from config is used in _find_onnx_model."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        output_dir = root_dir / "conversion" / "test"
        output_dir.mkdir(parents=True)
        mock_build_path.return_value = output_dir
        mock_process = Mock()
        mock_process.stdout = iter([])  # Empty iterator for stdout
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        # Test with custom filename_pattern
        conversion_config = resolved_conversion_config.copy()
        conversion_config["output"]["filename_pattern"] = "custom_{quantization}_model.onnx"
        
        with patch("conversion.orchestration.detect_platform") as mock_detect, \
             patch("conversion.orchestration.create_naming_context") as mock_create_context, \
             patch("conversion.orchestration.build_mlflow_run_name") as mock_build_name, \
             patch("conversion.orchestration.build_mlflow_tags") as mock_build_tags, \
             patch("conversion.orchestration.mlflow") as mock_mlflow, \
             patch("conversion.orchestration.MlflowClient") as mock_client_class, \
             patch("conversion.orchestration.update_mlflow_index"), \
             patch("conversion.orchestration.build_mlflow_run_key"), \
             patch("conversion.orchestration.build_mlflow_run_key_hash"), \
             patch("conversion.orchestration._find_onnx_model") as mock_find_onnx, \
             patch("conversion.orchestration.load_conversion_config") as mock_load_config:
            
            mock_detect.return_value = "local"
            mock_create_context.return_value = Mock()
            mock_build_name.return_value = "test_run_name"
            mock_build_tags.return_value = {}
            mock_mlflow.get_tracking_uri.return_value = None
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_experiment = Mock()
            mock_experiment.experiment_id = "test_exp_id"
            mock_client.get_experiment_by_name.return_value = mock_experiment
            mock_client.create_run.return_value = Mock(info=Mock(run_id="test_run_id"))
            mock_find_onnx.return_value = output_dir / "custom_fp32_model.onnx"
            mock_load_config.return_value = conversion_config
            
            # Create a mock parent training output dir
            parent_training_dir = tmp_path / "final_training" / "local" / "distilbert" / "spec_test_exec_test" / "v1"
            parent_training_dir.mkdir(parents=True)
            (parent_training_dir / "checkpoint").mkdir()
            
            try:
                execute_conversion(
                    root_dir=root_dir,
                    config_dir=config_dir,
                    parent_training_output_dir=parent_training_dir,
                    parent_spec_fp="test_spec_fp",
                    parent_exec_fp="test_exec_fp",
                    experiment_config=Mock(spec=ExperimentConfig),
                    conversion_experiment_name="test_conversion",
                    platform="local",
                )
            except Exception:
                pass
            
            # Verify _find_onnx_model was called with filename_pattern
            assert mock_find_onnx.called
            # _find_onnx_model is called with positional args: (output_dir, quantization, filename_pattern)
            call_args = mock_find_onnx.call_args[0]  # Positional arguments
            assert len(call_args) >= 3
            assert call_args[2] == "custom_{quantization}_model.onnx"  # filename_pattern is 3rd arg

