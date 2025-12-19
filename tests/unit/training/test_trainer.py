"""Tests for training loop and model training functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import pytest
import torch


class TestPrepareDataLoaders:
    """Tests for prepare_data_loaders function."""

    @patch("training.trainer.ResumeNERDataset")
    @patch("training.trainer.DataLoader")
    @patch("training.trainer.DataCollatorForTokenClassification")
    def test_prepare_data_loaders_basic(self, mock_collator_class, mock_loader_class, mock_dataset_class):
        """Test basic data loader preparation."""
        from training.trainer import prepare_data_loaders
        
        config = {
            "model": {
                "preprocessing": {"max_length": 128},
                "backbone": "distilbert-base-uncased",
            },
            "training": {"batch_size": 8},
        }
        dataset = {
            "train": [{"text": "test1", "annotations": []}, {"text": "test2", "annotations": []}],
            "validation": [{"text": "val1", "annotations": []}],
        }
        mock_tokenizer = MagicMock()
        label2id = {"O": 0, "PERSON": 1}
        
        mock_train_ds = MagicMock()
        mock_val_ds = MagicMock()
        mock_dataset_class.side_effect = [mock_train_ds, mock_val_ds]
        
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_loader_class.side_effect = [mock_train_loader, mock_val_loader]
        
        train_loader, val_loader = prepare_data_loaders(
            config, dataset, mock_tokenizer, label2id
        )
        
        assert train_loader == mock_train_loader
        assert val_loader == mock_val_loader
        assert mock_dataset_class.call_count == 2

    @patch("training.trainer.ResumeNERDataset")
    @patch("training.trainer.DataLoader")
    @patch("training.trainer.DataCollatorForTokenClassification")
    def test_prepare_data_loaders_with_indices(self, mock_collator_class, mock_loader_class, mock_dataset_class):
        """Test data loader preparation with fold indices."""
        from training.trainer import prepare_data_loaders
        
        config = {
            "model": {
                "preprocessing": {"max_length": 128},
                "backbone": "distilbert-base-uncased",
            },
            "training": {"batch_size": 8},
        }
        dataset = {
            "train": [
                {"text": "test1", "annotations": []},
                {"text": "test2", "annotations": []},
                {"text": "test3", "annotations": []},
            ],
            "validation": [{"text": "val1", "annotations": []}],
        }
        mock_tokenizer = MagicMock()
        label2id = {"O": 0, "PERSON": 1}
        train_indices = [0, 2]
        val_indices = [0]
        
        mock_train_ds = MagicMock()
        mock_val_ds = MagicMock()
        mock_dataset_class.side_effect = [mock_train_ds, mock_val_ds]
        
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_loader_class.side_effect = [mock_train_loader, mock_val_loader]
        
        train_loader, val_loader = prepare_data_loaders(
            config, dataset, mock_tokenizer, label2id,
            train_indices=train_indices, val_indices=val_indices
        )
        
        assert train_loader == mock_train_loader
        assert val_loader == mock_val_loader

    @patch("training.trainer.ResumeNERDataset")
    @patch("training.trainer.DataLoader")
    @patch("training.trainer.DataCollatorForTokenClassification")
    def test_prepare_data_loaders_use_all_data(self, mock_collator_class, mock_loader_class, mock_dataset_class):
        """Test data loader preparation with use_all_data=True."""
        from training.trainer import prepare_data_loaders
        
        config = {
            "model": {
                "preprocessing": {"max_length": 128},
                "backbone": "distilbert-base-uncased",
            },
            "training": {"batch_size": 8},
        }
        dataset = {
            "train": [{"text": "test1", "annotations": []}],
            "validation": [{"text": "val1", "annotations": []}],
        }
        mock_tokenizer = MagicMock()
        label2id = {"O": 0, "PERSON": 1}
        
        mock_train_ds = MagicMock()
        mock_dataset_class.return_value = mock_train_ds
        
        mock_train_loader = MagicMock()
        mock_loader_class.return_value = mock_train_loader
        
        train_loader, val_loader = prepare_data_loaders(
            config, dataset, mock_tokenizer, label2id, use_all_data=True
        )
        
        assert train_loader == mock_train_loader
        assert val_loader is None
        assert mock_dataset_class.call_count == 1

    @patch("training.trainer.ResumeNERDataset")
    @patch("training.trainer.DataLoader")
    @patch("training.trainer.DataCollatorForTokenClassification")
    def test_prepare_data_loaders_deberta_batch_size_cap(self, mock_collator_class, mock_loader_class, mock_dataset_class):
        """Test that DeBERTa batch size is capped."""
        from training.trainer import prepare_data_loaders
        
        config = {
            "model": {
                "preprocessing": {"max_length": 128},
                "backbone": "microsoft/deberta-v3-base",
            },
            "training": {"batch_size": 16},
        }
        dataset = {
            "train": [{"text": "test1", "annotations": []}],
            "validation": [{"text": "val1", "annotations": []}],
        }
        mock_tokenizer = MagicMock()
        label2id = {"O": 0, "PERSON": 1}
        
        mock_train_ds = MagicMock()
        mock_val_ds = MagicMock()
        mock_dataset_class.side_effect = [mock_train_ds, mock_val_ds]
        
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_loader_class.side_effect = [mock_train_loader, mock_val_loader]
        
        train_loader, val_loader = prepare_data_loaders(
            config, dataset, mock_tokenizer, label2id
        )
        
        call_args = mock_loader_class.call_args_list[0]
        assert call_args[1]["batch_size"] == 8

    @patch("training.trainer.ResumeNERDataset")
    @patch("training.trainer.DataLoader")
    @patch("training.trainer.DataCollatorForTokenClassification")
    def test_prepare_data_loaders_val_split_fallback(self, mock_collator_class, mock_loader_class, mock_dataset_class):
        """Test validation split fallback when no validation data."""
        from training.trainer import prepare_data_loaders
        
        config = {
            "model": {
                "preprocessing": {"max_length": 128},
                "backbone": "distilbert-base-uncased",
            },
            "training": {"batch_size": 8, "val_split_divisor": 10},
        }
        dataset = {
            "train": [{"text": f"test{i}", "annotations": []} for i in range(20)],
        }
        mock_tokenizer = MagicMock()
        label2id = {"O": 0, "PERSON": 1}
        
        mock_train_ds = MagicMock()
        mock_val_ds = MagicMock()
        mock_dataset_class.side_effect = [mock_train_ds, mock_val_ds]
        
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_loader_class.side_effect = [mock_train_loader, mock_val_loader]
        
        train_loader, val_loader = prepare_data_loaders(
            config, dataset, mock_tokenizer, label2id
        )
        
        assert train_loader == mock_train_loader
        assert val_loader == mock_val_loader


class TestCreateOptimizerAndScheduler:
    """Tests for create_optimizer_and_scheduler function."""

    @patch("training.trainer.get_linear_schedule_with_warmup")
    def test_create_optimizer_and_scheduler_defaults(self, mock_scheduler_fn):
        """Test optimizer and scheduler creation with defaults."""
        from training.trainer import create_optimizer_and_scheduler
        
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
        config = {
            "training": {},
        }
        total_steps = 100
        
        mock_scheduler = MagicMock()
        mock_scheduler_fn.return_value = mock_scheduler
        
        optimizer, scheduler, max_grad_norm = create_optimizer_and_scheduler(
            mock_model, config, total_steps
        )
        
        assert isinstance(optimizer, torch.optim.AdamW)
        assert scheduler == mock_scheduler
        assert max_grad_norm == 1.0

    @patch("training.trainer.get_linear_schedule_with_warmup")
    def test_create_optimizer_and_scheduler_custom(self, mock_scheduler_fn):
        """Test optimizer and scheduler creation with custom values."""
        from training.trainer import create_optimizer_and_scheduler
        
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
        config = {
            "training": {
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "warmup_steps": 10,
                "max_grad_norm": 2.0,
                "warmup_steps_divisor": 5,
            },
        }
        total_steps = 100
        
        mock_scheduler = MagicMock()
        mock_scheduler_fn.return_value = mock_scheduler
        
        optimizer, scheduler, max_grad_norm = create_optimizer_and_scheduler(
            mock_model, config, total_steps
        )
        
        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[0]["weight_decay"] == 0.01
        assert max_grad_norm == 2.0
        mock_scheduler_fn.assert_called_once()

    @patch("training.trainer.get_linear_schedule_with_warmup")
    def test_create_optimizer_and_scheduler_warmup_capped(self, mock_scheduler_fn):
        """Test that warmup steps are capped by max_warmup_steps."""
        from training.trainer import create_optimizer_and_scheduler
        
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
        config = {
            "training": {
                "warmup_steps": 50,
                "warmup_steps_divisor": 10,
            },
        }
        total_steps = 100
        
        mock_scheduler = MagicMock()
        mock_scheduler_fn.return_value = mock_scheduler
        
        optimizer, scheduler, max_grad_norm = create_optimizer_and_scheduler(
            mock_model, config, total_steps
        )
        
        call_kwargs = mock_scheduler_fn.call_args[1]
        assert call_kwargs["num_warmup_steps"] == 10


class TestRunTrainingLoop:
    """Tests for run_training_loop function."""

    def test_run_training_loop_basic(self):
        """Test basic training loop execution."""
        from training.trainer import run_training_loop
        
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        # Loss tensor needs requires_grad=True and grad_fn for backward() to work
        # Create from a computation to get a grad_fn
        x = torch.tensor(0.5, requires_grad=True)
        loss_tensor = x * 2.0
        mock_outputs.loss = loss_tensor
        # Make model callable and return outputs
        mock_model.__call__ = MagicMock(return_value=mock_outputs)
        
        mock_train_loader = MagicMock()
        mock_batch = {
            "input_ids": torch.zeros(2, 10),
            "labels": torch.zeros(2, 10),
        }
        mock_train_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        mock_train_loader.__len__ = Mock(return_value=1)
        
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()
        device = torch.device("cpu")
        
        run_training_loop(
            mock_model, mock_train_loader, mock_optimizer,
            mock_scheduler, epochs=1, max_grad_norm=1.0, device=device
        )
        
        mock_model.train.assert_called_once()
        mock_optimizer.step.assert_called_once()
        mock_scheduler.step.assert_called_once()
        mock_optimizer.zero_grad.assert_called_once()

    def test_run_training_loop_multiple_epochs(self):
        """Test training loop with multiple epochs."""
        from training.trainer import run_training_loop
        
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        # Loss tensor needs requires_grad=True and grad_fn for backward() to work
        # Create from a computation to get a grad_fn
        x = torch.tensor(0.5, requires_grad=True)
        loss_tensor = x * 2.0
        mock_outputs.loss = loss_tensor
        # Make model callable and return outputs
        mock_model.__call__ = MagicMock(return_value=mock_outputs)
        
        mock_train_loader = MagicMock()
        mock_batch = {
            "input_ids": torch.zeros(2, 10),
            "labels": torch.zeros(2, 10),
        }
        # __iter__ is called once per epoch, and each time it should return an iterator
        # that yields the batch. Use side_effect to return a new iterator each time.
        def make_iterator():
            return iter([mock_batch])
        mock_train_loader.__iter__ = Mock(side_effect=make_iterator)
        mock_train_loader.__len__ = Mock(return_value=1)
        
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()
        device = torch.device("cpu")
        
        run_training_loop(
            mock_model, mock_train_loader, mock_optimizer,
            mock_scheduler, epochs=3, max_grad_norm=1.0, device=device
        )
        
        assert mock_optimizer.step.call_count == 3
        assert mock_scheduler.step.call_count == 3


class TestSaveCheckpoint:
    """Tests for save_checkpoint function."""

    def test_save_checkpoint_success(self, temp_dir):
        """Test successful checkpoint saving."""
        from training.trainer import save_checkpoint
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        output_dir = temp_dir
        
        save_checkpoint(mock_model, mock_tokenizer, output_dir)
        
        checkpoint_path = output_dir / "checkpoint"
        assert checkpoint_path.exists()
        mock_model.save_pretrained.assert_called_once_with(checkpoint_path)
        mock_tokenizer.save_pretrained.assert_called_once_with(checkpoint_path)

    def test_save_checkpoint_creates_directory(self, temp_dir):
        """Test that checkpoint directory is created if it doesn't exist."""
        from training.trainer import save_checkpoint
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        output_dir = temp_dir / "outputs"
        
        save_checkpoint(mock_model, mock_tokenizer, output_dir)
        
        checkpoint_path = output_dir / "checkpoint"
        assert checkpoint_path.exists()


class TestTrainModel:
    """Tests for train_model function."""

    @patch("training.trainer.save_checkpoint")
    @patch("training.trainer.run_training_loop")
    @patch("training.trainer.create_optimizer_and_scheduler")
    @patch("training.trainer.prepare_data_loaders")
    @patch("training.trainer.create_model_and_tokenizer")
    @patch("training.trainer.build_label_list")
    @patch("training.trainer.evaluate_model")
    def test_train_model_basic(self, mock_evaluate, mock_build_labels, mock_create_model,
                               mock_prepare_loaders, mock_create_optimizer, mock_run_loop, mock_save):
        """Test basic model training."""
        from training.trainer import train_model
        
        config = {
            "data": {"schema": {"entity_types": ["PERSON", "ORG"]}},
            "model": {"backbone": "distilbert-base-uncased"},
            "training": {"epochs": 1, "batch_size": 8},
        }
        dataset = {
            "train": [{"text": "test", "annotations": []}],
            "validation": [{"text": "val", "annotations": []}],
        }
        output_dir = Path("/output")
        
        mock_build_labels.return_value = ["O", "PERSON", "ORG"]
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_device = torch.device("cpu")
        mock_create_model.return_value = (mock_model, mock_tokenizer, mock_device)
        
        mock_train_loader = MagicMock()
        mock_train_loader.__len__ = Mock(return_value=10)
        mock_val_loader = MagicMock()
        mock_prepare_loaders.return_value = (mock_train_loader, mock_val_loader)
        
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()
        mock_create_optimizer.return_value = (mock_optimizer, mock_scheduler, 1.0)
        
        mock_evaluate.return_value = {"f1": 0.85, "loss": 0.5}
        
        metrics = train_model(config, dataset, output_dir)
        
        assert metrics == {"f1": 0.85, "loss": 0.5}
        mock_evaluate.assert_called_once()
        mock_save.assert_called_once()

    @patch("training.trainer.save_checkpoint")
    @patch("training.trainer.run_training_loop")
    @patch("training.trainer.create_optimizer_and_scheduler")
    @patch("training.trainer.prepare_data_loaders")
    @patch("training.trainer.create_model_and_tokenizer")
    @patch("training.trainer.build_label_list")
    @patch("training.trainer.load_fold_splits")
    def test_train_model_with_fold_splits(self, mock_load_splits, mock_build_labels, mock_create_model,
                                         mock_prepare_loaders, mock_create_optimizer, mock_run_loop, mock_save):
        """Test model training with fold splits."""
        from training.trainer import train_model
        
        config = {
            "data": {"schema": {"entity_types": ["PERSON"]}},
            "model": {"backbone": "distilbert-base-uncased"},
            "training": {
                "epochs": 1,
                "batch_size": 8,
                "fold_idx": 0,
                "fold_splits_file": "/splits.json",
            },
        }
        dataset = {
            "train": [{"text": "test", "annotations": []}],
        }
        output_dir = Path("/output")
        
        mock_build_labels.return_value = ["O", "PERSON"]
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_device = torch.device("cpu")
        mock_create_model.return_value = (mock_model, mock_tokenizer, mock_device)
        
        mock_load_splits.return_value = ([(0, 1), (2, 3)], {})
        
        mock_train_loader = MagicMock()
        mock_train_loader.__len__ = Mock(return_value=10)
        mock_val_loader = MagicMock()
        mock_prepare_loaders.return_value = (mock_train_loader, mock_val_loader)
        
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()
        mock_create_optimizer.return_value = (mock_optimizer, mock_scheduler, 1.0)
        
        metrics = train_model(config, dataset, output_dir)
        
        mock_load_splits.assert_called_once()
        mock_prepare_loaders.assert_called_once()

    @patch("training.trainer.save_checkpoint")
    @patch("training.trainer.run_training_loop")
    @patch("training.trainer.create_optimizer_and_scheduler")
    @patch("training.trainer.prepare_data_loaders")
    @patch("training.trainer.create_model_and_tokenizer")
    @patch("training.trainer.build_label_list")
    def test_train_model_use_all_data(self, mock_build_labels, mock_create_model,
                                     mock_prepare_loaders, mock_create_optimizer, mock_run_loop, mock_save):
        """Test model training with use_all_data=True."""
        from training.trainer import train_model
        
        config = {
            "data": {"schema": {"entity_types": ["PERSON"]}},
            "model": {"backbone": "distilbert-base-uncased"},
            "training": {
                "epochs": 1,
                "batch_size": 8,
                "use_all_data": True,
            },
        }
        dataset = {
            "train": [{"text": "test", "annotations": []}],
        }
        output_dir = Path("/output")
        
        mock_build_labels.return_value = ["O", "PERSON"]
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_device = torch.device("cpu")
        mock_create_model.return_value = (mock_model, mock_tokenizer, mock_device)
        
        mock_train_loader = MagicMock()
        mock_train_loader.__len__ = Mock(return_value=10)
        mock_prepare_loaders.return_value = (mock_train_loader, None)
        
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()
        mock_create_optimizer.return_value = (mock_optimizer, mock_scheduler, 1.0)
        
        metrics = train_model(config, dataset, output_dir)
        
        assert "note" in metrics
        assert "No validation set" in metrics["note"]

    @patch("training.trainer.load_fold_splits")
    def test_train_model_invalid_fold_idx(self, mock_load_splits):
        """Test model training with invalid fold index."""
        from training.trainer import train_model
        
        config = {
            "data": {"schema": {"entity_types": ["PERSON"]}},
            "model": {"backbone": "distilbert-base-uncased"},
            "training": {
                "epochs": 1,
                "fold_idx": 5,
                "fold_splits_file": "/splits.json",
            },
        }
        dataset = {"train": []}
        output_dir = Path("/output")
        
        mock_load_splits.return_value = ([(0, 1), (2, 3)], {})
        
        with pytest.raises(ValueError, match="Fold index.*out of range"):
            train_model(config, dataset, output_dir)

