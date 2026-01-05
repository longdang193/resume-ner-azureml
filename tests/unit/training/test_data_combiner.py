"""Unit tests for data combiner module."""

import json
import tempfile
from pathlib import Path
import pytest

from training.data_combiner import combine_datasets


@pytest.fixture
def old_dataset(tmp_path):
    """Create a temporary old dataset."""
    dataset_dir = tmp_path / "old_dataset"
    dataset_dir.mkdir()
    
    train_data = [
        {"text": "Old sample 1", "entities": []},
        {"text": "Old sample 2", "entities": []},
    ]
    val_data = [
        {"text": "Old val 1", "entities": []},
    ]
    
    with open(dataset_dir / "train.json", "w") as f:
        json.dump(train_data, f)
    with open(dataset_dir / "validation.json", "w") as f:
        json.dump(val_data, f)
    
    return dataset_dir


@pytest.fixture
def new_dataset(tmp_path):
    """Create a temporary new dataset."""
    dataset_dir = tmp_path / "new_dataset"
    dataset_dir.mkdir()
    
    train_data = [
        {"text": "New sample 1", "entities": []},
        {"text": "New sample 2", "entities": []},
        {"text": "New sample 3", "entities": []},
    ]
    val_data = [
        {"text": "New val 1", "entities": []},
    ]
    
    with open(dataset_dir / "train.json", "w") as f:
        json.dump(train_data, f)
    with open(dataset_dir / "validation.json", "w") as f:
        json.dump(val_data, f)
    
    return dataset_dir


class TestCombineDatasets:
    """Test dataset combination strategies."""
    
    def test_new_only_strategy(self, new_dataset):
        """Test 'new_only' strategy uses only new dataset."""
        result = combine_datasets(
            old_dataset_path=None,
            new_dataset_path=new_dataset,
            strategy="new_only",
        )
        
        assert len(result["train"]) == 3
        assert len(result["validation"]) == 1
        assert all("New" in item["text"] for item in result["train"])
    
    def test_combined_strategy(self, old_dataset, new_dataset):
        """Test 'combined' strategy merges and shuffles datasets."""
        result = combine_datasets(
            old_dataset_path=old_dataset,
            new_dataset_path=new_dataset,
            strategy="combined",
            random_seed=42,
        )
        
        assert len(result["train"]) == 5  # 2 old + 3 new
        assert len(result["validation"]) == 2  # 1 old + 1 new
        
        # Check all samples are present
        train_texts = [item["text"] for item in result["train"]]
        assert "Old sample 1" in train_texts
        assert "New sample 1" in train_texts
    
    def test_append_strategy(self, old_dataset, new_dataset):
        """Test 'append' strategy appends without shuffling."""
        result = combine_datasets(
            old_dataset_path=old_dataset,
            new_dataset_path=new_dataset,
            strategy="append",
        )
        
        assert len(result["train"]) == 5
        assert len(result["validation"]) == 2
        
        # Check order: old samples first, then new
        train_texts = [item["text"] for item in result["train"]]
        assert train_texts[0] == "Old sample 1"
        assert train_texts[1] == "Old sample 2"
        assert train_texts[2] == "New sample 1"
    
    def test_invalid_strategy(self, old_dataset, new_dataset):
        """Test raises ValueError for invalid strategy."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            combine_datasets(
                old_dataset_path=old_dataset,
                new_dataset_path=new_dataset,
                strategy="invalid",
            )
    
    def test_combined_requires_old_dataset(self, new_dataset):
        """Test 'combined' strategy requires old dataset."""
        with pytest.raises(ValueError, match="Old dataset path required"):
            combine_datasets(
                old_dataset_path=None,
                new_dataset_path=new_dataset,
                strategy="combined",
            )
    
    def test_append_requires_old_dataset(self, new_dataset):
        """Test 'append' strategy requires old dataset."""
        with pytest.raises(ValueError, match="Old dataset path required"):
            combine_datasets(
                old_dataset_path=None,
                new_dataset_path=new_dataset,
                strategy="append",
            )
    
    def test_new_dataset_not_found(self, tmp_path):
        """Test raises FileNotFoundError when new dataset doesn't exist."""
        with pytest.raises(FileNotFoundError):
            combine_datasets(
                old_dataset_path=None,
                new_dataset_path=tmp_path / "nonexistent",
                strategy="new_only",
            )
    
    def test_old_dataset_not_found(self, tmp_path, new_dataset):
        """Test raises FileNotFoundError when old dataset doesn't exist."""
        with pytest.raises(FileNotFoundError):
            combine_datasets(
                old_dataset_path=tmp_path / "nonexistent",
                new_dataset_path=new_dataset,
                strategy="combined",
            )
    
    def test_no_validation_in_new_dataset(self, old_dataset, tmp_path):
        """Test handles new dataset without validation set."""
        new_dataset_no_val = tmp_path / "new_no_val"
        new_dataset_no_val.mkdir()
        
        train_data = [{"text": "New sample", "entities": []}]
        with open(new_dataset_no_val / "train.json", "w") as f:
            json.dump(train_data, f)
        
        result = combine_datasets(
            old_dataset_path=old_dataset,
            new_dataset_path=new_dataset_no_val,
            strategy="combined",
        )
        
        assert len(result["train"]) == 3  # 2 old + 1 new
        assert len(result["validation"]) == 1  # Only old validation
    
    def test_create_validation_split(self, tmp_path):
        """Test creates validation split when none exists."""
        old_dataset = tmp_path / "old"
        old_dataset.mkdir()
        with open(old_dataset / "train.json", "w") as f:
            json.dump([{"text": f"Sample {i}", "entities": []} for i in range(10)], f)
        
        new_dataset = tmp_path / "new"
        new_dataset.mkdir()
        with open(new_dataset / "train.json", "w") as f:
            json.dump([{"text": f"New {i}", "entities": []} for i in range(10)], f)
        
        result = combine_datasets(
            old_dataset_path=old_dataset,
            new_dataset_path=new_dataset,
            strategy="combined",
            validation_ratio=0.2,
            random_seed=42,
        )
        
        # Should have created validation split
        assert len(result["train"]) + len(result["validation"]) == 20
        assert len(result["validation"]) > 0

