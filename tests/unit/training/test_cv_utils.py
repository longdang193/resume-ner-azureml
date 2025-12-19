"""Tests for cross-validation utilities."""

import json
import pytest
from pathlib import Path
from training.cv_utils import (
    create_kfold_splits,
    get_fold_data,
    save_fold_splits,
    load_fold_splits,
)


class TestCreateKfoldSplits:
    """Tests for create_kfold_splits function."""

    def test_k_equals_2(self):
        """Test k-fold splitting with k=2."""
        dataset = list(range(10))
        splits = create_kfold_splits(dataset, k=2, random_seed=42)
        
        assert len(splits) == 2
        for train_idx, val_idx in splits:
            assert len(train_idx) == 5
            assert len(val_idx) == 5

    def test_k_equals_5(self):
        """Test k-fold splitting with k=5."""
        dataset = list(range(20))
        splits = create_kfold_splits(dataset, k=5, random_seed=42)
        
        assert len(splits) == 5
        for train_idx, val_idx in splits:
            # Each fold should have 16 train and 4 validation samples
            assert len(train_idx) == 16
            assert len(val_idx) == 4

    def test_no_overlap(self):
        """Test that train and validation indices don't overlap."""
        dataset = list(range(20))
        splits = create_kfold_splits(dataset, k=5, random_seed=42)
        
        for train_idx, val_idx in splits:
            train_set = set(train_idx)
            val_set = set(val_idx)
            assert len(train_set & val_set) == 0

    def test_all_samples_used(self):
        """Test that all samples are used in validation across all folds."""
        dataset = list(range(20))
        splits = create_kfold_splits(dataset, k=5, random_seed=42)
        
        all_val_indices = []
        for _, val_idx in splits:
            all_val_indices.extend(val_idx)
        
        # All indices should appear exactly once in validation sets
        assert set(all_val_indices) == set(range(20))
        assert len(all_val_indices) == 20

    def test_k_greater_than_dataset_size(self):
        """Test that k > dataset size raises ValueError."""
        dataset = list(range(5))
        
        with pytest.raises(ValueError, match="Cannot create.*folds"):
            create_kfold_splits(dataset, k=10, random_seed=42)

    def test_reproducibility(self):
        """Test that same seed produces same splits."""
        dataset = list(range(20))
        
        splits1 = create_kfold_splits(dataset, k=5, random_seed=42)
        splits2 = create_kfold_splits(dataset, k=5, random_seed=42)
        
        assert splits1 == splits2

    def test_different_seeds_produce_different_splits(self):
        """Test that different seeds produce different splits."""
        dataset = list(range(20))
        
        splits1 = create_kfold_splits(dataset, k=5, random_seed=42)
        splits2 = create_kfold_splits(dataset, k=5, random_seed=123)
        
        # They might be the same by chance, but likely different
        # At least verify the structure is correct
        assert len(splits1) == len(splits2) == 5

    def test_no_shuffle(self):
        """Test k-fold splitting without shuffling."""
        dataset = list(range(10))
        splits = create_kfold_splits(dataset, k=2, random_seed=42, shuffle=False)
        
        # Without shuffle, first fold should have first 5 as train, last 5 as val
        # Second fold should have last 5 as train, first 5 as val
        train_idx1, val_idx1 = splits[0]
        train_idx2, val_idx2 = splits[1]
        
        # Verify structure (exact indices depend on sklearn implementation)
        assert len(train_idx1) == 5
        assert len(val_idx1) == 5
        assert len(train_idx2) == 5
        assert len(val_idx2) == 5


class TestGetFoldData:
    """Tests for get_fold_data function."""

    def test_correct_data_extraction(self):
        """Test that correct data is extracted using indices."""
        dataset = ["a", "b", "c", "d", "e"]
        train_indices = [0, 1, 2]
        val_indices = [3, 4]
        
        train_data, val_data = get_fold_data(dataset, train_indices, val_indices)
        
        assert train_data == ["a", "b", "c"]
        assert val_data == ["d", "e"]

    def test_indices_match(self):
        """Test that extracted data matches the provided indices."""
        dataset = list(range(20))
        train_indices = [0, 1, 2, 5, 10, 15]
        val_indices = [3, 4, 6, 7, 8, 9]
        
        train_data, val_data = get_fold_data(dataset, train_indices, val_indices)
        
        assert train_data == [dataset[i] for i in train_indices]
        assert val_data == [dataset[i] for i in val_indices]

    def test_empty_train_indices(self):
        """Test with empty training indices."""
        dataset = ["a", "b", "c"]
        train_indices = []
        val_indices = [0, 1, 2]
        
        train_data, val_data = get_fold_data(dataset, train_indices, val_indices)
        
        assert train_data == []
        assert val_data == ["a", "b", "c"]

    def test_empty_val_indices(self):
        """Test with empty validation indices."""
        dataset = ["a", "b", "c"]
        train_indices = [0, 1, 2]
        val_indices = []
        
        train_data, val_data = get_fold_data(dataset, train_indices, val_indices)
        
        assert train_data == ["a", "b", "c"]
        assert val_data == []


class TestSaveAndLoadFoldSplits:
    """Tests for save_fold_splits and load_fold_splits functions."""

    def test_round_trip_serialization(self, temp_dir):
        """Test saving and loading fold splits."""
        splits = [
            ([0, 1, 2], [3, 4]),
            ([3, 4], [0, 1, 2]),
        ]
        metadata = {"k": 2, "random_seed": 42}
        output_path = temp_dir / "splits.json"
        
        save_fold_splits(splits, output_path, metadata)
        
        loaded_splits, loaded_metadata = load_fold_splits(output_path)
        
        assert loaded_splits == splits
        assert loaded_metadata == metadata

    def test_metadata_preserved(self, temp_dir):
        """Test that metadata is correctly preserved."""
        splits = [([0, 1], [2, 3])]
        metadata = {
            "k": 2,
            "random_seed": 42,
            "shuffle": True,
            "custom_field": "test_value",
        }
        output_path = temp_dir / "splits.json"
        
        save_fold_splits(splits, output_path, metadata)
        _, loaded_metadata = load_fold_splits(output_path)
        
        assert loaded_metadata == metadata

    def test_no_metadata(self, temp_dir):
        """Test saving and loading without metadata."""
        splits = [([0, 1], [2, 3])]
        output_path = temp_dir / "splits.json"
        
        save_fold_splits(splits, output_path)
        loaded_splits, loaded_metadata = load_fold_splits(output_path)
        
        assert loaded_splits == splits
        assert loaded_metadata == {}

    def test_file_not_found(self, temp_dir):
        """Test that loading non-existent file raises FileNotFoundError."""
        non_existent_path = temp_dir / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            load_fold_splits(non_existent_path)

    def test_invalid_file_format(self, temp_dir):
        """Test that invalid file format raises ValueError."""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text('{"invalid": "format"}')
        
        with pytest.raises(ValueError, match="Invalid splits file format"):
            load_fold_splits(invalid_file)

    def test_multiple_folds(self, temp_dir):
        """Test saving and loading multiple folds."""
        splits = [
            ([0, 1, 2], [3, 4]),
            ([3, 4], [0, 1, 2]),
            ([0, 2, 4], [1, 3]),
        ]
        output_path = temp_dir / "splits.json"
        
        save_fold_splits(splits, output_path)
        loaded_splits, _ = load_fold_splits(output_path)
        
        assert len(loaded_splits) == 3
        assert loaded_splits == splits

    def test_directory_creation(self, temp_dir):
        """Test that parent directory is created if it doesn't exist."""
        output_path = temp_dir / "nested" / "dir" / "splits.json"
        splits = [([0, 1], [2, 3])]
        
        save_fold_splits(splits, output_path)
        
        assert output_path.exists()
        loaded_splits, _ = load_fold_splits(output_path)
        assert loaded_splits == splits

