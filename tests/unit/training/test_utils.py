"""Tests for training utility functions."""

import pytest
from unittest.mock import patch, MagicMock
from training.utils import set_seed


class TestSetSeed:
    """Tests for set_seed function."""

    @patch("training.utils.torch")
    def test_set_seed_with_value(self, mock_torch):
        """Test setting seed with a specific value."""
        set_seed(42)
        
        mock_torch.manual_seed.assert_called_once_with(42)
        mock_torch.cuda.manual_seed_all.assert_called_once_with(42)

    @patch("training.utils.torch")
    def test_set_seed_with_none(self, mock_torch):
        """Test that setting seed with None does nothing."""
        set_seed(None)
        
        mock_torch.manual_seed.assert_not_called()
        mock_torch.cuda.manual_seed_all.assert_not_called()

    @patch("training.utils.torch")
    def test_set_seed_with_zero(self, mock_torch):
        """Test setting seed with zero."""
        set_seed(0)
        
        mock_torch.manual_seed.assert_called_once_with(0)
        mock_torch.cuda.manual_seed_all.assert_called_once_with(0)

    @patch("training.utils.torch")
    def test_set_seed_with_negative(self, mock_torch):
        """Test setting seed with negative value."""
        set_seed(-1)
        
        mock_torch.manual_seed.assert_called_once_with(-1)
        mock_torch.cuda.manual_seed_all.assert_called_once_with(-1)

    @patch("training.utils.torch")
    def test_set_seed_multiple_calls(self, mock_torch):
        """Test setting seed multiple times."""
        set_seed(42)
        set_seed(123)
        set_seed(456)
        
        assert mock_torch.manual_seed.call_count == 3
        assert mock_torch.cuda.manual_seed_all.call_count == 3

