"""Tests for JSON cache utilities."""

import json
import pytest
from pathlib import Path
from shared.json_cache import save_json, load_json


class TestSaveJson:
    """Tests for save_json function."""

    def test_save_simple_dict(self, temp_dir):
        """Test saving a simple dictionary."""
        output_file = temp_dir / "test.json"
        data = {"key1": "value1", "key2": 123}
        
        save_json(output_file, data)
        
        assert output_file.exists()
        with open(output_file, "r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_nested_dict(self, temp_dir):
        """Test saving nested dictionary."""
        output_file = temp_dir / "test.json"
        data = {
            "outer": {
                "inner": {
                    "key": "value",
                },
            },
            "list": [1, 2, 3],
        }
        
        save_json(output_file, data)
        
        with open(output_file, "r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_create_parent_directory(self, temp_dir):
        """Test that parent directory is created if it doesn't exist."""
        output_file = temp_dir / "nested" / "deep" / "test.json"
        data = {"key": "value"}
        
        save_json(output_file, data)
        
        assert output_file.exists()
        assert output_file.parent.exists()

    def test_pretty_printing(self, temp_dir):
        """Test that JSON is pretty-printed."""
        output_file = temp_dir / "test.json"
        data = {"key1": "value1", "key2": "value2"}
        
        save_json(output_file, data)
        
        content = output_file.read_text()
        # Should contain newlines (pretty-printed)
        assert "\n" in content


class TestLoadJson:
    """Tests for load_json function."""

    def test_load_existing_file(self, temp_dir):
        """Test loading an existing JSON file."""
        json_file = temp_dir / "test.json"
        data = {"key1": "value1", "key2": 123}
        with open(json_file, "w") as f:
            json.dump(data, f)
        
        loaded = load_json(json_file)
        
        assert loaded == data

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading a non-existent file returns None."""
        non_existent = temp_dir / "nonexistent.json"
        
        loaded = load_json(non_existent)
        
        assert loaded is None

    def test_load_nonexistent_file_with_default(self, temp_dir):
        """Test loading a non-existent file returns default value."""
        non_existent = temp_dir / "nonexistent.json"
        default = {"default": "value"}
        
        loaded = load_json(non_existent, default=default)
        
        assert loaded == default

    def test_load_invalid_json(self, temp_dir):
        """Test loading invalid JSON raises an error."""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("invalid json content {")
        
        with pytest.raises(json.JSONDecodeError):
            load_json(invalid_file)

    def test_load_empty_file(self, temp_dir):
        """Test loading empty JSON file."""
        empty_file = temp_dir / "empty.json"
        empty_file.write_text("")
        
        with pytest.raises(json.JSONDecodeError):
            load_json(empty_file)

