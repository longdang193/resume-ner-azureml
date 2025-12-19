"""Tests for YAML utilities."""

import pytest
from pathlib import Path
from shared.yaml_utils import load_yaml


class TestLoadYaml:
    """Tests for load_yaml function."""

    def test_load_valid_yaml(self, temp_dir):
        """Test loading a valid YAML file."""
        yaml_file = temp_dir / "test.yaml"
        yaml_file.write_text("key1: value1\nkey2: 123\n")
        
        data = load_yaml(yaml_file)
        
        assert data["key1"] == "value1"
        assert data["key2"] == 123

    def test_load_nested_yaml(self, temp_dir):
        """Test loading YAML with nested structures."""
        yaml_file = temp_dir / "test.yaml"
        yaml_file.write_text(
            "outer:\n"
            "  inner:\n"
            "    key: value\n"
            "  list:\n"
            "    - item1\n"
            "    - item2\n"
        )
        
        data = load_yaml(yaml_file)
        
        assert data["outer"]["inner"]["key"] == "value"
        assert data["outer"]["list"] == ["item1", "item2"]

    def test_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        non_existent = Path("/nonexistent/file.yaml")
        
        with pytest.raises(FileNotFoundError, match="YAML file not found"):
            load_yaml(non_existent)

    def test_invalid_yaml(self, temp_dir):
        """Test that invalid YAML raises an error."""
        yaml_file = temp_dir / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(Exception):  # yaml.YAMLError or similar
            load_yaml(yaml_file)

    def test_empty_yaml(self, temp_dir):
        """Test loading empty YAML file."""
        yaml_file = temp_dir / "empty.yaml"
        yaml_file.write_text("")
        
        data = load_yaml(yaml_file)
        
        assert data is None or data == {}

    def test_yaml_with_comments(self, temp_dir):
        """Test loading YAML with comments."""
        yaml_file = temp_dir / "test.yaml"
        yaml_file.write_text(
            "# This is a comment\n"
            "key: value  # Inline comment\n"
            "# Another comment\n"
            "number: 42\n"
        )
        
        data = load_yaml(yaml_file)
        
        assert data["key"] == "value"
        assert data["number"] == 42

