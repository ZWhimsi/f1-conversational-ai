"""
Basic tests for utility functions.
"""

import pytest
from pathlib import Path
from utils.data_utils import load_json, save_json, load_jsonl, save_jsonl
from utils.config import Config


def test_config_loading():
    """Test configuration loading."""
    config = Config()
    assert config.config_dir.exists()


def test_data_utils():
    """Test data utility functions."""
    # Test data
    test_data = {"test": "data", "number": 42}
    
    # Test JSON operations
    test_file = Path("test_data.json")
    save_json(test_data, test_file)
    loaded_data = load_json(test_file)
    assert loaded_data == test_data
    
    # Cleanup
    test_file.unlink()


def test_jsonl_operations():
    """Test JSONL operations."""
    # Test data
    test_data = [
        {"text": "Hello", "label": 1},
        {"text": "World", "label": 2}
    ]
    
    # Test JSONL operations
    test_file = Path("test_data.jsonl")
    save_jsonl(test_data, test_file)
    loaded_data = load_jsonl(test_file)
    assert loaded_data == test_data
    
    # Cleanup
    test_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
