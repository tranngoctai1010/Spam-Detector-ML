# Build-in imports
import os

# Third-party imports
import pytest

# Internal imports
from src.utils.config_loader import ConfigLoader

# pytest tests/unit/test_src/test_utils/test_config_loader.py

@pytest.fixture
def temp_config_file(tmp_path):
    config_content = """
    key1: value1
    key2: value2
    """
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    return config_file

def test_get_config_success(temp_config_file):
    config_data = ConfigLoader.get_config(str(temp_config_file))  
    # When you give the full address (str(temp_config_file)), the computer finds the file anywhere, but just the name (temp_config_file.name) makes it look in the wrong place
    expected = {"key1": "value1", "key2": "value2"}
    print(config_data)
    print(expected)
    assert config_data == expected

def test_get_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        ConfigLoader.get_config("ksjhdfd.yaml")
