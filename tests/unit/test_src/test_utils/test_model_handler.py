# Build in imports
import os

# Third-party imports
import pytest

# Internal imports 
from src.utils.model_handler import ModelHandler

# pytest tests/unit/test_src/test_utils/test_model_handler.py

def test_save_object_success():
    object = {"test": "data"}
    folder_name = "test_folder"
    file_name = "test_file.pkl"

    ModelHandler.save_object(obj=object, folder_name=folder_name, file_name=file_name)
    file_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src", "models", folder_name, file_name)
    assert os.path.exists(file_path)

def test_save_object_none():
    with pytest.raises(ValueError):
        ModelHandler.save_object(None, folder_name="test_folder", file_name="test_file.pkl")

def test_load_object_success():
    object = {"test": "data"}
    folder_name = "test_folder"
    file_name = "test_file.pkl"

    loaded_obj = ModelHandler.load_object(folder_name=folder_name, file_name=file_name)
    assert loaded_obj == object

def test_load_object_file_not_found():
    with pytest.raises(FileNotFoundError):
        ModelHandler.load_object("ljhsdlfs", "skljdf.pkl")    