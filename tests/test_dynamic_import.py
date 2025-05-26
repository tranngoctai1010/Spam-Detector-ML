import pytest
from typing import Dict
from unittest.mock import Mock, patch
from auto_ml.factory import DynamicImportFactory 

# Mock a fake class
class MockClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

@pytest.fixture
def module_list():
    return ["mock_module"]

@pytest.fixture
def factory(module_list):
    return DynamicImportFactory(module_list)

def test_init_valid_module_list(module_list, factory):
    """Test initialization with a valid module_list."""
    assert factory.module_list == module_list
    assert factory._registry == {}

def test_init_invalid_module_list_type():
    """Test error when module_list is not a list."""
    with pytest.raises(ValueError, match="not_a_list must be a list, got <class 'str'>."):
        DynamicImportFactory("not_a_list")

def test_init_invalid_module_list_elements():
    """Test error when module_list contains non-string elements."""
    with pytest.raises(ValueError, match="All elements in \['mock_module', 123\] must be strings."):
        DynamicImportFactory(["mock_module", 123])

def test_import_callable_success(factory):
    """Test successful import of a callable."""
    with patch.dict("sys.modules", {"mock_module": Mock()}):
        mock_module = __import__("mock_module")
        mock_module.MockClass = MockClass
        callable_obj = factory._import_callable("MockClass")
        assert callable_obj == MockClass
        assert factory._registry["MockClass"] == MockClass  # Check cache

def test_import_callable_module_not_found():
    """Test case when module does not exist (skipped by factory)."""
    factory = DynamicImportFactory(["non_existent_module"])
    with patch.dict("sys.modules", {}):  # No mock_module in sys.modules
        with pytest.raises(ValueError, match="Callable 'MockClass' not found in any module"):
            factory._import_callable("MockClass")

def test_create_success(factory):
    """Test successful creation of an instance."""
    with patch.dict("sys.modules", {"mock_module": Mock()}):
        mock_module = __import__("mock_module")
        mock_module.MockClass = MockClass
        params = {"param1": "value1"}
        instance = factory.create("MockClass", params)
        assert isinstance(instance, MockClass)
        assert instance.kwargs == params

def test_create_invalid_class_name(factory):
    """Test error when class_name is not a string."""
    with pytest.raises(ValueError, match="123 must be a string, got <class 'int'>"):
        factory.create(123, {})

def test_create_invalid_params(factory):
    """Test error when params is not a dictionary."""
    with pytest.raises(ValueError, match="not_a_dict must be a dictionary, got <class 'str'>"):
        factory.create("MockClass", "not_a_dict")


# pytest tests/test_dynamic_import.py -v  