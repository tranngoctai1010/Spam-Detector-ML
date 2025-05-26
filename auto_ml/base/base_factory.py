
import inspect
from pathlib import Path
from typing import Callable
import yaml

def load_yaml_file(config_path: str):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        if not isinstance(config, dict):
            raise ValueError("Invalid YAML format: Expected a dictionary")
        return config
    except Exception:
        raise FileNotFoundError(f"Config file {config_path} not found")
    except Exception as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

class FileLoaderFactory:
    _functions: Dict[str, Callable] = {}
    
    @classmethod
    def create(cls, config_path: str):
        if not cls._functions:
            cls._functions = cls._get_functions()
        file_suffix_name = cls._detect_file_type(config_path)
        function_name = cls._suffix_handler(file_suffix_name)
        pass
    
    @staticmethod
    def _get_functions():
        current_module = inspect.getmodule(inspect.currentframe())
        functions = inspect.getmembers(current_module, inspect.isfunction)
        return functions
    
    @staticmethod
    def _detect_file_type(config_path: str):
        file_suffix = Path(config_path).suffix
        return file_suffix

    @staticmethod
    def _suffix_handler(file_suffix_name: str):
        file_suffix_name = file_suffix_name.replace(".", "")
        function_name = "load_" + file_suffix_name + "_file"
        return function_name


