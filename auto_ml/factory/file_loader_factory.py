from ..base import BaseFileLoader


import inspect
from pathlib import Path
from typing import Dict, Type

class FileLoaderFactory:
    _registry = {
        "yaml.safe_load"
    }
    
    @classmethod
    def load(cls, path: str) -> Dict:
        if not isinstance(path, str):
            raise ValueError(f"{path} must be a string.")
        
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {path} doesn't exist.")
        
        suffix = file_path.suffix
        if cls._registry is None:
            raise ValueError(f"No loader registered for file type: {suffix}")
        loader_class = cls._registry.get(suffix)
        
        loader_instance = loader_class()
        return loader_instance.load(path)



# class FileLoaderFactory:
#     _functions: Dict[str, Callable] = {}

#     @classmethod
#     def load(cls, config_path: str):
#         if not isinstance(config_path, str):
#             raise ValueError(f"{config_path} must be a string.")
#         if not cls._functions:
#             cls._functions = cls._get_functions()
#         file_suffix_name = cls._detect_file_type(config_path)
#         class_name = cls._suffix_handler(file_suffix_name)
#         cls._functions[class_name].load()
        
#     @staticmethod
#     def _get_functions():
#         subclasses = BaseFileLoader.__subclasses__()
#         subclass_dict = {cls.__name__.lower() for cls in subclasses}
#         return subclass_dict
    
#     @staticmethod
#     def _detect_file_type(config_path: str):
#         file_suffix = Path(config_path).suffix
#         return file_suffix

#     @staticmethod
#     def _suffix_handler(file_suffix_name: str):
#         file_suffix_name = file_suffix_name.replace(".", "")
#         class_name = file_suffix_name + "fileloader" 
#         return class_name







# from pathlib import Path
# from typing import Dict, Type
# from ..base import BaseFileLoader


# class FileLoaderFactory:
#     """Factory for loading files based on their file extension."""
    
#     _registry: Dict[str, Type[BaseFileLoader]] = {}

#     @classmethod
#     def register_loader(cls, suffix: str, loader_class: Type[BaseFileLoader]):
#         """Register a loader for a specific file extension.
        
#         Args:
#             suffix (str): File extension (e.g., 'json', 'yaml').
#             loader_class: The loader class to handle the file type.
#         """
#         cls._registry[suffix.lower()] = loader_class

#     @classmethod
#     def load(cls, config_path: str) -> None:
#         """Load a file using the appropriate loader based on its extension.
        
#         Args:
#             config_path (str): Path to the file to be loaded.
            
#         Raises:
#             ValueError: If config_path is not a string or no loader is found.
#             FileNotFoundError: If the file does not exist.
#         """
#         if not isinstance(config_path, str):
#             raise ValueError(f"config_path must be a string, got {type(config_path)}")
        
#         file_path = Path(config_path)
#         if not file_path.exists():
#             raise FileNotFoundError(f"File {config_path} does not exist")
        
#         suffix = file_path.suffix.lstrip(".").lower()
#         loader_class = cls._registry.get(suffix)
        
#         if not loader_class:
#             raise ValueError(f"No loader registered for file type: {suffix}")
        
#         loader_instance = loader_class()
#         return loader_instance.load(config_path)


# # Ví dụ đăng ký loader
# # from json_loader import JsonFileLoader
# # FileLoaderFactory.register_loader("json", JsonFileLoader)