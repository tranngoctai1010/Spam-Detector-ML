import importlib
import yaml
from typing import Dict, Any
import logging
from functools import partial
import sklearn 
import logging

def basic_logger(name: str):
    # Tạo logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Đặt level tối thiểu cho logger

    # Xóa handler mặc định (nếu có)
    logger.handlers.clear()

    # Tạo handler cho console, chỉ in INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Chỉ cho phép INFO
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # (Tùy chọn) Tạo handler cho file, ghi ERROR và cao hơn
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.ERROR)  # Chỉ ghi ERROR và CRITICAL
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Thêm handlers vào logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def log_and_raise(
        message: str,
        exception: Exception,
        logger = logging.Logger
    ):
    logger.error(message)
    raise exception(message)

# model_utils.py (load, save model)
# data_loader.py (Tạo 1 factory để biết được nên load file nào)
# Dùng pathlib để lấy đuôi file hoặc dùng split để tách đuôi file.
def load_yaml_file(config_path):
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

class ModelFactory():
    _registry: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        config_path: str,
        logger: logging.Logger = basic_logger(__name__),
        _file_loader = load_yaml_file,
        _error_handler = log_and_raise
    ):
        self._parse_config(config_path)
        self.logger = logger
        self._load_file = _file_loader
        self._log_and_raise = partial(_error_handler, logger)

    def create(self, model_name: str):
        if model_name not in self._registry.keys():
            self._log_and_raise(f"Model '{model_name}' not found in registry", ValueError)
        model = importlib.import_module(self._registry[model_name])
        return model()

    def create_all(self):
        model_dict = {}
        model_names = self._registry.keys()
        for model_name in model_names:
            model = importlib.import_module(model_name)
            model_dict.append(model)
        return model_dict

    def _parse_config(self, config_path: str):
        config = self._load_file(config_path)
        for name, info in config.items():
            self._registry[name] = info

    def _import_module(self, model_name: str):
        pass

import inspect
from pathlib import Path
from typing import Callable

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








# from injector import Injector, Module, singleton

# class DatabaseModule(Module):
#     def configure(self, binder):
#         binder.bind(Database, to=Database, scope=singleton)

# class LoggerModule(Module):
#     def configure(self, binder):
#         binder.bind(Logger, to=Logger, scope=singleton)

# _injector = None
# def get_injector() -> Injector:
#     global _injector
#     if _injector is None:
#         _injector = Injector([DatabaseModule(), LoggerModule()])
#     return _injector

# # Trong module user_service.py
# injector = get_injector()
# user_service = injector.get(UserService)

# # Trong module strategy_factory.py
# injector = get_injector()
# strategy = injector.get(StrategyFactory)


# import importlib
# import logging
# from typing import Dict, Any, ClassVar, Optional
# import yaml

# # Thiết lập logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ModelFactory:
#     """A factory class to create model instances from a YAML configuration.

#     Attributes:
#         _registry: A class-level dictionary storing model configurations.
#     """
#     _registry: ClassVar[Dict[str, Dict[str, Any]]] = {}

#     def __init__(self, config_path: Optional[str] = None) -> None:
#         """Initialize the ModelFactory with an optional config file.

#         Args:
#             config_path: Path to the YAML configuration file. If None, no config is loaded.

#         Raises:
#             FileNotFoundError: If the config file does not exist.
#             yaml.YAMLError: If the YAML file is invalid.
#         """
#         if config_path:
#             self.load_config(config_path)

#     def load_config(self, config_path: str) -> None:
#         """Load model configurations from a YAML file.

#         Args:
#             config_path: Path to the YAML configuration file.

#         Raises:
#             FileNotFoundError: If the config file does not exist.
#             yaml.YAMLError: If the YAML file is invalid.
#             ValueError: If the YAML content is not a valid dictionary.
#         """
#         try:
#             with open(config_path, 'r') as file:
#                 config = yaml.safe_load(file)
#             if not isinstance(config, dict):
#                 raise ValueError("Invalid YAML format: Expected a dictionary")
#             self._registry.update(config)
#             logger.info(f"Loaded config from {config_path}")
#         except FileNotFoundError:
#             logger.error(f"Config file {config_path} not found")
#             raise
#         except yaml.YAMLError as e:
#             logger.error(f"Error parsing YAML file: {e}")
#             raise

#     def create(self, model_name: str) -> Any:
#         """Create a model instance based on the model name.

#         Args:
#             model_name: Name of the model in the registry.

#         Returns:
#             An instance of the model class.

#         Raises:
#             ValueError: If the model name is not in the registry or configuration is invalid.
#             ImportError: If the module or class cannot be imported.
#         """
#         if model_name not in self._registry:
#             logger.error(f"Model '{model_name}' not found in registry")
#             raise ValueError(f"Model '{model_name}' not found in registry")

#         config = self._registry[model_name]
#         required_keys = {'module', 'class'}
#         if not all(key in config for key in required_keys):
#             logger.error(f"Invalid config for {model_name}: {required_keys} required")
#             raise ValueError(f"Invalid config for {model_name}: {required_keys} required")

#         try:
#             module = importlib.import_module(config['module'])
#             model_class = getattr(module, config['class'])
#             params = config.get('params', {})
#             logger.info(f"Creating {model_name} with params: {params}")
#             return model_class(**params)
#         except (ImportError, AttributeError) as e:
#             logger.error(f"Failed to create model {model_name}: {e}")
#             raise ValueError(f"Failed to create model {model_name}: {e}")

#     def create_all(self) -> Dict[str, Any]:
#         """Create instances of all models in the registry.

#         Returns:
#             A dictionary mapping model names to their instances.

#         Raises:
#             ValueError: If any model creation fails.
#         """
#         logger.info("Creating all models from registry")
#         return {name: self.create(name) for name in self._registry}