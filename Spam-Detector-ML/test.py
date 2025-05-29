import yaml
import importlib

class ModelFactory():
    _registry: Dict[str, Dict[str, Any]] = {}

    def __init__(self, config_path: str):
        self._parse_config(config_path)

    def create(self, model_name: str):
        if model_name not in self._registry.keys():
            raise ValueError(f"Model '{model_name}' not found in registry")
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
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
        except Exception as e:
            raise FileNotFoundError(f"")
        for name, info in config.items():
            self._registry[name] = info

    def _import_module(self, model_name: str):
        








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