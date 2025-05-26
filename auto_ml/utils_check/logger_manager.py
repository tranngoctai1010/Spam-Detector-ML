from abc import ABC, abstractmethod
import os
import logging
import logging.config
import yaml
from beartype import beartype
from typing import Optional


@beartype
class BaseLoggerManager(ABC):
    """
    ---
    **Abstract class for managing logging configuration and initialization.**

    _This class defines the common interface for logging management, including 
    methods for setting up logging and retrieving loggers. Subclasses must implement 
    the abstract methods to provide specific functionality for loading configuration 
    and applying it to the logging system._

    ---
    ### Class attributes:
    
    - _is_configured (bool = False): Flag indicating whether logging is configured.

    ---
    ### Methods:

    - setup_logging(config_path, log_path):  
        _Sets up the logging system using the provided configuration and log file path._

    - get_logger(config_path, log_path, name):  
        _Returns a logger instance configured according to the provided settings._

    - resolve_path(file_name, must_exist):  
        _Resolves the absolute path of the given file and ensures its existence if required._

    - _load_config_file(file_path):  
        _Loads the logging configuration from a config file._

    - _set_config_file_in_config(config, log_path)  
        _Sets the log file path in the provided logging configuration._

    - _apply_config(config):  
        _Applies the logging configuration to the logging system._
    """
    _is_configured: bool = False

    @classmethod
    @abstractmethod
    def setup_logging(cls, config_path: str, log_path: str) -> None:
        """
        **Sets up the logging system using the provided configuration and log file path.**

        ### Args:

        - config_path (str): The path to the logging configuration file.
        - log_path (str): The path where log files will be stored.

        _This method should be implemented by subclasses to provide specific logic
        for configuring the logging system. It must handle loading the configuration 
        and applying it._
        """
        pass

    @classmethod
    @abstractmethod
    def get_logger(cls, config_path: str, log_path: str, name: str | None = None) -> Optional[logging.Logger]:
        """
        **Returns a logger instance configured according to the provided settings.**

        ### Args:
            
        - config_path (str): The path to the logging configuration file.
        - log_path (str): The path where log files will be stored.
        - name (Optional[str]): The name of the logger. If None, the root logger is used.

        ### Returns:
        
        - Optional[logging.Logger]: A configured logger instance.

        _Subclasses must implement this method to initialize the logger using the
        provided configuration and return it._
        """
        pass

    @staticmethod
    def _resolve_path(file_path: str, must_exist: bool) -> str:
        """
        **Resolves the absolute path of the given file and ensures its existence if required.**

        ### Args:
            
        - file_path (str): The relative or absolute path of the file to resolve.
        - must_exist (bool): Flag indicating whether the file must exist. If True, thefile will be created if it does not exist.

        ### Returns:
            
        - str: The absolute path to the file.

        _If must_exist is True and the file does not exist, an empty file is created at 
        the specified path._
        """
        absolute_path = os.path.abspath(file_path)
        if must_exist and not os.path.exists(absolute_path):
            open(absolute_path, "a").close()
        return absolute_path

    @staticmethod
    @abstractmethod
    def _load_config_file(file_path: str) -> dict:
        """
        **Loads the logging configuration from a YAML file.**

        ### Args:
            
        - file_path (str): The path to the YAML configuration file.

        ### Returns:
            
        - dict: A dictionary representing the loaded configuration.

        _Subclasses must implement this method to handle the specific logic of reading
        and parsing the configuration file._
        """
        pass

    @staticmethod
    @abstractmethod
    def set_log_file_in_config(config: dict, log_path: str) -> None:
        """
        **Sets the log file path in the provided logging configuration.**

        ### Args:
            
        - config (dict): The logging configuration dictionary.
        - log_path (str): The path to the log file to be set in the configuration.

        _This method should be implemented by subclasses to modify the configuration 
        by setting the appropriate log file path._
        """
        pass

    @staticmethod
    @abstractmethod
    def _apply_config(config: dict) -> None:
        """
        **Applies the logging configuration to the logging system.**

        ### Args:
            
        - config (dict): The logging configuration dictionary.

        _This method should be implemented by subclasses to apply the given configuration
        to the logging system. It is responsible for ensuring that the configuration is 
        correctly interpreted and applied._
        """
        pass


@beartype
class PythonLoggerManager(BaseLoggerManager):
    """
    ---
    **Concrete implementation of BaseLoggerManager for managing logging in Python.**

    _This class provides the actual implementation for setting up logging, 
    loading the configuration, and retrieving loggers._
    """
    @classmethod
    def setup_logging(cls, config_path: str, log_path: str) -> None:
        """
        **Sets up the logging system using the provided configuration and log file path.**

        ### Args:

        - config_path (str): The path to the logging configuration file.
        - log_path (str): The path where log files will be stored.

        _This method should be implemented by subclasses to provide specific logic
        for configuring the logging system. It must handle loading the configuration 
        and applying it._
        """
        if cls._is_configured:
            return
        
        try:
            absolute_config_path = cls._resolve_path(config_path, must_exist=False)
            absolute_log_path = cls._resolve_path(log_path, must_exist=True)

            config = cls._load_config_file(absolute_config_path)
            cls.set_log_file_in_config(config, absolute_log_path)
            cls._apply_config(config)

            cls._is_configured = True
        except Exception as e:
            cls._is_configured = False 
            raise

    @classmethod
    def get_logger(cls, config_path: str, log_path: str, name: str | None) -> Optional[logging.Logger]:
        """
        **Returns a logger instance configured according to the provided settings.**

        ### Args:
            
        - config_path (str): The path to the logging configuration file.
        - log_path (str): The path where log files will be stored.
        - name (Optional[str]): The name of the logger. If None, the root logger is used.

        ### Returns:
        
        - Optional[logging.Logger]: A configured logger instance.

        _Subclasses must implement this method to initialize the logger using the
        provided configuration and return it._
        """
        if not cls._is_configured:
            cls.setup_logging(config_path, log_path)
        
        return logging.getLogger(name)
    
    @staticmethod
    def _load_config_file(file_path: str) -> dict:
        """
        **Loads the logging configuration from a YAML file.**

        ### Args:
            
        - file_path (str): The path to the YAML configuration file.

        ### Returns:

        - dict: A dictionary representing the loaded configuration.

        _Subclasses must implement this method to handle the specific logic of reading
        and parsing the configuration file._
        """
        try:
            with open(file_path, "r") as file:
                config = yaml.safe_load(file)
                if not isinstance(config, dict):
                    raise ValueError(f"YAML configuration must be a dictonary.")
                return config
        except yaml.YAMLError as e:
            raise ValueError(f"YAML Parsing Error in {file_path}: {e}")

    @staticmethod
    def set_log_file_in_config(config: dict, log_path: str) -> None:
        """
        **Sets the log file path in the provided logging configuration.**

        ### Args:
            
        - config (dict): The logging configuration dictionary.
        - log_path (str): The path to the log file to be set in the configuration.

        _This method should be implemented by subclasses to modify the configuration 
        by setting the appropriate log file path._
        """
        handlers = config.get("handlers", {})
        file_handler = handlers.get("file_handler")
        if file_handler and "filename" in file_handler:
            file_handler["filename"] = log_path
        
    @staticmethod
    def _apply_config(config: dict) -> None:
        """
        **Applies the logging configuration to the logging system.**

        ### Args:
            
        - config (dict): The logging configuration dictionary.

        _This method should be implemented by subclasses to apply the given configuration
        to the logging system. It is responsible for ensuring that the configuration is 
        correctly interpreted and applied._
        """
        try:
            logging.config.dictConfig(config)
        except ValueError as e:
            raise ValueError(f"Error applying logging configuration: {e}")
        

#Baselogger
# ILogger



















# # utils/logger_manager.py
# import logging
# import os
# import yaml

# class PythonLoggerManager:
#     """Manage logging configuration and initialization."""
#     @classmethod
#     def setup_logging(cls, log_path, config_path=None, level=logging.INFO):
#         """Set up logging with file output and optional YAML config.
        
#         Args:
#             log_path (str): Path to log file.
#             config_path (str, optional): Path to YAML config file.
#             level (int): Logging level (default: logging.INFO).
#         """
#         log_path = os.path.abspath(log_path)
#         if config_path:
#             config = cls._load_config_file(config_path)
#             cls._set_log_file_in_config(config, log_path)
#             logging.config.dictConfig(config)
#         else:
#             logging.basicConfig(
#                 filename=log_path,
#                 level=level,
#                 format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#             )

#     @classmethod
#     def get_logger(cls, name=None):
#         """Get a logger instance.
        
#         Args:
#             name (str, optional): Logger name. Defaults to root logger.
        
#         Returns:
#             logging.Logger: Configured logger.
#         """
#         return logging.getLogger(name)

#     @staticmethod
#     def _load_config_file(file_path):
#         try:
#             with open(file_path, "r") as f:
#                 return yaml.safe_load(f)
#         except Exception as e:
#             raise ValueError(f"Failed to load config {file_path}: {e}")

#     @staticmethod
#     def _set_log_file_in_config(config, log_path):
#         handlers = config.get("handlers", {})
#         if "file_handler" in handlers:
#             handlers["file_handler"]["filename"] = log_path






























# # utils/exception_handler.py
# import logging
# import traceback
# from functools import wraps

# class PythonExceptionHandler:
#     """Handle and log exceptions with context manager or decorator."""
#     def __init__(self, log_prefix, logger=None):
#         self.log_prefix = log_prefix
#         self.logger = logger or logging.getLogger(__name__)

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if exc_type:
#             self.logger.error(f"{self.log_prefix} {exc_type.__name__}: {exc_val}\n{traceback.format_exc()}")
#         return False  # Luôn ném lỗi

#     @staticmethod
#     def log_exception(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             logger = logging.getLogger(__name__)
#             try:
#                 return func(*args, **kwargs)
#             except Exception as e:
#                 prefix = f"[{func.__name__}]"
#                 if args and hasattr(args[0], '__class__'):
#                     prefix = f"[{args[0].__class__.__name__}.{func.__name__}]"
#                 logger.error(f"{prefix} {type(e).__name__}: {e}\n{traceback.format_exc()}")
#                 raise
#         return wrapper