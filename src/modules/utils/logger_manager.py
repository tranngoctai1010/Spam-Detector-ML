# Build-in imports
import os
import logging
import logging.config

# Third-party imports
import yaml

class LoggerManager:
    """
    [LoggerManager] - Manages logging through the configuration from the logging_config.yaml file.
    
    Attributes:
        _is_setup (bool): Flag variable to check whether logging has been set up.
        
    Methods:
        setup_logging(): Set up logging configuration if not already set up.
        get_logger(name): Get a logger instance with the specified name. Default to __name__ if the name not provided.
    """
    _is_setup = False   
    
    @classmethod
    def setup_logging(cls):
        """
        [LoggerManager][setup_logging] - Set up logging configuration if not already set up.
        
        Raises:
            FileNotFoundError: The logging_config.yaml file not found.
            ValueError: Error when reading the logging_config.yaml file.
        """
        if cls._is_setup:
           return

        file_name = "logging_config.yaml"
        logging_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", file_name)
        
        if not os.path.exists(logging_path):
            raise FileNotFoundError(f"[LoggerManager][setup_logging] - The {file_name} file was not found.")
        
        try:
            with open(logging_path, mode="r") as file:
                config = yaml.safe_load(file)
            logging.config.dictConfig(config)
        except yaml.YAMLError as e:
            raise ValueError(f"[loggerManager][setup_logging] - Error when reading the logging_config.yaml file:\n{e}")
        
        cls._is_setup = True
    
    @classmethod
    def get_logger(cls, name=None) -> logging.getLogger:
        """
        [LoggerManager][get_logger] - Get a logger instance with the specified name.

        Args:
            name (str, optional): The name of the logger. If none, uses the default module name.

        Returns:
            logging.getLogger: A logger instance with the specified name.
        """
        if not cls._is_setup:
            cls.setup_logging()
        return logging.getLogger(name or __name__)
        