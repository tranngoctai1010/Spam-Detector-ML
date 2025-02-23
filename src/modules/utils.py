#Built-in imports
import os
import logging
import logging.config

#Third-party imports
import joblib
import yaml

# Internal imports
from src.modules.training_modules.base_trainer import ModelStorage

class LoggerManager:
    """
    [LoggerManager] - Manages logging through the configuration from the logging_config.yaml file.
    
    Attributes:
        _is_setup (bool): Flag variable to check whether logging has been set up.
        
    Methods:
        setup_logging(): Set up logging configuration if not already set up.
        get_logger(name): 
    """
    _is_setup = False   
    
    @classmethod
    def setup_logging(cls):
        """
        Set up logging configuration if not already set up.
        
        Raises:
            FileNotFoundError: The logging_config.yaml file not found.
            ValueError: Error when reading the logging_config.yaml file.
        """
        if cls._is_setup:
           return

        file_name = "logging_config.yaml"
        logging_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", file_name)
        
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
        Get a logger instance with the specified name.

        Args:
            name (str, optional): The name of the logger. If none, uses the default module name.

        Returns:
            logging.getLogger: A logger instance with the specified name.
        """
        if not cls._is_setup:
            cls.setup_logging()
        return logging.getLogger(name or __name__)
        
# Use logger
logger = LoggerManager.get_logger()


class ConfigLoader:
    """
    [ConfigLoader] - Loads the configuration file.
    
    Methods:
        get_config(file_name): Loads the configuration file.
    """
    @classmethod
    def get_config(cls, file_name: str) -> dict:
        """
        [load] - Loads the configuration file.

        Args:
            file_name (str): Name of the YAML configuration file.
        
        Returns:
            dict: Configuration data parsed from the file.
            
        Raises:
            FileNotFoundError: If the configuration file not found.
            yaml.YAMLError: If there is an error parsing the logging configuration file.
        """
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs" ,file_name)
        if not os.path.exists(config_path):
            raise FileNotFoundError("[ConfigLoader][load] - Configuration file not found.")

        try:
            with open(config_path, mode="r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file) or {}
            logger.info("[ConfigLoader][load] - Loaded configuration from %s", config_path)
            return config_data
        except yaml.YAMLError as e:
            logger.error("[ConfigLoader][load] - Error when reading the configuration file:\n%s", e)
            raise



class ModelHandler(ModelStorage):
    """
    [ModelHandler] - Manages storage/retrieval model and GridSearchCV.
    
    This class inherits from ModelStorage and add methods for saving and loading model-related objects, such as best estimators and search objects
    
    Methods:
        _save_object(prefix): Save objects related to the file with the specified prefix
        _load_object(prefix): Load objects related to the file with the specified prefix
        
    
    """
    def __init__(self):
       super().__init__()

    def _save_object(self, prefix):
        try:
            logger.info("[ModelHandler][_save_object] - Saving the best estimator .....")
            joblib.dump(self.best_estimator, f"{prefix}_best_estimator.pkl")
            logger.info("[ModelHandler][_save_object] - The best estimator has been saved .....")
            
            logger.info("[ModelHandler][_save_object] - Saving the search objects .....")
            joblib.dump(self.search_objects, f"{prefix}_search_objects.pkl")
            logger.info("[ModelHandler][_save_object] - The search objects has been saved .....")
            
        except ValueError as e:
            logger.error(f"[ModelHandler][_load_object] - Error while loading the objects: {e}")
            raise
        
    def _load_object(self, prefix, best_estimator=True, search_objects=True):
        try:
            if best_estimator == True:
                logger.info("[ModelHandler][_load_object] - Loading the best estimator .....")
                self.best_estimator = joblib.load(f"{prefix}_best_estimator.pkl")
                logger.info("[ModelHandler][_load_object] - The best estimator has been loaded .....")
                
            if search_objects == True:
                logger.info("[ModelHandler][_load_object] - Loading the search objects .....")
                self.search_objects = joblib.load(f"{prefix}_search_objects.pkl")  
                logger.info("[ModelHandler][_load_object] - The search objects has been loaded .....")
                
        except FileNotFoundError as e:
            logger.error(f"[ModelHandler][_load_object] - File not found: {e}")
            raise
        except ValueError as e:
            logger.error(f"[ModelHandler][_load_object] - Error while loading the objects: {e}")
            raise