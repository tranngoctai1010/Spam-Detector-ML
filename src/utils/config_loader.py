# Build-in imports
import os 
import traceback

# Third-party imports
import yaml

#Internal imports
from src.utils.logger_manager import LoggerManager


# Get logger
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
        [ConfigLoader][load] - Loads the configuration file.

        Args:
            file_name (str): Name of the YAML configuration file.
        
        Returns:
            dict: Configuration data parsed from the file.
            
        Raises:
            FileNotFoundError: If the configuration file not found.
        """
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs" ,file_name)
        if not os.path.exists(config_path):
            raise FileNotFoundError("[ConfigLoader][load] - Configuration file not found.")

        try:
            with open(config_path, mode="r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file) or {}
            logger.info("[ConfigLoader][load] - Loaded configuration from %s", config_path)
            return config_data
        except Exception as e:
            logger.error("[ConfigLoader][load] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
            raise

