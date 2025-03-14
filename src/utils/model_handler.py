# Build-in imports
import os
import traceback

# Third-party imports
import joblib

#Internal imports
from src.utils.logger_manager import LoggerManager

# Get logger
logger = LoggerManager.get_logger()


class ModelHandler:
    """
    [ModelHandler] - Utility class for saving and loading models and related objects using joblib.
    
    Methods:
        save_object(obj, file_name): Save an object to a file using joblib.
        load_object(file_name): Load an object from a file using joblib.
    """
    
    @staticmethod
    def save_object(obj, folder_name, file_name):
        """
        [ModelHandler][save_object] - Save an object to a file using joblib.
        
        Args:
            obj (object): The object to save.
            folder_name (str): The name of the model folder to save.
            file_name (str): The name of the model file to save.
        """
        try:
            logger.debug("[ModelHandler][save_object] - Object type: %s, Value: %s", type(obj), obj)
            if obj is None:
                logger.error("[ModelHandler][save_object]- Cannot save None object.\n%s", traceback.format_exc())
                raise ValueError("[ModelHandler][save_object]- Cannot save None object.")
            
            folder_path = os.path.join(os.path.dirname(__file__), "..", "models", folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(folder_path, file_name)
            joblib.dump(obj, file_path)
            logger.info("[ModelHandler][save_object] - Object saved successfully")
        except Exception as e:
            logger.error("[ModelHandler][save_object] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
            raise

    @staticmethod
    def load_object(folder_name, file_name):
        """
        [ModelHandler][load_object] - Load an object from a file using joblib.
        
        Args:
            folder_name (str): The name of the model folder to load.
            file_name (str): The name of the model file to load.
        
        Returns:
            object: The loaded object.
        
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        try:
            file_path = os.path.join(os.path.dirname(__file__), "..", "models", folder_name, file_name)
            if not os.path.exists(file_path):
                logger.error("[ModelHandler][load_object] - File not found.\n%s", traceback.format_exc())
                raise FileNotFoundError(f"[ModelHandler][load_object] - File not found.")

            obj = joblib.load(file_path)  
            logger.info("[ModelHandler][load_object] - Object loaded successfully")
            return obj
        except Exception as e:
            logger.error("[ModelHandler][load_object] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
            raise