# Build-in imports
import os
from abc import ABC, abstractmethod

# Third-party imports
from beartype import beartype
import joblib

# nó là base word 
beartype
class BaseModelHandler(ABC):
    """
    ---
    **Abstract base class for handling model serialization (`saving` and `loading`).**
    
    _**This class provides the interface for saving and loading objects in derived classes.
    It also includes a method to validate and create directories for saving files.**_

    ---
    ### Methods:

    - save_object(obj: object, folder_name: str, file_name: str) -> None:  
        Abstract method to save an object to a specified file.
        
    - load_object(path: str): -> None:  
        Abstract method to load an object from a file.

    - validation_and_create_directory(folder_path: str, file_name: str) -> str:  
        Validates if the directory exists, creates it if necessary, and returns the file path.
    """
    @staticmethod
    @abstractmethod
    def save_object(cls, obj: object, folder_name: str, file_name: str) -> None:
        """
        ---
        **Abstract method to save an object to a specified file.**

        ### Args:

        - cls (type): The class type (inherited classes).
        - obj (object): The object to be saved.
        - folder_name (str): The folder where the object will be saved.
        - file_name (str): The name of the file where the object will be saved.

        ### Raises:

        - NotImplementedError: If not implemented in the derived class.
        """
        pass

    @classmethod
    @abstractmethod
    def load_object(cls, path: str) -> None:
        """
        **Abstract method to load an object from a file.**

        ### Args:

        - cls (type): The class type (inherited classes).
        - path (str): The file path where the object is stored.

        ### Returns:
        - object: The loaded object.

        ### Raises:
        
        - NotImplementedError: If not implemented in the derived class.
        """
        pass

    @classmethod
    def validation_and_create_directory(folder_path: str, file_name: str) -> str:
        """
        **Validates the existence of the specified directory and file. Creates them if necessary.**

        ### Args:
            
        - folder_path (str): The path to the directory where the file will be saved.
        - file_name (str): The name of the file to be saved in the directory.

        ### Returns:
            
        - str: The complete file path of the saved file.

        ### Raises:
        
        - FileNotFoundError: If the folder does not exist and cannot be created.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"File not found: {folder_path}")

        file_path = os.path.join(folder_path, file_name)

        if not os.path.exists(folder_path):
            with open(file_path, "w") as file:
                pass
        return file_path


class MLModelHandler(BaseModelHandler):
    """
    ---
    **A concrete class for handling machine learning model saving and loading using `joblib`.**

    _**Inherits from BaseModelHandler and implements methods to save and load models.**_
    
    ---
    ### Methods:

    - save_object(obj: object, folder_path: str, file_name: str) -> None:  
        Saves a model object to a specified path using joblib.
        
    - load_object(path: str) -> None:  
        Loads a model object from a specified path using joblib.
    """
    @classmethod
    def save_object(cls, obj: object, folder_path: str, file_name: str) -> None:
        """
        **Saves the provided object to the specified directory and file.**

        ### Args:
            
        - obj (object): The object (model) to be saved.
        - folder_path (str): The folder path to store the object.
        - file_name (str): The name of the file to save the object.

        ### Raises:
            
        - ValueError: If the object to be saved is None.
        - FileNotFoundError: If the directory is not found or cannot be created.
        """
        if obj is None:
            raise  ValueError(f"Cannot save None object.")
        
        file_path = cls.validation_and_create_directory(folder_path, file_name)
        joblib.dump(obj, file_path)

    @classmethod
    def load_object(cls, path: str):
        """
        **Loads an object from the specified path.**

        ### Args:
            
        - path (str): The path where the object is stored.

        ### Returns:

        - object: The loaded object (model).

        ### Raises:
            
        - FileNotFoundError: If the file is not found at the specified path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        obj = joblib.load(path)
        return obj
    