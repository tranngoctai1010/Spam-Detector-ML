# Build-in import
import traceback

# Internal imports
from src.processing_data.process_emails import process_emails
from src.training_model.classification import TrainClassification
from src.training_model.regression import TrainRegression
from src.utils.logger_manager import LoggerManager
from src.utils.model_handler import ModelHandler
from src.utils.config_loader import ConfigLoader


# Get logger
logger = LoggerManager.get_logger()

# Get configuration
full_config = ConfigLoader.get_config(file_name="scripts_config.yaml")
config = full_config["train_model.py"]
process_function_config = config["process_function"]
predictive_modeling_config = config["predictive_modeling"]
folder_name_config = config["folder_name"]
name_config = config["name"]


def train_model(process_function, predictive_modeling, folder_name, name):
    """
    [train_model] - Runs the pipeline to train for models.

    Args:
        process_function (function): A function that processes the dataset (the name of the file in the "processing_data" folder).  
        predictive_modeling (class): A class used for training the model. 
        folder_name (str): The name of the folder in the models folder.
        name (str): The name of the dataset in the "datasets" folder. 
        
    Steps:
        1. Process data.
        2. Train the model.
        3. Save the trained model and search objects.
        
    Raises:
        ValueError: If an invalid argument is provided.
        Exception: If an error occurs during execution.
    """
    # Validate args 
    try:
        config_map = {
            process_function.__name__: process_function_config,
            predictive_modeling.__name__: predictive_modeling_config,
            folder_name: folder_name_config,
            name: name_config
        }
        for key, valid_value in config_map.items():
            if key not in valid_value:
                raise ValueError(f"[train_model] - Invalid value {key}. Choose one of {valid_value}")
    except Exception as e:
        logger.error("[train_model] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
    
    try:
        #Step 1: Data preprocessing
        x_train, x_test, y_train, y_test = process_function()

        #Step 2: Train the model 
        model = predictive_modeling(x_train, x_test, y_train, y_test)
        model.train()

        #Step 3: Save the model and gridsearch objects
        ModelHandler.save_object(obj=model.best_estimator, folder_name=folder_name, file_name=f"best_{name}_model.pkl")
        ModelHandler.save_object(obj=model.search_objects, folder_name=folder_name, file_name=f"{name}_search_objects.pkl")
    except Exception as e:
        logger.error("[train_model] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
        raise