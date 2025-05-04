# Build-in import
import traceback
from argparse import ArgumentParser
from typing import Tuple

#Third-party imports
import numpy as np

# Internal imports
from src.processing_data.process_emails import process_emails
from src.training_model.classification import TrainClassification
from src.training_model.regression import TrainRegression
from src.utils.logger_manager import LoggerManager
from src.utils.model_handler import ModelHandler
from src.utils.exception_handler import ExceptionHandler

# scripts.train_pipeline.py

# Get logger
logger = LoggerManager.get_logger(log_file="scripts.log")

ExceptionHandler.set_default_log_file("scripts.log")

FUNCTION_MAP = {
    "process_emails": process_emails
}

MODEL_MAP = {
    "TrainClassification": TrainClassification,
    "TrainRegression": TrainRegression
}

def get_training_arg():
    parser = ArgumentParser(description="training model")
    parser.add_argument("--process_function_str", "-f", type=str, default=None, help="The name of a function")
    parser.add_argument("--predictive_modeling_str", "-c", type=str, default=None, help="The name of a class used for training")
    parser.add_argument("--folder_name", "-fn", type=str, default=None, help="The name of the folder model")
    parser.add_argument("--name", "-n", type=str, default=None, help="The name of model")
    args = parser.parse_args()
    return args

@ExceptionHandler.log_exceptions
def validate_arg(process_function, predictive_modeling, folder_name, name):
    """
    [validate_arg] - Validate input arguments for training.

    Args:
        process_function (function): Function to process the dataset.
        predictive_modeling (class): Class used for model training.
        folder_name (str): Name of folder to save model.
        name (str): Model name.

    Raises:
        ValueError: If an invalid argument is provided.
    """
    CONFIG_MAP = {
        process_function.__name__: ["process_emails"],
        predictive_modeling.__name__: ["TrainClassification", "TrainRegression"],
        folder_name: ["email_model"],
        name: ["email"]
    }
    for key, valid_value in CONFIG_MAP.items():
        if key not in valid_value:
            logger.error("[validate_arg] - Invalid value %s. Choose one of %s.\n%s", key, valid_value, traceback.format_exc())
            raise ValueError(f"[validate_arg] - Invalid value {key}. Choose one of {valid_value}")

def process_data(process_function: callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    [process_data] - Preprocess data and return train-test-split.

    Args + Return:
        process_function (function): A function that processes the dataset.
    """
    return process_function()

def train_model(predictive_modeling, x_train_processed, x_test_processed, y_train, y_test):
    """
    [train_model] - Train a predictive model using the provided training and testing data.

    Args:
        predictive_modeling (class): A class used for training the model.
        x_train_processed (np.ndarray): The processed training feature.
        x_test_processed (np.ndarray): The processed testing feature.
        y_train (np.ndarray): The target labels for training data.
        y_test (np.ndarray): The target labels for testing data.

    Return:
        model (object): The trained model object, which has attribute "best_estimator" and "search_objects".
    """
    model = predictive_modeling(x_train_processed, x_test_processed, y_train, y_test)
    model.train()
    return model

@ExceptionHandler.log_exceptions
def save_trained_model(model, folder_name, name):
    """
    [save_trained_model] - Save the best estimator and gridsearch objects to the specified folder.

    Args:
        model (object): The trained model object, which has attribute "best_estimator" and "search_objects".
        folder_name (str): The name of the folder in the models folder. Must be ["email_model"].
        name (str): The name of model. Must be one of ["email"].
    """
    ModelHandler.save_object(obj=model.best_estimator, folder_name=folder_name, file_name=f"best_{name}_model.pkl")
    ModelHandler.save_object(obj=model.search_objects, folder_name=folder_name, file_name=f"{name}_search_objects.pkl")

@ExceptionHandler.log_exceptions
def train_pipeline(process_function_str, predictive_modeling_str, folder_name, name):
    """
    [train_model] - Runs the pipeline to train for models.

    Args:
        process_function_str (str): The name of a function that processes the dataset (the name of the file in the "processing_data" folder).  
        predictive_modeling_str (srt): The name of a class used for training the model. 
        folder_name (str): The name of the folder in the models folder. Must be ["email_model"].
        name (str): The name of model. Must be one of ["email"].
        
    Steps:
        1. Validate input arguments for training
        2. Process data.
        3. Train the model.
        4. Save the trained model and search objects.
        
    Raises:
        Exception: If an error occurs during execution.
    """
    process_function = FUNCTION_MAP[process_function_str]
    predictive_modeling = MODEL_MAP[predictive_modeling_str]

    # Step 1: Validate input arguments for training.
    validate_arg(process_function, predictive_modeling, folder_name, name)
    
    #Step 2: Data preprocessing
    x_train_processed, x_test_processed, y_train, y_test = process_data(process_function)

    #Step 3: Train the model 
    model = train_model(predictive_modeling, x_train_processed, x_test_processed, y_train, y_test)

    #Step 4: Save the model and gridsearch objects
    save_trained_model(model, folder_name, name)