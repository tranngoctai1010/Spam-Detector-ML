#Built-in Python libraries
import os
import logging
import logging.config

#Third-party libraries
import joblib
import yaml


def setup_logging(config_path="configs/logging_config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)

def get_logger():
    setup_logging()
    return logging.getLogger(__name__)


#Load config YAML file
try:
    with open(config_path="configs/modules_config.yaml", mode="r") as file:
        full_config = yaml.safe_load(file)
        config = full_config["training_modules/"]["classification.py"]
except Exception as e:
    logging.error("Error when using dev_config.yaml file:\n %s", e)
    raise


class ModelHandler:
    """
    This class to manages storage/retrieval model and GridSearchCV.

    Atributes: 
        best_estimator: Store the best model.
        search_objects: Store The GridSearchCV object.

    Methods:
        save_model(): 
            Save the best model to a file.

            Arguments:
                model (estimator): The best model have been optimized. 
                filename (str): The destination file path.

        load_model(): 
            Load the best model from a file.

            Arguments:
                filename (str): The file path of the saved model.

        save_gridsearch(): 
            gridsearch_objects (dict): Stores all optimized GridSearchCV objects (Use self.search_objects in base_trainer.py).
            Save the GridSearchCV object to a file.

            Arguments:
                filename (str): The destination file path.
            
        load_gridsearch(): 
            Load the GridSearchCV object from a file.

            Arguments:
                filename (str): The file path of the saved GridSearchCv.

    """

    def __init__(self):
        self.best_estimator = None
        self.search_objects = {}

    def save_model(self, model, filename):

        if model:
            try:
                joblib.dump(model, filename)
                logging.info("Model has been saved to %s", filename)
            except Exception as e:
                logging.error("Error when saving the model:\n %s", e)

        else:
            logging.warning("No have model to save.")
    
    def load_model(self, filename):

        if os.path.exists(filename):
            try:
                self.best_estimator = joblib.load(filename)
                logging.info("Model has been loaded to file.")
            except Exception as e:
                logging.error("Error when loading model:\n %s", e)

        else: 
            logging.warning("No have model to load.")

    def save_gridsearch(self, gridsearch_objects, filename):

        if gridsearch_objects:
            try:
                joblib.dump(gridsearch_objects, filename)
                logging.info("GridSearchCV object has been saved to %s", filename)
            except Exception as e:
                logging.error("Error when saving the GridSearchCv object:\n %s",e)

        else:
            logging.warning("No have GridSearchCV object to save.")

    def load_gridsearch(self, filename):

        if os.path.exists(filename):
            try:
                self.search_objects = joblib.load(filename)
                logging.info("GridSearchCV object has been loaded.")
            except Exception as e:
                logging.error("Error when loading GridSearchCV object:\n %s", e)

        else:
            logging.warning("No have GridSearch to load.")