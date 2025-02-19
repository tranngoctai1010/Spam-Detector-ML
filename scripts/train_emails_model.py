from modules.training_modules.classification import TrainClassification
from modules.process_emails import process_emails
from modules.utils import ModelHandler
import logging
import yaml

#Config logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="logs/scripts.log",
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#Load configuration file
file_path = "configs/scripts_config.yaml"
with open(file_path, "r") as file:
    full_config = yaml.safe_load(file)
    config = full_config["train_model.py"]
    
def run_pipeline():
    """
    Run the entire pipeline (processing, training, prediction, ...)
    """
    try:
        #Step 1: Data preprocessing
        logging.info("Starting data preprocessing .....")
        x_train, x_test, y_train, y_test = process_emails()

        #Step 2: Train the model 
        logging.info("Starting model training .....")
        model = TrainClassification(x_train, x_test, y_train, y_test)
        model.train()

        #Step 3: Save the model and gridsearch objects
        model_handler = ModelHandler()
        logging.info("Starting to save the model .....")
        model_handler.save_model(model=model.best_estimator , filename=config["save_model"])
        logging.info("Starting to save gridsearch objects .....")
        model_handler.save_gridsearch(gridsearch_objects=model.search_objects, filename=config["save_gridsearch"])
    except Exception as e:
        logging.error("Error when running train_model.py file.")
        raise