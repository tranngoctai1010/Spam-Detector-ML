# Built-in Python libraries
import traceback
import os

# Third-party libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import LabelEncoder

# Internal imports
from src.utils.model_handler import ModelHandler
from src.utils.logger_manager import LoggerManager
from src.utils.config_loader import ConfigLoader


# Get logger
logger = LoggerManager.get_logger()

# Get configuration
full_config = ConfigLoader.get_config(file_name="processing_data_config.yaml")
try:
    config = full_config["process_emails.py"]
except Exception as e:
    logger.error(f"[base_trainer.py] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
    raise


def process_emails():
    """
    This function processes data for the emails.csv file.        
    
    Steps: 
        1. Read file.
        2. Train test split dataset.
        3. Build a pipeline.
    """
    
    #Step1: Read file 
    try:
        file_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "emails.csv")
        data = pd.read_csv(file_path)
    except Exception as e:
        logger.error("[process_emails] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
        raise
    
    
    #Step2: Train test split dataset
    try:
        data = data.dropna()

        x = data["text"]
        y = data["label"]

        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    except Exception as e:
        logger.error("[process_emails] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
        raise


    #Step3: Build a pipeline 
    try:
        # Config for hyperparameters of data preprocessing
        stop_words_ = config["stop_words"]
        ngram_range_ = tuple(config["ngram_range"])
        max_features_ = config["max_features"]
        percentile_ = config["percentile"]
        
        pipeline = Pipeline(steps=[
            ("vectorizer", TfidfVectorizer(stop_words=stop_words_, ngram_range=ngram_range_, max_features=max_features_)),
            ("select_feature", SelectPercentile(chi2, percentile=percentile_)),
        ])

        x_train_processed = pipeline.fit_transform(x_train, y_train)
        x_test_processed = pipeline.transform(x_test)
        ModelHandler.save_object(pipeline, folder_name="email_model", file_name="emails_processing_pipeline.pkl")
        
    except Exception as e:
        logger.error("[process_emails] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
        raise
        
    return x_train_processed, x_test_processed, y_train, y_test