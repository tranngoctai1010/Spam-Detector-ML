from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, r2_score, mean_squared_error, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt 
import logging 
import yaml

#Config logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="logs/dev.log",
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Load config YAML file
config_path="configs/modules_config.yaml"
try:
    with open(config_path, "r") as file:
        full_config = yaml.safe_load(file)
        config = full_config["training_modules/"]["classification.py"] 
except FileNotFoundError:
    logging.error("Config not found at %s. Please check file path.", config_path)
    raise
except KeyError as e:
    logging.error("Missing key in configuration file:\n %s", e)
    raise
except yaml.YAMLError as e:
    logging.error("Error parsing YAML file:\n %s", e)
    raise 
except Exception as e:
    logging.error("Error:\n %s", e)
    raise


class BaseTrainer:
    """
    BaseTrainer is generic class to train machine learning models that use GridSearchCV or RandomizedGridSearchCV.

    Attributes:
        x_train, x_test, y_train, y_test: 
            Training and testing datasets.

        scoring: 
            Metric is used to evaluate model.

        models: 
            A dictionary that contains the model to be trained.

        param_grids: 
            A dictionary of hyperparameters for the optimization search process.

        search_objects: 
            Stores the GridSearchCV or RandomizedSearchCV for each model. 

    Methods: 
        validate_data(): 
            Validates if training data is not None.

        train_model(): 
            Trains multiple models using GridSearchCV or RandomizedSearchCV and select the best one based on scoring.

            Arguments:
                 use_random_search (bool): If True, uses RandomizedSearchCV instead of GridSearchCV.

        predict():
            Uses the best model to predict test set.
    """
    def __init__(self, x_train, x_test, y_train, y_test, scoring, models, param_grids):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.scoring = scoring
        self.models = models
        self.param_grids = param_grids
        self.best_model_name = None
        self.best_estimator = None
        self.best_score = -1
        self.best_params = None
        self.y_predict = None
        self.search_objects = {}    #Store search objects for each model

    def validate_data(self):
        """
        Validates if training data is not None.
        """
        try:
            if self.x_train is None or self.y_train is None:
                logging.error("Training data must not be empty.")
                raise
        except Exception as e:
            logging.error("Error during validation data in validate_data():\n %s", e)

    def train_model(self, use_random_search=False):
        """
        Trains multiple models using GridSearchCV or RandomizedSearchCV and select the best one based on scoring.

        Arguments:
            use_random_search (bool): If True, uses RandomizedSearchCV instead of GridSearchCV.
        """
        try:
            self.validate_data()
            for name, model in self.models.items():
                if name not in self.param_grids:
                    logging.error("Optimal parameter not found.")
                    raise RuntimeError("All models failed to train. Stopping execution.")
                
                logging.info("Training %s....", name)
                
                try:
                    if use_random_search:
                        gridsearch = RandomizedSearchCV(estimator=model, param_distributions=self.param_grids[name], scoring=self.scoring, n_iter=config["searchCV"]["n_iter"], n_jobs=config["searchCV"]["n_jobs"], cv=config["searchCV"]["cv"], verbose=config["searchCV"]["verbose"])
                    else:
                        gridsearch = GridSearchCV(estimator=model, param_grid=self.param_grids[name], scoring=self.scoring, n_jobs=config["searchCV"]["n_jobs"], cv=config["searchCV"]["cv"], verbose=config["searchCV"]["verbose"])
                    gridsearch.fit(self.x_train, self.y_train)
                except Exception as e:
                    logging.error("Error occurred when training model %s:\n %s", name, e)
                    raise

                self.search_objects[name] = gridsearch      #Store the search objects

                if gridsearch.best_score_ > self.best_score:
                    self.best_estimator = gridsearch.best_estimator_
                    self.best_score = gridsearch.best_score_
                    self.best_params = gridsearch.best_params_
                    self.best_model_name = name

                if self.best_estimator is None:
                    logging.critical("No valid model was successfully trained.")
                    raise RuntimeError("All models failed to train.")

        except Exception as e:
            logging.critical("Critical error in train_model():\n %s", e)
            raise

    def predict(self):
        """
        Uses the best trained model to make predictions on the test set.
        """
        try:
            if self.best_estimator:
                self.y_predict = self.best_estimator.predict(self.x_test)
            else:
                raise ValueError(f"The model not trained. Call train_model() function first.")
        except Exception as e:
            logging.error("Error predict model in predict():\n %s", e)
            raise