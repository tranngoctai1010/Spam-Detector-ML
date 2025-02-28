import traceback

# Third-party libraries
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Internal imports
from src.utils.logger_manager import LoggerManager
from src.utils.config_loader import ConfigLoader


# Create logger
logger = LoggerManager.get_logger()

# Get configuration 
full_config = ConfigLoader.get_config(file_name="training_model_config.yaml")
try:
    config = full_config["base_trainer.py"]
except Exception as e:
    logger.error(f"[base_trainer.py] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
    raise


class BaseTrainer():
    """
    [BaseTrainer] - Generic class to train machine learning models that use GridSearchCV or RandomizedGridSearchCV.
    
    Attributes:
        x_train, x_test, y_train, y_test (numpy.ndarray): Training and testing datasets.
        scoring (str): Metric is used to evaluate model.
        models (dict[str, object]): A dictionary that contains the model to be trained.
        param_grids (dict[str, dict]): A dictionary of hyperparameters for the optimization search process.
        best_model_name (str): Name of the best model.
        best_estimator (object): The best-performing model selected by GridSearchCV or a manually trained model.
        best_score (float): The highest score achieved by the model.
        best_params (dict): The best hyperparameters found for the model.
        self.search_objects (dict[str, object]): Stores search objects for each model that has been optimized by GridSearchCV.
        y_predict (numpy.ndarrray): Predicted target values from the best model.

    Methods:
        validate_data(): Validates if training and testing data is not None.
        train_model(use_random_search): Trains multiple models using GridSearchCV or RandomizedSearchCV and select the best one based on scoring.
        predict(): Uses the best model to predict test set.
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
        self.search_objects = {}
        self.y_predict = None

    def validate_data(self):
        """
        [BaseTrainer][validate_data] - Validates if training data is not None.
        
        Raises: 
            ValueError: If x_train or y_train is None, indicating missing traing data.
        """
        if self.x_train is None or self.y_train is None:
            raise ValueError("[BaseTrainer][validate_data] - Training data must not be empty.")

    def train_model(self, use_random_search=False):
        """
        [BaseTrainer][train_model] - Trains multiple models using GridSearchCV or RandomizedSearchCV and select the best one based on scoring.

        Args:
            use_random_search (bool): If True, uses RandomizedSearchCV instead of GridSearchCV.
            
        Raises:
            ValueError: If there are missing model names in param_grids.
            RuntimeError: If all models fail to train.
        """
        self.validate_data()
        missing_keys = set(self.models.keys()) - set(self.param_grids.keys())

        if missing_keys:
            raise ValueError(f"[BaseTrainer][train_model] - Missing model name {missing_keys}.")
        
        for name, model in self.models.items():
            
            try:
                logger.info("[BaseTrainer][train_model] - Training %s model .....", name)
                
                try:
                    # Config for hyperparameters of RandomizedSearchCV.
                    n_iter_ = config["n_iter"]
                    n_jobs_ = config["n_jobs"]
                    cv_ = config["cv"]
                    verbose_ = config["verbose"]
                except Exception as e:
                    logger.error("[BaseTrainer][train_model] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
                    raise
                    
                if use_random_search:
                    gridsearch = RandomizedSearchCV(estimator=model, param_distributions=self.param_grids[name], scoring=self.scoring, n_iter=n_iter_, n_jobs=n_jobs_, cv=cv_, verbose=verbose_)
                    
                else:
                    gridsearch = GridSearchCV(estimator=model, param_grid=self.param_grids[name], scoring=self.scoring, n_jobs=n_jobs_, cv=cv_, verbose=verbose_)
                    
                gridsearch.fit(self.x_train, self.y_train)
            except Exception as e:
                logger.error("[BaseTrainer][train_model] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
                raise

            self.search_objects[name] = gridsearch      #Store the search objects

            if gridsearch.best_score_ > self.best_score:
                self.best_estimator = gridsearch.best_estimator_
                self.best_score = gridsearch.best_score_
                self.best_params = gridsearch.best_params_
                self.best_model_name = name

        if self.best_estimator is None:
            raise RuntimeError("[BaseTrainer][train_model] - All models failed to train.")

    def predict(self):
        """
        Uses the best trained model to make predictions on the test set.
        
        Raises:
            ValueError: If the model is not trained.
        """
        if self.best_estimator:
            logger.info("[BaseTrainer][predict] - Using model %s to make predictions.", self.best_model_name)
            self.y_predict = self.best_estimator.predict(self.x_test)
        else:
            raise ValueError(f"[BaseTrainer][predict] - The model not trained. Call train_model() function first.")
    
        
    