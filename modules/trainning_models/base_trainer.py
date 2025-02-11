from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, r2_score, mean_squared_error, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt 
import logging 

#Basic configuration for logging 
logging.basicConfig(level=logging.INFO)

class BaseTrainer:
    """
    BaseTrainer is generic class to train machine learning models that use GridSearchCV or RandomizedGridSearchCV.

    Attributes:
        x_train, x_test, y_train, y_test: Training and testing datasets.
        scoring: Metric is used to evaluate model.
        models: A dictionary that contains the model to be trained.
        param_grids: A dictionary of hyperparameters for the optimization search process.
        search_objects: Stores the GridSearchCV or RandomizedSearchCV for each model.

    Methods: 
        train_model(): Trains and chooses the model based on evaluation metric.
        predict(): Uses the best model to predict test set.
        save_model(filename): Saves the best model to a file.
        load_model(filename): Loads the model from a file.
        save_gridsearch(filename): Saves the GridSearch object to a file.
        load_gridsearch(filename): Loads the GridSearch object to a file.
        validate_data(): Validate train set.
    """
    def __init__(self, x_train, x_test, y_train, y_test, scoring, models, param_grids):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.scoring = scoring 
        self.models = models
        self.param_grids = param_grids
        self.best_estimator = None
        self.best_score = -1
        self.best_params = None
        self.y_predict = None
        self.search_objects = {}    #Store search objects for each model

    def validate_data(self):
        """
        Validates if training data is not None.
        """
        if self.x_train is not None or self.y_train is not None:
            raise ValueError("Training data must not be empty.")

    def train_model(self, use_random_search=False):
        """
        Trains multiple models using GridSearchCV or RandomizedSearchCV and select the best one based on scoring.

        Arguments:
            use_random_search (bool): If True, uses RandomizedSearchCV instead of GridSearchCV
        """
        self.validate_data()
        for name, model in self.models.items():
            if name not in self.param_grids:
                raise KeyError("Optimal parameter not found.")
            
            logging.info(f"Training {name}....")
            gridsearch = None
            if use_random_search:
                gridsearch = RandomizedSearchCV(estimator=model, param_distributions=self.param_grids[name], scoring=self.scoring, n_iter=10, n_jobs=-1, cv=5, verbose=4)
            else:
                gridsearch = GridSearchCV(estimator=model, param_grid=self.param_grids[name], scoring=self.scoring, n_jobs=-1, cv=5, verbose=4)
            gridsearch.fit(self.x_train, self.y_train)

            self.search_objects[name] = gridsearch      #Store the search objects

            if gridsearch.best_score_ > self.best_score:
                self.best_estimator = gridsearch.best_estimator_
                self.best_score = gridsearch.best_score_
                self.best_params = gridsearch.best_params_

        return self.best_estimator

    def predict(self):
        """
        Uses the best trained model to make predictions on the test set.
        """
        if self.best_estimator:
            self.y_predict = self.best_estimator.predict(self.x_test)
            return self.y_predict
        else:
            raise ValueError(f"The model not trained. Call train_model() function first.")
        
    