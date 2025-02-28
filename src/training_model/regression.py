# Build-in imports
import traceback

# Third-party imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error


# Internal imports
from src.training_model.base_trainer import BaseTrainer
from src.utils.logger_manager import LoggerManager
from src.utils.config_loader import ConfigLoader


# Get logger
logger = LoggerManager.get_logger()

# Get configuration
full_config = ConfigLoader.get_config(file_name="training_model_config.yaml")
try:
    config = full_config["regression.py"]
except Exception as e:
    logger.error("[regression.py] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())


class TrainRegression(BaseTrainer):
    """
    [TrainClassification] - Extend BaseTrainer for classification tasks.

    Arguments:
        x_train, x_test, y_train, y_test (np.ndarray): Training and testing datasets.

    Atributes:
        models (dict[str, object]): Store algorihms.

    Methods:
        train(use_random_search): Train all sefecified models and select the best one.
        evaluate(): Evaluates classification model and return classification report.
    """
    def __init__(self, x_train, x_test, y_train, y_test):
        try:
            # Config for hyperparameters of algorithm.
            scoring_ = config["scoring"]
            param_grids_ = config["param_grids"]
            random_state_ = config["random_state"]
        except Exception as e:
            logger.error("[TrainRegression][__init__] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
            raise

        try:
            self.models = {
                "RandomForestRegressor": RandomForestRegressor(random_state=random_state_),
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "SVR": SVR()
            }
        except Exception as e:
            logger.error("[TrainRegression][__init__] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
            raise

        # Call the parent class contructor
        super.__init__(x_train, x_test, y_train, y_test, scoring_, self.models, param_grids_)

    def train(self, use_random_seach=False):
        """
        [TrainRegression][train] - Train all sefecified models and select the best one.

        Args:
            use_random_seach (bool, optional): If True, uses RandomizedSearchCV instead of GridSearchCV. Defaults to False.
        """

        logger.info("[TrainRegression][train] - Training models with scoring: %s", self.scoring)
        self.train_model()      #Train all models defined in self.models

    def evaluate(self):
        """
        [TrainRegression][evaluate] - Evaluates classification model and return classification report.
        """
        try:
            if self.y_predict is None:
                self.preidct()

            report1 = r2_score(self.y_test, self.y_predict)
            report2 = mean_squared_error(self.y_test, self.y_predict)
            logger.info("[TrainRegression][evaluate] - R2 score for the best model %s is:\n%s", self.best_model_name, report1)
            logger.info("[TrainRegression][evaluate] - Mean squared error for the best model %s is:\n%s", self.best_model_name, report2)
        except Exception as e:
            logger.error("[TrainRegression][evaluate] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())


