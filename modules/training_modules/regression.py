import yaml
import logging
from base_trainer import BaseTrainer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Config logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="logs/dev.log",
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#Load config YAML file
config_path="configs/modules_config.yaml"
try:
    with open(config_path, "r") as file:
        full_config = yaml.safe_load(file)
        config = full_config["training_modules/"]["regression.py"] 
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


class TrainRegression(BaseTrainer):
    """
    TrainClassification extend BaseTrainer for classification tasks.

    Arguments:
        x_train, x_test, y_train, y_test: 
            Training and testing datasets.

    Atributes:
        scoring: 
            Metric is used to evaluate model.

        param_grids: 
            A dictionary of hyperparameters for the optimization search process.

        random_state:
            Used to control the randomness of the algorithm.

        models: 
            Store algorithms.

    Methods:
        train(): 
            Train all sefecified models and select the best one.

            Arguments: 
                use_random_search (bool): If True, uses RandomizedSearchCV instead of GridSearchCV.

        predict(): 
            Uses the best model to predict test set (method of BaseTrainer).
            
        evaluate():  
            Evaluates classification model and return classification report.

        plot_confusion_matrix(): 
            Displays the confusion matrix.
    """
    def __init__(self, x_train, x_test, y_train, y_test):
        
        self.scoring = config["scoring"]
        self.param_grids = config["param_grids"]
        self.random_state = config["random_state"]

        try:
            self.models = {
                "RandomForestRegressor": RandomForestRegressor(random_state=self.random_state),
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "SVR": SVR()
            }
        except KeyError as e:
            logging.error("Missing the key in config:\n %s", e)
            raise
        except TypeError as e:
            logging.error("Invalid file in config:\n %s", e)
            raise
        except ValueError as e:
            logging.error("Invalid parameter value:\n %s", e)
        except Exception as e:
            logging.error("Error initializing models:\n %s", e)
            raise

        # Call the parent class contructor
        super.__init__(x_train, x_test, y_train, y_test, self.scoring, self.models, self.param_grids)

    def train(self, use_random_seach=False):

        logging.info("Training models with scoring: %s", self.scoring)
        self.train_model()      #Train all models defined in self.models

    def evaluate(self):

        try:
            if self.y_predict is None:
                self.preidct()

            report1 = r2_score(self.y_test, self.y_predict)
            report2 = mean_squared_error(self.y_test, self.y_predict)
            logging.info("R2 score for the best model %s is:\n%s", self.best_model_name, report1)
            logging.info("Mean squared error for the best model %s is:\n%s", self.best_model_name, report2)
        except Exception as e:
            logging.error("Error during model evaluation in evaluate().")

    def plot_confusion_matrix(self):

        try:
            display = ConfusionMatrixDisplay.from_estimator(self.best_estimator, self.x_test, self.y_test)
            plt.title("Confusion matrix - The best model")
            plt.show()
        except Exception as e:
            logging.error("Error displaying confusion matrix in plot_confusion_matrix():\n %s", e)
            raise

