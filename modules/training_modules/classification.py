import yaml
import logging
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from base_trainer import BaseTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

#Config logging
logging.basicConfig(
    level=logging.DEBUG,
    filename="logs/dev.log",
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)


#Load config YAML file
config_path="configs/dev_config.yaml"
try:
    with open(config_path, "r") as file:
        full_config = yaml.safe_load(file)
        config = full_config["modules/"]["training_modules/"]["classification.py"] 
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


class TrainClassification(BaseTrainer):
    """
    TrainClassification extend BaseTrainer for classification tasks.

    Methods:
        evaluate(): Evaluates classification model and return classification report.
        plot_confusion_matrix(): Displays the confusion matrix.
    """
    def __init__(self, x_train, x_test, y_train, y_test):

        self.scoring = config["scoring"]
        self.param_grids = config["param_grids"]
        self.random_state = config["random_state"]

        try:
            available_models = {
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=self.random_state),
                "RandomForestClassifier": RandomForestClassifier(random_state=self.random_state),
                "LinearSVC": LinearSVC(max_iter=1000, random_state=self.random_state),
                "GaussianNB": GaussianNB()
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

        #Call the parent class constructor 
        super().__init__(x_train, x_test, y_train, y_test, self.scoring, available_models, self.param_grids)

    def train(self):
        """
        Train all sefecified models and select the best one.
        """
        logging.info("Training model with scoring %s.", self.scoring)
        self.train_model()      #Train all models defined in available_models

    def evaluate(self):
        """
        Evaluate the best classificatoin model and return classification report.
        """
        try:
            if self.y_predict is None:
                self.predict()
            report = classification_report(self.y_test, self.y_predict)
            logging.info("Classification report for the best model is %s:\n%s", self.best_model_name, report)
        except Exception as e:
            logging.error("Error during model evaluation in evaluate(): %s", e)
            raise

    def plot_confusion_matrix(self):
        """
        Display confusion matrix for the best trained model.
        """
        try:
            display = ConfusionMatrixDisplay.from_estimator(self.best_estimator, self.x_test, self.y_test)
            plt.title("Confusion matrix - The best model")
            plt.show()
        except Exception as e:
            logging.error("Error displaying confusion matrix in plot_confusion_matrix():\n %s", e)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate synthetic dataset with 500 samples, 10 features, and 2 classes
    x, y = make_classification(n_samples=500, n_features=10, n_informative=7, n_redundant=3, n_classes=2, random_state=42)

    # Split dataset: 80% train, 20% test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    train_classification = TrainClassification(x_train, x_test, y_train, y_test)

    train_classification.train()
    train_classification.evaluate()
    # train_classification.plot_confusion_matrix()

