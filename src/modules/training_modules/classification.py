# Build-in imports
import traceback

#Third-party imports
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# Internal imports
from src.modules.training_modules.base_trainer import BaseTrainer
from src.modules.utils.logger_manager import LoggerManager
from src.modules.utils.config_loader import ConfigLoader


# Get logger
logger = LoggerManager.get_logger()

# Get configuration
full_config = ConfigLoader.get_config(file_name="modules_config.yaml")
try:
    config = full_config["training_modules"]["classification.py"]
except Exception as e:
    logger.error(f"[classification.py] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
    raise


class TrainClassification(BaseTrainer):
    """
    [TrainClassification] - Extend BaseTrainer for classification tasks.

    Arguments:
        x_train, x_test, y_train, y_test (np.ndarray): Training and testing datasets.

    Atributes:
        models (dict[str, object]): Store algorihms.

    Methods:
        train(use_random_search): Train all sefecified models and select the best one.
        evaluate(): Evaluates classification model and return classification report.
        plot_confusion_matrix(): Displays the confusion matrix.
    """
    def __init__(self, x_train, x_test, y_train, y_test):

        try:
            # Config for hyperparameters of algorithm.
            scoring_ = config["scoring"]
            param_grids_ = config["param_grids"]
            random_state_ = config["random_state"]
            max_iter_ = config["max_iter"]
        except Exception as e:
            logger.error("[TrainClassification][__init__] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
            raise

        try:
            self.models = {
                "LogisticRegression": LogisticRegression(max_iter=max_iter_, random_state=random_state_),
                "RandomForestClassifier": RandomForestClassifier(random_state=random_state_),
                "LinearSVC": LinearSVC(max_iter=max_iter_, random_state=random_state_),
                "GaussianNB": GaussianNB(),
                "MultinomialNB": MultinomialNB()
            }
        except Exception as e:
            logger.error("[TrainClassification][__init__] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
            raise
            
        #Call the parent class constructor 
        super().__init__(x_train, x_test, y_train, y_test, scoring_, self.models, param_grids_)

    def train(self, use_random_search = False):
        """
        [TrainClassification][train] - Train all sefecified models and select the best one.

        Args:
            use_random_search (bool, optional): If True, uses RandomizedSearchCV instead of GridSearchCV. Defaults to False.
        """

        logger.info("[TrainClassification][train]- Training model with scoring: %s.", self.scoring)
        self.train_model()      #Train all models defined in self.models

    def evaluate(self):
        """
        [TrainClassification][evaluate] - Evaluates classification model and return classification report.
        """
        try:
            if self.y_predict is None:
                self.predict()
            report = classification_report(self.y_test, self.y_predict)
            logger.info("[TrainClassification][evaluate] - Classification report for the best model is %s:\n%s", self.best_model_name, report)
        except Exception as e:
            logger.error("[TrainClassification][evaluate] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
            raise

    def plot_confusion_matrix(self):
        try:
            display = ConfusionMatrixDisplay.from_estimator(self.best_estimator, self.x_test, self.y_test)
            plt.title("[TrainClassification][plot_confusion_matrix] - Confusion matrix - The best model")
            plt.show()
        except Exception as e:
            logger.error("[TrainClassification][plot_confusion_matrix] - Error %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
            raise

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

