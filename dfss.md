import yaml
import logging
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from modules.train_models.base_trainer import BaseTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

class TrainClassification(BaseTrainer):
    """
    TrainClassification extends BaseTrainer for classification tasks.

    Methods:
        evaluate(): Evaluates the classification model and returns a classification report.
        plot_confusion_matrix(): Displays the confusion matrix.
    """
    def __init__(self, x_train, x_test, y_train, y_test, config_path='configs/pipeline_config.yaml'):
        # Load config from YAML file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.scoring = config['training']['scoring']
        self.param_grids = config['hyperparameters']
        self.random_state = config['data']['random_state']

        # Define available models
        self.available_models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "RandomForestClassifier": RandomForestClassifier(random_state=self.random_state),
            "LinearSVC": LinearSVC(max_iter=1000, random_state=self.random_state),
            "GaussianNB": GaussianNB()
        }

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        super().__init__(x_train, x_test, y_train, y_test, self.scoring, self.available_models, self.param_grids)

    def train(self):
        """
        Train all specified models and select the best one.
        """
        logging.info(f"Training models with scoring: {self.scoring}")
        self.train_model()  # Train all models defined in available_models

    def evaluate(self):
        """
        Evaluates the best trained classification model and returns a classification report.
        """
        if self.y_predict is None:
            self.predict()
        report = classification_report(self.y_test, self.y_predict)
        logging.info(f"Classification Report for Best Model:\n{report}")
        print(report)

    def plot_confusion_matrix(self):
        """
        Displays the confusion matrix for the best trained model.
        """
        disp = ConfusionMatrixDisplay.from_estimator(self.best_estimator, self.x_test, self.y_test)
        plt.title(f'Confusion Matrix - Best Model')
        plt.show()
