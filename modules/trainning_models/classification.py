import yaml
import logging
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from base_trainer import BaseTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

class TrainClassification:
    """
    TrainClassification extend BaseTrainer for classification tasks.

    Methods:
        evaluate(): Evaluates classification model and return classification report.
        plot_confusion_matrix(): Displays the confusion matrix.
    """
    def __init__(self, x_train, x_test, y_train, y_test, config_path="configs/dev_config.yaml"):
        #Load config YAML file
        with open(config_path, "r") as file:
            full_config = yaml.safe_load(file)
            config = full_config["classification"]