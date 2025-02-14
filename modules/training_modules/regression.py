import yaml
import logging
from base_trainer import BaseTrainer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

logging.basicConfig(level=logging.INFO)

class TrainRegression(BaseTrainer):
    """
    TrainRegression extends BaseTrainer for Regression tasks.
    """
    def __init__(self, x_train, x_test, y_train, y_test, config_path="configs/dev_config.yaml"):
        #Load config YAML file
        with open(config_path, "r") as file:
            all_config = yaml.safe_load(file)
            config = sdaasd