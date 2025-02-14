from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, r2_score, mean_squared_error, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

class BaseTrainer:
    """
    BaseTrainer is a generic class for training machine learning models using GridSearchCV or RandomizedSearchCV.

    Attributes:
        x_train, x_test, y_train, y_test: Training and testing datasets.
        scoring: Metric used for evaluating models.
        models: Dictionary of models to be trained.
        param_grids: Dictionary of hyperparameters for grid search.
        search_objects: Stores GridSearchCV or RandomizedSearchCV objects for each model.

    Methods:
        train_model(): Trains models and selects the best one based on the scoring metric.
        predict(): Uses the best model to make predictions on the test set.
        save_model(filename): Saves the best model to a file.
        load_model(filename): Loads a model from a file.
        save_gridsearch(filename): Saves the GridSearchCV object to a file.
        load_gridsearch(filename): Loads the GridSearchCV object from a file.
        validate_data(): Validates the training data.
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
        self.search_objects = {}  # Store search objects for each model

    def validate_data(self):
        """
        Validates if training data is not None.
        """
        if self.x_train is None or self.y_train is None:
            raise ValueError("Dữ liệu huấn luyện không được để trống.")

    def train_model(self, use_random_search=False):
        """
        Trains multiple models using GridSearchCV or RandomizedSearchCV and selects the best one based on scoring.
        Args:
            use_random_search (bool): If True, uses RandomizedSearchCV instead of GridSearchCV.
        """
        self.validate_data()
        for name, model in self.models.items():
            if name not in self.param_grids:
                raise KeyError(f"Không tìm thấy tham số tối ưu cho {name}")

            logging.info(f"Training {name}...")
            if use_random_search:
                search = RandomizedSearchCV(estimator=model, param_distributions=self.param_grids[name], 
                                            scoring=self.scoring, n_iter=10, n_jobs=-1, cv=5, verbose=4)
            else:
                search = GridSearchCV(estimator=model, param_grid=self.param_grids[name], 
                                      scoring=self.scoring, n_jobs=-1, cv=5, verbose=4)
            search.fit(self.x_train, self.y_train)
            
            self.search_objects[name] = search  # Store the search object

            if search.best_score_ > self.best_score:
                self.best_estimator = search.best_estimator_
                self.best_score = search.best_score_
                self.best_params = search.best_params_

        return self.best_estimator

    def predict(self):
        """
        Uses the best trained model to make predictions on the test set.
        """
        if self.best_estimator:
            self.y_predict = self.best_estimator.predict(self.x_test)
            return self.y_predict
        else:
            raise ValueError("Model chưa được train. Hãy gọi train_model() trước.")

    def save_model(self, filename='best_model.pkl'):
        """
        Saves the best trained model to a file.
        """
        if self.best_estimator:
            joblib.dump(self.best_estimator, filename)
            logging.info(f"Model đã lưu vào {filename}")
        else:
            raise ValueError("Không có model để lưu.")

    def load_model(self, filename='best_model.pkl'):
        """
        Loads a model from a file.
        """
        self.best_estimator = joblib.load(filename)
        logging.info(f"Model đã được tải từ {filename}")

    def save_gridsearch(self, filename='gridsearch.pkl'):
        """
        Saves all GridSearchCV or RandomizedSearchCV objects to a file.
        """
        if self.search_objects:
            joblib.dump(self.search_objects, filename)
            logging.info(f"Tất cả các đối tượng GridSearchCV đã lưu vào {filename}")
        else:
            raise ValueError("Không có đối tượng GridSearchCV nào để lưu.")

    def load_gridsearch(self, filename='gridsearch.pkl'):
        """
        Loads all GridSearchCV or RandomizedSearchCV objects from a file.
        """
        self.search_objects = joblib.load(filename)
        logging.info(f"Tất cả các đối tượng GridSearchCV đã được tải từ {filename}")


from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor

class TrainClassification(BaseTrainer):
    """
    TrainClassification extends BaseTrainer for classification tasks.

    Methods:
        evaluate(): Evaluates the classification model and returns a classification report.
        plot_confusion_matrix(): Displays the confusion matrix.
    """
    def __init__(self, x_train, x_test, y_train, y_test, scoring='accuracy'):
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "LinearSVC": LinearSVC(max_iter=1000, random_state=42),
            "GaussianNB": GaussianNB()
        }

        param_grids = {
            "LogisticRegression": {"C": [0.1, 1, 10], "solver": ["liblinear", "lbfgs"]},
            "RandomForestClassifier": {"n_estimators": [100, 200], "max_depth": [5, 10]},
            "LinearSVC": {"C": [0.01, 0.1, 1], "dual": [True, False]},
            "GaussianNB": {"var_smoothing": [1e-9, 1e-8]}
        }

        super().__init__(x_train, x_test, y_train, y_test, scoring, models, param_grids)

    def evaluate(self):
        """
        Evaluates the classification model and returns a detailed classification report.
        """
        if self.y_predict is not None:
            logging.info(f"Best Parameters: {self.best_params}")
            logging.info(f"Best Score: {self.best_score}")
            return classification_report(self.y_test, self.y_predict)
        else:
            raise ValueError("Chưa có dự đoán để đánh giá.")

    def plot_confusion_matrix(self):
        """
        Displays the confusion matrix for classification results.
        """
        if self.y_predict is not None:
            disp = ConfusionMatrixDisplay.from_estimator(self.best_estimator, self.x_test, self.y_test)
            plt.title('Confusion Matrix')
            plt.show()
        else:
            raise ValueError("Chưa có dự đoán để hiển thị.")


class TrainRegression(BaseTrainer):
    """
    TrainRegression extends BaseTrainer for regression tasks.

    Methods:
        evaluate(metric): Evaluates the regression model using r2 or mse metric.
    """
    def __init__(self, x_train, x_test, y_train, y_test, scoring='r2'):
        models = {
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
            "LinearRegression": LinearRegression(),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "SVR": SVR()
        }

        param_grids = {
            "RandomForestRegressor": {"n_estimators": [100, 200], "max_depth": [10, 20]},
            "LinearRegression": {},
            "KNeighborsRegressor": {"n_neighbors": [3, 5], "weights": ["uniform", "distance"]},
            "SVR": {"kernel": ["linear", "rbf"], "C": [0.1, 1]}
        }

        super().__init__(x_train, x_test, y_train, y_test, scoring, models, param_grids)

    def evaluate(self, metric='r2'):
        """
        Evaluates the regression model using the specified metric (r2 or mse).
        """
        if self.y_predict is not None:
            logging.info(f"Best Parameters: {self.best_params}")
            logging.info(f"Best Score: {self.best_score}")
            if metric == "r2":
                return r2_score(self.y_test, self.y_predict)
            elif metric == "mse":
                return mean_squared_error(self.y_test, self.y_predict)
            else:
                raise ValueError("Metric không hợp lệ.")
        else:
            raise ValueError("Chưa có dự đoán để đánh giá.")
