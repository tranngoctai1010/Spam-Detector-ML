from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

class train_classification:
    def __init__(self, x_train, x_test, y_train, y_test, sc):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.sc = sc    
        self.y_predict = None
        
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=100, random_state=42),
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "LinearSVC": LinearSVC(max_iter=100, random_state=42),
            "GaussianNB": GaussianNB()
        }
        
        self.param_grids = {
            "LogisticRegression": {
                "C": [0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]              
            },
            "RandomForestClassifier": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [5, 10, 15] if len(x_train) < 10000 else [10, 20, 30] if len(x_train) < 50000 else [20, 40, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
                "bootstrap": [True, False]
            },
            "LinearSVC": {
                "C": [0.01, 0.1, 1, 10, 100],
                "max_iter": [1000, 5000, 10000],
                "dual": [True, False]
            },
            "GaussianNB": {
                "var_smoothing": [1e-9, 1e-8, 1e-7]
            }
        }
        
        self.best_estimator = None
        self.best_score = -1
        self.best_params = None
    
    def train_model(self):
        for name, model in self.models.items():
            gridsearch = GridSearchCV(estimator=model, param_grid=self.param_grids[name], scoring=self.sc, n_jobs=-1, cv=5, verbose=4)
            gridsearch.fit(self.x_train, self.y_train)
            if self.best_score < gridsearch.best_score_:
                self.best_estimator = gridsearch.best_estimator_
                self.best_score = gridsearch.best_score_
                self.best_params = gridsearch.best_params_
        return self.best_estimator
    
    def predict(self):
        if self.best_estimator is not None:
            self.y_predict = self.best_estimator.predict(self.x_test)
            return self.y_predict
        else:
            raise Exception("Model is not trained yet. Please call the 'train()' method first.")
    
    def evaluate(self):
        if self.y_predict is not None:
            return classification_report(self.y_test, self.y_predict)
    


class train_regression:
    def __init__(self, x_train, x_test, y_train, y_test, sc):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.sc = sc
        self.y_predict = None

        self.models = {
            "RandomForest": RandomForestRegressor(random_state=42),
            "LinearRegression": LinearRegression(),
            "KNN": KNeighborsRegressor(),
            "SVR": SVR()
        }

        self.param_grids = {
            "RandomForest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
                "bootstrap": [True, False]
            },
            "LinearRegression": {},  # Linear regression doesn't require optimization
            "KNN": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
            },
            "SVR": {
                "kernel": ["linear", "poly", "rbf"],
                "C": [0.1, 1, 10, 100],
                "epsilon": [0.01, 0.1, 0.2],
                "gamma": ["scale", "auto"]
            }
        }

        self.best_model = None
        self.best_params = None
        self.best_score = -1

    def train_model(self):
        for name, model in self.models.items():
            gridsearch = GridSearchCV(estimator=model, param_grid=self.param_grids[name], scoring=self.sc, n_jobs=-1, cv=5, verbose=4)
            gridsearch.fit(self.x_train, self.y_train)
            if self.best_score < gridsearch.best_score_:
                self.best_estimator = gridsearch.best_estimator_
                self.best_score = gridsearch.best_score_
                self.best_params = gridsearch.best_params_
        return self.best_estimator

    def predict(self):
        if self.best_estimator is not None:
            self.y_predict = self.best_estimator.predict(self.x_test)
            return self.y_predict
        else:
            raise Exception("Model is not trained yet. Please call the 'train()' method first.")
    
    def evaluate(self, metric):
        if self.y_predict is not None:
            if metric == "r2_score":
                return r2_score(self.y_test, self.y_predict)
            if metric == "mean_squared_error":
                return mean_squared_error(self.y_test, self.y_predict)
        