from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, r2_score, mean_squared_error

#Automatically find the optimal for classification
def auto_train_classification(x, y, sc):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=100)
    }

    param_grids = {
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        },
        "LogisticRegression": {
            "C": [0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"]
        }
    }

    best_model = None
    best_params = None
    best_score = 0
    for name, model in models.items():
        grid = GridSearchCV(estimator=model, param_grid=param_grids[name], scoring=sc, cv=5, verbose=4, n_jobs=-1)
        grid.fit(x_train, y_train)
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_params = grid.best_params_
            best_model = grid.best_estimator_

    print(f"Best model: {best_model}")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")
    return best_model

def auto_train_regression(x, y, sc):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "LinearRegression": LinearRegression()
    }

    param_grids = {
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        },
        "LinearRegression": {}  #Linear regression doesn't require optimization
    }

    best_model = None
    best_params = None
    best_score = 0
    for name, model in models.items():
        grid = GridSearchCV(estimator=model, param_grid=param_grids[name], scoring=sc, cv=5, verbose=4, n_jobs=-1)
        grid.fit(x_train, y_train)
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_params = grid.best_params_
            best_model = grid.best_estimator_

    print(f"Best model: {best_model}")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")
    return best_model