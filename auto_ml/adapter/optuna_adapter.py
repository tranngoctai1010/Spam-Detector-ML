from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
import optuna

class OptunaAdapter(BaseEstimator):
    def __init__(self, estimator, param_distributions, n_trials=20, cv=5):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_trials = n_trials
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = {}

    def fit(self, X, y):
        def objective(trial):
            params = {key: trial.suggest_int(key, *values) if isinstance(values, tuple) else values
                      for key, values in self.param_distributions.items()}
            self.estimator.set_params(**params)
            score = cross_val_score(self.estimator, X, y, cv=self.cv).mean()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        self.best_estimator_ = self.estimator.set_params(**study.best_params)
        self.best_estimator_.fit(X, y)
        self.cv_results_ = {'params': [t.params for t in study.trials], 
                           'mean_test_score': [t.value for t in study.trials]}
        return self

    def predict(self, X):
        if self.best_estimator_ is None:
            raise ValueError("Model not fitted yet.")
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        if self.best_estimator_ is None:
            raise ValueError("Model not fitted yet.")
        return self.best_estimator_.score(X, y)

    def set_params(self, **params):
        self.param_distributions.update(params)
        return self