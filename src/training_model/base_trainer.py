from __future__ import annotations
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Any, Optional, Protocol, TypeVar, Tuple, Type, Literal, Union
import logging

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import matplotlib.pyplot as plt
from tqdm.rich import tqdm


# Data definitions
XData = np.ndarray
YData = np.ndarray
Dataset = Tuple[XData, XData, YData, YData]
ModelDict = Dict[str, BaseEstimator]
ParamGrid = Dict[str, Any]
SearchStrategyName = Literal["grid", "random"]
SearchCVType = TypeVar("SearchCVType", GridSearchCV, RandomizedSearchCV)
ModelFactory = Union[ClassificationModelFactory, ]


class SearchStrategy(Protocol):
    """Protocol for hyperparameter search strategy."""
    def __init__(
        self,
        model: BaseEstimator,
        param_grid: Dict[str, Any],
        scoring: str,
        config: SearchConfig
    ): ...

    def __getattr__(self, name): ...


class GridSearchStrategy:
    """Wrapper around GridSearchCV with transparent method access."""
    def __init__(
        self,
        model: BaseEstimator,
        param_grid: ParamGrid,
        scoring: str,
        config: SearchConfig
    ):
        self.search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=config.n_jobs,
            cv=config.cv,
            verbose=config.verbose
        )
    def __getattr__(self, name):
        """Delegate attribute access to internal GridSearchCV."""
        return getattr(self.search, name)
    

class RandomSearchStrategy:
    """Wrapper around RandomizedSearchCV with transparent method access."""
    def __init__(
        self,
        model: BaseEstimator,
        param_grid: ParamGrid,
        scoring: str,
        config: SearchConfig
    ):
        self.search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=config.n_iter,
            scoring=scoring,
            n_jobs=config.n_jobs,
            cv=config.cv,
            verbose=config.verbose
        )

    def __getattr__(self, name):
        """Delegate attribute access to internal RandomizedSearchCV."""
        return getattr(self.search, name)


class SearchStrategyFactory:
    """Factory for creating search strategies."""

    _strategies: Dict[str, Type[SearchStrategy]] = {
        "grid": GridSearchStrategy,
        "random": RandomSearchStrategy
    }

    @classmethod
    def get_strategy(cls, strategy_name: SearchStrategyName) -> SearchStrategy:
        strategy_class = cls._strategies.get(strategy_name, GridSearchStrategy)
        return strategy_class()




class BaseTrainerConfig:
    ...


# Base Trainer
class BaseTrainer:
    """Base class for model training."""
    
    def __init__(
        self,
        estimator: ModelDict | BaseEstimator,
        search_config: SearchConfig,
        search_factory: SearchStrategyName,
        logger: logging.Logger | None = None
    ) -> None:
        self._x_train, self._x_test, self._y_train, self._y_test = dataset
        self._config = config
        self._logger = logger
        self._model_factory = model_factory
        self._search_factory = search_factory
        self._models = model_factory.create_models(
            config.max_iter,
            config.random_state
        )
        self._best_estimator: Optional[BaseEstimator] = None
        self._best_model_name: Optional[str] = None
        self._best_score: float = -float("inf")
        self._best_params: Optional[Dict[str, Any]] = None
        self._y_predict: Optional[YData] = None
        self._search_objects: Dict[str, Any] = {}

    def train(self, use_random_search: bool = False) -> BaseEstimator:
        """Train all models and select the best one."""
        strategy = self._search_factory.get_strategy("random" if use_random_search else "grid")
        self._validate_input()

        for name, model in self._models.items():
            self._logger.info("Training model: %s", name)
            with tqdm(total=1, desc=f"Training {name}", bar_format="{desc}: {percentage:3.0f}%|{bar}|") as pbar:
                search = self._train_single_model(name, model, strategy)
                self._search_objects[name] = search
                self._update_best_model(name, search)
                pbar.update(1)

        if not self._best_estimator:
            self._logger.error("No models trained successfully")
            raise RuntimeError("All models failed to train")
        return self._best_estimator

    def predict(self) -> YData:
        """Generate predictions using the best model."""
        if not self._best_estimator:
            self._logger.error("No trained model available")
            raise ValueError("Model not trained. Call train() first.")
        self._logger.info("Predicting with model: %s", self._best_model_name)
        self._y_predict = self._best_estimator.predict(self._x_test)
        return self._y_predict

    def evaluate(self) -> str:
        """Evaluate the model and return classification report."""
        if self._y_predict is None:
            self.predict()
        report = classification_report(self._y_test, self._y_predict)
        self._logger.info("Evaluation for %s:\n%s", self._best_model_name, report)
        return report

    def plot_confusion_matrix(self) -> None:
        """Plot confusion matrix for the best model."""
        if not self._best_estimator:
            raise ValueError("Model not trained. Call train() first.")
        display = ConfusionMatrixDisplay.from_estimator(
            self._best_estimator,
            self._x_test,
            self._y_test
        )
        plt.title(f"Confusion Matrix - {self._best_model_name}")
        plt.show()

    def _train_single_model(
        self,
        name: str,
        model: BaseEstimator,
        strategy: SearchStrategy
    ) -> Any:
        param_grid = self._config.param_grids.get(name, {})
        search = strategy.create_search(
            model,
            param_grid,
            self._config.scoring,
            self._config.search_config
        )
        return strategy.fit(search, self._x_train, self._y_train)

    def _validate_input(self) -> None:
        missing_keys = set(self._models.keys()) - set(self._config.param_grids.keys())
        if missing_keys:
            self._logger.error("Missing param grids for models: %s", missing_keys)
            raise ValueError(f"Missing param grids for models: {missing_keys}")

    def _update_best_model(self, name: str, search: Any) -> None:
        if search.best_score_ > self._best_score:
            self._best_score = search.best_score_
            self._best_estimator = search.best_estimator_
            self._best_params = search.best_params_
            self._best_model_name = name
            self._logger.info("New best model: %s with score %.4f", name, self._best_score)


class ClassificationTrainer(BaseTrainer):

    def __init__(
        self,
        estimator: ModelDict | BaseEstimator,
        search_config: SearchConfig,
        search_factory: SearchStrategyName,
        logger: logging.Logger | None = None
    ):










#     from __future__ import annotations
# from abc import ABC, abstractmethod
# from collections import namedtuple
# from dataclasses import dataclass
# from typing import Dict, Any, Optional, Protocol, TypeVar, Tuple, Type
# import logging

# import numpy as np
# from sklearn.base import BaseEstimator
# from sklearn.metrics import classification_report, ConfusionMatrixDisplay
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVC
# from sklearn.naive_bayes import GaussianNB, MultinomialNB
# import matplotlib.pyplot as plt
# from tqdm.rich import tqdm

# # Type definitions
# XData = np.ndarray
# YData = np.ndarray
# Dataset = Tuple[XData, XData, YData, YData]
# ModelDict = Dict[str, BaseEstimator]
# ParamGrid = Dict[str, Dict[str, list]]
# T = TypeVar("T", GridSearchCV, RandomizedSearchCV)


# # Configuration classes
# @dataclass(frozen=True)
# class SearchConfig:
#     """Immutable configuration for hyperparameter search."""
#     n_iter: int = 10
#     n_jobs: int = -1
#     cv: int = 5
#     verbose: int = 1


# @dataclass(frozen=True)
# class TrainerConfig:
#     """Immutable configuration for classification trainer."""
#     scoring: str
#     param_grid: ParamGrid
#     random_state: int
#     max_iter: int
#     search_config: SearchConfig

#     @classmethod
#     def from_dict(cls, config_dict: Dict[str, Any]) -> TrainerConfig:
#         """Create TrainerConfig from dictionary."""
#         return cls(
#             scoring=config_dict["scoring"],
#             param_grid=config_dict["param_grid"],
#             random_state=config_dict["random_state"],
#             max_iter=config_dict["max_iter"],
#             search_config=SearchConfig(**config_dict.get("search_config", {}))
#         )


# # Protocols
# class Logger(Protocol):
#     """Protocol for logging interface."""
#     def info(self, msg: str, *args: Any) -> None: ...
#     def error(self, msg: str, *args: Any) -> None: ...


# class SearchStrategy(Protocol):
#     """Protocol for hyperparameter search strategy."""
#     def create_search(
#         self,
#         model: BaseEstimator,
#         param_grid: Dict[str, Any],
#         scoring: str,
#         config: SearchConfig
#     ) -> T: ...

#     def fit(
#         self,
#         search: T,
#         x_train: XData,
#         y_train: YData
#     ) -> T: ...


# # Search Strategy Implementations
# class GridSearchStrategy:
#     """Implementation of grid search strategy."""
    
#     def create_search(
#         self,
#         model: BaseEstimator,
#         param_grid: Dict[str, Any],
#         scoring: str,
#         config: SearchConfig
#     ) -> GridSearchCV:
#         return GridSearchCV(
#             estimator=model,
#             param_grid=param_grid,
#             scoring=scoring,
#             n_jobs=config.n_jobs,
#             cv=config.cv,
#             verbose=config.verbose
#         )

#     def fit(
#         self,
#         search: GridSearchCV,
#         x_train: XData,
#         y_train: YData
#     ) -> GridSearchCV:
#         return search.fit(x_train, y_train)


# class RandomSearchStrategy:
#     """Implementation of random search strategy."""
    
#     def create_search(
#         self,
#         model: BaseEstimator,
#         param_grid: Dict[str, Any],
#         scoring: str,
#         config: SearchConfig
#     ) -> RandomizedSearchCV:
#         return RandomizedSearchCV(
#             estimator=model,
#             param_distributions=param_grid,
#             n_iter=config.n_iter,
#             scoring=scoring,
#             n_jobs=config.n_jobs,
#             cv=config.cv,
#             verbose=config.verbose
#         )

#     def fit(
#         self,
#         search: RandomizedSearchCV,
#         x_train: XData,
#         y_train: YData
#     ) -> RandomizedSearchCV:
#         return search.fit(x_train, y_train)


# class SearchStrategyFactory:
#     """Factory for creating search strategies."""
    
#     _strategies: Dict[str, Type[SearchStrategy]] = {
#         "grid": GridSearchStrategy,
#         "random": RandomSearchStrategy
#     }

#     @classmethod
#     def get_strategy(cls, strategy_name: str) -> SearchStrategy:
#         strategy_class = cls._strategies.get(strategy_name, GridSearchStrategy)
#         return strategy_class()


# # Model Factory
# class ClassificationModelFactory:
#     """Factory for creating classification models."""
    
#     @staticmethod
#     def create_models(max_iter: int, random_state: int) -> ModelDict:
#         return {
#             "LogisticRegression": LogisticRegression(
#                 max_iter=max_iter,
#                 random_state=random_state
#             ),
#             "RandomForestClassifier": RandomForestClassifier(
#                 random_state=random_state
#             ),
#             "LinearSVC": LinearSVC(
#                 max_iter=max_iter,
#                 random_state=random_state
#             ),
#             "GaussianNB": GaussianNB(),
#             "MultinomialNB": MultinomialNB()
#         }


# # Base Trainer
# class BaseTrainer:
#     """Base class for model training."""
    
#     def __init__(
#         self,
#         dataset: Dataset,
#         config: TrainerConfig,
#         logger: Logger,
#         model_factory: ClassificationModelFactory,
#         search_factory: SearchStrategyFactory
#     ) -> None:
#         self._x_train, self._x_test, self._y_train, self._y_test = dataset
#         self._config = config
#         self._logger = logger
#         self._model_factory = model_factory
#         self._search_factory = search_factory
#         self._models = model_factory.create_models(
#             config.max_iter,
#             config.random_state
#         )
#         self._best_estimator: Optional[BaseEstimator] = None
#         self._best_model_name: Optional[str] = None
#         self._best_score: float = -float("inf")
#         self._best_params: Optional[Dict[str, Any]] = None
#         self._y_predict: Optional[YData] = None
#         self._search_objects: Dict[str, Any] = {}

#     def train(self, use_random_search: bool = False) -> BaseEstimator:
#         """Train all models and select the best one."""
#         strategy = self._search_factory.get_strategy("random" if use_random_search else "grid")
#         self._validate_input()

#         for name, model in self._models.items():
#             self._logger.info("Training model: %s", name)
#             with tqdm(total=1, desc=f"Training {name}", bar_format="{desc}: {percentage:3.0f}%|{bar}|") as pbar:
#                 search = self._train_single_model(name, model, strategy)
#                 self._search_objects[name] = search
#                 self._update_best_model(name, search)
#                 pbar.update(1)

#         if not self._best_estimator:
#             self._logger.error("No models trained successfully")
#             raise RuntimeError("All models failed to train")
#         return self._best_estimator

#     def predict(self) -> YData:
#         """Generate predictions using the best model."""
#         if not self._best_estimator:
#             self._logger.error("No trained model available")
#             raise ValueError("Model not trained. Call train() first.")
#         self._logger.info("Predicting with model: %s", self._best_model_name)
#         self._y_predict = self._best_estimator.predict(self._x_test)
#         return self._y_predict

#     def evaluate(self) -> str:
#         """Evaluate the model and return classification report."""
#         if self._y_predict is None:
#             self.predict()
#         report = classification_report(self._y_test, self._y_predict)
#         self._logger.info("Evaluation for %s:\n%s", self._best_model_name, report)
#         return report

#     def plot_confusion_matrix(self) -> None:
#         """Plot confusion matrix for the best model."""
#         if not self._best_estimator:
#             raise ValueError("Model not trained. Call train() first.")
#         display = ConfusionMatrixDisplay.from_estimator(
#             self._best_estimator,
#             self._x_test,
#             self._y_test
#         )
#         plt.title(f"Confusion Matrix - {self._best_model_name}")
#         plt.show()

#     def _train_single_model(
#         self,
#         name: str,
#         model: BaseEstimator,
#         strategy: SearchStrategy
#     ) -> Any:
#         param_grid = self._config.param_grid.get(name, {})
#         search = strategy.create_search(
#             model,
#             param_grid,
#             self._config.scoring,
#             self._config.search_config
#         )
#         return strategy.fit(search, self._x_train, self._y_train)

#     def _validate_input(self) -> None:
#         missing_keys = set(self._models.keys()) - set(self._config.param_grid.keys())
#         if missing_keys:
#             self._logger.error("Missing param grids for models: %s", missing_keys)
#             raise ValueError(f"Missing param grids for models: {missing_keys}")

#     def _update_best_model(self, name: str, search: Any) -> None:
#         if search.best_score_ > self._best_score:
#             self._best_score = search.best_score_
#             self._best_estimator = search.best_estimator_
#             self._best_params = search.best_params_
#             self._best_model_name = name
#             self._logger.info("New best model: %s with score %.4f", name, self._best_score)


# # Classification Trainer
# class ClassificationTrainer(BaseTrainer):
#     """Concrete implementation for classification training."""
    
#     def __init__(
#         self,
#         dataset: Dataset,
#         config: Dict[str, Any],
#         logger: Logger,
#         model_factory: ClassificationModelFactory = ClassificationModelFactory(),
#         search_factory: SearchStrategyFactory = SearchStrategyFactory()
#     ) -> None:
#         super().__init__(
#             dataset=dataset,
#             config=TrainerConfig.from_dict(config),
#             logger=logger,
#             model_factory=model_factory,
#             search_factory=search_factory
#         )


# # Main execution
# def main() -> None:
#     """Main execution function."""
#     from sklearn.datasets import make_classification
#     from sklearn.model_selection import train_test_split

#     # Mock config (thay bằng ConfigLoader thực tế trong production)
#     config = {
#         "scoring": "accuracy",
#         "param_grid": {
#             "LogisticRegression": {"C": [0.1, 1.0, 10.0]},
#             "RandomForestClassifier": {"n_estimators": [50, 100, 200]},
#             "LinearSVC": {"C": [0.1, 1.0, 10.0]},
#             "GaussianNB": {},
#             "MultinomialNB": {}
#         },
#         "random_state": 42,
#         "max_iter": 1000,
#         "search_config": {}
#     }

#     # Setup logger
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     # Prepare dataset
#     x, y = make_classification(
#         n_samples=500,
#         n_features=10,
#         n_informative=7,
#         n_redundant=3,
#         n_classes=2,
#         random_state=42
#     )
#     dataset = train_test_split(x, y, test_size=0.2, random_state=42)

#     # Initialize and run trainer
#     trainer = ClassificationTrainer(
#         dataset=dataset,
#         config=config,
#         logger=logger
#     )
#     trainer.train(use_random_search=True)
#     trainer.evaluate()
#     # trainer.plot_confusion_matrix()


# if __name__ == "__main__":
#     main()













# from src.utils.config_loader import ConfigLoader

# class ConfigProvider:
#     def get_config(self) -> Dict[str, Any]:
#         return ConfigLoader.get_config("training_model_config.yaml")["classification"]

# config = ConfigProvider().get_config()
# trainer = ClassificationTrainer(
#     dataset=dataset,
#     config=config,
#     logger=logging.getLogger("trainer")
# )






# import unittest
# from unittest.mock import Mock

# class TestClassificationTrainer(unittest.TestCase):
#     def test_train(self):
#         mock_logger = Mock(spec=Logger)
#         mock_factory = Mock(spec=ClassificationModelFactory)
#         mock_factory.create_models.return_value = {"mock_model": Mock()}
        
#         trainer = ClassificationTrainer(
#             dataset=(np.array([]), np.array([]), np.array([]), np.array([])),
#             config={"scoring": "accuracy", "param_grid": {}, "random_state": 42, "max_iter": 1000},
#             logger=mock_logger,
#             model_factory=mock_factory
#         )
#         # Test logic here