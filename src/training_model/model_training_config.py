from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, Type, Literal, Union
import numpy as np
from sklearn.base import BaseEstimator

# Bayesian Optimization
# Successive Halving
train
tune
config

# Type definitions  
XData = np.ndarray
YData = np.ndarray
Dataset = tuple[XData, XData, YData, YData]
ModelDict = Dict[str, BaseEstimator]
ParamGrid = Dict[str, Any]
ModelConfigName = Literal["classification", "regression"]
ModelConfigClass = Union["ClassificationModelConfig", "RegressionModelConfig"]


@dataclass
def BaseConfig(ABC):
    @classmethod
    @abstractmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> BaseConfig:
        pass

@dataclass(frozen=True)
def SearchConfig(BaseConfig):

    n_iter: int
    n_jobs: int
    cv: int
    verbose: int

    def __post_init__(self):
        """Validate configuration values after initialization."""
        if self.n_iter <= 0:
            raise ValueError("n_iter must be positive")
        if self.cv < 1:
            raise ValueError("cv must be at least 1")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> SearchConfig:
        required_keys = {"n_iter", "n_jobs", "cv", "verbose"}
        missing_keys = required_keys - set(config_dict.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in config_dict: {missing_keys}")
        return cls(
            n_iter=config_dict["n_iter"],
            n_jobs=config_dict["n_jobs"],
            cv=config_dict["cv"],
            verbose=config_dict["verbose"]
        )


@dataclass(frozen=True)
class ClassificationModelConfig(BaseConfig):
    """Immutable configuration for classification trainer."""
    scoring: str
    param_grid: ParamGrid
    random_state: int
    max_iter: int

    def __post_init__(self):
        """Validate configuration values after initialization."""
        if not self.param_grid:
            raise ValueError("param_grid cannot be empty")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ClassificationModelConfig":
        """Create ClassificationModelConfig from dictionary."""
        required_keys = {"scoring", "param_grid", "random_state", "max_iter"}
        missing_keys = required_keys - set(config_dict.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in config_dict: {missing_keys}")
        
        return cls(
            scoring=config_dict["scoring"],
            param_grid=config_dict["param_grid"],
            random_state=config_dict["random_state"],
            max_iter=config_dict["max_iter"],
        )


@dataclass(frozen=True)
class RegressionModelConfig(BaseConfig):
    """Immutable configuration for regression trainer."""
    scoring: str
    param_grid: ParamGrid
    random_state: int
    max_iter: int

    def __post_init__(self):
        """Validate configuration values after initialization."""
        if not self.param_grid:
            raise ValueError("param_grid cannot be empty")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ClassificationModelConfig":
        """Create ClassificationModelConfig from dictionary."""
        required_keys = {"scoring", "param_grid", "random_state", "max_iter"}
        missing_keys = required_keys - set(config_dict.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in config_dict: {missing_keys}")
        
        return cls(
            scoring=config_dict["scoring"],
            param_grid=config_dict["param_grid"],
            random_state=config_dict["random_state"],
            max_iter=config_dict["max_iter"],
        )
    
    @staticmethod
    def _validate_required_keys()

class BaseModelConfig(BaseConfig):





































# from dataclasses import dataclass
# from typing import Dict, Any


# @dataclass(frozen=True)
# class BaseModelConfig:
#     """Base configuration for machine learning models.

#     Attributes:
#         scoring: Metric used for model evaluation (e.g., 'accuracy', 'mse').
#         param_grid: Dictionary of hyperparameter search space.
#         random_state: Seed for reproducibility.
#         max_iter: Maximum number of iterations for training.
#     """
#     scoring: str
#     param_grid: Dict[str, Any]
#     random_state: int
#     max_iter: int

#     def __post_init__(self) -> None:
#         """Validate configuration attributes after initialization.

#         Raises:
#             ValueError: If param_grid is empty, max_iter is non-positive, or
#                 random_state is negative.
#         """
#         if not self.param_grid:
#             raise ValueError("param_grid must not be empty.")
#         if self.max_iter <= 0:
#             raise ValueError("max_iter must be positive.")
#         if self.random_state < 0:
#             raise ValueError("random_state must be non-negative.")

#     @classmethod
#     def from_dict(cls, config: Dict[str, Any]) -> "BaseModelConfig":
#         """Create a configuration from a dictionary.

#         Args:
#             config: Dictionary containing configuration parameters.

#         Returns:
#             An instance of BaseModelConfig.

#         Raises:
#             ValueError: If required keys are missing or invalid.
#         """
#         required_keys = {"scoring", "param_grid", "random_state", "max_iter"}
#         cls._validate_required_keys(config, required_keys)

#         return cls(
#             scoring=config["scoring"],
#             param_grid=config["param_grid"],
#             random_state=config["random_state"],
#             max_iter=config["max_iter"],
#         )

#     @staticmethod
#     def _validate_required_keys(config: Dict[str, Any], required_keys: set) -> None:
#         """Validate that all required keys are present in the config dictionary.

#         Args:
#             config: Dictionary to validate.
#             required_keys: Set of required key names.

#         Raises:
#             ValueError: If any required keys are missing.
#         """
#         missing_keys = required_keys - config.keys()
#         if missing_keys:
#             raise ValueError(f"Missing required configuration keys: {missing_keys}")


# @dataclass(frozen=True)
# class ClassificationModelConfig(BaseModelConfig):
#     """Configuration for classification models, extending base config.

#     Attributes:
#         class_weight: Strategy for handling imbalanced classes (e.g., 'balanced').
#     """
#     class_weight: str

#     @classmethod
#     def from_dict(cls, config: Dict[str, Any]) -> "ClassificationModelConfig":
#         """Create a classification model config from a dictionary.

#         Args:
#             config: Dictionary containing configuration parameters, including
#                 those from BaseModelConfig and 'class_weight'.

#         Returns:
#             An instance of ClassificationModelConfig.

#         Raises:
#             ValueError: If required keys are missing or invalid.
#         """
#         required_keys = {"scoring", "param_grid", "random_state", "max_iter", "class_weight"}
#         cls._validate_required_keys(config, required_keys)

#         return cls(**config)


# @dataclass(frozen=True)
# class RegressionModelConfig(BaseModelConfig):
#     """Configuration for regression models, extending base config.

#     Attributes:
#         early_stopping_rounds: Number of rounds for early stopping.
#     """
#     early_stopping_rounds: int

#     def __post_init__(self) -> None:
#         """Validate configuration attributes after initialization.

#         Extends base validation to check early_stopping_rounds.

#         Raises:
#             ValueError: If early_stopping_rounds is non-positive or base validation fails.
#         """
#         super().__post_init__()
#         if self.early_stopping_rounds <= 0:
#             raise ValueError("early_stopping_rounds must be positive.")

#     @classmethod
#     def from_dict(cls, config: Dict[str, Any]) -> "RegressionModelConfig":
#         """Create a regression model config from a dictionary.

#         Args:
#             config: Dictionary containing configuration parameters, including
#                 those from BaseModelConfig and 'early_stopping_rounds'.

#         Returns:
#             An instance of RegressionModelConfig.

#         Raises:
#             ValueError: If required keys are missing or invalid.
#         """
#         required_keys = {
#             "scoring",
#             "param_grid",
#             "random_state",
#             "max_iter",
#             "early_stopping_rounds",
#         }
#         cls._validate_required_keys(config, required_keys)

#         return cls(**config)

# # Example usage
# if __name__ == "__main__":
#     # Sample configuration dictionary
#     config_dict = {
#         "search_config": {
#             "n_iter": 10,
#             "n_jobs": -1,
#             "cv": 5,
#             "verbose": 1
#         },
#         "model_config": {
#             "scoring": "accuracy",
#             "param_grid": {"n_estimators": [50, 100, 200]},
#             "random_state": 42,
#             "max_iter": 100,
#             "search_config": {
#                 "n_iter": 10,
#                 "n_jobs": -1,
#                 "cv": 5,
#                 "verbose": 1
#             }
#         }
#     }

#     # Create TrainerConfig
#     try:
#         trainer_config = TrainerConfig.from_dict(config_dict, trainer_type="classification")
#         print("TrainerConfig created successfully:")
#         print(trainer_config.to_dict())
#     except ValueError as e:
#         print(f"Error creating TrainerConfig: {e}")