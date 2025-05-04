from .gradient_boosting import GradientBoostingRegressorWrapper
from .linear_regression import LinearRegressionWrapper
from .random_forest import RandomForestRegressorWrapper
from .ridge import RidgeWrapper
from .svr import SVRWrapper

__all__ = [
    "GradientBoostingRegressorWrapper",
    "LinearRegressionWrapper",
    "RandomForestRegressorWrapper",
    "RidgeWrapper",
    "SVRWrapper"
]