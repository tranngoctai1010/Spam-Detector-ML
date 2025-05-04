from typing import Dict, Any, Type

from _base import BaseFactoryImpl
from ..models.classifiers import (
    GradientBoostingClassifierWrapper,
    KNeighborsClassifierWrapper,
    LogisticRegressionWrapper,
    RandomForestClassifierWrapper,
    SVCWrapper,
)
from ..models.regressors import (
    GradientBoostingRegressorWrapper,
    LinearRegressionWrapper,
    RandomForestRegressorWrapper,
    RidgeWrapper,
    SVRWrapper,
)
from utils import (
    UnknownModelTypeError,
    UnknownTaskTypeError
)

class ClassifierFactory(BaseFactoryImpl):
    """
    Factory for creating classification models.

    Support models:
        - gradient_boosting: GradientBoostingClassifierWrapper
        - knn: KNeighborsClassifierWrapper
        - logistic_regression: LogisticRegressionWrapper
        - random_forest: RandomForestClassifierWrapper
        - svm: SVCWrapper

    Example:
        >>> factory = ClassifierFactory()
        >>> model = factory.create_model("random_forest", estimator=100) 
    """
    _models = {
        "gradient_boosting": GradientBoostingClassifierWrapper,
        "knn": KNeighborsClassifierWrapper,
        "logistic_regression": LogisticRegressionWrapper,
        "random_forest": RandomForestClassifierWrapper,
        "svm": SVCWrapper
    }


class RegressorFactory(BaseFactoryImpl):
    """
    Factory for creating regression models.

    Support models:
        - gradient_boosting: GradientBoostingRegressorWrapper
        - linear_regression: LinearRegressionWrapper
        - random_forest: RandomForestRegressorWrapper
        - ridge: RidgeWrapper
        - svr: SVRWrapper

    Example:
        >>> factory = RegressorFactory()
        >>> model = factory.create_model("random_forest", estimator=100) 
    """
    _models = {
        "gradient_boosting": GradientBoostingRegressorWrapper,
        "linear_regression": LinearRegressionWrapper,
        "random_forest": RandomForestRegressorWrapper,
        "ridge": RidgeWrapper,
        "svr": SVRWrapper
    }


class ModelFactory(BaseFactoryImpl):
    """
    General factory for creating models based on task type.

    Supported tasks:
        - classification: ClassifierFactory
        - regression: RegressorFactory
    
    Example:
        >>> factory = ModelFactory()
        
    """
    _factories = {
        "classification": ClassifierFactory,
        "regression": RegressorFactory
    }