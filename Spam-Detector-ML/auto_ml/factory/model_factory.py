from typing import Dict, Any, Type
import yaml

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

# class ModelFactory(BaseFactoryImpl):
#     """
#     General factory for creating models based on task type.

#     Supported tasks:
#         - classification: ClassifierFactory
#         - regression: RegressorFactory
    
#     Example:
#         >>> factory = ModelFactory()
        
#     """
#     _factories = {
#         "classification": ClassifierFactory,
#         "regression": RegressorFactory
#     }


import importlib

class ModelFactory():
    _registry: Dict[str, Dict[str, Any]] = {}

    def __init__(self, config_path: str):
        self._parse_config(config_path)

    def create(self, model_name: str):
        if model_name not in self._registry.keys():
            raise ValueError(f"Model '{model_name}' not found in registry")
        model = importlib.import_module(self._registry[model_name])
        return model()

    def create_all(self):
        model_dict = {}
        model_names = self._registry.keys()
        for model_name in model_names:
            model = importlib.import_module(model_name)
            model_dict.append(model)
        return model_dict

    def _parse_config(self, config_path: str):
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
        except Exception as e:
            raise FileNotFoundError(f"")
        for name, info in config.items():
            self._registry[name] = info

    def _import_module(self, model_name: str):
        


# from typing import List, Type, Dict, Any
# from ..base.model import BaseModelImpl
# import yaml

# class ModelFactory:
#     _registry: Dict[str, Dict[str, Any]] = {}

#     @classmethod
#     def load_config(cls, config_path: str):
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
#         for name, model_info in config['models'].items():
#             cls._registry[name] = model_info

#     @classmethod
#     def create(cls, model_name: str) -> BaseModelImpl:
#         if model_name not in cls._registry:
#             raise ValueError(f"Model {model_name} not found in registry")
#         model_info = cls._registry[model_name]
#         model_class = model_info['class']  # Giả sử là class đã được import
#         return model_class()  # Tạo instance

#     @classmethod
#     def create_all(cls) -> List[BaseModelImpl]:
#         return [cls.create(name) for name in cls._registry]

# # Sử dụng
# if __name__ == "__main__":
#     ModelFactory.load_config("models.yaml")
#     model = ModelFactory.create("GradientBoostingClassifier")
#     print(model.get_param_grid())