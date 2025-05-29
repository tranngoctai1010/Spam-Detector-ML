# Method: fit(X, y), predict(X), score(X, y), set_params(**params).
# Attribute: best_params_, best_score_, best_estimator_, cv_results_.

# GridSearchCV
# RandomizedSearchCV
# HalvingGridSearchCV
# HalvingRandomSearchCV
# BayesSearchCV
# OptunaSearchCV (từ optuna-integration)

# joblib.Parallel

# Tham số được lưu vào config giúp người khác thay đổi dễ dnagf 

from typing import Any, Dict, Type, Callable, Tuple
import logging
from pathlib import Path

import yaml
import json
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV
)
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Ca
from optuna.integration.sklearn import OptunaSearchCV
from imblearn.pipeline import Pipeline

from ..base import BaseSearchCV
from ..factory import DynamicImportFactory

CLASSIFICATION_CONFIG_PATH = "../configs/classification.yaml"
REGRESSION_CONFIG_PATH = "../configs/regression.yaml"

class BaseAutoTrainer:

    SEARCH_CONFIG = {
        GridSearchCV: {"param_key": "param_grid", "extra_params": {}},
        RandomizedSearchCV: {"param_key": "param_distributions", "extra_params": {"n_iter": 10}},
        HalvingGridSearchCV: {"param_key": "param_grid", "extra_params": {"factor": 3}},
        HalvingRandomSearchCV: {"param_key": "param_distributions", "extra_params": {"n_iter": 10, "factor": 3}},
        BayesSearchCV: {"param_key": "search_spaces", "extra_params": {"n_iter": 32}},
        OptunaSearchCV: {"param_key": "param_distributions", "extra_params": {"n_trials": 100}},
    }

    def __init__(
        self,
        search_cv: BaseSearchCV,
        config_path: str,
        process_steps: list[tuple] | None = None,
        process_param_grid: dict[str, Any] | None = None
    ): 
        self._validate_input()
        self.search_cv = search_cv
        self.config_path = config_path
        self.process_steps = process_steps or []
        self.process_param_grid = process_param_grid or {}
        self.best_estimators_ = []
        self.best_params_ = []
        self.best_scores_ = []

    def fit(self, X, y):
        config = self._load_hyperparam_config(self.config_path)
        model_list, param_grid_list = self._generate_model_and_get_param_grid(config)

        self.best_estimators_ = []
        self.best_params_ = []
        self.best_scores_ = []

        search_type = type(self.search_cv)
        search_config = self.SEARCH_CONFIG[search_type]
        

        for model, param_grid in zip(model_list, param_grid_list):
            steps = [("model", model)]
            param_grid = self._prefix_param_keys("model", param_grid)
            if not self.process_steps:
                steps = self.process_steps + [("model", model)]
                param_grid = param_grid.update(self.process_param_grid)
            pipeline = Pipeline(steps=steps)
            seach_cv = self.search_cv.

    # def _validate_input(self):
    #     if (self.process_steps is None) != (self.process_param_grid is None):
    #     Xác thực sao cho mỗi values phải có ít nhất là 2 giá trị nữa
    #         ...


    def _convert_param_grid(
        self,
        search_type: type,
        param_grid: Dict[str, Any]
    ):
        if search_type in (GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV):
            return param_grid
        
        if search_type == BayesSearchCV:
            return {
                param: (
                    Integer(min(values)), Integer(max(values)) if isinstance(values[0], int)
                    else Real(min(values), Real(max(values)), prior="uniform")
                )
                if isinstance(values, list) and len(values) > 1 and isinstance(values[0], (int, float))
                else Categorical(values)
                for param, values in param_grid.items()
            }
        
        if search_type == OptunaSearchCV:
            from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
            return {
                param: (IntDistribution(min(values), max(values)) if isinstance(values[0], int)
                        else FloatDistribution(min(values), max(values)))
                if isinstance(values, list) and len(values) > 1 and isinstance(values[0], (int, float))
                else CategoricalDistribution(values)
                for param, values in param_grid.items()
            }
        raise ValueError(f"Search type is not supported: {search_type}")
        
    @staticmethod
    def _load_hyperparam_config(file_path: str) -> Dict[str, Dict[str, Any]]:
        if not isinstance(file_path, str):
            raise ValueError(f"File path must be a string.")
        
        path = Path(file_path)
        if not path.is_file():
                raise FileNotFoundError(f"Configuration file not found: {path}")
        if path.suffix not in (".yaml", ".yml", ".json"):
            raise ValueError(f"Unsupported file extension: {path.suffix}. Use .yaml, .yml, or .json")

        with path.open("r", encoding="utf-8") as file:
            if path.suffix in (".yaml", ".yml"):
                config = yaml.safe_load(file)
            else:  
                config = json.load(file)

        if config is None:
            raise RuntimeError("Configuration file is empty or invalid")
        if not isinstance(config, dict):
            raise RuntimeError("Configuration must be a dictionary")
        if not all(isinstance(key, str) and isinstance(value, dict) for key, value in config.items()):
            raise ValueError("All keys must be strings, and values must be dictionaries")
        if not all(all(isinstance(sub_k, str) for sub_k in v.keys()) for v in config.values()):
            raise ValueError("All keys in nested dictionaries must be strings")
        
        return config
    
    @staticmethod
    def _generate_model_and_get_param_grid(
        config: Dict[str, Dict[str, Any]],
        dynamic_import = DynamicImportFactory
    ) -> Tuple[list[object], list[dict[str, Any]]]:
        module_list = list(config.keys())
        class_name_list = [class_name for sub_config in config.values() for class_name in sub_config.keys()]
        param_grid_list = [class_name for sub_config in config.values() for class_name in sub_config.values()]
        dynamic_import = dynamic_import(module_list)
        model_list = []
        for class_name in class_name_list:
            model = dynamic_import.create(class_name)
            model_list.append(model)
        return model_list, param_grid_list

    @staticmethod
    def _prefix_param_keys(prefix: str, param_list: Dict[str, Any]) -> Dict[str, Any]:
        return {f"{prefix}__{key}": value for key, value in param_list.items()}
    

class ClassificationTrainer(BaseAutoTrainer):
    def __init__(
        self,
        search_cv: BaseSearchCV = Type[GridSearchCV],
        config_path: str = CLASSIFICATION_CONFIG_PATH,
        process_steps: list[tuple] | None = None,
        process_param_grid: dict[str, Any] | None = None,
    ):
        super.__init__(
            search_cv,
            config_path,
            process_steps,
            process_param_grid
        )


class RegressionTrainer(BaseAutoTrainer): ...




# === GridSearchCV ===
# Methods:
# - fit
# - get_params
# - predict
# - score
# Attributes:
# - best_estimator
# - best_params_
# - cv_results

# === RandomizedSearchCV ===
# Methods: [similar to GridSearchCV]
# Attributes: [similar to GridSearchCV]

# === BayesSearchCV ===
# Methods: [similar to GridSearchCV]
# Attributes: [similar to GridSearchCV, plus some extras like optimizer_result_]

# === Optuna Study ===
# Methods:
# - best_trial
# - optimize
# - trials
# Attributes:
# - best_params
# - best_value

from pathlib import Path
import yaml
import json
from typing import Dict, Any, Tuple, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, BaseSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from optuna.integration.sklearn import OptunaSearchCV
from sklearn.base import BaseEstimator
import numpy as np

# Import optuna.distributions ở đầu file với xử lý lỗi
try:
    from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
except ImportError:
    IntDistribution = FloatDistribution = CategoricalDistribution = None

class BaseAutoTrainer:
    """Lớp cơ sở để tự động huấn luyện mô hình với tìm kiếm siêu tham số."""
    
    SEARCH_CONFIG = {
        GridSearchCV: {'param_key': 'param_grid', 'extra_params': {}},
        RandomizedSearchCV: {'param_key': 'param_distributions', 'extra_params': {'n_iter': 10}},
        HalvingGridSearchCV: {'param_key': 'param_grid', 'extra_params': {'factor': 3}},
        HalvingRandomSearchCV: {'param_key': 'param_distributions', 'extra_params': {'n_iter': 10, 'factor': 3}},
        BayesSearchCV: {'param_key': 'search_spaces', 'extra_params': {'n_iter': 32}},
        OptunaSearchCV: {'param_key': 'param_distributions', 'extra_params': {'n_trials': 100}}
    }

    def __init__(
        self,
        search_cv: BaseSearchCV,
        config_path: Optional[str] = None,
        process_steps: Optional[List[Tuple[str, Any]]] = None,
        process_param_grid: Optional[Dict[str, Any]] = None
    ):
        self._validate_input(search_cv, config_path, process_steps, process_param_grid)
        self.search_cv = search_cv
        self.config_path = config_path
        self.process_steps = process_steps or []
        self.process_param_grid = process_param_grid or {}
        self.best_estimators_ = []
        self.best_params_ = []
        self.best_scores_ = []

    def _validate_input(
        self,
        search_cv: BaseSearchCV,
        config_path: Optional[str],
        process_steps: Optional[List[Tuple[str, Any]]],
        process_param_grid: Optional[Dict[str, Any]]
    ) -> None:
        """Kiểm tra tính hợp lệ của các tham số đầu vào."""
        if (process_steps is None) != (process_param_grid is None):
            raise ValueError("process_steps và process_param_grid phải cùng là None hoặc không.")
        if not isinstance(search_cv, BaseSearchCV):
            raise TypeError("search_cv phải là instance của BaseSearchCV.")
        
        # Nếu search_cv đã có estimator, kiểm tra config_path và các tham số
        if hasattr(search_cv, 'estimator') and search_cv.estimator is not None:
            if not isinstance(search_cv.estimator, (Pipeline, BaseEstimator)):
                raise TypeError("search_cv.estimator phải là Pipeline hoặc BaseEstimator.")
            param_key = self.SEARCH_CONFIG[type(search_cv)]['param_key']
            if not hasattr(search_cv, param_key) or getattr(search_cv, param_key) is None:
                raise ValueError(f"search_cv phải có {param_key} nếu đã có estimator.")
            if config_path is not None:
                print("Cảnh báo: config_path sẽ bị bỏ qua vì search_cv đã có estimator.")
        else:
            if not isinstance(config_path, str):
                raise TypeError("config_path phải là chuỗi khi search_cv không có estimator.")
        
        if type(search_cv) not in self.SEARCH_CONFIG:
            raise ValueError(f"search_cv không được hỗ trợ: {type(search_cv)}")

    @staticmethod
    def _load_hyperparam_config(file_path: str) -> Dict[str, Dict[str, Any]]:
        """Tải cấu hình siêu tham số từ file YAML hoặc JSON."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        if path.suffix not in (".yaml", ".yml", ".json"):
            raise ValueError(f"Định dạng file không hỗ trợ: {path.suffix}")

        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) if path.suffix in (".yaml", ".yml") else json.load(file)

        if not config or not isinstance(config, dict):
            raise RuntimeError("File cấu hình trống hoặc không hợp lệ.")
        if not all(isinstance(k, str) and isinstance(v, dict) and all(isinstance(sk, str) for sk in v)
                   for k, v in config.items()):
            raise ValueError("Cấu hình phải là từ điển với khóa là chuỗi và giá trị là từ điển.")
        
        return config

    @staticmethod
    def _convert_param_grid(param_grid: Dict[str, Any], search_type: type) -> Dict[str, Any]:
        """Chuyển đổi param_grid sang định dạng phù hợp với search_type."""
        if search_type in (GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV):
            return param_grid
        if search_type == BayesSearchCV:
            return {
                param: (Integer(min(values), max(values)) if isinstance(values[0], int)
                        else Real(min(values), max(values), prior='uniform'))
                if isinstance(values, list) and len(values) > 1 and isinstance(values[0], (int, float))
                else Categorical(values)
                for param, values in param_grid.items()
            }
        if search_type == OptunaSearchCV:
            if IntDistribution is None:
                raise ImportError("optuna.distributions không được cài đặt. Vui lòng cài optuna.")
            return {
                param: (IntDistribution(min(values), max(values)) if isinstance(values[0], int)
                        else FloatDistribution(min(values), max(values)))
                if isinstance(values, list) and len(values) > 1 and isinstance(values[0], (int, float))
                else CategoricalDistribution(values)
                for param, values in param_grid.items()
            }
        raise ValueError(f"search_type không được hỗ trợ: {search_type}")

    @staticmethod
    def _generate_model_and_get_param_grid(
        config: Dict[str, Dict[str, Any]],
        dynamic_import
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """Tạo danh sách mô hình và không gian tham số từ cấu hình."""
        models, param_grids = [], []
        for module, sub_config in config.items():
            for class_name, param_grid in sub_config.items():
                models.append(dynamic_import.create(f"{module}.{class_name}"))
                param_grids.append(param_grid)
        return models, param_grids

    @staticmethod
    def _prefix_param_keys(prefix: str, param_grid: Dict[str, Any]) -> Dict[str, Any]:
        """Thêm tiền tố cho các khóa trong param_grid."""
        return {f"{prefix}__{key}": value for key, value in param_grid.items()}

    def fit(self, X, y):
        """Huấn luyện mô hình với tìm kiếm siêu tham số."""
        search_type = type(self.search_cv)
        search_config = self.SEARCH_CONFIG[search_type]
        param_key = search_config['param_key']
        extra_params = search_config['extra_params']

        # Trường hợp 1: search_cv đã có estimator và param_grid
        if hasattr(self.search_cv, 'estimator') and self.search_cv.estimator is not None:
            # Lấy param_grid từ search_cv
            final_param_grid = getattr(self.search_cv, param_key)
            # Gộp process_param_grid nếu có
            if self.process_param_grid:
                final_param_grid = {**final_param_grid, **self.process_param_grid}
                final_param_grid = self._convert_param_grid(final_param_grid, search_type)
            
            # Kết hợp process_steps với estimator
            estimator = self.search_cv.estimator
            if self.process_steps:
                if isinstance(estimator, Pipeline):
                    steps = self.process_steps + estimator.steps
                else:
                    steps = self.process_steps + [("model", estimator)]
                estimator = Pipeline(steps=steps)

            # Tạo instance search_cv mới với các tham số cập nhật
            search_cv_params = {
                'estimator': estimator,
                param_key: final_param_grid,
                'cv': getattr(self.search_cv, 'cv', 5),
                'scoring': getattr(self.search_cv, 'scoring', None),
                'n_jobs': getattr(self.search_cv, 'n_jobs', -1),
                'verbose': getattr(self.search_cv, 'verbose', 1),
                **extra_params
            }
            search_cv = search_type(**search_cv_params)
            search_cv.fit(X, y)
            
            # Lưu kết quả
            self.best_estimators_ = [search_cv.best_estimator_]
            self.best_params_ = [search_cv.best_params_]
            self.best_scores_ = [search_cv.best_score_]
        else:
            # Trường hợp 2: search_cv chưa có estimator, dùng config_path
            if not self.config_path:
                raise ValueError("config_path phải được cung cấp khi search_cv không có estimator.")
            config = self._load_hyperparam_config(self.config_path)
            models, param_grids = self._generate_model_and_get_param_grid(config, dynamic_import=...)  # Giả sử dynamic_import được cung cấp
            
            self.best_estimators_ = []
            self.best_params_ = []
            self.best_scores_ = []

            for model, param_grid in zip(models, param_grids):
                steps = self.process_steps + [("model", model)]
                pipeline = Pipeline(steps=steps)
                
                final_param_grid = {
                    **self._prefix_param_keys("model", param_grid),
                    **self.process_param_grid
                }
                final_param_grid = self._convert_param_grid(final_param_grid, search_type)

                search_cv_params = {
                    'estimator': pipeline,
                    param_key: final_param_grid,
                    'cv': getattr(self.search_cv, 'cv', 5),
                    'scoring': getattr(self.search_cv, 'scoring', None),
                    'n_jobs': getattr(self.search_cv, 'n_jobs', -1),
                    'verbose': getattr(self.search_cv, 'verbose', 1),
                    **extra_params
                }

                search_cv = search_type(**search_cv_params)
                search_cv.fit(X, y)

                self.best_estimators_.append(search_cv.best_estimator_)
                self.best_params_.append(search_cv.best_params_)
                self.best_scores_.append(search_cv.best_score_)

        return self

    def predict(self, X):
        """Dự đoán với mô hình tốt nhất."""
        if not self.best_estimators_:
            raise ValueError("Chưa huấn luyện mô hình. Gọi fit() trước.")
        best_idx = np.argmax(self.best_scores_)
        return self.best_estimators_[best_idx].predict(X)

    def get_best_model(self) -> Tuple[Any, Dict[str, Any], float]:
        """Trả về mô hình tốt nhất, tham số và điểm số."""
        if not self.best_estimators_:
            raise ValueError("Chưa huấn luyện mô hình.")
        best_idx = np.argmax(self.best_scores_)
        return (
            self.best_estimators_[best_idx],
            self.best_params_[best_idx],
            self.best_scores_[best_idx]
        )