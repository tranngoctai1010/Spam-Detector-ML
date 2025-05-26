from typing import Callable, Any
import logging

from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class AutoModelTrainer: # Dùng all() để validate input
    def __init__(
        self,
        models: list[object],
        model_selection: object,
        model_param_grid: list[tuple[str, Any]],
        process_param_grid: list[tuple[str, Any]] | None = None
    ):
        self.models = models
        self.model_selection = model_selection
        self.model_param_grid = model_param_grid
        self.process_param_grid = process_param_grid
        
        if not process_param_grid:
            self.model_param_grid = self._join_string("model", self.model_param_grid)
        
        self.param_grid = self.model_param_grid | self.process_param_grid

        
    def fit(self, X, y):
        for model in self.models:
            pipeline_step = 
            self.model_selection.set_params(estimator=)
        
    def save_results(self)
    
    @staticmethod
    def _join_string(prefix: str, param_list: list[tuple]):
        return [(f"{prefix}__{key}", value) for key, value in param_list]
    
