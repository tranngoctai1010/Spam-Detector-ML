from typing import Dict, List, Union, Any, Type

from ..base.model import BaseModel

class BaseModelImpl(BaseModel):
    model_class = None
    default_params = {}

    def __init__(self, *args, **kwargs):
        self.model = None
        if self.model_class is not None:
            self.model = self.model_class(*args, **kwargs)

    def get_model(self):
        if self.model is None:
            raise NotImplementedError("Model not initialized.")
        return self.model
    
    def get_default_params(self) -> Dict:
        return self.default_params
    
    def set_param_grid(self, param_grid: Dict[str, Any]) -> Dict:
        self.default_params = param_grid
        return self.default_params
