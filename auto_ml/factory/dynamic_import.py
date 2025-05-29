from typing import Callable, Dict, Any
import importlib

class DynamicImportFactory:
    """
    ---
    A factory class to dynamically import and initialize classes or callables from a list of modules.

    The factory searches through the provided list of module paths to find and instantiate the specified callable.

    Attributes:
        module_list (List[str]): A list of module paths (e.g., ["sklearn.ensemble", "sklearn.linear_model"]).
        _registry (Dict[str, Callable]): Cache for imported callables.
    
    ---
    """
    def __init__(self, module_list: list[str]) -> None:
        """
        Initialize the factory with a list of module paths.

        Args:
            module_list (list[str]): A list of module paths (e.g., ["sklearn.ensemble"]).

        Raises:
            ValueError: If module_list is not a list or contains non-string elements.

        Examples:
            >>> from auto_ml.factory import DynamicImportFactory
            >>> module_list = ["sklearn.ensemble"]
            >>> model = DynamicImportFactory(module_list)
            >>> params = {
            ...     "n_estimators": 100,
            ...     "criterion": "gini",
            ...     "max_depth": 2,
            ...     "n_jobs": -1
            ... }
            >>> rf_model = model.create("RandomForestClassifier", params)
            
        ---
        """
        if not isinstance(module_list, list):
            raise ValueError(f"{module_list} must be a list, got {type(module_list)}.")
        if not all(isinstance(module, str) for module in module_list):
            raise ValueError(f"All elements in {module_list} must be strings.")
        
        self.module_list = module_list
        self._registry: Dict[str, Callable] = {}

    def _import_callable(self, name: str) -> Callable:
        """
        Import a callable (class or function) from one of the modules in module_list.

        Args:
            name (str): The name of the callable to import.

        Returns:
            Callable: The imported callable.

        Raises:
            ValueError: If the name is not found in any module or the object is not callable.
            ModuleNotFoundError: If a module in module_list cannot be found.

        Examples:
            >>> factory = DynamicImportFactory(["sklearn.ensemble"])
            >>> callable_obj = factory._import_callable("RandomForestClassifier")
        """
        if name in self._registry:
            return self._registry[name]
        
        for module_path in self.module_list:
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, name):
                    callable_obj = getattr(module, name)
                    if not callable(callable_obj):
                        continue  # Skip if not callable
                    self._registry[name] = callable_obj
                    return callable_obj
            except ModuleNotFoundError:
                continue  # Skip invalid modules
        
        raise ValueError(f"Callable '{name}' not found in any module in {self.module_list}.")

    def create(self, class_name: str, params: Dict[str, Any] = {}) -> Any:
        """
        **Create an instance of a callable dynamically.**

        ### Args:
            
        - class_name (str): The name of the callable to instantiate.
        - params (Dict[str, Any]): A dictionary of parameters to pass to the callable.

        ### Returns:
        
        - Any: An instance of the callable initialized with the provided parameters.

        ### Raises:
        
        - ValueError: If class_name is not a string, params is not a dictionary, or class_name is not found.

        ### Examples:
        
            >>> factory = DynamicImportFactory(["sklearn.ensemble"])
            >>> params = {"n_estimators": 100, "max_depth": 2}
            >>> rf_model = factory.create("RandomForestClassifier", params)
        """
        if not isinstance(class_name, str):
            raise ValueError(f"{class_name} must be a string, got {type(class_name)}.")
        if not isinstance(params, dict):
            raise ValueError(f"{params} must be a dictionary, got {type(params)}.")
        
        callable_obj = self._import_callable(class_name)
        return callable_obj(**params)