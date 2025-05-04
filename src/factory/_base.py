from typing import Dict, Any, Type, Optional, Protocol, Union 
from types import MappingProxyType
import logging

from pydantic import BaseModel, Field, field_validator

from ..base import BaseFactory

# Custom exceptions
class ModelFactoryError(Exception):
    """Base exception for model factory errors."""
    def __init__(self, message: str, context: Optional[dict] = None):
        self.context = context or {}
        super().__init__(message)

class UnknownTaskTypeError(ModelFactoryError):
    """Raised when an unsupported task type is provided."""
    def __init__(self, task_type: str, valid_tasks: list[str]):
        message = f"Unknown task type: '{task_type}. Valid type: {",".join(valid_tasks)}."
        super().__init__(message, {"task_type": task_type, "valid_tasks": valid_tasks})

class UnknownModelTypeError(ModelFactoryError):
    """Raised when an unsupported model type is provided."""
    def __init__(self, task_type: str, valid_tasks: list[str]):
        message = f"Unknown model type: '{task_type}. Valid type: {",".join(valid_tasks)}."
        super().__init__(message, {"task_type": task_type, "valid_tasks": valid_tasks})

class ConfigurationError(ModelFactoryError):
    """Raised when model configuration is invalid."""
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, {"field": field})

# Validator interface
class ValidatorInterface(Protocol):
    """
    Interface for validating identifiers (e.g., model_type, task_type).

    Example:
        >>> def custom_validator(value: str, field_name: str) -> str:
        ...
    """
    def __call__(self, value: str, field_name: str) -> str:
        pass

# # Logging Càn tạo 1 class Ilogger với 3 thuộc tính là error, infor
#     def log_with_verbose(message: str, level: str):
#         """Log a message if logger used."""
#         if 

# Validator
def validate_identifier(value: str, field_name: str) -> str:
    """
    Validate and normalize an identifier (e.g., model_type, task_type).

    Args:
        value: The identifier to validate.
        field_name: Name of the field for error reporting.

    Returns: 
        Normalized identifier (lowercase, stripped).

    Raises:
        ConfigurationError: If the identifier is invalid.
    """
    if not isinstance(value, str):
        raise ConfigurationError(f"{field_name} must be a string", field_name)
    stripped = value.strip().lower()
    if not stripped:
        raise ConfigurationError(f"{field_name} cannot be empty and whitespace", field_name)
    return stripped


class ModelConfig(BaseModel):
    """
    Configuration for machine learning model.

    Attributes:
        model_type: Type of model (e.g., 'random_forest')
        params: Model-specific parameters (e.g., {'n_estimators': 100})

    Example:
        >>> config = ModelConfig(model_type="random_forest", params: "estimators": 100)
    """
    model_type: str = Field(
        ...,
        min_length=1, 
        description="Model type (e.g., 'random_forest')"
    )    
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Custom parameter for model"
    )

    # Validator instance (can be replaced)
    _validator: ValidatorInterface = validate_identifier

    @field_validator("model_type")
    def validate_model_type(cls, value: str) -> str:
        """Validate and nomalize model_type using the configured validator."""
        return cls._validator(value, "model_type")

    model_config = {"extra": "forbid"} 

    @classmethod
    def set_validator(cls, validator: ValidatorInterface) -> None:
        """
        Set a custom validator for the model configuration.

        Args:
            validator: Validator implementing ValidatorInterface.

        Example:
            >>> def custom_validator(value: str, field_name: str) -> str:
            ... return value.strip().upper()
            >>> ModelConfig.set_validator(custom_validator)
        """
        cls._validator = validator

class BaseFactoryImpl(BaseFactory):
    """
    Base implementation for model factories.

    Attributes:
        _models: Imutable mapping of model types to classes._
    """
    def __init__(
        self,
        models: Optional[Dict[str, Type[Any]]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the factory with an optional mode registry.

        Args:
            models: Dictionary mapping model types to model classes.
        """
        self._models = MappingProxyType(models or {})
        self._logger = logger

    def register_model(self, model_type: str, model_class: Type[Any]) -> None:
        """
        Register a new model type.

        Args:
            model_type: Type of the model (e.g., "random_forest").

        Raises:
            ConfigurationError: If model_type is invalid or already registered.
        """
        validated_model_type = validate_identifier(model_type, "model_type")
        if validated_model_type in self._model:
            raise ConfigurationError(f"Model type {validated_model_type} already registed")
        
        new_models = dict(self._models)
        new_models[model_type] = model_class

        self._models = MappingProxyType(new_models)
        if self._logger:
            self._logger.info(f"Registed model {validated_model_type}.")

    def create_model(
        self,
        model_type: str,
        params,
        **kwargs: Any
    ) -> Any:
        """
        Create a model instance from configuration.

        Args:
            config: Model configuration (ConfigInterface or mode_type string).
            **kwargs: Keyword arguments for model initialization.

        Returns:
            Model instance.

        Raises:
            UnknownModelTypeError: If model_type is not registered.
            ConfigurationError: If config is invalid.   
        """