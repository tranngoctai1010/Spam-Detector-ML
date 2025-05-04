from typing import Any, Dict, Optional, Type, Union, Protocol, List
from types import MappingProxyType
import logging
from pydantic import BaseModel, Field, field_validator
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom exceptions
class ModelFactoryError(Exception):
    """Base exception for ModelFactory errors."""
    def __init__(self, message: str, context: Optional[dict] = None):
        self.context = context or {}
        super().__init__(message)

class UnknownTaskTypeError(ModelFactoryError):
    """Raised when an unsupported task type is provided."""
    def __init__(self, task_type: str, valid_tasks: list[str]):
        message = f"Unknown task type: '{task_type}'. Valid types: {', '.join(valid_tasks)}."
        super().__init__(message, {"task_type": task_type, "valid_tasks": valid_tasks})

class UnknownModelTypeError(ModelFactoryError):
    """Raised when an unsupported model type is provided."""
    def __init__(self, model_type: str, valid_models: list[str]):
        message = f"Unknown model type: '{model_type}'. Valid types: {', '.join(valid_models)}."
        super().__init__(message, {"model_type": model_type, "valid_models": valid_models})

class ConfigurationError(ModelFactoryError):
    """Raised when model configuration is invalid."""
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, {"field": field})

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
        raise ConfigurationError(f"{field_name} cannot be empty or whitespace", field_name)
    return stripped

# Configuration interface
class ConfigInterface(Protocol):
    """Interface for model configuration."""
    model_type: str
    params: Dict[str, Any]

# Pydantic model
class ModelConfig(BaseModel):
    """
    Configuration for a machine learning model.

    Attributes:
        model_type: Type of the model (e.g., 'random_forest').
        params: Model-specific parameters (e.g., {'n_estimators': 100}).

    Example:
        >>> config = ModelConfig(model_type="random_forest", params={"n_estimators": 100})
    """
    model_type: str = Field(..., min_length=1, description="Type of the model (e.g., 'random_forest')")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom model parameters")

    @field_validator("model_type")
    def validate_model_type(cls, value: str) -> str:
        """Validate and normalize model_type."""
        return validate_identifier(value, "model_type")

    @field_validator("params")
    def validate_params(cls, value: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model-specific parameters based on model_type.

        Args:
            value: Parameters to validate.
            values: Other field values (e.g., model_type).

        Returns:
            Validated parameters.

        Raises:
            ConfigurationError: If parameters are invalid.
        """
        model_type = values.get("model_type")
        if model_type == "random_forest":
            if value.get("n_estimators") and (not isinstance(value["n_estimators"], int) or value["n_estimators"] <= 0):
                raise ConfigurationError("n_estimators must be a positive integer", "params.n_estimators")
        return value

    model_config = {"extra": "forbid"}

# Model builder (Builder Pattern)
class ModelBuilder:
    """
    Builder for constructing model instances with complex initialization.

    Attributes:
        model_class: The class of the model to instantiate.
        params: Parameters for model initialization.
        args: Positional arguments for model initialization.
        kwargs: Keyword arguments for model initialization.

    Example:
        >>> builder = ModelBuilder(model_class=RandomForestClassifierWrapper)
        >>> builder.with_params({"n_estimators": 100}).build()
    """
    def __init__(self, model_class: Type[Any]):
        self.model_class = model_class
        self.params: Dict[str, Any] = {}
        self.args: tuple = ()
        self.kwargs: Dict[str, Any] = {}

    def with_params(self, params: Optional[Dict[str, Any]] = None) -> "ModelBuilder":
        """
        Set model parameters.

        Args:
            params: Dictionary of model parameters.

        Returns:
            Self for method chaining.
        """
        self.params = params or {}
        return self

    def with_args(self, *args: Any) -> "ModelBuilder":
        """
        Set positional arguments.

        Args:
            *args: Positional arguments for model initialization.

        Returns:
            Self for method chaining.
        """
        self.args = args
        return self

    def with_kwargs(self, **kwargs: Any) -> "ModelBuilder":
        """
        Set keyword arguments.

        Args:
            **kwargs: Keyword arguments for model initialization.

        Returns:
            Self for method chaining.
        """
        self.kwargs = kwargs
        return self

    def build(self) -> Any:
        """
        Construct the model instance.

        Returns:
            An instance of the model class.

        Raises:
            ConfigurationError: If initialization fails.
        """
        try:
            return self.model_class(*self.args, **self.params, **self.kwargs)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize model: {str(e)}", "build")

# Factory interface
class ModelFactoryInterface(ABC):
    """
    Interface for model factories.

    Methods:
        register_model: Register a new model type.
        create_model: Create a model instance from configuration.
    """
    @abstractmethod
    def register_model(self, model_type: str, model_class: Type[Any]) -> None:
        pass

    @abstractmethod
    def create_model(self, config: Union[ConfigInterface, str], *args: Any, **kwargs: Any) -> Any:
        pass

# Base factory implementation (Factory Method Pattern)
class BaseFactoryImpl(ModelFactoryInterface):
    """
    Base implementation for model factories.

    Attributes:
        _models: Immutable mapping of model types to classes.
    """
    def __init__(self, models: Optional[Dict[str, Type[Any]]] = None):
        """
        Initialize the factory with an optional model registry.

        Args:
            models: Dictionary mapping model types to model classes.
        """
        self._models = MappingProxyType(models or {})

    def register_model(self, model_type: str, model_class: Type[Any]) -> None:
        """
        Register a new model type.

        Args:
            model_type: Type of the model (e.g., 'random_forest').
            model_class: Class of the model.

        Raises:
            ConfigurationError: If model_type is invalid or already registered.
        """
        validated_model_type = validate_identifier(model_type, "model_type")
        if validated_model_type in self._models:
            raise ConfigurationError(f"Model type '{validated_model_type}' already registered", "model_type")

        new_models = dict(self._models)
        new_models[validated_model_type] = model_class
        self._models = MappingProxyType(new_models)
        logger.info(f"Registered model: {validated_model_type}")

    def create_model(self, config: Union[ConfigInterface, str], *args: Any, **kwargs: Any) -> Any:
        """
        Create a model instance from configuration.

        Args:
            config: Model configuration (ConfigInterface or model_type string).
            *args: Positional arguments for model initialization.
            **kwargs: Keyword arguments for model initialization.

        Returns:
            Model instance.

        Raises:
            UnknownModelTypeError: If model_type is not registered.
            ConfigurationError: If config is invalid.
        """
        if isinstance(config, str):
            config = ModelConfig(model_type=config)

        if not isinstance(config, ConfigInterface):
            raise ConfigurationError("Config must be a string or ConfigInterface", "config")

        model_type = validate_identifier(config.model_type, "model_type")
        model_class = self._models.get(model_type)
        if not model_class:
            raise UnknownModelTypeError(model_type, list(self._models.keys()))

        logger.info(f"Creating model: {model_type}")
        return ModelBuilder(model_class).with_params(config.params).with_args(*args).with_kwargs(**kwargs).build()

# Factory registry for dynamic factory management
class FactoryRegistry:
    """
    Registry for dynamically managing task-specific factories.

    Attributes:
        _factories: Mapping of task types to factory classes.
    """
    def __init__(self):
        self._factories: Dict[str, Type[BaseFactoryImpl]] = {}

    def register_factory(self, task_type: str, factory: Type[BaseFactoryImpl]) -> None:
        """
        Register a new factory for a task type.

        Args:
            task_type: Type of the task (e.g., 'clustering').
            factory: Factory class for the task.

        Raises:
            ConfigurationError: If task_type is invalid or already registered.
        """
        validated_task_type = validate_identifier(task_type, "task_type")
        if validated_task_type in self._factories:
            raise ConfigurationError(f"Task type '{validated_task_type}' already registered", "task_type")
        self._factories[validated_task_type] = factory
        logger.info(f"Registered factory: {validated_task_type}")

    def get_factory(self, task_type: str) -> Type[BaseFactoryImpl]:
        """
        Get the factory for a task type.

        Args:
            task_type: Type of the task.

        Returns:
            Factory class.

        Raises:
            UnknownTaskTypeError: If task_type is not registered.
        """
        validated_task_type = validate_identifier(task_type, "task_type")
        factory_class = self._factories.get(validated_task_type)
        if not factory_class:
            raise UnknownTaskTypeError(validated_task_type, list(self._factories.keys()))
        return factory_class

# Model wrappers (placeholders)
class GradientBoostingClassifierWrapper:
    def __init__(self, *args, **kwargs):
        pass

class KNeighborsClassifierWrapper:
    def __init__(self, *args, **kwargs):
        pass

class LogisticRegressionWrapper:
    def __init__(self, *args, **kwargs):
        pass

class RandomForestClassifierWrapper:
    def __init__(self, *args, **kwargs):
        pass

class SVCWrapper:
    def __init__(self, *args, **kwargs):
        pass

class GradientBoostingRegressorWrapper:
    def __init__(self, *args, **kwargs):
        pass

class LinearRegressionWrapper:
    def __init__(self, *args, **kwargs):
        pass

class RandomForestRegressorWrapper:
    def __init__(self, *args, **kwargs):
        pass

class RidgeWrapper:
    def __init__(self, *args, **kwargs):
        pass

class SVRWrapper:
    def __init__(self, *args, **kwargs):
        pass

# Classifier factory
class ClassifierFactory(BaseFactoryImpl):
    """
    Factory for classification models.
    """
    def __init__(self):
        super().__init__({
            "gradient_boosting": GradientBoostingClassifierWrapper,
            "knn": KNeighborsClassifierWrapper,
            "logistic_regression": LogisticRegressionWrapper,
            "random_forest": RandomForestClassifierWrapper,
            "svm": SVCWrapper,
        })

# Regressor factory
class RegressorFactory(BaseFactoryImpl):
    """
    Factory for regression models.
    """
    def __init__(self):
        super().__init__({
            "gradient_boosting": GradientBoostingRegressorWrapper,
            "linear_regression": LinearRegressionWrapper,
            "random_forest": RandomForestRegressorWrapper,
            "ridge": RidgeWrapper,
            "svr": SVRWrapper,
        })

# General factory (Abstract Factory Pattern)
class ModelFactory(ModelFactoryInterface):
    """
    General factory for creating models based on task type.

    Example:
        >>> factory = ModelFactory()
        >>> config = ModelConfig(model_type="random_forest", params={"n_estimators": 100})
        >>> model = factory.create_model("classification", config)
    """
    def __init__(self):
        self.registry = FactoryRegistry()
        # Register default factories
        self.registry.register_factory("classification", ClassifierFactory)
        self.registry.register_factory("regression", RegressorFactory)

    def register_model(self, model_type: str, model_class: Type[Any]) -> None:
        """
        Not implemented. Use register_factory instead.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError("Use register_factory instead")

    def register_factory(self, task_type: str, factory: Type[BaseFactoryImpl]) -> None:
        """
        Register a new factory for a task type.

        Args:
            task_type: Type of the task (e.g., 'clustering').
            factory: Factory class for the task.
        """
        self.registry.register_factory(task_type, factory)

    def create_model(self, task_type: str, config: Union[ConfigInterface, str], *args: Any, **kwargs: Any) -> Any:
        """
        Create a model based on task type and configuration.

        Args:
            task_type: Type of the task (e.g., 'classification').
            config: Model configuration (ConfigInterface or model_type string).
            *args: Positional arguments for model initialization.
            **kwargs: Keyword arguments for model initialization.

        Returns:
            Model instance.

        Raises:
            UnknownTaskTypeError: If task_type is not registered.
            ConfigurationError: If config is invalid.
        """
        factory_class = self.registry.get_factory(task_type)
        logger.info(f"Creating model for task: {task_type}")
        return factory_class().create_model(config, *args, **kwargs)

# Pipeline (Facade Pattern)
class Pipeline:
    """
    Facade for running a machine learning pipeline (preprocessing, training, prediction).

    Example:
        >>> pipeline = Pipeline()
        >>> config = ModelConfig(model_type="random_forest", params={"n_estimators": 100})
        >>> pipeline.run("classification", config)
    """
    def __init__(self):
        self.factory = ModelFactory()
        self.preprocessor = PreprocessorStrategy()
        self.trainer = TrainingStrategy()
        self.observers: List[Observer] = [LoggerObserver()]

    def notify_observers(self, event: str) -> None:
        """
        Notify all observers of an event.

        Args:
            event: Event description.
        """
        for observer in self.observers:
            observer.update(event)

    def run(self, task_type: str, config: ModelConfig) -> Any:
        """
        Run the pipeline: preprocess, train, and return the model.

        Args:
            task_type: Type of task (e.g., 'classification').
            config: Model configuration.

        Returns:
            Trained model instance.
        """
        self.notify_observers(f"Starting pipeline for task: {task_type}")
        data = self.preprocessor.preprocess()
        model = self.factory.create_model(task_type, config)
        self.trainer.train(model, data)
        self.notify_observers(f"Completed pipeline for task: {task_type}")
        return model

# Strategy Pattern for preprocessing and training
class PreprocessorStrategy:
    """
    Strategy for preprocessing data.
    """
    def preprocess(self) -> Any:
        """
        Preprocess data (placeholder).

        Returns:
            Preprocessed data.
        """
        return "preprocessed_data"

class TrainingStrategy:
    """
    Strategy for training models.
    """
    def train(self, model: Any, data: Any) -> None:
        """
        Train the model (placeholder).

        Args:
            model: Model instance.
            data: Training data.
        """
        pass

# Observer Pattern for logging events
class Observer:
    """
    Interface for observers.
    """
    def update(self, event: str) -> None:
        pass

class LoggerObserver(Observer):
    """
    Observer for logging pipeline events.
    """
    def update(self, event: str) -> None:
        """
        Log an event.

        Args:
            event: Event description.
        """
        logger.info(event)

# Example usage
if __name__ == "__main__":
    # Create a pipeline
    pipeline = Pipeline()
    # Create a model configuration
    config = ModelConfig(model_type="random_forest", params={"n_estimators": 100})
    # Run the pipeline
    model = pipeline.run("classification", config)














    from typing import Any, Dict, Optional, Type, Union, Protocol
from types import MappingProxyType
import logging
from pydantic import BaseModel, Field, field_validator
from abc import ABC, abstractmethod

# Configuration error
class ConfigurationError(Exception):
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.field = field

# Validator
def validate_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{field_name} must be a string", field_name)
    stripped = value.strip().lower()
    if not stripped:
        raise ConfigurationError(f"{field_name} cannot be empty or whitespace", field_name)
    return stripped

# Configuration interface
class ConfigInterface(Protocol):
    model_type: str
    params: Dict[str, Any]

# Pydantic model
class ModelConfig(BaseModel):
    model_type: str = Field(..., min_length=1, description="Type of the model")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model parameters")

    @field_validator("model_type")
    def validate_model_type(cls, value: str) -> str:
        return validate_identifier(value, "model_type")

    @field_validator("params")
    def validate_params(cls, value: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
        model_type = values.get("model_type")
        if model_type == "random_forest":
            if value.get("n_estimators") and (not isinstance(value["n_estimators"], int) or value["n_estimators"] <= 0):
                raise ConfigurationError("n_estimators must be a positive integer", "params.n_estimators")
        return value

    model_config = {"extra": "forbid"}

# Model builder
class ModelBuilder:
    def __init__(self, model_class: Type[Any]):
        self.model_class = model_class
        self.params: Dict[str, Any] = {}
        self.args: tuple = ()
        self.kwargs: Dict[str, Any] = {}

    def with_params(self, params: Optional[Dict[str, Any]] = None) -> "ModelBuilder":
        self.params = params or {}
        return self

    def with_args(self, *args: Any) -> "ModelBuilder":
        self.args = args
        return self

    def with_kwargs(self, **kwargs: Any) -> "ModelBuilder":
        self.kwargs = kwargs
        return self

    def build(self) -> Any:
        try:
            return self.model_class(*self.args, **self.params, **self.kwargs)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize model: {str(e)}", "build")

# Factory interface
class ModelFactoryInterface(ABC):
    @abstractmethod
    def register_model(self, model_type: str, model_class: Type[Any]) -> None:
        pass

    @abstractmethod
    def create_model(self, model_type: str, params: Optional[Dict[str, Any]] = None, *args: Any, **kwargs: Any) -> Any:
        pass

# Base factory implementation
class BaseFactoryImpl(ModelFactoryInterface):
    """
    Base implementation for model factories.
    """
    def __init__(self, models: Optional[Dict[str, Type[Any]]] = None):
        """
        Initialize the factory.

        Args:
            models: Dictionary mapping model types to model classes.
        """
        self._models = MappingProxyType(models or {})
        self._logger = logging.getLogger(__name__)

    def register_model(self, model_type: str, model_class: Type[Any]) -> None:
        """
        Register a new model type.
        """
        validated_model_type = validate_identifier(model_type, "model_type")
        if validated_model_type in self._models:
            self._logger.error(f"Model type '{validated_model_type}' already registered")
            raise ConfigurationError(f"Model type '{validated_model_type}' already registered", "model_type")

        new_models = dict(self._models)
        new_models[validated_model_type] = model_class
        self._models = MappingProxyType(new_models)
        self._logger.info(f"Registered model: {validated_model_type}")

    def create_model(self, model_type: str, params: Optional[Dict[str, Any]] = None, *args: Any, **kwargs: Any) -> Any:
        """
        Create a model instance from model type and parameters.

        Args:
            model_type: Type of the model (e.g., 'random_forest').
            params: Model parameters (e.g., {'n_estimators': 100}). Defaults to empty dict.
            *args: Positional arguments for model initialization.
            **kwargs: Keyword arguments for model initialization.

        Returns:
            Model instance.

        Raises:
            ConfigurationError: If model_type is invalid or not registered.
        """
        # Create and validate ModelConfig internally
        try:
            config = ModelConfig(model_type=model_type, params=params or {})
        except Exception as e:
            self._logger.error(f"Invalid configuration: {str(e)}")
            raise ConfigurationError(f"Invalid configuration: {str(e)}", "config")

        model_type = validate_identifier(config.model_type, "model_type")
        model_class = self._models.get(model_type)
        if not model_class:
            self._logger.error(f"Unknown model type: {model_type}")
            raise ConfigurationError(f"Unknown model type: '{model_type}'", "model_type")

        self._logger.info(f"Creating model: {model_type}")
        return ModelBuilder(model_class).with_params(config.params).with_args(*args).with_kwargs(**kwargs).build()