from abc import ABC, abstractmethod

class BaseFactory(ABC):
    """
    Base factory for modelfatories

    Methods:
        register_model: Register a new model type.
        crate_model: Create a model instance from configuration.
    """
    @classmethod
    @abstractmethod
    def register_model(cls):
        pass

    @classmethod
    @abstractmethod
    def create_model(cls): 
        pass