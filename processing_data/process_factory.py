from abc import ABC, abstractmethod

class BaseProcess(ABC):

    @staticmethod
    @abstractmethod
    def process() -> tuple[dict, dict]:
        pass

class ProcessFactory:
    @staticmethod
    def create(class_name: str):
        subclasses = BaseProcess.__subclasses__()
        subclass_dict = {cls.__name__.lower() for cls in subclasses}
        obj = subclass_dict[class_name]
        return obj()
    
