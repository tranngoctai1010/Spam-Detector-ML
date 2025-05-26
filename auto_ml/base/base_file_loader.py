from abc import ABC, abstractmethod
from typing import Dict

class BaseFileLoader(ABC):
    @classmethod
    def load(path: str) -> Dict:
        pass
    
    @classmethod
    def save(path: str) -> None:
        pass
    