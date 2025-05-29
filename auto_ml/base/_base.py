from typing import Protocol

class BaseSearchCV(Protocol):

    def fit(self): ...

    def predict(self): ...