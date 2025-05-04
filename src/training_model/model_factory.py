from __future__ import annotations
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Any, Optional, Protocol, TypeVar, Tuple, Type, Literal, Union
import logging

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import matplotlib.pyplot as plt
from tqdm.rich import tqdm

ModelDict = Dict[str, BaseEstimator]


class ClassificationModelFactory:
    """Factory for creating classification models."""
    
    @staticmethod
    def create_models(max_iter: int, random_state: int) -> ModelDict:
        return {
            "LogisticRegression": LogisticRegression(
                max_iter=max_iter,
                random_state=random_state
            ),
            "RandomForestClassifier": RandomForestClassifier(
                random_state=random_state
            ),
            "LinearSVC": LinearSVC(
                max_iter=max_iter,
                random_state=random_state
            ),
            "GaussianNB": GaussianNB(),
            "MultinomialNB": MultinomialNB()
        }
