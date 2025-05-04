from sklearn.neighbors import KNeighborsClassifier
from .._base import BaseModelImpl

class KNeighborsClassifierWrapper(BaseModelImpl):
    model_class = KNeighborsClassifier
    default_params = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [20, 30, 40],
        'p': [1, 2],
        'metric': ['minkowski', 'manhattan', 'euclidean']
    }
    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
            weights = weights
        )