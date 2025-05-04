from sklearn.svm import SVR

from .._base import BaseModelImpl


class SVRWrapper(BaseModelImpl):
    model_class = SVR
    default_params = {
        'kernel': ['rbf', 'linear', 'poly'],
        'degree': [2, 3], 
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'coef0': [0.0, 1.0],
        'C': [0.1, 1.0, 10.0, 100.0],
        'epsilon': [0.01, 0.1, 0.5],
    }

    def __init__(
        self,
        *,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        C=1.0,
        epsilon=0.1,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1,
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            nu=0.0,
            epsilon=epsilon,
            verbose=verbose,
            shrinking=shrinking,
            probability=False,
            cache_size=cache_size,
            class_weight=None,
            max_iter=max_iter,
            random_state=None,
        )