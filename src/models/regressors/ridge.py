from sklearn.linear_model import Ridge

from .._base import BaseModelImpl


class RidgeWrapper(BaseModelImpl):
    model_class = Ridge
    default_params = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
    'fit_intercept': [True, False],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
    'tol': [1e-5, 1e-4, 1e-3],
    'positive': [False, True],
    'random_state': [None, 42]
    }

    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=1e-4,
        solver="auto",
        positive=False,
        random_state=None,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            positive=positive,
            random_state=random_state,
        )