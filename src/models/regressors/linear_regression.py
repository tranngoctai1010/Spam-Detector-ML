from sklearn.linear_model import LinearRegression
from .._base import BaseModelImpl

class LinearRegressionWrapper(BaseModelImpl):
    model_class = LinearRegression
    default_params = {
        'fit_intercept': [True, False],
        'positive': [True, False]
    }

    def __init__(
        self,
        *,
        fit_intercept=True,
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive,
        )