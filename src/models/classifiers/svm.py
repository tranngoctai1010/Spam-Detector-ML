from sklearn.svm import SVC
from .._base import BaseModelImpl

class SVCWrapper(BaseModelImpl):
    model_class = SVC
    default_params = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
        'class_weight': [None, 'balanced']
    }
    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            nu=0.0,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )