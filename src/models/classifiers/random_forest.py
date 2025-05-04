from typing import Optional, Union, Dict, List

from sklearn.ensemble import RandomForestClassifier

from .._base import BaseModelImpl


class RandomForestClassifierWrapper(BaseModelImpl):
    """
    ---
    A wrapper for sklearn's [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

    This class provides a simplified interface for using RandomForestClassifier with predefined hyperparameter grids for tuning.

    ### Parameters:

    - Uses the same parameters as RandomForestClassifier. See the official documentation for details.

    ### Methods:

    - **get_model()**: Returns the underlying RandomForestClassifier model instance.
    - **get_param_grid()**: Returns a dictionary of hyperparameter grids for tuning the model.
    """
    model_class = RandomForestClassifier
    default_params: Dict[str, List[Union[int, float, str, None]]] = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'class_weight': [None, 'balanced']
    }

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )