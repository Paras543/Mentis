"""
Model registry ("zoo") for Mentis's automated model comparison.

Keeping model instantiation in one registry -- rather than scattered
`if/else` blocks in the trainer -- means adding a new algorithm is a
one-line addition here, following the Open/Closed Principle.
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mentis.constants import RANDOM_STATE
from mentis.utils.logger import get_logger

logger = get_logger(__name__)

# Optional gradient-boosting libraries. Each is imported defensively so
# Mentis remains fully usable even if a given package isn't installed --
# it just quietly drops that model from the zoo instead of crashing.
_OPTIONAL_MODELS: dict[str, Callable[[], Any]] = {}

try:
    from xgboost import XGBClassifier, XGBRegressor

    _OPTIONAL_MODELS["XGBoost"] = lambda: XGBClassifier(
        random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0
    )
    _OPTIONAL_MODELS["XGBoost_regressor"] = lambda: XGBRegressor(
        random_state=RANDOM_STATE, verbosity=0
    )
except ImportError:
    logger.warning("xgboost not installed; skipping XGBoost models.")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    _OPTIONAL_MODELS["LightGBM"] = lambda: LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    _OPTIONAL_MODELS["LightGBM_regressor"] = lambda: LGBMRegressor(
        random_state=RANDOM_STATE, verbose=-1
    )
except ImportError:
    logger.warning("lightgbm not installed; skipping LightGBM models.")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    _OPTIONAL_MODELS["CatBoost"] = lambda: CatBoostClassifier(
        random_state=RANDOM_STATE, verbose=False
    )
    _OPTIONAL_MODELS["CatBoost_regressor"] = lambda: CatBoostRegressor(
        random_state=RANDOM_STATE, verbose=False
    )
except ImportError:
    logger.warning("catboost not installed; skipping CatBoost models.")


def get_classification_models(selected: list[str] | None = None) -> dict[str, Any]:
    """
    Build a dict of fresh, unfitted classification model instances.

    Args:
        selected: Optional list of model names to restrict the zoo to
            (e.g. ["Random Forest", "XGBoost"]). If None, all available
            models are returned.

    Returns:
        Dict mapping model name -> unfitted estimator instance.

    Examples:
        >>> models = get_classification_models(["Logistic Regression"])
        >>> list(models.keys())
        ['Logistic Regression']
    """
    zoo: dict[str, Any] = {
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
    }

    if "XGBoost" in _OPTIONAL_MODELS:
        zoo["XGBoost"] = _OPTIONAL_MODELS["XGBoost"]()
    if "LightGBM" in _OPTIONAL_MODELS:
        zoo["LightGBM"] = _OPTIONAL_MODELS["LightGBM"]()
    if "CatBoost" in _OPTIONAL_MODELS:
        zoo["CatBoost"] = _OPTIONAL_MODELS["CatBoost"]()

    if selected:
        zoo = {name: model for name, model in zoo.items() if name in selected}

    return zoo


def get_regression_models(selected: list[str] | None = None) -> dict[str, Any]:
    """
    Build a dict of fresh, unfitted regression model instances.

    Args:
        selected: Optional list of model names to restrict the zoo to.
            If None, all available models are returned.

    Returns:
        Dict mapping model name -> unfitted estimator instance.

    Examples:
        >>> models = get_regression_models(["Ridge"])
        >>> list(models.keys())
        ['Ridge']
    """
    zoo: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Extra Trees Regressor": ExtraTreesRegressor(random_state=RANDOM_STATE),
    }

    if "XGBoost_regressor" in _OPTIONAL_MODELS:
        zoo["XGBoost Regressor"] = _OPTIONAL_MODELS["XGBoost_regressor"]()
    if "LightGBM_regressor" in _OPTIONAL_MODELS:
        zoo["LightGBM Regressor"] = _OPTIONAL_MODELS["LightGBM_regressor"]()
    if "CatBoost_regressor" in _OPTIONAL_MODELS:
        zoo["CatBoost Regressor"] = _OPTIONAL_MODELS["CatBoost_regressor"]()

    if selected:
        zoo = {name: model for name, model in zoo.items() if name in selected}

    return zoo


def get_model_zoo(task: str, selected: list[str] | None = None) -> dict[str, Any]:
    """
    Return the appropriate model zoo for a given task type.

    Args:
        task: "classification" or "regression".
        selected: Optional restriction to specific model names.

    Returns:
        Dict mapping model name -> unfitted estimator instance.

    Raises:
        ValueError: If `task` is not "classification" or "regression".

    Examples:
        >>> zoo = get_model_zoo("classification")
        >>> "Random Forest" in zoo
        True
    """
    if task == "classification":
        return get_classification_models(selected)
    if task == "regression":
        return get_regression_models(selected)
    raise ValueError(f"Unsupported task type: {task!r}")






