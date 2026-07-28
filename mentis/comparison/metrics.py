"""
Metric computation for classification and regression model comparison.

Centralizing metric logic here means the trainer and leaderboard never
duplicate scikit-learn calls, and adding a new metric only requires a
change in one place.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from mentis.utils.logger import get_logger

logger = get_logger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for the positive class
            (binary) or full probability matrix (multiclass), used for
            ROC AUC. If None, ROC AUC is omitted.

    Returns:
        Dict mapping metric name -> value. Keys: "accuracy",
        "precision", "recall", "f1", and "roc_auc" if computable.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> metrics = compute_classification_metrics(y_true, y_pred)
        >>> round(metrics["accuracy"], 2)
        0.75
    """
    average = "binary" if len(np.unique(y_true)) == 2 else "macro"

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            elif y_proba.ndim == 1:
                auc = roc_auc_score(y_true, y_proba)
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            metrics["roc_auc"] = float(auc)
        except ValueError as exc:
            logger.warning(f"Could not compute ROC AUC: {exc}")

    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute standard regression metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        Dict with keys "rmse", "mae", "mape", "r2".

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([3.0, 5.0, 2.5])
        >>> y_pred = np.array([2.8, 5.1, 2.4])
        >>> metrics = compute_regression_metrics(y_true, y_pred)
        >>> metrics["r2"] > 0.9
        True
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    try:
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
    except ValueError:
        mape = float("nan")

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def compute_metrics(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Dispatch to the correct metric set based on task type.

    Args:
        task: "classification" or "regression".
        y_true: Ground-truth values.
        y_pred: Predicted values.
        y_proba: Predicted probabilities (classification only).

    Returns:
        Dict of computed metrics for the given task.

    Raises:
        ValueError: If `task` is not "classification" or "regression".

    Examples:
        >>> import numpy as np
        >>> compute_metrics("regression", np.array([1.0, 2.0]), np.array([1.1, 1.9]))["mae"] < 0.2
        True
    """
    if task == "classification":
        return compute_classification_metrics(y_true, y_pred, y_proba)
    if task == "regression":
        return compute_regression_metrics(y_true, y_pred)
    raise ValueError(f"Unsupported task type: {task!r}")


def primary_metric_for_task(task: str) -> str:
    """
    Return the default metric used to rank models on the leaderboard.

    Args:
        task: "classification" or "regression".

    Returns:
        "f1" for classification, "r2" for regression.

    Examples:
        >>> primary_metric_for_task("classification")
        'f1'
    """
    return "f1" if task == "classification" else "r2"


def is_higher_better(metric_name: str) -> bool:
    """
    Whether a higher value of the given metric indicates better
    performance (used to sort the leaderboard correctly).

    Args:
        metric_name: Name of the metric, e.g. "rmse", "f1", "r2".

    Returns:
        False for error-style metrics (rmse, mae, mape); True otherwise.

    Examples:
        >>> is_higher_better("rmse")
        False
        >>> is_higher_better("f1")
        True
    """
    lower_is_better = {"rmse", "mae", "mape"}
    return metric_name not in lower_is_better


