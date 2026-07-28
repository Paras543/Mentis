"""
Curve computation for model evaluation (ROC, PR, calibration,
residuals, learning curves).

These functions return plain data structures rather than plotting
directly, so `visualization/charts.py` (and any future output format)
can render them without this module depending on matplotlib/plotly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve as sk_calibration_curve
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    precision_recall_curve as sk_pr_curve,
    roc_curve as sk_roc_curve,
)
from sklearn.model_selection import learning_curve as sk_learning_curve

from mentis.constants import RANDOM_STATE
from mentis.exceptions import ExplainabilityError
from mentis.utils.logger import get_logger

logger = get_logger(__name__)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """
    Compute a confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dict with "matrix" (list of lists) and "labels" (sorted unique
        class labels used to build the matrix).

    Examples:
        >>> import numpy as np
        >>> result = compute_confusion_matrix(np.array([0, 1, 1]), np.array([0, 1, 0]))
        >>> result["labels"]
        [0, 1]
    """
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    matrix = sk_confusion_matrix(y_true, y_pred, labels=labels)
    return {"matrix": matrix.tolist(), "labels": labels}


def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]]:
    """
    Compute ROC curve points for binary classification.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probability/score for the positive class.

    Returns:
        Dict with "fpr", "tpr", "thresholds" lists.

    Raises:
        ExplainabilityError: If the curve cannot be computed (e.g. only
            one class present in `y_true`).

    Examples:
        >>> import numpy as np
        >>> curve = compute_roc_curve(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.3]))
        >>> len(curve["fpr"]) == len(curve["tpr"])
        True
    """
    if len(np.unique(y_true)) < 2:
        raise ExplainabilityError("Could not compute ROC curve: y_true must contain at least two classes.")

    try:
        fpr, tpr, thresholds = sk_roc_curve(y_true, y_score)
    except ValueError as exc:
        raise ExplainabilityError(f"Could not compute ROC curve: {exc}") from exc

    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}


def compute_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]]:
    """
    Compute precision-recall curve points for binary classification.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probability/score for the positive class.

    Returns:
        Dict with "precision", "recall", "thresholds" lists.

    Raises:
        ExplainabilityError: If the curve cannot be computed.

    Examples:
        >>> import numpy as np
        >>> curve = compute_pr_curve(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.3]))
        >>> len(curve["precision"]) == len(curve["recall"])
        True
    """
    try:
        precision, recall, thresholds = sk_pr_curve(y_true, y_score)
    except ValueError as exc:
        raise ExplainabilityError(f"Could not compute PR curve: {exc}") from exc

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """
    Compute a calibration (reliability) curve for binary classification.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of bins to group predictions into.

    Returns:
        Dict with "mean_predicted_value" and "fraction_of_positives"
        lists.

    Raises:
        ExplainabilityError: If the curve cannot be computed.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        >>> y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.4])
        >>> curve = compute_calibration_curve(y_true, y_prob, n_bins=4)
        >>> len(curve["mean_predicted_value"]) <= 4
        True
    """
    try:
        fraction_of_positives, mean_predicted_value = sk_calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
    except ValueError as exc:
        raise ExplainabilityError(f"Could not compute calibration curve: {exc}") from exc

    return {
        "mean_predicted_value": mean_predicted_value.tolist(),
        "fraction_of_positives": fraction_of_positives.tolist(),
    }


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, list[float]]:
    """
    Compute residuals for a regression model.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        Dict with "predicted", "residuals" lists, where
        `residuals[i] = y_true[i] - y_pred[i]`.

    Examples:
        >>> import numpy as np
        >>> result = compute_residuals(np.array([3.0, 5.0]), np.array([2.5, 5.5]))
        >>> result["residuals"]
        [0.5, -0.5]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    return {"predicted": y_pred.tolist(), "residuals": residuals.tolist()}


def compute_learning_curve(
    estimator: Any,
    X: Any,
    y: Any,
    cv: int = 5,
    scoring: str | None = None,
    train_sizes: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """
    Compute a learning curve (train/validation score vs. training set
    size) for a given estimator.

    Args:
        estimator: An unfitted scikit-learn-compatible estimator.
        X: Full feature matrix.
        y: Full target vector.
        cv: Number of cross-validation folds.
        scoring: Scikit-learn scoring string. If None, the estimator's
            default `.score()` is used.
        train_sizes: Fractions of the training set to use at each
            point. Defaults to `np.linspace(0.1, 1.0, 5)`.

    Returns:
        Dict with "train_sizes", "train_scores_mean", "train_scores_std",
        "val_scores_mean", "val_scores_std".

    Raises:
        ExplainabilityError: If the learning curve cannot be computed.

    Examples:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, random_state=42)
        >>> curve = compute_learning_curve(LogisticRegression(), X, y, cv=3)  # doctest: +SKIP
    """
    sizes = train_sizes if train_sizes is not None else np.linspace(0.1, 1.0, 5)

    try:
        res = sk_learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            scoring=scoring,
            train_sizes=sizes,
            random_state=RANDOM_STATE,
        )
        train_sizes_abs, train_scores, val_scores = res[0], res[1], res[2]
    except Exception as exc:  # noqa: BLE001 - surface as domain error
        raise ExplainabilityError(f"Could not compute learning curve: {exc}") from exc

    return {
        "train_sizes": train_sizes_abs.tolist(),
        "train_scores_mean": train_scores.mean(axis=1).tolist(),
        "train_scores_std": train_scores.std(axis=1).tolist(),
        "val_scores_mean": val_scores.mean(axis=1).tolist(),
        "val_scores_std": val_scores.std(axis=1).tolist(),
    }


