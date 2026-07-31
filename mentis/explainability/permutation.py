"""
Permutation importance computation for Mentis.

Model-agnostic: works with any fitted scikit-learn-compatible
estimator, unlike SHAP which needs model-specific explainers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance as sk_permutation_importance

from mentis.constants import RANDOM_STATE
from mentis.exceptions import ExplainabilityError
from mentis.utils.logger import get_logger

logger = get_logger(__name__)


def compute_permutation_importance(
    model: Any,
    X: Any,
    y: Any,
    scoring: str | None = None,
    n_repeats: int = 10,
) -> dict[str, Any]:
    """
    Compute permutation importance for a fitted model.

    Args:
        model: A fitted scikit-learn-compatible estimator.
        X: Feature matrix used for evaluation.
        y: Target vector used for evaluation.
        scoring: Scikit-learn scoring string. If None, the estimator's
            default scorer is used.
        n_repeats: Number of times each feature is permuted.

    Returns:
        Dict mapping feature name -> {"importance_mean", "importance_std"},
        sorted by `importance_mean` descending.

    Raises:
        ExplainabilityError: If importance computation fails (e.g. the
            model is unfitted, or `X`/`y` are malformed).

    Examples:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=50, random_state=42)
        >>> model = LogisticRegression().fit(X, y)
        >>> result = compute_permutation_importance(model, X, y, n_repeats=3)
        >>> len(result) == X.shape[1]
        True
    """
    try:
        result: Any = sk_permutation_importance(
            model,
            X,
            y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
        )
    except Exception as exc:
        raise ExplainabilityError(f"Could not compute permutation importance: {exc}") from exc

    feature_names = (
        list(X.columns)
        if hasattr(X, "columns")
        else [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
    )

    importances = {
        str(name): {
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for name, mean, std in zip(
            feature_names, result.importances_mean, result.importances_std, strict=False
        )
    }

    return dict(
        sorted(importances.items(), key=lambda item: item[1]["importance_mean"], reverse=True)
    )
