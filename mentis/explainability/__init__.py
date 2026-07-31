"""
Explainability subpackage: SHAP values, permutation importance, and
evaluation curves for Mentis.
"""

from mentis.explainability.curves import (
    compute_calibration_curve,
    compute_confusion_matrix,
    compute_learning_curve,
    compute_pr_curve,
    compute_residuals,
    compute_roc_curve,
)
from mentis.explainability.permutation import compute_permutation_importance
from mentis.explainability.shap_explainer import compute_shap_values

__all__ = [
    "compute_shap_values",
    "compute_permutation_importance",
    "compute_confusion_matrix",
    "compute_roc_curve",
    "compute_pr_curve",
    "compute_calibration_curve",
    "compute_residuals",
    "compute_learning_curve",
]
