"""
SHAP-based explainability for Mentis.

SHAP's API varies significantly across explainer types (Tree, Linear,
Kernel), so this module tries the fast, model-specific explainer first
and falls back to the slower, model-agnostic `KernelExplainer` only
when necessary -- keeping that complexity out of callers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mentis.exceptions import ExplainabilityError
from mentis.utils.helpers import truncate_dataframe_for_heavy_ops
from mentis.utils.logger import get_logger

logger = get_logger(__name__)

# Cap the background/sample size for SHAP so KernelExplainer (which is
# O(n) in the number of background samples) stays tractable.
_MAX_SHAP_ROWS = 500


def compute_shap_values(model: Any, X: Any) -> dict[str, Any]:
    """
    Compute SHAP values for a fitted model.

    Tries `shap.TreeExplainer` first (fast, exact for tree-based
    models). Falls back to `shap.KernelExplainer` (slower,
    model-agnostic) if the model type isn't tree-based or
    `TreeExplainer` raises.

    Args:
        model: A fitted scikit-learn-compatible estimator.
        X: Feature matrix to explain. Large inputs are sampled down to
            keep `KernelExplainer` tractable.

    Returns:
        Dict with:
            - "shap_values": list of lists (n_samples x n_features)
            - "feature_names": list of feature names
            - "base_value": expected value of the model output
            - "explainer_type": "tree" or "kernel"

    Raises:
        ExplainabilityError: If SHAP is not installed, or if both the
            tree and kernel explainers fail.

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=50, random_state=42)
        >>> model = RandomForestClassifier(random_state=42).fit(X, y)
        >>> result = compute_shap_values(model, X)  # doctest: +SKIP
    """
    try:
        import shap
    except ImportError as exc:
        raise ExplainabilityError(
            "shap is not installed. Install it with `pip install shap` to use "
            "SHAP-based explainability."
        ) from exc

    X_sample = truncate_dataframe_for_heavy_ops(X, _MAX_SHAP_ROWS) if hasattr(X, "sample") else X

    feature_names = (
        list(X_sample.columns)
        if hasattr(X_sample, "columns")
        else [f"feature_{i}" for i in range(np.asarray(X_sample).shape[1])]
    )

    try:
        return _compute_with_tree_explainer(shap, model, X_sample, feature_names)
    except Exception as tree_exc:  # noqa: BLE001 - fall back to kernel explainer
        logger.warning(f"TreeExplainer failed ({tree_exc}); falling back to KernelExplainer.")
        try:
            return _compute_with_kernel_explainer(shap, model, X_sample, feature_names)
        except Exception as kernel_exc:  # noqa: BLE001 - both explainers exhausted
            raise ExplainabilityError(
                f"SHAP explanation failed. TreeExplainer error: {tree_exc}. "
                f"KernelExplainer error: {kernel_exc}."
            ) from kernel_exc


def _compute_with_tree_explainer(
    shap: Any, model: Any, X_sample: Any, feature_names: list[str]
) -> dict[str, Any]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Multiclass/binary classifiers may return a list of arrays (one
    # per class) or a 3D ndarray (n_samples, n_features, n_classes);
    # use the positive/last class for a single summary.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).flatten()[-1]

    return {
        "shap_values": np.asarray(shap_values).tolist(),
        "feature_names": feature_names,
        "base_value": float(base_value),
        "explainer_type": "tree",
    }


def _compute_with_kernel_explainer(
    shap: Any, model: Any, X_sample: Any, feature_names: list[str]
) -> dict[str, Any]:
    predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict

    # KernelExplainer needs a small background set, not the full sample.
    background_size = min(50, len(X_sample))
    background = (
        shap.sample(X_sample, background_size)
        if hasattr(shap, "sample")
        else X_sample[:background_size]
    )

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_sample, silent=True)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).flatten()[-1]

    return {
        "shap_values": np.asarray(shap_values).tolist(),
        "feature_names": feature_names,
        "base_value": float(base_value),
        "explainer_type": "kernel",
    }
