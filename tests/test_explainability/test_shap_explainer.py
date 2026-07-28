"""
Tests for mentis/explainability/shap_explainer.py — SHAP values.

SHAP is an optional dependency. Tests are skipped automatically when
the `shap` package is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

shap = pytest.importorskip("shap", reason="shap not installed — skipping SHAP tests")

from mentis.exceptions import ExplainabilityError
from mentis.explainability.shap_explainer import compute_shap_values


class TestComputeShapValues:
    def test_returns_required_keys(self, fitted_clf, X_test_clf):
        result = compute_shap_values(fitted_clf, X_test_clf)
        for key in ("shap_values", "feature_names", "base_value", "explainer_type"):
            assert key in result

    def test_explainer_type_is_tree(self, fitted_clf, X_test_clf):
        result = compute_shap_values(fitted_clf, X_test_clf)
        assert result["explainer_type"] == "tree"

    def test_shap_values_shape(self, fitted_clf, X_test_clf):
        result = compute_shap_values(fitted_clf, X_test_clf)
        shap_vals = np.array(result["shap_values"])
        assert shap_vals.ndim == 2
        assert shap_vals.shape[1] == X_test_clf.shape[1]

    def test_feature_names_match_columns(self, fitted_clf, X_test_clf):
        result = compute_shap_values(fitted_clf, X_test_clf)
        assert result["feature_names"] == list(X_test_clf.columns)

    def test_base_value_is_float(self, fitted_clf, X_test_clf):
        result = compute_shap_values(fitted_clf, X_test_clf)
        assert isinstance(result["base_value"], float)

    def test_regression_model(self, fitted_reg, X_test_reg):
        result = compute_shap_values(fitted_reg, X_test_reg)
        assert result["explainer_type"] in ("tree", "kernel")
        assert len(result["feature_names"]) == X_test_reg.shape[1]

    def test_large_input_is_sampled(self, fitted_clf, X_test_clf):
        """When input > 500 rows, SHAP should sample it down."""
        import pandas as pd
        big_X = pd.concat([X_test_clf] * 20, ignore_index=True)
        assert len(big_X) > 500
        result = compute_shap_values(fitted_clf, big_X)
        # Should complete without error; shap_values rows <= 500
        shap_vals = np.array(result["shap_values"])
        assert shap_vals.shape[0] <= 500

    def test_raises_when_both_explainers_fail(self):
        """A model that's not sklearn-compatible should raise ExplainabilityError."""
        class FakeModel:
            pass

        import pandas as pd
        X = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ExplainabilityError):
            compute_shap_values(FakeModel(), X)
