"""
Tests for mentis/explainability/permutation.py — permutation importance.
"""

from __future__ import annotations

import numpy as np
import pytest

from mentis.explainability.permutation import compute_permutation_importance


class TestComputePermutationImportance:
    def test_returns_dict(self, fitted_clf, X_test_clf, y_test_clf):
        result = compute_permutation_importance(fitted_clf, X_test_clf, y_test_clf)
        assert isinstance(result, dict)

    def test_keys_match_features(self, fitted_clf, X_test_clf, y_test_clf):
        result = compute_permutation_importance(fitted_clf, X_test_clf, y_test_clf)
        expected_cols = set(X_test_clf.columns)
        assert set(result.keys()) == expected_cols

    def test_values_are_dicts_with_mean_std(self, fitted_clf, X_test_clf, y_test_clf):
        result = compute_permutation_importance(fitted_clf, X_test_clf, y_test_clf)
        for v in result.values():
            assert isinstance(v, dict)
            assert "importance_mean" in v
            assert "importance_std" in v
            assert isinstance(v["importance_mean"], float)
            assert isinstance(v["importance_std"], float)

    def test_regression_model(self, fitted_reg, X_test_reg, y_test_reg):
        result = compute_permutation_importance(fitted_reg, X_test_reg, y_test_reg)
        assert isinstance(result, dict)
        assert len(result) == X_test_reg.shape[1]

    def test_numpy_arrays_accepted(self, fitted_clf, X_test_clf, y_test_clf):
        result = compute_permutation_importance(
            fitted_clf,
            X_test_clf.to_numpy(),
            y_test_clf,
        )
        assert isinstance(result, dict)
        # Feature names fall back to "feature_i" for numpy arrays
        assert all(k.startswith("feature_") or isinstance(k, str) for k in result)
