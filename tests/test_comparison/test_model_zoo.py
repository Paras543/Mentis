"""
Tests for mentis/comparison/model_zoo.py — model registry.
"""

from __future__ import annotations

import pytest

from mentis.comparison.model_zoo import (
    get_classification_models,
    get_model_zoo,
    get_regression_models,
)


class TestGetClassificationModels:
    def test_returns_nonempty_dict(self):
        zoo = get_classification_models()
        assert len(zoo) > 0

    def test_contains_core_models(self):
        zoo = get_classification_models()
        for name in ("Random Forest", "Logistic Regression", "Decision Tree"):
            assert name in zoo

    def test_selected_filter(self):
        zoo = get_classification_models(selected=["Random Forest", "Decision Tree"])
        assert set(zoo.keys()) == {"Random Forest", "Decision Tree"}

    def test_empty_selected_returns_empty(self):
        zoo = get_classification_models(selected=["NonExistentModel"])
        assert zoo == {}

    def test_models_are_unfitted(self):
        """Models returned must not have been fitted yet."""
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        zoo = get_classification_models()
        for name, model in zoo.items():
            with pytest.raises(NotFittedError):
                check_is_fitted(model)


class TestGetRegressionModels:
    def test_returns_nonempty_dict(self):
        zoo = get_regression_models()
        assert len(zoo) > 0

    def test_contains_core_models(self):
        zoo = get_regression_models()
        for name in ("Linear Regression", "Ridge", "Random Forest Regressor"):
            assert name in zoo

    def test_selected_filter(self):
        zoo = get_regression_models(selected=["Ridge"])
        assert list(zoo.keys()) == ["Ridge"]


class TestGetModelZoo:
    def test_classification_zoo(self):
        zoo = get_model_zoo("classification")
        assert len(zoo) > 0

    def test_regression_zoo(self):
        zoo = get_model_zoo("regression")
        assert len(zoo) > 0

    def test_unsupported_task_raises(self):
        with pytest.raises(ValueError, match="Unsupported task"):
            get_model_zoo("clustering")

    def test_selected_passed_through(self):
        zoo = get_model_zoo("classification", selected=["Logistic Regression"])
        assert "Logistic Regression" in zoo
        assert len(zoo) == 1
