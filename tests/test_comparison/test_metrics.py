"""
Tests for mentis/comparison/metrics.py — metric computation functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from mentis.comparison.metrics import (
    compute_classification_metrics,
    compute_metrics,
    compute_regression_metrics,
    is_higher_better,
    primary_metric_for_task,
)


class TestComputeClassificationMetrics:
    def test_perfect_binary_classification(self):
        y = np.array([0, 1, 0, 1])
        metrics = compute_classification_metrics(y, y)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)

    def test_all_wrong_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_returns_required_keys(self):
        y = np.array([0, 1, 0, 1])
        metrics = compute_classification_metrics(y, y)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics

    def test_roc_auc_with_proba(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_classification_metrics(y_true, y_pred, y_proba)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] == pytest.approx(1.0)

    def test_no_roc_auc_without_proba(self):
        y = np.array([0, 1, 0, 1])
        metrics = compute_classification_metrics(y, y)
        assert "roc_auc" not in metrics

    def test_multiclass_uses_macro(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["f1"] == pytest.approx(1.0)


class TestComputeRegressionMetrics:
    def test_perfect_regression(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_regression_metrics(y, y)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)

    def test_returns_required_keys(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = compute_regression_metrics(y, y + 0.1)
        for key in ("rmse", "mae", "mape", "r2"):
            assert key in metrics

    def test_rmse_positive(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics["rmse"] > 0

    def test_r2_below_1_for_imperfect(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics["r2"] < 1.0


class TestComputeMetricsDispatch:
    def test_classification_dispatch(self):
        y = np.array([0, 1, 0, 1])
        metrics = compute_metrics("classification", y, y)
        assert "accuracy" in metrics
        assert "rmse" not in metrics

    def test_regression_dispatch(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics("regression", y, y)
        assert "rmse" in metrics
        assert "accuracy" not in metrics

    def test_unsupported_task_raises(self):
        y = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Unsupported task"):
            compute_metrics("clustering", y, y)


class TestPrimaryMetric:
    def test_classification_primary_metric(self):
        assert primary_metric_for_task("classification") == "f1"

    def test_regression_primary_metric(self):
        assert primary_metric_for_task("regression") == "r2"


class TestIsHigherBetter:
    def test_rmse_lower_is_better(self):
        assert is_higher_better("rmse") is False

    def test_mae_lower_is_better(self):
        assert is_higher_better("mae") is False

    def test_mape_lower_is_better(self):
        assert is_higher_better("mape") is False

    def test_f1_higher_is_better(self):
        assert is_higher_better("f1") is True

    def test_r2_higher_is_better(self):
        assert is_higher_better("r2") is True

    def test_accuracy_higher_is_better(self):
        assert is_higher_better("accuracy") is True
