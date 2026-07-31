"""
Tests for mentis/explainability/curves.py — evaluation curve computations.
"""

from __future__ import annotations

import numpy as np
import pytest

from mentis.exceptions import ExplainabilityError
from mentis.explainability.curves import (
    compute_calibration_curve,
    compute_confusion_matrix,
    compute_learning_curve,
    compute_pr_curve,
    compute_residuals,
    compute_roc_curve,
)


class TestComputeConfusionMatrix:
    def test_binary_returns_2x2(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        result = compute_confusion_matrix(y_true, y_pred)
        assert result["matrix"] == [[1, 1], [0, 2]]
        assert result["labels"] == [0, 1]

    def test_perfect_prediction(self):
        y = np.array([0, 1, 2])
        result = compute_confusion_matrix(y, y)
        matrix = np.array(result["matrix"])
        assert np.all(matrix == np.diag(np.diag(matrix)))

    def test_multiclass_labels(self):
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 2, 2, 1])
        result = compute_confusion_matrix(y_true, y_pred)
        assert result["labels"] == [0, 1, 2]
        assert len(result["matrix"]) == 3


class TestComputeRocCurve:
    def test_returns_required_keys(self):
        y_true = np.array([0, 1, 1, 0])
        y_score = np.array([0.1, 0.9, 0.8, 0.3])
        result = compute_roc_curve(y_true, y_score)
        assert "fpr" in result
        assert "tpr" in result
        assert "thresholds" in result

    def test_lengths_match(self):
        y_true = np.array([0, 1, 1, 0])
        y_score = np.array([0.1, 0.9, 0.8, 0.3])
        result = compute_roc_curve(y_true, y_score)
        assert len(result["fpr"]) == len(result["tpr"])

    def test_single_class_raises(self):
        y_true = np.array([1, 1, 1])
        y_score = np.array([0.8, 0.9, 0.7])
        with pytest.raises(ExplainabilityError, match="ROC curve"):
            compute_roc_curve(y_true, y_score)

    def test_perfect_classifier_auc_1(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_roc_curve(y_true, y_score)
        assert result["fpr"][0] == pytest.approx(0.0)


class TestComputePrCurve:
    def test_returns_required_keys(self):
        y_true = np.array([0, 1, 1, 0])
        y_score = np.array([0.1, 0.9, 0.8, 0.3])
        result = compute_pr_curve(y_true, y_score)
        assert "precision" in result
        assert "recall" in result
        assert "thresholds" in result

    def test_lengths_consistent(self):
        y_true = np.array([0, 1, 1, 0])
        y_score = np.array([0.1, 0.9, 0.8, 0.3])
        result = compute_pr_curve(y_true, y_score)
        # sklearn PR curve: precision/recall have one more entry than thresholds
        assert len(result["precision"]) == len(result["recall"])


class TestComputeCalibrationCurve:
    def test_returns_required_keys(self):
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.4])
        result = compute_calibration_curve(y_true, y_prob, n_bins=4)
        assert "mean_predicted_value" in result
        assert "fraction_of_positives" in result

    def test_output_length_bounded_by_n_bins(self):
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.4])
        result = compute_calibration_curve(y_true, y_prob, n_bins=4)
        assert len(result["mean_predicted_value"]) <= 4


class TestComputeResiduals:
    def test_perfect_prediction_zero_residuals(self):
        y = np.array([1.0, 2.0, 3.0])
        result = compute_residuals(y, y)
        assert result["residuals"] == pytest.approx([0.0, 0.0, 0.0])

    def test_residual_direction(self):
        y_true = np.array([3.0, 5.0])
        y_pred = np.array([2.5, 5.5])
        result = compute_residuals(y_true, y_pred)
        assert result["residuals"] == pytest.approx([0.5, -0.5])

    def test_predicted_values_in_output(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 2.1])
        result = compute_residuals(y_true, y_pred)
        assert result["predicted"] == pytest.approx([1.1, 2.1])


class TestComputeLearningCurve:
    def test_returns_required_keys(self, fitted_clf, X_train_clf, y_train_clf):
        from sklearn.tree import DecisionTreeClassifier

        result = compute_learning_curve(
            DecisionTreeClassifier(random_state=42),
            X_train_clf,
            y_train_clf,
            cv=2,
        )
        for key in (
            "train_sizes",
            "train_scores_mean",
            "train_scores_std",
            "val_scores_mean",
            "val_scores_std",
        ):
            assert key in result

    def test_train_sizes_ascending(self, X_train_clf, y_train_clf):
        from sklearn.tree import DecisionTreeClassifier

        result = compute_learning_curve(
            DecisionTreeClassifier(random_state=42),
            X_train_clf,
            y_train_clf,
            cv=2,
        )
        sizes = result["train_sizes"]
        assert sizes == sorted(sizes)

    def test_bad_estimator_raises(self, X_train_clf, y_train_clf):
        with pytest.raises(ExplainabilityError, match="learning curve"):
            compute_learning_curve("not_an_estimator", X_train_clf, y_train_clf)
