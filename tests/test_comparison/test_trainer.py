"""
Tests for mentis/comparison/trainer.py — ModelTrainer orchestrator.
"""

from __future__ import annotations

import pytest

from mentis.comparison.leaderboard import Leaderboard, ModelResult
from mentis.comparison.trainer import ModelTrainer
from mentis.exceptions import ModelError


class TestModelTrainerInit:
    def test_valid_classification_task(self):
        trainer = ModelTrainer(task="classification")
        assert trainer.task == "classification"
        assert trainer.primary_metric == "f1"

    def test_valid_regression_task(self):
        trainer = ModelTrainer(task="regression")
        assert trainer.task == "regression"
        assert trainer.primary_metric == "r2"

    def test_invalid_task_raises(self):
        with pytest.raises(Exception):
            ModelTrainer(task="clustering")


class TestModelTrainerRun:
    def test_returns_leaderboard_clf(self, X_train_clf, X_test_clf, y_train_clf, y_test_clf):
        trainer = ModelTrainer(
            task="classification",
            models=["Logistic Regression", "Decision Tree"],
            cv=2,
        )
        lb = trainer.run(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
        assert isinstance(lb, Leaderboard)

    def test_returns_leaderboard_reg(self, X_train_reg, X_test_reg, y_train_reg, y_test_reg):
        trainer = ModelTrainer(
            task="regression",
            models=["Linear Regression", "Ridge"],
            cv=2,
        )
        lb = trainer.run(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
        assert isinstance(lb, Leaderboard)

    def test_leaderboard_has_results(self, X_train_clf, X_test_clf, y_train_clf, y_test_clf):
        trainer = ModelTrainer(
            task="classification",
            models=["Logistic Regression"],
            cv=2,
        )
        lb = trainer.run(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
        assert len(lb.results) >= 1

    def test_best_model_is_a_model_result(self, X_train_clf, X_test_clf, y_train_clf, y_test_clf):
        trainer = ModelTrainer(
            task="classification",
            models=["Logistic Regression", "Decision Tree"],
            cv=2,
        )
        lb = trainer.run(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
        assert isinstance(lb.best_model(), ModelResult)

    def test_all_results_have_metrics(self, X_train_clf, X_test_clf, y_train_clf, y_test_clf):
        trainer = ModelTrainer(
            task="classification",
            models=["Logistic Regression"],
            cv=2,
        )
        lb = trainer.run(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
        for r in lb.results:
            if r.error is None:
                assert "f1" in r.metrics

    def test_length_mismatch_raises(self, X_train_clf, X_test_clf, y_train_clf, y_test_clf):
        trainer = ModelTrainer(task="classification", models=["Logistic Regression"], cv=2)
        with pytest.raises(Exception):
            trainer.run(X_train_clf, X_test_clf, y_train_clf[:5], y_test_clf)

    def test_all_models_fail_raises_model_error(
        self, X_train_clf, X_test_clf, y_train_clf, y_test_clf
    ):
        """Passing string data to models that require numeric should fail all."""
        import pandas as pd

        X_bad = pd.DataFrame({"a": ["x", "y"] * (len(X_train_clf) // 2)})
        X_test_bad = pd.DataFrame({"a": ["x", "y"] * (len(X_test_clf) // 2)})
        trainer = ModelTrainer(
            task="classification",
            models=["Logistic Regression"],
            cv=2,
        )
        with pytest.raises(ModelError):
            trainer.run(X_bad, X_test_bad, y_train_clf, y_test_clf)


class TestModelSizeMb:
    def test_size_positive(self, fitted_clf):
        size = ModelTrainer._model_size_mb(fitted_clf)
        assert size > 0

    def test_size_plausible_range(self, fitted_clf):
        """A small Random Forest should be measurable in KB to MB, not bytes."""
        size = ModelTrainer._model_size_mb(fitted_clf)
        assert 0.001 < size < 500  # between 1 KB and 500 MB

    def test_size_returns_float(self, fitted_clf):
        size = ModelTrainer._model_size_mb(fitted_clf)
        assert isinstance(size, float)
