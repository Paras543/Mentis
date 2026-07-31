"""
Training orchestrator for Mentis's automated model comparison.

`ModelTrainer` fits every model in a given zoo against a train/test
split, times training and inference, runs cross-validation, and
collects results into `ModelResult` objects -- isolating one model's
failure from crashing the entire comparison run.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
from sklearn.model_selection import cross_val_score

from mentis.comparison.leaderboard import Leaderboard, ModelResult, build_leaderboard
from mentis.comparison.metrics import compute_metrics, primary_metric_for_task
from mentis.comparison.model_zoo import get_model_zoo
from mentis.exceptions import ModelError
from mentis.utils.helpers import timer
from mentis.utils.logger import get_logger
from mentis.utils.validators import validate_matching_length, validate_task_type

logger = get_logger(__name__)


class ModelTrainer:
    """
    Trains and evaluates multiple models for automated comparison.

    Examples:
        >>> trainer = ModelTrainer(task="classification")
        >>> leaderboard = trainer.run(X_train, X_test, y_train, y_test)  # doctest: +SKIP
    """

    def __init__(
        self,
        task: str,
        models: list[str] | None = None,
        cv: int = 5,
    ) -> None:
        """
        Args:
            task: "classification" or "regression".
            models: Optional list of model names to restrict comparison
                to. If None, all available models for the task are used.
            cv: Number of cross-validation folds.

        Raises:
            ValidationError: If `task` is not a supported value.
        """
        validate_task_type(task)
        self.task = task
        self.models = models
        self.cv = cv
        self.primary_metric = primary_metric_for_task(task)

    def run(
        self,
        X_train: Any,
        X_test: Any,
        y_train: Any,
        y_test: Any,
    ) -> Leaderboard:
        """
        Train every model in the zoo and return a ranked `Leaderboard`.

        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training targets.
            y_test: Test targets.

        Returns:
            A `Leaderboard` of `ModelResult`s, ranked by the task's
            primary metric.

        Raises:
            ModelError: If every single model in the zoo fails to train
                (indicating a systemic input problem rather than a
                per-model issue).

        Examples:
            >>> trainer = ModelTrainer(task="regression")
            >>> lb = trainer.run(X_train, X_test, y_train, y_test)  # doctest: +SKIP
            >>> lb.best_model().model_name  # doctest: +SKIP
        """
        validate_matching_length(X_train, y_train, names=["X_train", "y_train"])
        validate_matching_length(X_test, y_test, names=["X_test", "y_test"])

        zoo = get_model_zoo(self.task, selected=self.models)
        if not zoo:
            raise ModelError(
                f"No models available for task '{self.task}'. "
                "Check that at least one relevant library is installed."
            )

        results: list[ModelResult] = []
        for name, model in zoo.items():
            logger.info(f"Training model: {name}...")
            result = self._train_and_evaluate(name, model, X_train, X_test, y_train, y_test)
            results.append(result)

        if all(r.error is not None for r in results):
            raise ModelError(
                "All models failed to train. This usually indicates a problem "
                "with the input data (e.g. wrong dtypes, NaNs, or mismatched shapes)."
            )

        return build_leaderboard(self.task, results, self.primary_metric)

    def _train_and_evaluate(
        self,
        name: str,
        model: Any,
        X_train: Any,
        X_test: Any,
        y_train: Any,
        y_test: Any,
    ) -> ModelResult:
        try:
            with timer() as train_timer:
                model.fit(X_train, y_train)

            with timer() as inference_timer:
                y_pred = model.predict(X_test)

            y_proba = None
            if self.task == "classification" and hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                except Exception:  # noqa: BLE001 - proba is best-effort
                    y_proba = None

            metrics = compute_metrics(self.task, np.asarray(y_test), np.asarray(y_pred), y_proba)

            cv_scores = self._safe_cross_validate(model, X_train, y_train)

            feature_importances = self._extract_feature_importances(model, X_train)

            memory_mb = self._model_size_mb(model)

            return ModelResult(
                model_name=name,
                metrics=metrics,
                cv_scores=cv_scores,
                cv_mean=float(np.mean(cv_scores)) if cv_scores else 0.0,
                cv_std=float(np.std(cv_scores)) if cv_scores else 0.0,
                training_time_seconds=train_timer["elapsed_seconds"],
                inference_time_seconds=inference_timer["elapsed_seconds"],
                memory_usage_mb=memory_mb,
                feature_importances=feature_importances,
            )

        except Exception as exc:  # noqa: BLE001 - isolate failures per-model
            logger.warning(f"Model '{name}' failed: {exc}")
            return ModelResult(model_name=name, error=str(exc))

    def _safe_cross_validate(self, model: Any, X_train: Any, y_train: Any) -> list[float]:
        scoring = "f1_weighted" if self.task == "classification" else "r2"
        try:
            scores = cross_val_score(model, X_train, y_train, cv=self.cv, scoring=scoring)
            return [float(s) for s in scores]
        except Exception as exc:  # noqa: BLE001 - CV is best-effort, not fatal
            logger.warning(f"Cross-validation skipped: {exc}")
            return []

    @staticmethod
    def _model_size_mb(model: Any) -> float:
        """Return serialised model size in MB using pickle (accurate for sklearn models)."""
        try:
            return len(pickle.dumps(model)) / 1_048_576
        except Exception:  # noqa: BLE001 - best-effort; don't crash training over this
            return 0.0

    @staticmethod
    def _extract_feature_importances(model: Any, X_train: Any) -> dict[str, float] | None:
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances = np.abs(coef).flatten() if coef.ndim > 1 else np.abs(coef)

        if importances is None:
            return None

        try:
            feature_names = (
                list(X_train.columns)
                if hasattr(X_train, "columns")
                else [f"feature_{i}" for i in range(len(importances))]
            )
            return {str(f): float(v) for f, v in zip(feature_names, importances, strict=False)}
        except Exception:  # noqa: BLE001 - importances are a bonus, not critical
            return None
