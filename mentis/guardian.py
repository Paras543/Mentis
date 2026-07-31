"""
Public-facing entry point for Mentis.

`Guardian` is a Facade over the library's independent subsystems
(scanner, comparison, explainability, reporting, deployment,
monitoring). It exists purely to give users one clean, discoverable
API surface -- all real logic lives in the dedicated subpackages, kept
fully decoupled from this class.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from mentis.comparison.leaderboard import Leaderboard
from mentis.comparison.trainer import ModelTrainer
from mentis.config import MentisConfig
from mentis.deployment.checker import DeploymentChecker, DeploymentResult
from mentis.explainability.curves import (
    compute_calibration_curve,
    compute_confusion_matrix,
    compute_pr_curve,
    compute_residuals,
    compute_roc_curve,
)
from mentis.explainability.permutation import compute_permutation_importance
from mentis.explainability.shap_explainer import compute_shap_values
from mentis.monitoring.dashboard import ModelMonitor, MonitoringSnapshot
from mentis.monitoring.drift import DriftDetector
from mentis.reporting.report_builder import ReportBuilder
from mentis.scanner.dataset_scanner import DatasetScanner
from mentis.scanner.result import ScanResult
from mentis.utils.logger import get_logger
from mentis.validation.auditor import AuditResult, PipelineAuditor
from mentis.validation.fairness import BiasDetector, FairnessResult

logger = get_logger(__name__)


class Guardian:
    """
    Main entry point for the Mentis library.

    `Guardian` wires together Mentis's subsystems behind a single,
    consistent API, so users write:

        >>> from mentis import Guardian
        >>> guardian = Guardian()
        >>> result = guardian.scan(df)  # doctest: +SKIP

    instead of importing and orchestrating half a dozen internal
    classes themselves.

    Args:
        config: Optional `MentisConfig`. If omitted, sensible defaults
            are used for every subsystem.

    Attributes:
        config: The active `MentisConfig` for this Guardian instance.
        last_scan_result: The most recent `ScanResult`, populated after
            calling `.scan()`. Useful for `.generate_report()` later on.
    """

    def __init__(self, config: MentisConfig | None = None) -> None:
        self.config: MentisConfig = config or MentisConfig()
        self._scanner = DatasetScanner()

        # Populated by their respective methods as subsystems are called.
        self.last_scan_result: ScanResult | None = None
        self.last_comparison_result: Any | None = None
        self.last_explain_result: dict[str, Any] | None = None
        self.last_audit_result: Any | None = None
        self.last_deployment_result: Any | None = None
        self.last_bias_result: Any | None = None
        self.last_drift_result: dict[str, Any] | None = None
        self.last_monitoring_result: Any | None = None

    @classmethod
    def from_yaml(cls, path: str) -> Guardian:
        """
        Construct a `Guardian` from a YAML configuration file.

        Args:
            path: Path to a Mentis YAML config file.

        Returns:
            A configured `Guardian` instance.

        Examples:
            >>> guardian = Guardian.from_yaml("mentis.yaml")  # doctest: +SKIP
        """
        return cls(config=MentisConfig.from_yaml(path))

    def scan(self, df: pd.DataFrame, target: str | None = None) -> ScanResult:
        """
        Run the Dataset Scanner against a dataframe.

        Args:
            df: The dataframe to inspect.
            target: Optional target/label column name. If omitted,
                falls back to `self.config.project.target`.

        Returns:
            A `ScanResult` describing dataset health, per-column
            profiles, and any issues found.

        Raises:
            DatasetError: If `df` is invalid or empty, or `target` is
                specified but not present in `df`.

        Examples:
            >>> from mentis import Guardian
            >>> guardian = Guardian()
            >>> result = guardian.scan(df)  # doctest: +SKIP
            >>> result.summary["total_findings"]  # doctest: +SKIP
        """
        resolved_target = target or self.config.project.target
        result = self._scanner.scan(df, target=resolved_target)
        self.last_scan_result = result
        return result

    def compare_models(
        self,
        X_train: Any,
        X_test: Any,
        y_train: Any,
        y_test: Any,
        task: str | None = None,
        models: list[str] | None = None,
        cv: int | None = None,
        **kwargs: Any,
    ) -> Leaderboard:
        """
        Train and compare multiple ML models on the given split.

        Args:
            X_train: Training feature matrix.
            X_test:  Test feature matrix.
            y_train: Training target vector.
            y_test:  Test target vector.
            task: "classification" or "regression". Falls back to
                `self.config.project.task` if omitted.
            models: Optional list of model names to restrict the
                comparison to (e.g. ["Random Forest", "XGBoost"]).
                Falls back to `self.config.comparison.models`.
            cv: Number of cross-validation folds. Falls back to
                `self.config.comparison.cv`.
            **kwargs: Reserved for future extensions.

        Returns:
            A `Leaderboard` of `ModelResult`s ranked by the task's
            primary metric (F1 for classification, R² for regression).

        Raises:
            ValidationError: If `task` is unsupported or array lengths
                mismatch.
            ModelError: If every model in the zoo fails to train.

        Examples:
            >>> guardian = Guardian()
            >>> lb = guardian.compare_models(X_train, X_test, y_train, y_test)  # doctest: +SKIP
            >>> lb.best_model().model_name  # doctest: +SKIP
        """
        resolved_task = task or self.config.project.task
        resolved_models = models if models is not None else self.config.comparison.models
        resolved_cv = cv if cv is not None else self.config.comparison.cv

        trainer = ModelTrainer(task=resolved_task, models=resolved_models, cv=resolved_cv)
        leaderboard = trainer.run(X_train, X_test, y_train, y_test)

        self.last_comparison_result = leaderboard
        return leaderboard

    def explain(
        self,
        model: Any,
        X: Any,
        y: Any | None = None,
        task: str | None = None,
        y_pred: Any | None = None,
        y_proba: Any | None = None,
        include_shap: bool = True,
        include_permutation: bool = True,
        include_curves: bool = True,
    ) -> dict[str, Any]:
        """
        Generate explainability artifacts for a fitted model: SHAP
        values, permutation importance, and evaluation curves.

        Args:
            model: A fitted scikit-learn-compatible estimator.
            X: Feature matrix to explain (e.g. X_test).
            y: True target values, required for permutation importance
                and most curves. Optional if only SHAP is requested.
            task: "classification" or "regression". Falls back to
                `self.config.project.task` if omitted.
            y_pred: Precomputed predictions on `X`. If omitted and `y`
                is provided, `model.predict(X)` is called internally.
            y_proba: Precomputed positive-class probabilities on `X`
                (classification only), used for ROC/PR/calibration
                curves. If omitted, `model.predict_proba(X)` is used
                when available.
            include_shap: Whether to compute SHAP values.
            include_permutation: Whether to compute permutation
                importance. Requires `y`.
            include_curves: Whether to compute evaluation curves
                (confusion matrix, ROC/PR/calibration for
                classification; residuals for regression). Requires `y`.

        Returns:
            Dict with keys among "shap", "permutation_importance",
            "confusion_matrix", "roc_curve", "pr_curve",
            "calibration_curve", "residuals" -- only the ones
            successfully computed are included. Individual failures are
            logged as warnings and skipped rather than aborting the
            whole call.

        Raises:
            ExplainabilityError: If every requested artifact fails to
                compute.

        Examples:
            >>> guardian = Guardian()
            >>> report = guardian.explain(model, X_test, y_test)  # doctest: +SKIP
            >>> report["shap"]["explainer_type"]  # doctest: +SKIP
        """
        resolved_task = task or self.config.project.task
        report: dict[str, Any] = {}

        if include_shap:
            try:
                report["shap"] = compute_shap_values(model, X)
            except Exception as exc:  # noqa: BLE001 - one artifact failing shouldn't block others
                logger.warning(f"SHAP computation skipped: {exc}")

        if y is not None:
            resolved_pred = y_pred if y_pred is not None else model.predict(X)

            if include_permutation:
                try:
                    report["permutation_importance"] = compute_permutation_importance(model, X, y)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Permutation importance skipped: {exc}")

            if include_curves:
                if resolved_task == "classification":
                    try:
                        report["confusion_matrix"] = compute_confusion_matrix(y, resolved_pred)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"Confusion matrix skipped: {exc}")

                    resolved_proba = y_proba
                    if resolved_proba is None and hasattr(model, "predict_proba"):
                        try:
                            proba_matrix = model.predict_proba(X)
                            resolved_proba = (
                                proba_matrix[:, 1] if proba_matrix.ndim == 2 else proba_matrix
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(f"predict_proba unavailable: {exc}")

                    if resolved_proba is not None:
                        try:
                            report["roc_curve"] = compute_roc_curve(y, resolved_proba)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(f"ROC curve skipped: {exc}")
                        try:
                            report["pr_curve"] = compute_pr_curve(y, resolved_proba)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(f"PR curve skipped: {exc}")
                        try:
                            report["calibration_curve"] = compute_calibration_curve(
                                y, resolved_proba
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(f"Calibration curve skipped: {exc}")
                else:
                    try:
                        report["residuals"] = compute_residuals(y, resolved_pred)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"Residuals computation skipped: {exc}")
        elif include_permutation or include_curves:
            logger.warning(
                "include_permutation/include_curves requested but 'y' was not provided; skipping."
            )

        if not report:
            from mentis.exceptions import ExplainabilityError

            raise ExplainabilityError(
                "All explainability computations failed or were skipped. "
                "Provide 'y' for permutation importance/curves, or check that "
                "shap is installed for SHAP values."
            )

        self.last_explain_result = report
        return report

    def audit_pipeline(self, project_path: str = ".") -> AuditResult:
        """
        Audit an ML project's structure and production readiness.

        Args:
            project_path: Root directory of the project to audit.

        Returns:
            An `AuditResult` with per-check findings and a Production
            Readiness Score (0-100).

        Examples:
            >>> guardian = Guardian()
            >>> result = guardian.audit_pipeline(".")  # doctest: +SKIP
            >>> result.score  # doctest: +SKIP
        """
        auditor = PipelineAuditor()
        result = auditor.audit(project_path)
        self.last_audit_result = result
        return result

    def deploy_check(self, project_path: str = ".") -> DeploymentResult:
        """
        Check a project's deployment readiness.

        Args:
            project_path: Root directory of the project to check.

        Returns:
            A `DeploymentResult` with per-check findings, detected web
            framework, and a Deployment Score (0-100).

        Examples:
            >>> guardian = Guardian()
            >>> result = guardian.deploy_check(".")  # doctest: +SKIP
            >>> result.score  # doctest: +SKIP
        """
        checker = DeploymentChecker()
        result = checker.check(project_path)
        self.last_deployment_result = result
        return result

    def check_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        target: str | None = None,
        prediction: str | None = None,
    ) -> dict[str, Any]:
        """
        Detect data drift between a reference dataset and current data.

        Args:
            reference: Baseline dataset (e.g. training data).
            current: New/production dataset to compare against.
            target: Optional target column name, enables target drift.
            prediction: Optional prediction column name, enables
                prediction drift.

        Returns:
            Dict with "dataset_drift", "drift_share", "feature_drift",
            and optionally "target_drift" / "prediction_drift".

        Raises:
            ValidationError: If inputs are empty or malformed, or if
                evidently is not installed.

        Examples:
            >>> guardian = Guardian()
            >>> report = guardian.check_drift(reference_df, current_df)  # doctest: +SKIP
            >>> report["dataset_drift"]  # doctest: +SKIP
        """
        detector = DriftDetector()
        result = detector.detect(reference, current, target=target, prediction=prediction)
        self.last_drift_result = result
        return result

    def check_bias(
        self,
        y_true: Any,
        y_pred: Any,
        sensitive_features: dict[str, Any],
    ) -> list[FairnessResult]:
        """
        Audit a classifier's predictions for bias across sensitive
        features (e.g. gender, race, age group).

        Args:
            y_true: Ground-truth binary labels.
            y_pred: Predicted binary labels.
            sensitive_features: Dict mapping feature name -> array-like
                of group values.

        Returns:
            List of `FairnessResult`, one per sensitive feature, with
            demographic parity, equal opportunity, and equalized odds
            differences.

        Raises:
            ValidationError: If inputs are malformed or fairlearn is
                not installed.

        Examples:
            >>> sens = {"gender": df["gender"]}  # doctest: +SKIP
            >>> results = guardian.check_bias(y_test, y_pred, sens)  # doctest: +SKIP
            >>> results[0].demographic_parity_difference  # doctest: +SKIP
        """
        detector = BiasDetector()
        results = detector.detect(y_true, y_pred, sensitive_features)
        self.last_bias_result = results
        return results

    def monitor(
        self,
        current_data: pd.DataFrame,
        predictions: Any,
        current_metrics: dict[str, float] | None = None,
        baseline_metrics: dict[str, float] | None = None,
        degradation_tolerance: float = 0.05,
    ) -> MonitoringSnapshot:
        """
        Build a monitoring snapshot: performance degradation vs.
        baseline, prediction/feature distributions, and data quality.

        Args:
            current_data: Current feature data (e.g. today's inference
                batch).
            predictions: Model predictions on `current_data`.
            current_metrics: Optional currently-observed performance
                metrics.
            baseline_metrics: Optional baseline/training-time metrics
                to compare against.
            degradation_tolerance: Fractional drop above which a
                metric is flagged as degraded.

        Returns:
            A `MonitoringSnapshot` summarizing model and data health.

        Raises:
            ValidationError: If `current_data` is empty.

        Examples:
            >>> guardian = Guardian()
            >>> snapshot = guardian.monitor(current_df, preds)  # doctest: +SKIP
            >>> snapshot.degraded  # doctest: +SKIP
        """
        monitor = ModelMonitor()
        snapshot = monitor.snapshot(
            current_data,
            predictions,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            degradation_tolerance=degradation_tolerance,
        )
        self.last_monitoring_result = snapshot
        return snapshot

    def generate_report(
        self,
        output_path: str | None = None,
        fmt: str | None = None,
    ) -> str:
        """
        Generate a professional report from the most recent scan/
        comparison/explain/audit/deployment/bias/drift/monitoring
        results.

        Args:
            output_path: Directory to write the report into. Falls
                back to `self.config.report.output_dir`.
            fmt: "html", "markdown", or "pdf". Falls back to
                `self.config.report.format`.

        Returns:
            Path to the generated report file.

        Raises:
            ReportGenerationError: If `fmt` is unsupported, no results
                are available to report on, or rendering fails.

        Examples:
            >>> guardian = Guardian()
            >>> guardian.scan(df)  # doctest: +SKIP
            >>> path = guardian.generate_report()  # doctest: +SKIP
        """
        resolved_output_dir = output_path or self.config.report.output_dir
        resolved_fmt = fmt or self.config.report.format

        builder = ReportBuilder()
        return builder.build(self, output_dir=resolved_output_dir, fmt=resolved_fmt)

    def __repr__(self) -> str:
        return f"<Guardian task={self.config.project.task!r} target={self.config.project.target!r}>"
