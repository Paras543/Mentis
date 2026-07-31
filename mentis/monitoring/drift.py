"""
Data Drift detection for Mentis, built on top of Evidently.

Wraps Evidently's drift detection for feature, prediction, and target
drift, returning plain dicts so callers (Guardian, reporting) don't
need to depend on Evidently's internal report objects directly.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from mentis.exceptions import ValidationError
from mentis.utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """
    Detects feature, prediction, and target drift between a reference
    dataset and a current (production) dataset using Evidently.

    Examples:
        >>> detector = DriftDetector()
        >>> report = detector.detect(reference_df, current_df)  # doctest: +SKIP
    """

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        target: str | None = None,
        prediction: str | None = None,
    ) -> dict[str, Any]:
        """
        Run drift detection between reference and current datasets.

        Args:
            reference: Baseline dataset (e.g. training data).
            current: New/production dataset to compare against.
            target: Optional target column name, enables target drift.
            prediction: Optional prediction column name, enables
                prediction drift.

        Returns:
            Dict with keys "dataset_drift" (bool), "drift_share"
            (float), "feature_drift" (dict of column -> drift info),
            and optionally "target_drift" / "prediction_drift".

        Raises:
            ValidationError: If `reference` or `current` is empty, or
                `target`/`prediction` is specified but missing from
                either dataframe.

        Examples:
            >>> detector = DriftDetector()
            >>> report = detector.detect(reference_df, current_df)  # doctest: +SKIP
            >>> report["dataset_drift"]  # doctest: +SKIP
        """
        if reference.empty or current.empty:
            raise ValidationError("Both 'reference' and 'current' datasets must be non-empty.")

        for col, label in [(target, "target"), (prediction, "prediction")]:
            if col is not None and (col not in reference.columns or col not in current.columns):
                raise ValidationError(
                    f"{label} column '{col}' missing from reference or current data."
                )

        try:
            try:
                from evidently.legacy.metric_preset import DataDriftPreset
                from evidently.legacy.report import Report
            except (ImportError, AttributeError):
                from evidently import Report  # type: ignore
                from evidently.presets import DataDriftPreset  # type: ignore
        except ImportError as exc:
            raise ValidationError(
                "evidently is not installed. Install it with `pip install evidently` "
                "to use drift detection."
            ) from exc

        column_mapping = None
        if target or prediction:
            try:
                from evidently.legacy.pipeline.column_mapping import ColumnMapping
            except (ImportError, AttributeError):
                try:
                    from evidently import ColumnMapping  # type: ignore
                except (ImportError, AttributeError):
                    ColumnMapping = None  # type: ignore

            if ColumnMapping is not None:
                column_mapping = ColumnMapping(target=target, prediction=prediction)

        report = Report(metrics=[DataDriftPreset()])
        _report_any: Any = report
        if column_mapping is not None:
            res_run = _report_any.run(
                reference_data=reference, current_data=current, column_mapping=column_mapping
            )
        else:
            res_run = _report_any.run(reference_data=reference, current_data=current)

        if hasattr(report, "as_dict"):
            result_dict = report.as_dict()
        elif res_run and hasattr(res_run, "dict"):
            result_dict = res_run.dict()
        else:
            result_dict = {}

        return self._parse_evidently_result(result_dict, target, prediction)

    @staticmethod
    def _parse_evidently_result(
        result_dict: dict[str, Any],
        target: str | None,
        prediction: str | None,
    ) -> dict[str, Any]:
        parsed: dict[str, Any] = {
            "dataset_drift": False,
            "drift_share": 0.0,
            "feature_drift": {},
        }

        try:
            metrics = result_dict.get("metrics", [])
            for metric in metrics:
                result = metric.get("result", {})
                if "dataset_drift" in result:
                    parsed["dataset_drift"] = bool(result.get("dataset_drift", False))
                if "share_of_drifted_columns" in result:
                    parsed["drift_share"] = float(result.get("share_of_drifted_columns", 0.0))
                elif "drift_share" in result and not parsed["drift_share"]:
                    parsed["drift_share"] = float(result.get("drift_share", 0.0))

                if "drift_by_columns" in result:
                    for col, col_info in result["drift_by_columns"].items():
                        entry = {
                            "drift_detected": bool(col_info.get("drift_detected", False)),
                            "drift_score": float(col_info.get("drift_score", 0.0)),
                            "stattest_name": col_info.get("stattest_name"),
                        }
                        if target and col == target:
                            parsed["target_drift"] = entry
                        elif prediction and col == prediction:
                            parsed["prediction_drift"] = entry
                        else:
                            parsed["feature_drift"][col] = entry

            if any(info.get("drift_detected", False) for info in parsed["feature_drift"].values()):
                parsed["dataset_drift"] = True
            elif "target_drift" in parsed and parsed["target_drift"].get("drift_detected"):
                parsed["dataset_drift"] = True
            elif "prediction_drift" in parsed and parsed["prediction_drift"].get("drift_detected"):
                parsed["dataset_drift"] = True
        except Exception as exc:  # noqa: BLE001 - best-effort parsing of Evidently's internal shape
            logger.warning(f"Could not fully parse Evidently drift report: {exc}")

        return parsed
