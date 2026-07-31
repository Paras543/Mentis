"""
Monitoring dashboard data for Mentis.

Computes model degradation signals, prediction/feature distribution
summaries, and data quality snapshots over time -- returned as plain
dicts so the reporting module (or any external dashboard) can render
them without depending on this module's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mentis.exceptions import ValidationError
from mentis.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MonitoringSnapshot:
    """
    A single monitoring snapshot summarizing model and data health.

    Attributes:
        performance_metrics: Current metric values (e.g. accuracy, r2).
        performance_delta: Difference vs. a baseline, per metric, if a
            baseline was provided.
        degraded: Whether any metric dropped below its configured
            tolerance relative to baseline.
        prediction_distribution: Summary stats of predictions
            (mean, std, min, max, quantiles).
        feature_distribution: Per-feature summary stats.
        data_quality: Dict with "missing_pct", "duplicate_pct",
            "row_count".
    """

    performance_metrics: dict[str, float] = field(default_factory=dict)
    performance_delta: dict[str, float] = field(default_factory=dict)
    degraded: bool = False
    prediction_distribution: dict[str, float] = field(default_factory=dict)
    feature_distribution: dict[str, dict[str, float]] = field(default_factory=dict)
    data_quality: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "performance_metrics": self.performance_metrics,
            "performance_delta": self.performance_delta,
            "degraded": self.degraded,
            "prediction_distribution": self.prediction_distribution,
            "feature_distribution": self.feature_distribution,
            "data_quality": self.data_quality,
        }

    def __repr__(self) -> str:
        return f"<MonitoringSnapshot degraded={self.degraded} metrics={self.performance_metrics}>"


class ModelMonitor:
    """
    Builds monitoring snapshots for a deployed model: performance
    degradation vs. a baseline, prediction/feature distributions, and
    data quality signals.

    Examples:
        >>> monitor = ModelMonitor()
        >>> snapshot = monitor.snapshot(current_df, predictions)  # doctest: +SKIP
    """

    def snapshot(
        self,
        current_data: pd.DataFrame,
        predictions: Any,
        current_metrics: dict[str, float] | None = None,
        baseline_metrics: dict[str, float] | None = None,
        degradation_tolerance: float = 0.05,
    ) -> MonitoringSnapshot:
        """
        Build a monitoring snapshot from current production data.

        Args:
            current_data: Current feature data (e.g. today's inference
                batch).
            predictions: Model predictions on `current_data`.
            current_metrics: Optional dict of currently-observed
                performance metrics (e.g. from labeled feedback data).
            baseline_metrics: Optional dict of baseline/training-time
                metrics to compare against.
            degradation_tolerance: Fractional drop (relative to
                baseline) above which a metric is flagged as degraded.

        Returns:
            A `MonitoringSnapshot` summarizing model and data health.

        Raises:
            ValidationError: If `current_data` is empty.

        Examples:
            >>> monitor = ModelMonitor()
            >>> snapshot = monitor.snapshot(
            ...     df, preds, {"accuracy": 0.80}, {"accuracy": 0.90}
            ... )  # doctest: +SKIP
            >>> snapshot.degraded  # doctest: +SKIP
            True
        """
        if current_data.empty:
            raise ValidationError("'current_data' must not be empty.")

        performance_delta: dict[str, float] = {}
        degraded = False

        if current_metrics and baseline_metrics:
            for name, baseline_value in baseline_metrics.items():
                current_value = current_metrics.get(name)
                if current_value is None or baseline_value == 0:
                    continue
                delta = (current_value - baseline_value) / abs(baseline_value)
                performance_delta[name] = round(delta, 4)
                if delta < -degradation_tolerance:
                    degraded = True

        prediction_distribution = self._summarize_array(predictions)
        feature_distribution = self._summarize_features(current_data)
        data_quality = self._data_quality(current_data)

        return MonitoringSnapshot(
            performance_metrics=current_metrics or {},
            performance_delta=performance_delta,
            degraded=degraded,
            prediction_distribution=prediction_distribution,
            feature_distribution=feature_distribution,
            data_quality=data_quality,
        )

    @staticmethod
    def _summarize_array(values: Any) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return {}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
        }

    @classmethod
    def _summarize_features(cls, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            summary[str(col)] = cls._summarize_array(df[col].dropna().to_numpy())
        return summary

    @staticmethod
    def _data_quality(df: pd.DataFrame) -> dict[str, float]:
        n_rows = len(df)
        missing_pct = float(df.isnull().mean().mean() * 100) if n_rows else 0.0
        duplicate_pct = float(df.duplicated().mean() * 100) if n_rows else 0.0
        return {
            "row_count": float(n_rows),
            "missing_pct": round(missing_pct, 2),
            "duplicate_pct": round(duplicate_pct, 2),
        }
