"""
Leaderboard construction for Mentis model comparison results.

Separated from the trainer so ranking/formatting logic can evolve
(e.g. weighted scoring, custom tie-breakers) without touching the
training loop itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mentis.comparison.metrics import is_higher_better


@dataclass
class ModelResult:
    """
    Result of training and evaluating a single model.

    Attributes:
        model_name: Name of the model (e.g. "Random Forest").
        metrics: Dict of metric name -> value on the test set.
        cv_scores: Cross-validation scores (primary metric) per fold.
        cv_mean: Mean cross-validation score.
        cv_std: Standard deviation of cross-validation scores.
        training_time_seconds: Wall-clock time spent fitting the model.
        inference_time_seconds: Wall-clock time spent predicting on the
            test set.
        memory_usage_mb: Approximate memory footprint of the fitted
            model.
        feature_importances: Optional dict of feature name -> importance
            score, if the model exposes one.
        error: Populated if training/evaluation failed for this model,
            instead of raising and aborting the whole comparison.
    """

    model_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    training_time_seconds: float = 0.0
    inference_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    feature_importances: dict[str, float] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Leaderboard:
    """
    Ranked collection of `ModelResult`s produced by a model comparison
    run.

    Attributes:
        task: "classification" or "regression".
        primary_metric: The metric used to rank models.
        results: All `ModelResult`s, sorted best-to-worst by
            `primary_metric`.
    """

    task: str
    primary_metric: str
    results: list[ModelResult] = field(default_factory=list)

    def best_model(self) -> ModelResult | None:
        """Return the top-ranked `ModelResult`, or None if empty."""
        successful = [r for r in self.results if r.error is None]
        return successful[0] if successful else None

    def to_dataframe(self):
        """
        Return the leaderboard as a pandas DataFrame, sorted by rank.

        Returns:
            A `pandas.DataFrame` with one row per model.

        Examples:
            >>> lb = Leaderboard(task="classification", primary_metric="f1")
            >>> lb.to_dataframe().empty
            True
        """
        import pandas as pd

        rows = []
        for r in self.results:
            row = {"model": r.model_name, "error": r.error}
            row.update(r.metrics)
            row["cv_mean"] = r.cv_mean
            row["cv_std"] = r.cv_std
            row["training_time_s"] = r.training_time_seconds
            row["inference_time_s"] = r.inference_time_seconds
            row["memory_mb"] = r.memory_usage_mb
            rows.append(row)

        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "primary_metric": self.primary_metric,
            "results": [r.to_dict() for r in self.results],
        }

    def __repr__(self) -> str:
        best = self.best_model()
        best_name = best.model_name if best else "N/A"
        return (
            f"<Leaderboard task={self.task!r} best_model={best_name!r}"
            f" n_models={len(self.results)}>"
        )


def build_leaderboard(
    task: str,
    model_results: list[ModelResult],
    primary_metric: str,
) -> Leaderboard:
    """
    Sort model results into a ranked `Leaderboard`.

    Args:
        task: "classification" or "regression".
        model_results: Unsorted list of `ModelResult`s.
        primary_metric: Metric name used for ranking.

    Returns:
        A `Leaderboard` with `results` sorted best-to-worst. Models
        that errored are placed last, in original order.

    Examples:
        >>> results = [
        ...     ModelResult(model_name="A", metrics={"f1": 0.8}),
        ...     ModelResult(model_name="B", metrics={"f1": 0.9}),
        ... ]
        >>> lb = build_leaderboard("classification", results, "f1")
        >>> lb.results[0].model_name
        'B'
    """
    higher_better = is_higher_better(primary_metric)

    successful = [r for r in model_results if r.error is None]
    failed = [r for r in model_results if r.error is not None]

    successful.sort(
        key=lambda r: r.metrics.get(
            primary_metric, float("-inf") if higher_better else float("inf")
        ),
        reverse=higher_better,
    )

    return Leaderboard(task=task, primary_metric=primary_metric, results=successful + failed)
