"""
Chart generation for Mentis.

Consumes the plain data structures produced by scanner/comparison/
explainability/monitoring modules and renders them as matplotlib
figures, saved to disk. Kept separate from those modules so they never
need a plotting dependency themselves.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mentis.exceptions import ValidationError
from mentis.utils.helpers import ensure_directory
from mentis.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_DPI = 120
_DEFAULT_FIGSIZE = (8, 6)


class ChartGenerator:
    """
    Generates and saves standard ML diagnostic charts as PNG files.

    Examples:
        >>> charts = ChartGenerator(output_dir="mentis_charts")
        >>> path = charts.correlation_heatmap(df)  # doctest: +SKIP
    """

    def __init__(self, output_dir: str = "mentis_charts") -> None:
        """
        Args:
            output_dir: Directory where generated chart PNGs are saved.
                Created automatically if it does not exist.
        """
        self.output_dir = ensure_directory(output_dir)

    def correlation_heatmap(
        self, df: pd.DataFrame, filename: str = "correlation_heatmap.png"
    ) -> str:
        """
        Plot a correlation heatmap for numerical columns.

        Args:
            df: Dataframe to compute correlations from.
            filename: Output filename within `output_dir`.

        Returns:
            Path to the saved PNG.

        Raises:
            ValidationError: If there are fewer than 2 numerical columns.

        Examples:
            >>> charts = ChartGenerator()
            >>> path = charts.correlation_heatmap(df)  # doctest: +SKIP
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            raise ValidationError("Need at least 2 numerical columns for a correlation heatmap.")

        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        fig.colorbar(im, ax=ax)
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        return self._save(fig, filename)

    def missing_value_heatmap(self, df: pd.DataFrame, filename: str = "missing_heatmap.png") -> str:
        """
        Plot a heatmap showing missing-value locations across the
        dataframe.

        Args:
            df: Dataframe to inspect.
            filename: Output filename within `output_dir`.

        Returns:
            Path to the saved PNG.

        Examples:
            >>> charts = ChartGenerator()
            >>> path = charts.missing_value_heatmap(df)  # doctest: +SKIP
        """
        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
        ax.imshow(df.isnull(), cmap="viridis", aspect="auto", interpolation="none")
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")
        ax.set_yticks([])
        ax.set_title("Missing Value Heatmap")
        fig.tight_layout()
        return self._save(fig, filename)

    def distribution(
        self,
        series: pd.Series,
        filename: str = "distribution.png",
        title: str | None = None,
    ) -> str:
        """
        Plot the distribution of a single column (histogram for
        numerical, bar chart of value counts for categorical).

        Args:
            series: Column to visualize.
            filename: Output filename within `output_dir`.
            title: Optional chart title. Defaults to the series name.

        Returns:
            Path to the saved PNG.

        Examples:
            >>> charts = ChartGenerator()
            >>> path = charts.distribution(df["target"])  # doctest: +SKIP
        """
        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
        if pd.api.types.is_numeric_dtype(series):
            ax.hist(series.dropna(), bins=30, color="#4C72B0", edgecolor="white")
        else:
            counts = series.value_counts()
            ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
            ax.tick_params(axis="x", rotation=45)
        ax.set_title(title or f"Distribution of {series.name}")
        fig.tight_layout()
        return self._save(fig, filename)

    def feature_importance(
        self,
        importances: dict[str, float],
        filename: str = "feature_importance.png",
        top_n: int = 20,
    ) -> str:
        """
        Plot a horizontal bar chart of feature importances.

        Args:
            importances: Dict mapping feature name -> importance score.
            filename: Output filename within `output_dir`.
            top_n: Maximum number of features to display, highest first.

        Returns:
            Path to the saved PNG.

        Raises:
            ValidationError: If `importances` is empty.

        Examples:
            >>> charts = ChartGenerator()
            >>> path = charts.feature_importance({"age": 0.4, "income": 0.6})  # doctest: +SKIP
        """
        if not importances:
            raise ValidationError("'importances' must not be empty.")

        sorted_items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        names, values = zip(*sorted_items, strict=False)

        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
        ax.barh(range(len(names)), values, color="#55A868")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        fig.tight_layout()
        return self._save(fig, filename)

    def shap_summary(self, shap_result: dict[str, Any], filename: str = "shap_summary.png") -> str:
        """
        Plot a SHAP-style summary bar chart (mean absolute SHAP value
        per feature) from a `compute_shap_values` result.

        Args:
            shap_result: Output of
                `mentis.explainability.compute_shap_values`.
            filename: Output filename within `output_dir`.

        Returns:
            Path to the saved PNG.

        Examples:
            >>> charts = ChartGenerator()
            >>> path = charts.shap_summary(shap_result)  # doctest: +SKIP
        """
        values = np.asarray(shap_result["shap_values"])
        feature_names = shap_result["feature_names"]
        mean_abs = np.abs(values).mean(axis=0)
        importances = dict(zip(feature_names, mean_abs.tolist(), strict=False))
        return self.feature_importance(importances, filename=filename)

    def roc_curve(self, curve: dict[str, list[float]], filename: str = "roc_curve.png") -> str:
        """
        Plot an ROC curve from a `compute_roc_curve` result.

        Args:
            curve: Output of
                `mentis.explainability.compute_roc_curve`.
            filename: Output filename within `output_dir`.

        Returns:
            Path to the saved PNG.

        Examples:
            >>> charts = ChartGenerator()
            >>> path = charts.roc_curve(curve)  # doctest: +SKIP
        """
        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
        ax.plot(curve["fpr"], curve["tpr"], color="#4C72B0", label="ROC")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        fig.tight_layout()
        return self._save(fig, filename)

    def pr_curve(self, curve: dict[str, list[float]], filename: str = "pr_curve.png") -> str:
        """
        Plot a precision-recall curve from a `compute_pr_curve` result.

        Args:
            curve: Output of `mentis.explainability.compute_pr_curve`.
            filename: Output filename within `output_dir`.

        Returns:
            Path to the saved PNG.

        Examples:
            >>> charts = ChartGenerator()
            >>> path = charts.pr_curve(curve)  # doctest: +SKIP
        """
        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
        ax.plot(curve["recall"], curve["precision"], color="#C44E52")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        fig.tight_layout()
        return self._save(fig, filename)

    def calibration_curve(
        self, curve: dict[str, list[float]], filename: str = "calibration_curve.png"
    ) -> str:
        """
        Plot a calibration curve from a `compute_calibration_curve`
        result.

        Args:
            curve: Output of
                `mentis.explainability.compute_calibration_curve`.
            filename: Output filename within `output_dir`.

        Returns:
            Path to the saved PNG.

        Examples:
            >>> charts = ChartGenerator()
            >>> path = charts.calibration_curve(curve)  # doctest: +SKIP
        """
        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
        ax.plot(
            curve["mean_predicted_value"],
            curve["fraction_of_positives"],
            marker="o",
            color="#4C72B0",
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        fig.tight_layout()
        return self._save(fig, filename)

    def residual_plot(
        self, residuals: dict[str, list[float]], filename: str = "residual_plot.png"
    ) -> str:
        """
        Plot residuals vs. predicted values from a `compute_residuals`
        result.

        Args:
            residuals: Output of
                `mentis.explainability.compute_residuals`.
            filename: Output filename within `output_dir`.

        Returns:
            Path to the saved PNG.

        Examples:
            >>> charts = ChartGenerator()
            >>> path = charts.residual_plot(residuals)  # doctest: +SKIP
        """
        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
        ax.scatter(residuals["predicted"], residuals["residuals"], alpha=0.6, color="#4C72B0")
        ax.axhline(0, linestyle="--", color="gray")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title("Residual Plot")
        fig.tight_layout()
        return self._save(fig, filename)

    def learning_curve(
        self, curve: dict[str, list[float]], filename: str = "learning_curve.png"
    ) -> str:
        """
        Plot a learning curve from a `compute_learning_curve` result.

        Args:
            curve: Output of
                `mentis.explainability.compute_learning_curve`.
            filename: Output filename within `output_dir`.

        Returns:
            Path to the saved PNG.

        Examples:
            >>> charts = ChartGenerator()
            >>> path = charts.learning_curve(curve)  # doctest: +SKIP
        """
        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
        sizes = curve["train_sizes"]
        ax.plot(sizes, curve["train_scores_mean"], label="Training score", color="#4C72B0")
        ax.plot(sizes, curve["val_scores_mean"], label="Validation score", color="#C44E52")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Score")
        ax.set_title("Learning Curve")
        ax.legend()
        fig.tight_layout()
        return self._save(fig, filename)

    def confusion_matrix(self, cm: dict[str, Any], filename: str = "confusion_matrix.png") -> str:
        """
        Plot a confusion matrix from a `compute_confusion_matrix` result.

        Args:
            cm: Output of
                `mentis.explainability.compute_confusion_matrix`.
            filename: Output filename within `output_dir`.

        Returns:
            Path to the saved PNG.

        Examples:
            >>> charts = ChartGenerator()
            >>> path = charts.confusion_matrix(cm)  # doctest: +SKIP
        """
        matrix = np.asarray(cm["matrix"])
        labels = cm["labels"]

        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return self._save(fig, filename)

    def _save(self, fig: plt.Figure, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        try:
            fig.savefig(path, dpi=_DEFAULT_DPI, bbox_inches="tight")
        finally:
            plt.close(fig)
        return path
