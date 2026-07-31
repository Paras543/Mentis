"""
Concrete implementations of individual dataset checks.

Each class here does exactly one job and returns `Finding` objects.
This keeps checks independently testable, independently toggleable
via config, and easy to reason about in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mentis.constants import (
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_ID_UNIQUENESS_THRESHOLD,
    DEFAULT_IMBALANCE_THRESHOLD,
    DEFAULT_MISSING_THRESHOLD,
    DEFAULT_NEAR_ZERO_VARIANCE_FREQ_RATIO,
    DEFAULT_NEAR_ZERO_VARIANCE_UNIQUE_PCT,
    DEFAULT_OUTLIER_ZSCORE_THRESHOLD,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
)
from mentis.scanner.base import BaseCheck
from mentis.scanner.result import Finding


class MissingValuesCheck(BaseCheck):
    """Flags columns with missing values above a configurable threshold."""

    name = "missing_values"

    def __init__(self, threshold: float = DEFAULT_MISSING_THRESHOLD) -> None:
        self.threshold = threshold

    def run(self, df: pd.DataFrame, **context: object) -> list[Finding]:
        findings: list[Finding] = []
        missing_pct = df.isnull().mean()

        for col, pct in missing_pct.items():
            if pct == 0:
                continue
            if pct >= self.threshold:
                severity = SEVERITY_CRITICAL
                suggestion = "Consider dropping this column or applying an imputation strategy."
            elif pct >= 0.10:
                severity = SEVERITY_WARNING
                suggestion = "Consider imputing missing values — missing rate is notable."
            else:
                severity = SEVERITY_INFO
                suggestion = "Consider imputing missing values before training."

            findings.append(
                Finding(
                    check_name=self.name,
                    severity=severity,
                    message=f"Column '{col}' has {pct:.1%} missing values.",
                    columns=[str(col)],
                    details={"missing_pct": float(pct)},
                    suggestion=suggestion,
                )
            )
        return findings


class DuplicateRowsCheck(BaseCheck):
    """Flags fully duplicated rows in the dataset."""

    name = "duplicate_rows"

    def run(self, df: pd.DataFrame, **context: object) -> list[Finding]:
        n_dupes = int(df.duplicated().sum())
        if n_dupes == 0:
            return []

        pct = float(n_dupes / len(df)) if len(df) else 0.0
        severity = SEVERITY_WARNING if pct > 0.05 else SEVERITY_INFO
        return [
            Finding(
                check_name=self.name,
                severity=severity,
                message=f"Found {n_dupes} duplicate rows ({pct:.1%} of dataset).",
                details={"duplicate_count": n_dupes, "duplicate_pct": pct},
                suggestion="Consider removing duplicate rows with df.drop_duplicates().",
            )
        ]


class DuplicateColumnsCheck(BaseCheck):
    """Flags columns that are exact duplicates of one another."""

    name = "duplicate_columns"

    def run(self, df: pd.DataFrame, **context: object) -> list[Finding]:
        findings: list[Finding] = []
        seen: dict[tuple, str] = {}

        for col in df.columns:
            try:
                fingerprint = tuple(pd.util.hash_pandas_object(df[col], index=False))
            except TypeError:
                continue

            if fingerprint in seen:
                original = seen[fingerprint]
                findings.append(
                    Finding(
                        check_name=self.name,
                        severity=SEVERITY_WARNING,
                        message=f"Column '{col}' is a duplicate of '{original}'.",
                        columns=[original, str(col)],
                        suggestion=f"Consider dropping one of '{original}' or '{col}'.",
                    )
                )
            else:
                seen[fingerprint] = str(col)

        return findings


class ConstantColumnsCheck(BaseCheck):
    """Flags columns with a single unique value (zero variance / no signal)."""

    name = "constant_columns"

    def run(self, df: pd.DataFrame, **context: object) -> list[Finding]:
        findings: list[Finding] = []
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            if nunique <= 1:
                findings.append(
                    Finding(
                        check_name=self.name,
                        severity=SEVERITY_WARNING,
                        message=f"Column '{col}' is constant (only {nunique} unique value).",
                        columns=[str(col)],
                        suggestion=f"Drop '{col}' -- it carries no predictive signal.",
                    )
                )
        return findings


class NearZeroVarianceCheck(BaseCheck):
    """
    Flags numerical columns with near-zero variance: a small number of
    unique values with one value dominating the frequency distribution.
    Mirrors the heuristic used by R's caret::nearZeroVar.
    """

    name = "near_zero_variance"

    def __init__(
        self,
        freq_ratio_threshold: float = DEFAULT_NEAR_ZERO_VARIANCE_FREQ_RATIO,
        unique_pct_threshold: float = DEFAULT_NEAR_ZERO_VARIANCE_UNIQUE_PCT,
    ) -> None:
        self.freq_ratio_threshold = freq_ratio_threshold
        self.unique_pct_threshold = unique_pct_threshold

    def run(self, df: pd.DataFrame, **context: object) -> list[Finding]:
        findings: list[Finding] = []
        n_rows = len(df)
        if n_rows == 0:
            return findings

        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            value_counts = series.value_counts()
            if len(value_counts) < 2:
                continue  # already caught by ConstantColumnsCheck

            freq_ratio = float(value_counts.iloc[0] / value_counts.iloc[1])
            unique_pct = float(series.nunique() / n_rows)

            if freq_ratio > self.freq_ratio_threshold and unique_pct < self.unique_pct_threshold:
                findings.append(
                    Finding(
                        check_name=self.name,
                        severity=SEVERITY_WARNING,
                        message=(
                            f"Column '{col}' shows near-zero variance "
                            f"(freq_ratio={freq_ratio:.1f}, unique_pct={unique_pct:.1%})."
                        ),
                        columns=[str(col)],
                        details={"freq_ratio": freq_ratio, "unique_pct": unique_pct},
                        suggestion="This column may add noise rather than signal to a model.",
                    )
                )
        return findings


class HighCorrelationCheck(BaseCheck):
    """Flags pairs of numerical features with high absolute correlation."""

    name = "high_correlation"

    def __init__(self, threshold: float = DEFAULT_CORRELATION_THRESHOLD) -> None:
        self.threshold = threshold

    def run(self, df: pd.DataFrame, **context: object) -> list[Finding]:
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
            return []

        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        findings: list[Finding] = []
        for col in upper.columns:
            correlated = upper.index[upper[col] > self.threshold].tolist()
            for other in correlated:
                findings.append(
                    Finding(
                        check_name=self.name,
                        severity=SEVERITY_WARNING,
                        message=(
                            f"Columns '{other}' and '{col}' are highly correlated "
                            f"(|r|={upper.loc[other, col]:.2f})."
                        ),
                        columns=[str(other), str(col)],
                        details={"correlation": float(upper.loc[other, col])},
                        suggestion=(
                            f"Consider dropping one of '{other}' or '{col}'"
                            " to reduce multicollinearity."
                        ),
                    )
                )
        return findings


class OutlierCheck(BaseCheck):
    """Flags numerical columns containing statistical outliers via z-score."""

    name = "outliers"

    def __init__(self, z_threshold: float = DEFAULT_OUTLIER_ZSCORE_THRESHOLD) -> None:
        self.z_threshold = z_threshold

    def run(self, df: pd.DataFrame, **context: object) -> list[Finding]:
        findings: list[Finding] = []
        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            series = df[col].dropna()
            std = series.std()
            if std == 0 or pd.isna(std):
                continue

            z_scores = (series - series.mean()).abs() / std
            n_outliers = int((z_scores > self.z_threshold).sum())

            if n_outliers > 0:
                pct = float(n_outliers / len(series))
                findings.append(
                    Finding(
                        check_name=self.name,
                        severity=SEVERITY_INFO,
                        message=(
                            f"Column '{col}' has {n_outliers} potential outliers "
                            f"({pct:.1%}, |z|>{self.z_threshold})."
                        ),
                        columns=[str(col)],
                        details={"outlier_count": n_outliers, "outlier_pct": pct},
                        suggestion=(
                            "Investigate whether these are data errors" " or valid extreme values."
                        ),
                    )
                )
        return findings


class InfiniteValuesCheck(BaseCheck):
    """Flags numerical columns containing infinite values."""

    name = "infinite_values"

    def run(self, df: pd.DataFrame, **context: object) -> list[Finding]:
        findings: list[Finding] = []
        numeric_df = df.select_dtypes(include=np.number)

        for col in numeric_df.columns:
            n_inf = int(np.isinf(numeric_df[col]).sum())
            if n_inf > 0:
                findings.append(
                    Finding(
                        check_name=self.name,
                        severity=SEVERITY_CRITICAL,
                        message=f"Column '{col}' contains {n_inf} infinite value(s).",
                        columns=[str(col)],
                        details={"infinite_count": n_inf},
                        suggestion=(
                            "Replace infinite values with NaN or a bounded value"
                            " before training."
                        ),
                    )
                )
        return findings


class IDColumnCheck(BaseCheck):
    """
    Flags columns that look like identifier columns (near-100% unique
    values), which typically should not be used as model features.
    """

    name = "id_columns"

    def __init__(self, uniqueness_threshold: float = DEFAULT_ID_UNIQUENESS_THRESHOLD) -> None:
        self.uniqueness_threshold = uniqueness_threshold

    def run(self, df: pd.DataFrame, **context: object) -> list[Finding]:
        findings: list[Finding] = []
        n_rows = len(df)
        if n_rows == 0:
            return findings

        for col in df.columns:
            unique_pct = float(df[col].nunique(dropna=True) / n_rows)
            name_hint = str(col).lower() in {"id", "uuid", "index", "row_id"} or str(
                col
            ).lower().endswith("_id")

            if unique_pct >= self.uniqueness_threshold or name_hint:
                findings.append(
                    Finding(
                        check_name=self.name,
                        severity=SEVERITY_INFO,
                        message=(
                            f"Column '{col}' looks like an identifier column"
                            f" ({unique_pct:.1%} unique)."
                        ),
                        columns=[str(col)],
                        details={"unique_pct": unique_pct, "name_hint": name_hint},
                        suggestion=f"Exclude '{col}' from model features to avoid leakage/noise.",
                    )
                )
        return findings


class TargetImbalanceCheck(BaseCheck):
    """
    Flags class imbalance in the target column, if one is provided via
    context (context["target"]).
    """

    name = "target_imbalance"

    def __init__(self, threshold: float = DEFAULT_IMBALANCE_THRESHOLD) -> None:
        self.threshold = threshold

    def run(self, df: pd.DataFrame, **context: object) -> list[Finding]:
        target = context.get("target")
        if not target or target not in df.columns:
            return []

        value_counts = df[target].value_counts(normalize=True, dropna=True)
        if value_counts.empty:
            return []

        majority_pct = float(value_counts.iloc[0])
        if majority_pct >= self.threshold:
            return [
                Finding(
                    check_name=self.name,
                    severity=SEVERITY_WARNING,
                    message=(
                        f"Target column '{target}' is imbalanced: majority class "
                        f"represents {majority_pct:.1%} of samples."
                    ),
                    columns=[str(target)],
                    details={
                        "majority_class_pct": majority_pct,
                        "class_distribution": value_counts.to_dict(),
                    },
                    suggestion="Consider resampling, class weighting, or stratified evaluation.",
                )
            ]
        return []


class DataLeakageCheck(BaseCheck):
    """
    Flags features that are suspiciously perfectly (or near-perfectly)
    correlated with the target, which often indicates leakage.
    """

    name = "data_leakage"

    def __init__(self, threshold: float = 0.98) -> None:
        self.threshold = threshold

    def run(self, df: pd.DataFrame, **context: object) -> list[Finding]:
        target = context.get("target")
        if not target or target not in df.columns:
            return []

        if not pd.api.types.is_numeric_dtype(df[target]):
            return []

        findings: list[Finding] = []
        numeric_df = df.select_dtypes(include=np.number)

        for col in numeric_df.columns:
            if col == target:
                continue
            corr = float(numeric_df[col].corr(numeric_df[target]))
            if pd.notna(corr) and abs(corr) >= self.threshold:
                findings.append(
                    Finding(
                        check_name=self.name,
                        severity=SEVERITY_CRITICAL,
                        message=(
                            f"Column '{col}' is suspiciously correlated with target "
                            f"'{target}' (|r|={abs(corr):.3f}) -- possible data leakage."
                        ),
                        columns=[str(col), str(target)],
                        details={"correlation": corr},
                        suggestion=(
                            f"Investigate whether '{col}' would be available" " at prediction time."
                        ),
                    )
                )
        return findings


class MixedTypesCheck(BaseCheck):
    """Flags object columns containing a mix of Python types (e.g. str and int)."""

    name = "mixed_data_types"

    def run(self, df: pd.DataFrame, **context: object) -> list[Finding]:
        findings: list[Finding] = []
        object_cols = df.select_dtypes(include="object").columns

        for col in object_cols:
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            type_set = non_null.map(type).unique()
            if len(type_set) > 1:
                type_names = sorted(t.__name__ for t in type_set)
                findings.append(
                    Finding(
                        check_name=self.name,
                        severity=SEVERITY_WARNING,
                        message=f"Column '{col}' contains mixed types: {type_names}.",
                        columns=[str(col)],
                        details={"types_found": type_names},
                        suggestion=f"Cast '{col}' to a single consistent dtype.",
                    )
                )
        return findings
