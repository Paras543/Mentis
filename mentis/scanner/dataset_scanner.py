"""
Orchestrator for the Mentis Dataset Scanner.

`DatasetScanner` composes individual `BaseCheck` implementations (see
`checks.py`), runs them against a dataframe, and assembles a single
`ScanResult`. This is a Facade over many small, focused checks --
consumers never need to know about individual check classes unless
they want to customize the pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mentis.constants import CATEGORICAL_DTYPES, DATETIME_DTYPES, NUMERIC_DTYPES
from mentis.exceptions import DatasetError
from mentis.scanner.base import BaseCheck
from mentis.scanner.checks import (
    ConstantColumnsCheck,
    DataLeakageCheck,
    DuplicateColumnsCheck,
    DuplicateRowsCheck,
    HighCorrelationCheck,
    IDColumnCheck,
    InfiniteValuesCheck,
    MissingValuesCheck,
    MixedTypesCheck,
    NearZeroVarianceCheck,
    OutlierCheck,
    TargetImbalanceCheck,
)
from mentis.scanner.result import ColumnProfile, Finding, ScanResult
from mentis.utils.logger import get_logger

logger = get_logger(__name__)


DEFAULT_CHECKS: list[type[BaseCheck]] = [
    MissingValuesCheck,
    DuplicateRowsCheck,
    DuplicateColumnsCheck,
    ConstantColumnsCheck,
    NearZeroVarianceCheck,
    HighCorrelationCheck,
    OutlierCheck,
    InfiniteValuesCheck,
    IDColumnCheck,
    TargetImbalanceCheck,
    DataLeakageCheck,
    MixedTypesCheck,
]


class DatasetScanner:
    """
    Runs a configurable battery of checks against a pandas DataFrame and
    produces a structured `ScanResult`.

    Examples:
        >>> import pandas as pd
        >>> from mentis.scanner.dataset_scanner import DatasetScanner
        >>> df = pd.DataFrame({"a": [1, 1, 1, None], "b": [1, 2, 3, 4]})
        >>> scanner = DatasetScanner()
        >>> result = scanner.scan(df)
        >>> result.n_rows
        4

    Args:
        checks: Optional custom list of `BaseCheck` instances to run
            instead of the default battery. Allows full customization
            without touching library internals.
    """

    def __init__(self, checks: list[BaseCheck] | None = None) -> None:
        self.checks: list[BaseCheck] = checks or [check_cls() for check_cls in DEFAULT_CHECKS]

    def scan(
        self,
        df: pd.DataFrame,
        target: str | None = None,
    ) -> ScanResult:
        """
        Scan a dataframe and return a structured `ScanResult`.

        Args:
            df: The dataframe to inspect.
            target: Optional name of the target/label column. Enables
                target-aware checks such as class imbalance and data
                leakage detection.

        Returns:
            A `ScanResult` containing column profiles and findings.

        Raises:
            DatasetError: If `df` is not a DataFrame, is empty, or
                `target` is provided but not found in `df.columns`.

        Examples:
            >>> scanner = DatasetScanner()
            >>> result = scanner.scan(df, target="churn")  # doctest: +SKIP
        """
        self._validate_input(df, target)

        logger.info(f"Scanning dataframe with shape {df.shape}...")

        column_profiles = self._build_column_profiles(df, target)

        context: dict[str, Any] = {"target": target}
        findings: list[Finding] = []

        for check in self.checks:
            try:
                check_findings = check.run(df, **context)
                findings.extend(check_findings)
            except Exception as exc:  # noqa: BLE001 - isolate one bad check from the rest
                logger.warning(f"Check '{check.name}' failed to run: {exc}")

        summary = self._build_summary(findings)

        result = ScanResult(
            n_rows=len(df),
            n_columns=df.shape[1],
            memory_usage_mb=float(df.memory_usage(deep=True).sum() / 1_048_576),
            column_profiles=column_profiles,
            findings=findings,
            summary=summary,
        )

        logger.info(
            f"Scan complete: {len(result.critical_findings())} critical, "
            f"{len(result.warnings())} warnings, {len(result.info_findings())} info."
        )
        return result

    @staticmethod
    def _validate_input(df: pd.DataFrame, target: str | None) -> None:
        if not isinstance(df, pd.DataFrame):
            raise DatasetError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
        if df.empty:
            raise DatasetError("Cannot scan an empty DataFrame.")
        if target is not None and target not in df.columns:
            raise DatasetError(f"Target column '{target}' not found in dataframe columns.")

    @staticmethod
    def _infer_role(df: pd.DataFrame, col: str, target: str | None) -> str:
        if target is not None and col == target:
            return "target"

        dtype_str = str(df[col].dtype)
        if dtype_str in DATETIME_DTYPES or "datetime" in dtype_str:
            return "datetime"
        if dtype_str in NUMERIC_DTYPES:
            return "numerical"
        if dtype_str in CATEGORICAL_DTYPES:
            return "categorical"
        return "unknown"

    def _build_column_profiles(self, df: pd.DataFrame, target: str | None) -> list[ColumnProfile]:
        n_rows = len(df)
        profiles: list[ColumnProfile] = []

        for col in df.columns:
            series = df[col]
            missing_count = int(series.isnull().sum())
            unique_count = int(series.nunique(dropna=True))

            profiles.append(
                ColumnProfile(
                    name=str(col),
                    dtype=str(series.dtype),
                    role=self._infer_role(df, col, target),
                    missing_count=missing_count,
                    missing_pct=missing_count / n_rows if n_rows else 0.0,
                    unique_count=unique_count,
                    unique_pct=unique_count / n_rows if n_rows else 0.0,
                    is_constant=unique_count <= 1,
                    memory_usage_bytes=int(series.memory_usage(deep=True)),
                )
            )
        return profiles

    @staticmethod
    def _build_summary(findings: list[Finding]) -> dict[str, Any]:
        severity_counts = {"critical": 0, "warning": 0, "info": 0}
        for f in findings:
            if f.severity in severity_counts:
                severity_counts[f.severity] += 1

        checks_triggered = sorted({f.check_name for f in findings})

        return {
            "total_findings": len(findings),
            "severity_counts": severity_counts,
            "checks_triggered": checks_triggered,
        }
    

    