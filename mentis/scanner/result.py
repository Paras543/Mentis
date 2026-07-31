from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Finding:
    """
    Result of a single dataset check.

    Attributes:
        check_name: Check identifier (e.g. "missing_values").
        severity: "info", "warning", or "critical".
        message: Human-readable description of the issue.
        columns: Affected column name(s).
        details: Quantitative metadata about the finding.
        suggestion: Actionable guidance to address the finding.
    """

    check_name: str
    severity: str
    message: str
    columns: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str | None = None

    @property
    def column_names(self) -> list[str]:
        return self.columns

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation of this finding."""
        return asdict(self)


@dataclass
class ColumnProfile:
    """
    Per-column profiling summary.

    Attributes:
        name: Column name.
        dtype: Pandas dtype string.
        role: "numerical", "categorical", "datetime", "target", or "unknown".
        missing_count: Count of missing/null values.
        missing_pct: Fraction of missing values relative to row count.
        unique_count: Number of unique non-null values.
        unique_pct: Fraction of unique values relative to row count.
        is_constant: Whether the column carries a single unique value.
        memory_usage_bytes: Memory footprint of this column in bytes.
    """

    name: str
    dtype: str
    role: str
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    is_constant: bool
    memory_usage_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScanResult:
    """
    Full result of a DatasetScanner run.

    Attributes:
        n_rows: Number of rows in the dataset.
        n_columns: Number of columns in the dataset.
        memory_usage_mb: Total memory footprint of the dataframe in MB.
        column_profiles: Per-column profiling information.
        findings: All issues/observations discovered across all checks.
        summary: High-level aggregate statistics (counts by severity, etc.).
    """

    n_rows: int = 0
    n_columns: int = 0
    memory_usage_mb: float = 0.0
    column_profiles: list[ColumnProfile] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cols(self) -> int:
        return self.n_columns

    def critical_findings(self) -> list[Finding]:
        """Return only findings with severity == 'critical'."""
        return [f for f in self.findings if f.severity == "critical"]

    def warnings(self) -> list[Finding]:
        """Return only findings with severity == 'warning'."""
        return [f for f in self.findings if f.severity == "warning"]

    def info_findings(self) -> list[Finding]:
        """Return only findings with severity == 'info'."""
        return [f for f in self.findings if f.severity == "info"]

    def to_dict(self) -> dict[str, Any]:
        """Return a fully JSON-serializable representation of the scan."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "memory_usage_mb": self.memory_usage_mb,
            "column_profiles": [c.to_dict() for c in self.column_profiles],
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
        }

    def __repr__(self) -> str:
        crit = len(self.critical_findings())
        warn = len(self.warnings())
        return (
            f"<ScanResult rows={self.n_rows} cols={self.n_columns} "
            f"critical={crit} warnings={warn}>"
        )
