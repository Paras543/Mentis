"""
Scanner subpackage: dataset inspection and issue detection for Mentis.

Public entry point is `DatasetScanner`, used internally by
`Guardian.scan()`. Individual checks (`checks.py`) can also be imported
directly for advanced/custom pipelines.
"""

from mentis.scanner.dataset_scanner import DatasetScanner
from mentis.scanner.result import ColumnProfile, Finding, ScanResult

__all__ = [
    "DatasetScanner",
    "ScanResult",
    "ColumnProfile",
    "Finding",
]

