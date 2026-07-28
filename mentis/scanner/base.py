"""
Base classes and shared contracts for scanner checks.

Each individual check (missing values, duplicates, correlation, etc.)
is implemented as its own class conforming to `BaseCheck`. This makes
the scanner open for extension (add a new check by subclassing) but
closed for modification (the orchestrator never changes) -- the
Open/Closed Principle in practice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


from mentis.scanner.result import Finding


class BaseCheck(ABC):
    """
    Abstract base class for all dataset scanner checks.

    Subclasses implement `run`, returning a list of `Finding` objects.
    A check should never raise for "normal" data issues -- it should
    encode issues as findings. Checks may raise `DatasetError` only for
    truly unusable input (e.g. an empty dataframe where the check is
    fundamentally inapplicable).
    """

    #: Unique, stable identifier for this check (used in Finding.check_name)
    name: str = "base_check"

    @abstractmethod
    def run(self, df: pd.DataFrame, **context: object) -> list[Finding]:
        """
        Execute the check against the given dataframe.

        Args:
            df: The dataframe being scanned.
            **context: Additional shared context (e.g. target column name,
                task type, precomputed column roles) passed by the
                orchestrating `DatasetScanner`.

        Returns:
            A list of `Finding` objects. Empty list if no issues found.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"