"""
Rich-powered logging setup for Mentis.

Provides a single `get_logger` factory so every module gets a
consistently configured, colorized logger without duplicating
handler/formatter setup logic (DRY).
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False
_LOG_FORMAT = "%(message)s"


def _configure_root_logger(level: int = logging.INFO) -> None:
    """Configure the root 'mentis' logger exactly once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=True,
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="[%X]"))

    root = logging.getLogger("mentis")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a namespaced logger under the 'mentis' hierarchy.

    Args:
        name: Typically `__name__` of the calling module.

    Returns:
        A configured `logging.Logger` instance with Rich formatting.

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Scan started")  # doctest: +SKIP
    """
    _configure_root_logger()
    # Namespace under "mentis." so all library logs share one root config,
    # even when called from e.g. "mentis.scanner.dataset_scanner".
    if not name.startswith("mentis"):
        name = f"mentis.{name}"
    return logging.getLogger(name)


def set_log_level(level: int | str) -> None:
    """
    Adjust the verbosity of all Mentis logging at runtime.

    Args:
        level: A logging level such as `logging.DEBUG`, `logging.INFO`,
            or a string like "DEBUG", "WARNING".

    Examples:
        >>> import logging
        >>> set_log_level(logging.DEBUG)
        >>> set_log_level("WARNING")
    """
    _configure_root_logger()
    logging.getLogger("mentis").setLevel(level)
