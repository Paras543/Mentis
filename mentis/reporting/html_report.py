"""
HTML report rendering for Mentis, using Jinja2 templates.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from mentis.exceptions import ReportGenerationError
from mentis.utils.helpers import ensure_directory
from mentis.utils.logger import get_logger

logger = get_logger(__name__)

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")


class HtmlReportRenderer:
    """
    Renders a Mentis report context dict into a styled HTML file using
    the `report.html.j2` Jinja2 template.

    Examples:
        >>> renderer = HtmlReportRenderer()
        >>> path = renderer.render(context, "mentis_reports/report.html")  # doctest: +SKIP
    """

    def __init__(self, template_dir: str = _TEMPLATE_DIR) -> None:
        """
        Args:
            template_dir: Directory containing `report.html.j2` and
                `styles.css`.
        """
        self._env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "j2"]),
        )

    def render(self, context: dict[str, Any], output_path: str) -> str:
        """
        Render the report template with `context` and write it to disk.

        Args:
            context: Data to populate the template with (see
                `ReportBuilder.build_context`).
            output_path: Destination path for the rendered HTML file.

        Returns:
            The `output_path` written to.

        Raises:
            ReportGenerationError: If the template fails to render or
                the file cannot be written.

        Examples:
            >>> renderer = HtmlReportRenderer()
            >>> renderer.render({"generated_at": "now"}, "out/report.html")  # doctest: +SKIP
        """
        try:
            template = self._env.get_template("report.html.j2")
            html = template.render(**context)
        except Exception as exc:  # noqa: BLE001 - surface as domain error
            raise ReportGenerationError(f"Failed to render HTML report template: {exc}") from exc

        output_dir = os.path.dirname(output_path) or "."
        ensure_directory(output_dir)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
        except OSError as exc:
            raise ReportGenerationError(f"Failed to write HTML report to '{output_path}': {exc}") from exc

        return output_path
    

    