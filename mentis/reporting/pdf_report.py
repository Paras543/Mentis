"""
PDF report rendering for Mentis.

Converts an already-rendered HTML report into a PDF using WeasyPrint.
Kept as a thin, optional layer -- WeasyPrint has system-level
dependencies (Cairo/Pango), so failures here should never break HTML
report generation.
"""

from __future__ import annotations

import os

from mentis.exceptions import ReportGenerationError
from mentis.utils.helpers import ensure_directory
from mentis.utils.logger import get_logger

logger = get_logger(__name__)


class PdfReportRenderer:
    """
    Converts a rendered HTML report file into a PDF.

    Examples:
        >>> renderer = PdfReportRenderer()
        >>> path = renderer.render_from_html("out/report.html", "out/report.pdf")  # doctest: +SKIP
    """

    def render_from_html(self, html_path: str, output_path: str) -> str:
        """
        Convert an existing HTML report file into a PDF.

        Args:
            html_path: Path to a previously rendered HTML report.
            output_path: Destination path for the PDF.

        Returns:
            The `output_path` written to.

        Raises:
            ReportGenerationError: If WeasyPrint is not installed, the
                source HTML is missing, or conversion fails.

        Examples:
            >>> renderer = PdfReportRenderer()
            >>> renderer.render_from_html("out/report.html", "out/report.pdf")  # doctest: +SKIP
        """
        if not os.path.exists(html_path):
            raise ReportGenerationError(f"HTML report not found at '{html_path}'.")

        try:
            from weasyprint import HTML
        except ImportError as exc:
            raise ReportGenerationError(
                "weasyprint is not installed. Install it with `pip install weasyprint` "
                "to generate PDF reports."
            ) from exc

        output_dir = os.path.dirname(output_path) or "."
        ensure_directory(output_dir)

        try:
            HTML(filename=html_path).write_pdf(output_path)
        except Exception as exc:  # noqa: BLE001 - surface as domain error
            raise ReportGenerationError(f"Failed to generate PDF report: {exc}") from exc

        return output_path