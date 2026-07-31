"""
Report context building and orchestration for Mentis.

`ReportBuilder` gathers results from the scanner, comparison,
explainability, audit, deployment, and monitoring subsystems into one
context dict, then dispatches to the HTML/Markdown/PDF renderers.
Keeping assembly separate from rendering means adding a new output
format never touches this aggregation logic.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from mentis.exceptions import ReportGenerationError
from mentis.reporting.html_report import HtmlReportRenderer
from mentis.reporting.pdf_report import PdfReportRenderer
from mentis.utils.helpers import ensure_directory
from mentis.utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_FORMATS = {"html", "markdown", "pdf"}


class ReportBuilder:
    """
    Builds a professional report from a Guardian's accumulated results.

    Examples:
        >>> builder = ReportBuilder()
            >>> path = builder.build(
            ...     guardian, output_dir="reports", fmt="html"
            ... )  # doctest: +SKIP
    """

    def build(
        self,
        guardian: Any,
        output_dir: str = "mentis_reports",
        fmt: str = "html",
        filename: str | None = None,
    ) -> str:
        """
        Assemble a report from a `Guardian` instance's last results and
        write it to disk.

        Args:
            guardian: A `Guardian` instance holding `last_scan_result`,
                `last_comparison_result`, `last_explain_result`,
                `last_audit_result`, `last_deployment_result`,
                `last_bias_result`, `last_drift_result`,
                `last_monitoring_result` (any subset may be `None`).
            output_dir: Directory to write the report into.
            fmt: "html", "markdown", or "pdf".
            filename: Optional output filename (without extension
                inference beyond what `fmt` implies). Defaults to
                "mentis_report".

        Returns:
            Path to the generated report file.

        Raises:
            ReportGenerationError: If `fmt` is unsupported, no results
                are available to report on, or rendering fails.

        Examples:
            >>> builder = ReportBuilder()
            >>> path = builder.build(guardian, fmt="html")  # doctest: +SKIP
        """
        if fmt not in _SUPPORTED_FORMATS:
            raise ReportGenerationError(
                f"Unsupported report format: {fmt!r}. Expected one of {sorted(_SUPPORTED_FORMATS)}."
            )

        context = self.build_context(guardian)
        if not context["has_any_results"]:
            raise ReportGenerationError(
                "No results available to report on. Run scan(), compare_models(), "
                "explain(), audit_pipeline(), or deploy_check() before generate_report()."
            )

        ensure_directory(output_dir)
        base_name = filename or "mentis_report"

        if fmt == "markdown":
            return self._build_markdown(context, os.path.join(output_dir, f"{base_name}.md"))

        html_path = os.path.join(output_dir, f"{base_name}.html")
        HtmlReportRenderer().render(context, html_path)

        if fmt == "html":
            return html_path

        # fmt == "pdf": render HTML first, then convert
        pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
        return PdfReportRenderer().render_from_html(html_path, pdf_path)

    def build_context(self, guardian: Any) -> dict[str, Any]:
        """
        Gather a `Guardian`'s accumulated results into a single
        template-ready context dict.

        Args:
            guardian: A `Guardian` instance.

        Returns:
            Dict with keys: "generated_at", "scan", "comparison",
            "explain", "audit", "deployment", "bias", "drift",
            "monitoring", "has_any_results".

        Examples:
            >>> builder = ReportBuilder()
            >>> ctx = builder.build_context(guardian)  # doctest: +SKIP
            >>> ctx["has_any_results"]  # doctest: +SKIP
        """
        scan = getattr(guardian, "last_scan_result", None)
        comparison = getattr(guardian, "last_comparison_result", None)
        explain = getattr(guardian, "last_explain_result", None)
        audit = getattr(guardian, "last_audit_result", None)
        deployment = getattr(guardian, "last_deployment_result", None)
        bias = getattr(guardian, "last_bias_result", None)
        drift = getattr(guardian, "last_drift_result", None)
        monitoring = getattr(guardian, "last_monitoring_result", None)

        context: dict[str, Any] = {
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "scan": scan.to_dict() if scan is not None and hasattr(scan, "to_dict") else scan,
            "comparison": (
                comparison.to_dict()
                if comparison is not None and hasattr(comparison, "to_dict")
                else comparison
            ),
            "explain": explain,
            "audit": audit.to_dict() if audit is not None and hasattr(audit, "to_dict") else audit,
            "deployment": (
                deployment.to_dict()
                if deployment is not None and hasattr(deployment, "to_dict")
                else deployment
            ),
            "bias": [b.to_dict() for b in bias] if bias else None,
            "drift": drift,
            "monitoring": (
                monitoring.to_dict()
                if monitoring is not None and hasattr(monitoring, "to_dict")
                else monitoring
            ),
        }
        context["has_any_results"] = any(
            context[key] is not None
            for key in (
                "scan",
                "comparison",
                "explain",
                "audit",
                "deployment",
                "bias",
                "drift",
                "monitoring",
            )
        )
        return context

    def _build_markdown(self, context: dict[str, Any], output_path: str) -> str:
        lines: list[str] = [
            "# Mentis Report",
            f"_Generated: {context['generated_at']}_",
            "",
        ]

        if context["scan"]:
            lines += ["## Dataset Scan", f"```\n{context['scan']}\n```", ""]
        if context["comparison"]:
            lines += ["## Model Comparison", f"```\n{context['comparison']}\n```", ""]
        if context["explain"]:
            lines += ["## Explainability", f"```\n{context['explain']}\n```", ""]
        if context["audit"]:
            lines += ["## Pipeline Audit", f"```\n{context['audit']}\n```", ""]
        if context["deployment"]:
            lines += ["## Deployment Readiness", f"```\n{context['deployment']}\n```", ""]
        if context["bias"]:
            lines += ["## Bias / Fairness", f"```\n{context['bias']}\n```", ""]
        if context["drift"]:
            lines += ["## Data Drift", f"```\n{context['drift']}\n```", ""]
        if context["monitoring"]:
            lines += ["## Monitoring", f"```\n{context['monitoring']}\n```", ""]

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except OSError as exc:
            raise ReportGenerationError(
                f"Failed to write Markdown report to '{output_path}': {exc}"
            ) from exc

        return output_path
