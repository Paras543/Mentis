"""
Command-line interface for Mentis, built with Typer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from mentis import Guardian, MentisConfig
from mentis.exceptions import MentisError

app = typer.Typer(name="mentis", help="Mentis: an AI engineer's toolkit.")
console = Console()


def _load_guardian(config: str | None) -> Guardian:
    return Guardian.from_yaml(config) if config else Guardian()


def _load_dataframe(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        console.print(f"[red]File not found: {data_path}[/red]")
        raise typer.Exit(code=1)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


@app.command()
def scan(
    data: str = typer.Argument(..., help="Path to CSV/Parquet dataset."),
    target: str | None = typer.Option(None, help="Target column name."),
    config: str | None = typer.Option(None, help="Path to a Mentis YAML config."),
) -> None:
    """Run the Dataset Scanner on a CSV/Parquet file."""
    guardian = _load_guardian(config)
    df = _load_dataframe(data)

    with console.status("Scanning dataset..."):
        try:
            result = guardian.scan(df, target=target)
        except MentisError as exc:
            console.print(f"[red]Scan failed: {exc}[/red]")
            raise typer.Exit(code=1)

    console.print(result)
    for finding in result.warnings():
        console.print(f"[yellow]⚠ {finding.message}[/yellow] -> {finding.suggestion}")


@app.command()
def compare(
    config: str = typer.Argument(..., help="Path to a Mentis YAML config."),
    data: str = typer.Option(..., help="Path to CSV/Parquet dataset."),
) -> None:
    """Train and compare models using settings from a YAML config."""
    guardian = Guardian.from_yaml(config)
    df = _load_dataframe(data)

    target = guardian.config.project.target
    if not target:
        console.print("[red]Config must specify project.target for comparison.[/red]")
        raise typer.Exit(code=1)

    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with console.status("Training and comparing models..."):
        try:
            leaderboard = guardian.compare_models(X_train, X_test, y_train, y_test)
        except MentisError as exc:
            console.print(f"[red]Comparison failed: {exc}[/red]")
            raise typer.Exit(code=1)

    table = Table(title="Model Leaderboard")
    table.add_column("Model")
    table.add_column(leaderboard.primary_metric)
    for r in leaderboard.results:
        score = r.metrics.get(leaderboard.primary_metric, "N/A")
        table.add_row(r.model_name, f"{score:.4f}" if isinstance(score, float) else str(score))
    console.print(table)


@app.command()
def audit(
    project_path: str = typer.Argument(".", help="Path to the project to audit.")
) -> None:
    """Audit an ML project's structure and production readiness."""
    guardian = Guardian()
    with console.status("Auditing project..."):
        result = guardian.audit_pipeline(project_path)

    console.print(f"[bold]Production Readiness Score: {result.score}/100[/bold]")
    for finding in result.failed():
        console.print(f"[yellow]✗ {finding.name}[/yellow] ({finding.severity}) -> {finding.suggestion}")


@app.command(name="deploy-check")
def deploy_check(
    project_path: str = typer.Argument(".", help="Path to the project to check.")
) -> None:
    """Check a project's deployment readiness."""
    guardian = Guardian()
    with console.status("Checking deployment readiness..."):
        result = guardian.deploy_check(project_path)

    console.print(f"[bold]Deployment Score: {result.score}/100[/bold]")
    if result.detected_framework:
        console.print(f"Framework detected: {result.detected_framework}")
    for finding in result.failed():
        console.print(f"[yellow]✗ {finding.name}[/yellow] ({finding.severity}) -> {finding.suggestion}")


@app.command()
def report(
    data: str | None = typer.Option(None, help="Path to CSV/Parquet dataset (optional, enables scan)."),
    target: str | None = typer.Option(None, help="Target column name."),
    project_path: str = typer.Option(".", help="Project path for audit + deploy checks."),
    output_dir: str = typer.Option("mentis_reports", help="Output directory for the report."),
    fmt: str = typer.Option("html", help="Report format: html, markdown, or pdf."),
    config: str | None = typer.Option(None, help="Path to a Mentis YAML config."),
) -> None:
    """
    Generate a Mentis report: optionally scans a dataset, then always
    runs pipeline audit and deployment checks, and writes the report.
    """
    guardian = _load_guardian(config)

    if data:
        df = _load_dataframe(data)
        with console.status("Scanning dataset..."):
            try:
                guardian.scan(df, target=target)
            except MentisError as exc:
                console.print(f"[yellow]⚠ Scan skipped: {exc}[/yellow]")

    with console.status("Auditing project structure..."):
        try:
            guardian.audit_pipeline(project_path)
        except Exception as exc:
            console.print(f"[yellow]⚠ Audit skipped: {exc}[/yellow]")

    with console.status("Checking deployment readiness..."):
        try:
            guardian.deploy_check(project_path)
        except Exception as exc:
            console.print(f"[yellow]⚠ Deploy check skipped: {exc}[/yellow]")

    with console.status(f"Generating {fmt.upper()} report..."):
        try:
            path = guardian.generate_report(output_path=output_dir, fmt=fmt)
        except MentisError as exc:
            console.print(f"[red]Report generation failed: {exc}[/red]")
            raise typer.Exit(code=1)

    console.print(f"[green]✓ Report written to:[/green] {path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()



    