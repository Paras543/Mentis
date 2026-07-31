# Mentis

**An AI engineer's toolkit — dataset scanning, automated model comparison, explainability, pipeline auditing, deployment checks, drift and bias detection, monitoring, and professional reporting, all behind a single elegant API.**

[![PyPI version](https://img.shields.io/pypi/v/mentis.svg)](https://pypi.org/project/mentis/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

</div>

```python
from mentis import Guardian

guardian = Guardian()

guardian.scan(df)
guardian.compare_models(X_train, X_test, y_train, y_test)
guardian.explain(model, X_test, y_test)
guardian.audit_pipeline()
guardian.deploy_check()
guardian.generate_report()
```

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Guardian](#guardian)
- [Dataset Scanner](#dataset-scanner)
- [Model Comparison](#model-comparison)
- [Explainability](#explainability)
- [Pipeline Auditor](#pipeline-auditor)
- [Deployment Checker](#deployment-checker)
- [Data Drift](#data-drift)
- [Bias Detection](#bias-detection)
- [Monitoring](#monitoring)
- [Visualization](#visualization)
- [Reporting](#reporting)
- [Configuration](#configuration)
- [CLI](#cli)
- [Error Handling](#error-handling)
- [API Reference Summary](#api-reference-summary)
- [License](#license)

---

## Installation

Install the core library:

```bash
pip install mentis
```

Install with every optional integration (XGBoost, LightGBM, CatBoost, SHAP, Evidently, Fairlearn, WeasyPrint):

```bash
pip install "mentis[all]"
```

Or pick specific extras:

```bash
pip install "mentis[boosting]"        # xgboost, lightgbm, catboost
pip install "mentis[explainability]"  # shap
pip install "mentis[monitoring]"      # evidently
pip install "mentis[fairness]"        # fairlearn
pip install "mentis[pdf]"             # weasyprint
```

> If a module's optional dependency isn't installed, Mentis degrades gracefully — the relevant method raises a clear error telling you what to install, rather than crashing the whole library.

---

## Quick Start

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from mentis import Guardian

df = pd.read_csv("customers.csv")

guardian = Guardian()

# 1. Scan the dataset for quality issues
scan_result = guardian.scan(df, target="churn")
print(scan_result)

# 2. Compare models
X = df.drop(columns=["churn"])
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

leaderboard = guardian.compare_models(X_train, X_test, y_train, y_test)
best_model = leaderboard.best_model()
print(f"Best model: {best_model.model_name}")

# 3. Generate a report
guardian.generate_report()
```

---

## Core Concepts

Mentis is built around one idea: **a single object should be able to do everything**, without you having to learn ten different library APIs.

`Guardian` is a *facade* — a thin coordinating layer over independent subsystems (scanner, comparison, explainability, auditing, deployment, monitoring, reporting). Every subsystem can be used standalone by importing it directly from its subpackage, but `Guardian` is the recommended entry point for almost everything.

Each call to a `Guardian` method stores its result on the instance (`last_scan_result`, `last_comparison_result`, etc.), so `generate_report()` can assemble everything you've run so far without you having to pass results around manually.

---

## Guardian

```python
from mentis import Guardian
```

### Creating a Guardian

```python
guardian = Guardian()
```

Or from a YAML config file:

```python
guardian = Guardian.from_yaml("mentis.yaml")
```

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `config` | `MentisConfig` | Active configuration for this instance |
| `last_scan_result` | `ScanResult \| None` | Result of the most recent `.scan()` |
| `last_comparison_result` | `Leaderboard \| None` | Result of the most recent `.compare_models()` |
| `last_explain_result` | `dict \| None` | Result of the most recent `.explain()` |
| `last_audit_result` | `AuditResult \| None` | Result of the most recent `.audit_pipeline()` |
| `last_deployment_result` | `DeploymentResult \| None` | Result of the most recent `.deploy_check()` |
| `last_bias_result` | `list[FairnessResult] \| None` | Result of the most recent `.check_bias()` |
| `last_drift_result` | `dict \| None` | Result of the most recent `.check_drift()` |
| `last_monitoring_result` | `MonitoringSnapshot \| None` | Result of the most recent `.monitor()` |

---

## Dataset Scanner

Inspects a dataframe for data-quality issues before you spend time training on it.

```python
result = guardian.scan(df, target="churn")
```

### Signature

```python
guardian.scan(df: pd.DataFrame, target: str | None = None) -> ScanResult
```

### Checks performed

- Missing values & null percentage
- Duplicate rows and duplicate columns
- Constant / zero-variance / near-zero-variance columns
- Data types & mixed data types
- Categorical vs. numerical vs. date columns
- Class imbalance / target imbalance
- Infinite values
- Memory usage
- High correlation between features
- Outlier detection
- Potential data leakage
- Potential ID columns and other suspicious columns

### Working with results

```python
result = guardian.scan(df, target="churn")

print(result)                          # <ScanResult rows=... cols=... critical=... warnings=...>
print(result.summary)                  # severity counts, checks triggered

for finding in result.warnings():
    print(finding.message, "->", finding.suggestion)
```

### Example: fail a CI job on critical findings

```python
result = guardian.scan(df, target="churn")

if result.summary.get("critical", 0) > 0:
    raise SystemExit("Critical data quality issues found — see scan report.")
```

---

## Model Comparison

Trains and ranks multiple models automatically for classification or regression tasks.

```python
leaderboard = guardian.compare_models(X_train, X_test, y_train, y_test)
```

### Signature

```python
guardian.compare_models(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    task: str | None = None,      # "classification" | "regression"
    models: list[str] | None = None,
    cv: int | None = None,
) -> Leaderboard
```

### Models included

**Classification:** Random Forest, Logistic Regression, XGBoost, LightGBM, CatBoost, Gradient Boosting, Extra Trees, Decision Tree, SVM, KNN.

**Regression:** Linear Regression, Random Forest Regressor, XGBoost Regressor, LightGBM Regressor, CatBoost Regressor, ElasticNet, Lasso, Ridge, Gradient Boosting Regressor, Extra Trees Regressor.

> Boosted models (XGBoost/LightGBM/CatBoost) are only included if their libraries are installed — Mentis detects this automatically and logs a warning instead of failing.

### Metrics computed

- **Classification:** accuracy, precision, recall, F1, ROC AUC
- **Regression:** RMSE, MAE, MAPE, R²

Every model result also includes cross-validation scores, training time, inference time, memory usage, and feature importances (when the model exposes them).

### Working with the leaderboard

```python
print(leaderboard)                          # <Leaderboard task=... best_model=... n_models=...>

df = leaderboard.to_dataframe()             # full comparison as a pandas DataFrame
best = leaderboard.best_model()             # top-ranked ModelResult
print(best.model_name, best.metrics)
```

### Restricting to specific models

```python
leaderboard = guardian.compare_models(
    X_train, X_test, y_train, y_test,
    models=["Random Forest", "XGBoost", "Logistic Regression"],
    cv=10,
)
```

---

## Explainability

Generates SHAP values, permutation importance, and evaluation curves for a fitted model.

```python
report = guardian.explain(model, X_test, y_test)
```

### Signature

```python
guardian.explain(
    model: Any,
    X: Any,
    y: Any | None = None,
    task: str | None = None,
    y_pred: Any | None = None,
    y_proba: Any | None = None,
    include_shap: bool = True,
    include_permutation: bool = True,
    include_curves: bool = True,
) -> dict[str, Any]
```

### What gets generated

| Key | When | Description |
|---|---|---|
| `shap` | `include_shap=True` | SHAP values, feature names, base value |
| `permutation_importance` | `y` provided | Feature importance via permutation |
| `confusion_matrix` | classification + `y` | Confusion matrix |
| `roc_curve` | classification + probabilities available | ROC curve points |
| `pr_curve` | classification + probabilities available | Precision-recall curve points |
| `calibration_curve` | classification + probabilities available | Reliability curve points |
| `residuals` | regression + `y` | Predicted vs. residual values |

> Each artifact is computed independently — if SHAP fails (e.g. not installed) but permutation importance succeeds, you still get a partial report instead of a hard failure. `explain()` only raises if *every* requested artifact fails.

### Example: explainability-only, no labels needed

```python
# SHAP doesn't require y — just the fitted model and features
report = guardian.explain(model, X_test, include_permutation=False, include_curves=False)
print(report["shap"]["explainer_type"])   # "tree" or "kernel"
```

### Example: full report with precomputed predictions

```python
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

report = guardian.explain(model, X_test, y_test, y_pred=y_pred, y_proba=y_proba)
```

---

## Pipeline Auditor

Scores a project's structure against production-readiness best practices.

```python
result = guardian.audit_pipeline(".")
```

### Signature

```python
guardian.audit_pipeline(project_path: str = ".") -> AuditResult
```

### What it checks

README, requirements/dependency files, `.gitignore`, Dockerfile, `tests/`, GitHub Actions, logging config, config files, `.env.example`, model artifacts directory, `CHANGELOG.md`, pre-commit hooks, Makefile.

### Working with results

```python
result = guardian.audit_pipeline(".")

print(f"Production Readiness Score: {result.score}/100")

for finding in result.failed(severity="critical"):
    print(f"FAILED: {finding.name}: {finding.suggestion}")
```

---

## Deployment Checker

Scores a project's deployment readiness: containerization, orchestration, health endpoints, and framework detection.

```python
result = guardian.deploy_check(".")
```

### Signature

```python
guardian.deploy_check(project_path: str = ".") -> DeploymentResult
```

### What it checks

Dockerfile, `docker-compose.yml`, Kubernetes manifests, `.env.example`, secrets template, web framework detection (FastAPI/Flask), health/liveness/readiness endpoints, logging.

```python
result = guardian.deploy_check(".")

print(f"Deployment Score: {result.score}/100")
print(f"Framework detected: {result.detected_framework}")

for finding in result.failed():
    print(f"FAILED: {finding.name} ({finding.severity}) — {finding.suggestion}")
```

---

## Data Drift

Detects feature, target, and prediction drift between a reference dataset and current production data, powered by [Evidently](https://github.com/evidentlyai/evidently).

```python
report = guardian.check_drift(reference_df, current_df, target="churn")
```

### Signature

```python
guardian.check_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target: str | None = None,
    prediction: str | None = None,
) -> dict[str, Any]
```

### Return shape

```python
{
    "dataset_drift": True,
    "drift_share": 0.42,
    "feature_drift": {
        "age": {"drift_detected": True, "drift_score": 0.31, "stattest_name": "ks"},
        ...
    },
    "target_drift": {...},       # if target was provided
    "prediction_drift": {...},   # if prediction was provided
}
```

> Requires `evidently` — install with `pip install "mentis[monitoring]"`.

---

## Bias Detection

Audits a classifier's predictions for fairness across sensitive features, powered by [Fairlearn](https://github.com/fairlearn/fairlearn).

```python
results = guardian.check_bias(
    y_test,
    y_pred,
    sensitive_features={"gender": df["gender"], "age_group": df["age_group"]},
)
```

### Signature

```python
guardian.check_bias(
    y_true: Any,
    y_pred: Any,
    sensitive_features: dict[str, Any],
) -> list[FairnessResult]
```

Each `FairnessResult` includes demographic parity difference, equal opportunity difference, equalized odds difference, and per-group selection rates — one result per sensitive feature audited.

```python
for result in results:
    print(result.sensitive_feature, result.demographic_parity_difference)
```

> Requires `fairlearn` — install with `pip install "mentis[fairness]"`.

---

## Monitoring

Builds a health snapshot for a deployed model: performance degradation vs. a baseline, prediction/feature distributions, and data quality.

```python
snapshot = guardian.monitor(
    current_data=production_df,
    predictions=model.predict(production_df),
    current_metrics={"accuracy": 0.81},
    baseline_metrics={"accuracy": 0.90},
)
```

### Signature

```python
guardian.monitor(
    current_data: pd.DataFrame,
    predictions: Any,
    current_metrics: dict[str, float] | None = None,
    baseline_metrics: dict[str, float] | None = None,
    degradation_tolerance: float = 0.05,
) -> MonitoringSnapshot
```

```python
if snapshot.degraded:
    print("Model performance has degraded beyond tolerance:")
    print(snapshot.performance_delta)
```

---

## Visualization

Chart generation is available standalone via `ChartGenerator` and is used internally by `generate_report()`. Every chart method saves a PNG and returns its path.

```python
from mentis.visualization.charts import ChartGenerator

charts = ChartGenerator(output_dir="mentis_charts")

charts.correlation_heatmap(df)
charts.missing_value_heatmap(df)
charts.distribution(df["age"])
charts.feature_importance(best_model.feature_importances)
charts.shap_summary(report["shap"])
charts.roc_curve(report["roc_curve"])
charts.pr_curve(report["pr_curve"])
charts.calibration_curve(report["calibration_curve"])
charts.residual_plot(report["residuals"])
charts.learning_curve(curve)
charts.confusion_matrix(report["confusion_matrix"])
```

---

## Reporting

Assembles every result you've generated so far into a single professional report.

```python
path = guardian.generate_report()
```

### Signature

```python
guardian.generate_report(
    output_path: str | None = None,   # output directory
    fmt: str | None = None,           # "html" | "markdown" | "pdf"
) -> str
```

### Formats

| Format | Notes |
|---|---|
| `html` | Styled, self-contained HTML report (default) |
| `markdown` | Plain-text-friendly report for READMEs/PRs |
| `pdf` | Requires `weasyprint` (`pip install "mentis[pdf]"`) |

```python
guardian.scan(df, target="churn")
guardian.compare_models(X_train, X_test, y_train, y_test)
guardian.audit_pipeline(".")

path = guardian.generate_report(output_path="reports", fmt="html")
print(f"Report saved to {path}")
```

> `generate_report()` raises `ReportGenerationError` if nothing has been run yet — there's nothing to report on.

---

## Configuration

Mentis supports YAML-based configuration so CI pipelines and CLI runs don't need Python code.

### mentis.yaml

```yaml
project:
  task: classification
  target: churn

scanner:
  leakage: true
  missing_threshold: 0.30
  correlation_threshold: 0.90

comparison:
  cv: 5
  models: null   # null = use all available models

report:
  format: html
  output_dir: mentis_reports
```

### Loading it

```python
from mentis import Guardian

guardian = Guardian.from_yaml("mentis.yaml")
```

Any value you pass directly to a method (e.g. `guardian.compare_models(..., cv=10)`) overrides the config for that call only.

---

## CLI

Mentis ships a Typer-based CLI, installed automatically alongside the package.

```bash
mentis scan data.csv --target churn
mentis compare mentis.yaml --data data.csv
mentis audit .
mentis deploy-check .
```

### `mentis scan`

```bash
mentis scan data.csv --target churn [--config mentis.yaml]
```

Prints the scan summary and every warning-level finding with its suggestion.

### `mentis compare`

```bash
mentis compare mentis.yaml --data data.csv
```

Reads `project.target` from the config, splits the data, trains every configured model, and prints a ranked leaderboard table.

### `mentis audit`

```bash
mentis audit [project_path]
```

Prints the Production Readiness Score and every failed check.

### `mentis deploy-check`

```bash
mentis deploy-check [project_path]
```

Prints the Deployment Score, detected framework, and every failed check.

---

## Error Handling

Every error Mentis raises is a subclass of `MentisError`, so you can catch broadly or narrowly:

```python
from mentis.exceptions import (
    MentisError,
    DatasetError,
    ModelError,
    DeploymentError,
    ValidationError,
    ConfigurationError,
    ExplainabilityError,
    ReportGenerationError,
)

try:
    guardian.scan(df)
except DatasetError as e:
    print(f"Dataset problem: {e}")
except MentisError as e:
    print(f"Mentis error: {e}")
```

| Exception | Raised by |
|---|---|
| `DatasetError` | Scanner — invalid/empty/malformed dataframe |
| `ModelError` | Comparison — all models fail to train |
| `ValidationError` | Any module — bad input shape, unsupported task, missing dependency |
| `ConfigurationError` | Config loading — invalid YAML or schema |
| `ExplainabilityError` | Explainability — SHAP/curve computation fails |
| `DeploymentError` | Deployment checks — critical failure |
| `ReportGenerationError` | Reporting — unsupported format, nothing to report, render failure |

---

## API Reference Summary

| Method | Returns | Purpose |
|---|---|---|
| `guardian.scan(df, target=None)` | `ScanResult` | Dataset quality scan |
| `guardian.compare_models(X_train, X_test, y_train, y_test, ...)` | `Leaderboard` | Train & rank models |
| `guardian.explain(model, X, y=None, ...)` | `dict` | SHAP, importance, curves |
| `guardian.audit_pipeline(project_path=".")` | `AuditResult` | Production readiness score |
| `guardian.deploy_check(project_path=".")` | `DeploymentResult` | Deployment readiness score |
| `guardian.check_drift(reference, current, ...)` | `dict` | Feature/target/prediction drift |
| `guardian.check_bias(y_true, y_pred, sensitive_features)` | `list[FairnessResult]` | Fairness audit |
| `guardian.monitor(current_data, predictions, ...)` | `MonitoringSnapshot` | Degradation & distribution snapshot |
| `guardian.generate_report(output_path=None, fmt=None)` | `str` | Build HTML/Markdown/PDF report |

---

## License

MIT — see [`LICENSE`](LICENSE).# Mentis

## Documentation : https://mentis-documentation.vercel.app/docs/installation


## Read More 
 CasePaper: https://steady-twilight-ac2119.netlify.app/

 ## How I Solved The Bugs 
 Bugs Report: https://bejewelled-sorbet-7617d9.netlify.app/
