import pandas as pd
from mentis import Guardian


df = pd.read_csv("data.csv")
guardian = Guardian()


scan = guardian.scan(df, target="target")
print(f"Dataset scan complete: {scan.summary['total_findings']} issues found.")


X = df.drop(columns=["target"])
y = df["target"]
leaderboard = guardian.compare_models(X, X, y, y, task="classification")
print(f"Best Model: {leaderboard.best_model().model_name}")


guardian.audit_pipeline(".")
guardian.deploy_check(".")


report_path = guardian.generate_report(output_path="mentis_reports", fmt="html")
print(f"Report generated at: {report_path}")




