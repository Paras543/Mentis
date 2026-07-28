import pandas as pd
from mentis import Guardian

# 1. Load raw data and initialize Guardian
df = pd.read_csv("data.csv")
guardian = Guardian()

# 2. Scan raw dataset
guardian.scan(df, target="target")

# 3. Clean dataset manually
df_clean = df.drop(columns=["constant_col"]).dropna()

# 4. Train and compare models on cleaned data
X = df_clean.drop(columns=["target"])
y = df_clean["target"]
leaderboard = guardian.compare_models(X, X, y, y, task="classification")

# 5. Export HTML report with clean scan + model accuracy leaderboard
report_path = guardian.generate_report(output_path="mentis_reports", fmt="html")
print(f"Report generated at: {report_path}")
