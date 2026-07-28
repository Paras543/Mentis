"""
Scan-Only Example for Mentis

This script shows how a real user who ONLY wants to scan a dataset
and generate an HTML report containing ONLY the Dataset Scan findings
does so using Mentis.
"""

import pandas as pd
from mentis import Guardian

# 1. Load data
df = pd.read_csv("data.csv")

# 2. Initialize Guardian
guardian = Guardian()

# 3. Perform scan only
scan_result = guardian.scan(df, target="target")
print(f"Scan finished. Found {scan_result.summary['total_findings']} issues.")

# 4. Generate HTML report containing ONLY the scan section
report_path = guardian.generate_report(output_path="scan_reports", fmt="html")
print(f"HTML Scan Report saved to: {report_path}")
