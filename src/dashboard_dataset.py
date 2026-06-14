import os
import pandas as pd

print("=" * 50)
print("PHASE 9 : DASHBOARD DATASET")
print("=" * 50)

# Create dashboard folder
os.makedirs("dashboard", exist_ok=True)

# Load ranked dataset
df = pd.read_csv(
    "data/output/candidate_ranking.csv"
)

print("\nOriginal Shape:")
print(df.shape)

# ------------------------------------
# Select Dashboard Columns
# ------------------------------------

dashboard_columns = [
    "resume_id",
    "name",
    "job_role",
    "education",
    "recruiter_decision",
    "predicted_decision",
    "pm_fit_score",
    "candidate_rank",
    "pm_skill_count",
    "experience_score"
]

# Keep only existing columns
dashboard_columns = [
    col for col in dashboard_columns
    if col in df.columns
]

dashboard_df = df[dashboard_columns]

# ------------------------------------
# Add KPI Category
# ------------------------------------

dashboard_df["candidate_category"] = (
    dashboard_df["pm_fit_score"]
    .apply(
        lambda x:
        "High Potential"
        if x >= 75
        else (
            "Medium Potential"
            if x >= 50
            else "Low Potential"
        )
    )
)

# ------------------------------------
# Save Dashboard Dataset
# ------------------------------------

dashboard_df.to_csv(
    "dashboard/dashboard_data.csv",
    index=False
)

print("\nDashboard Dataset Created")

print("\nColumns:")
print(dashboard_df.columns.tolist())

print("\nRecords:")
print(len(dashboard_df))