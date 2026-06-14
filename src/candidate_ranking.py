import os
import pandas as pd
import joblib

# Create output folder
os.makedirs("data/output", exist_ok=True)

print("=" * 50)
print("PHASE 8 : CANDIDATE RANKING")
print("=" * 50)

# Load Dataset
df = pd.read_csv(
    "data/processed/feature_data.csv"
)

# Load Model
model = joblib.load(
    "models/random_forest_model.pkl"
)

# Load Label Encoder
le = joblib.load(
    "models/label_encoder.pkl"
)

# Features
features = [
    "pm_skill_count",
    "sql_flag",
    "analytics_flag",
    "agile_flag",
    "jira_flag",
    "certification_flag",
    "experience_score"
]

# Predict
predictions = model.predict(
    df[features]
)

df["predicted_decision"] = (
    le.inverse_transform(predictions)
)

# PM Fit Score
df["pm_fit_score"] = (
      df["pm_skill_count"] * 30
    + df["experience_score"] * 0.30
    + df["sql_flag"] * 10
    + df["analytics_flag"] * 10
    + df["agile_flag"] * 10
    + df["jira_flag"] * 10
    + df["certification_flag"] * 10
)

# Normalize Score
df["pm_fit_score"] = (
    df["pm_fit_score"]
    / df["pm_fit_score"].max()
) * 100

# Ranking
df["candidate_rank"] = (
    df["pm_fit_score"]
    .rank(
        ascending=False,
        method="dense"
    )
)

# Sort
df = df.sort_values(
    by="candidate_rank"
)

# Save
df.to_csv(
    "data/output/candidate_ranking.csv",
    index=False
)

print("\nTop 10 Candidates:")

print(
    df[
        [
            "name",
            "job_role",
            "pm_fit_score",
            "candidate_rank",
            "predicted_decision"
        ]
    ].head(10)
)

print("\nCandidate Ranking Created Successfully")