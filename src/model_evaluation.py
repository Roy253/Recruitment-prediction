import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ==================================
# CREATE REQUIRED FOLDERS
# ==================================

os.makedirs("visuals", exist_ok=True)
os.makedirs("data/output", exist_ok=True)

print("=" * 50)
print("PHASE 7 : MODEL EVALUATION")
print("=" * 50)

# ==================================
# LOAD DATASET
# ==================================

df = pd.read_csv(
    "data/processed/feature_data.csv"
)

print("\nDataset Loaded")
print("Rows:", len(df))

# ==================================
# FEATURES
# ==================================

features = [
    "pm_skill_count",
    "sql_flag",
    "analytics_flag",
    "agile_flag",
    "jira_flag",
    "certification_flag",
    "experience_score"
]

# Verify Features Exist

missing_features = [
    col for col in features
    if col not in df.columns
]

if missing_features:
    print("\nMissing Columns:")
    print(missing_features)
    exit()

X = df[features]

# ==================================
# TARGET
# ==================================

target = "recruiter_decision"

if target not in df.columns:
    print("\nTarget column not found!")
    exit()

y = df[target]

# ==================================
# LOAD LABEL ENCODER
# ==================================

le = joblib.load(
    "models/label_encoder.pkl"
)

y = le.transform(y)

# ==================================
# TRAIN TEST SPLIT
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==================================
# LOAD MODEL
# ==================================

model = joblib.load(
    "models/random_forest_model.pkl"
)

# ==================================
# PREDICTIONS
# ==================================

predictions = model.predict(X_test)

# ==================================
# METRICS
# ==================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted"
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted"
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

print("\nAccuracy :", round(accuracy * 100, 2))
print("Precision:", round(precision * 100, 2))
print("Recall   :", round(recall * 100, 2))
print("F1 Score :", round(f1 * 100, 2))

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

# ==================================
# SAVE METRICS
# ==================================

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4)
    ]
})

metrics_df.to_csv(
    "data/output/model_metrics.csv",
    index=False
)

print("\nMetrics Saved")

# ==================================
# CONFUSION MATRIX
# ==================================

cm = confusion_matrix(
    y_test,
    predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "visuals/confusion_matrix.png"
)

plt.close()

print("Confusion Matrix Saved")

# ==================================
# FEATURE IMPORTANCE
# ==================================

if hasattr(
    model,
    "feature_importances_"
):

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    plt.figure(
        figsize=(8,5)
    )

    plt.bar(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xticks(
        rotation=45
    )

    plt.title(
        "Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        "visuals/feature_importance.png"
    )

    plt.close()

    importance_df.to_csv(
        "data/output/feature_importance.csv",
        index=False
    )

    print("Feature Importance Saved")

print("\nPhase 7 Completed Successfully")