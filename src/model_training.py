import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("="*50)
print("PHASE 6 : MACHINE LEARNING")
print("="*50)

# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv(
    "data/processed/feature_data.csv"
)

print("\nDataset Shape:")
print(df.shape)

# -----------------------------------
# Target Variable
# -----------------------------------

target = "recruiter_decision"

# -----------------------------------
# Features
# -----------------------------------

features = [
    "pm_skill_count",
    "sql_flag",
    "analytics_flag",
    "agile_flag",
    "jira_flag",
    "certification_flag",
    "experience_score"
]

X = df[features]

# -----------------------------------
# Encode Target
# -----------------------------------

le = LabelEncoder()

y = le.fit_transform(
    df[target]
)

# Save Encoder
joblib.dump(
    le,
    "models/label_encoder.pkl"
)

# -----------------------------------
# Train Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# -----------------------------------
# Logistic Regression
# -----------------------------------

lr_model = LogisticRegression()

lr_model.fit(
    X_train,
    y_train
)

lr_pred = lr_model.predict(
    X_test
)

lr_accuracy = accuracy_score(
    y_test,
    lr_pred
)

print(
    "\nLogistic Regression Accuracy:",
    round(lr_accuracy*100,2),
    "%"
)

joblib.dump(
    lr_model,
    "models/logistic_regression_model.pkl"
)

# -----------------------------------
# Random Forest
# -----------------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print(
    "\nRandom Forest Accuracy:",
    round(rf_accuracy*100,2),
    "%"
)

joblib.dump(
    rf_model,
    "models/random_forest_model.pkl"
)

# -----------------------------------
# Best Model
# -----------------------------------

if rf_accuracy > lr_accuracy:
    print("\nBest Model: Random Forest")
else:
    print("\nBest Model: Logistic Regression")

print("\nPhase 6 Completed Successfully")