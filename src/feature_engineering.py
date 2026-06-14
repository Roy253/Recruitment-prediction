import pandas as pd

print("="*50)
print("PHASE 4 : FEATURE ENGINEERING")
print("="*50)

# Load cleaned data
df = pd.read_csv(
    "data/processed/cleaned_data.csv"
)

# -------------------------------------
# Show Columns
# -------------------------------------

print("\nAvailable Columns:")
print(df.columns.tolist())

# -------------------------------------
# Find Skills Column
# -------------------------------------

skills_col = None

for col in df.columns:
    if "skill" in col.lower():
        skills_col = col
        break

if skills_col is None:
    print("Skills column not found!")
    exit()

# -------------------------------------
# PM Skills Dictionary
# -------------------------------------

PM_SKILLS = [
    "product management",
    "product strategy",
    "roadmap",
    "roadmapping",
    "sql",
    "analytics",
    "power bi",
    "stakeholder management",
    "agile",
    "scrum",
    "jira",
    "figma",
    "user research",
    "market research",
    "ab testing",
    "a/b testing",
    "kpi",
    "wireframing",
    "data analysis"
]

# -------------------------------------
# PM Skill Count
# -------------------------------------

def count_pm_skills(text):

    text = str(text).lower()

    count = 0

    for skill in PM_SKILLS:
        if skill in text:
            count += 1

    return count

df["pm_skill_count"] = (
    df[skills_col]
    .apply(count_pm_skills)
)

# -------------------------------------
# Binary Skill Features
# -------------------------------------

df["sql_flag"] = (
    df[skills_col]
    .str.contains(
        "sql",
        case=False,
        na=False
    )
    .astype(int)
)

df["analytics_flag"] = (
    df[skills_col]
    .str.contains(
        "analytics",
        case=False,
        na=False
    )
    .astype(int)
)

df["agile_flag"] = (
    df[skills_col]
    .str.contains(
        "agile",
        case=False,
        na=False
    )
    .astype(int)
)

df["jira_flag"] = (
    df[skills_col]
    .str.contains(
        "jira",
        case=False,
        na=False
    )
    .astype(int)
)

# -------------------------------------
# Certification Feature
# -------------------------------------

cert_col = None

for col in df.columns:
    if "cert" in col.lower():
        cert_col = col
        break

if cert_col:

    df["certification_flag"] = (
        df[cert_col]
        .notnull()
        .astype(int)
    )

# -------------------------------------
# Experience Score
# -------------------------------------

experience_col = None

for col in df.columns:
    if "experience" in col.lower():
        experience_col = col
        break

if experience_col:

    try:

        df[experience_col] = pd.to_numeric(
            df[experience_col],
            errors="coerce"
        )

        max_exp = df[experience_col].max()

        if max_exp > 0:

            df["experience_score"] = (
                df[experience_col]
                / max_exp
            ) * 100

    except:
        pass

# -------------------------------------
# Save Feature Dataset
# -------------------------------------

df.to_csv(
    "data/processed/feature_data.csv",
    index=False
)

print("\nFeature Engineering Completed")

print("\nNew Features Created:")

feature_cols = [
    "pm_skill_count",
    "sql_flag",
    "analytics_flag",
    "agile_flag",
    "jira_flag",
    "certification_flag",
    "experience_score"
]

for col in feature_cols:
    if col in df.columns:
        print("✓", col)

print("\nFinal Dataset Shape:")
print(df.shape)

print(df.columns.tolist())


df.rename(columns={
    "experience_(years)": "experience_years",
    "salary_expectation_($)": "salary_expectation",
    "ai_score_(0-100)": "ai_score"
}, inplace=True)