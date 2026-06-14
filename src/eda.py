import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

print("="*50)
print("PHASE 3 : EXPLORATORY DATA ANALYSIS")
print("="*50)

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# -----------------------------
# Hiring Decision Distribution
# -----------------------------

decision_col = None

for col in df.columns:
    if "decision" in col.lower():
        decision_col = col
        break

if decision_col:

    plt.figure(figsize=(6,4))

    df[decision_col].value_counts().plot(
        kind="bar"
    )

    plt.title("Hiring Decision Distribution")
    plt.tight_layout()

    plt.savefig(
        "visuals/hiring_distribution.png"
    )

    plt.close()

# -----------------------------
# Experience Distribution
# -----------------------------

experience_col = None

for col in df.columns:
    if "experience" in col.lower():
        experience_col = col
        break

if experience_col:

    plt.figure(figsize=(6,4))

    df[experience_col].hist(
        bins=10
    )

    plt.title("Experience Distribution")

    plt.tight_layout()

    plt.savefig(
        "visuals/experience_distribution.png"
    )

    plt.close()

# -----------------------------
# Correlation Heatmap
# -----------------------------

numeric_df = df.select_dtypes(
    include=["int64","float64"]
)

if len(numeric_df.columns) > 1:

    plt.figure(figsize=(8,6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="Blues"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        "visuals/correlation_heatmap.png"
    )

    plt.close()

# -----------------------------
# Skills WordCloud
# -----------------------------

skills_col = None

for col in df.columns:
    if "skill" in col.lower():
        skills_col = col
        break

if skills_col:

    text = " ".join(
        df[skills_col].astype(str)
    )

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate(text)

    plt.figure(figsize=(12,6))

    plt.imshow(wc)

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        "visuals/top_skills_wordcloud.png"
    )

    plt.close()

print("\nEDA Completed Successfully")