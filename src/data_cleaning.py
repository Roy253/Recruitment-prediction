import os
import pandas as pd

print("="*50)
print("PHASE 2 : DATA CLEANING")
print("="*50)

# Load Dataset
files = os.listdir("data/raw")
csv_file = [f for f in files if f.endswith(".csv")][0]

df = pd.read_csv(
    os.path.join("data/raw", csv_file)
)

print("\nOriginal Shape:", df.shape)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Create Data Quality Report
quality_report = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values,
    "Data Type": df.dtypes.values
})

quality_report.to_csv(
    "data/processed/data_quality_report.csv",
    index=False
)

# Remove Duplicates
df = df.drop_duplicates()

# Standardize Column Names
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

# Fill Missing Values
text_columns = df.select_dtypes(include="object").columns
numeric_columns = df.select_dtypes(include=["int64","float64"]).columns

for col in text_columns:
    df[col] = df[col].fillna("Unknown")

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Remove Spaces
for col in text_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
    )

print("\nFinal Shape:", df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum().sum())

# Save Clean Dataset
df.to_csv(
    "data/processed/cleaned_data.csv",
    index=False
)

print("\nPhase 2 Completed Successfully")