import os
import pandas as pd

print("=" * 50)
print("AI Recruitment Prediction Project")
print("=" * 50)

# List files in raw folder
files = os.listdir("data/raw")

print("\nFiles Found:")
print(files)

# Read first CSV file
csv_file = [file for file in files if file.endswith(".csv")][0]

file_path = os.path.join("data/raw", csv_file)

df = pd.read_csv(file_path)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nFirst 5 Rows:")
print(df.head())