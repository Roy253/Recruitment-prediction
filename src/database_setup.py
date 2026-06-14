import sqlite3
import pandas as pd

print("="*50)
print("PHASE 5 : SQLITE DATABASE")
print("="*50)

# Load Feature Dataset
df = pd.read_csv(
    "data/processed/feature_data.csv"
)

print("\nDataset Loaded")
print("Rows:", len(df))

# Create Database Connection
conn = sqlite3.connect(
    "database/recruitment.db"
)

# Store Data
df.to_sql(
    "candidates",
    conn,
    if_exists="replace",
    index=False
)

print("\nTable Created Successfully")

# Verify Records
cursor = conn.cursor()

cursor.execute(
    "SELECT COUNT(*) FROM candidates"
)

count = cursor.fetchone()[0]

print(f"\nTotal Records Stored: {count}")

conn.commit()
conn.close()

print("\nDatabase Created Successfully")