import sqlite3
import pandas as pd

conn = sqlite3.connect(
    "database/recruitment.db"
)

query = """
SELECT *
FROM candidates
LIMIT 5
"""

df = pd.read_sql(
    query,
    conn
)

print(df)

conn.close()