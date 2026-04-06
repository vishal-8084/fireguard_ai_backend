import sqlite3

conn = sqlite3.connect("database/alerts.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    label TEXT,
    severity INTEGER,
    snapshot_path TEXT
)
""")

conn.commit()
conn.close()

print("✔ SQLite Database & alerts table created successfully.")
