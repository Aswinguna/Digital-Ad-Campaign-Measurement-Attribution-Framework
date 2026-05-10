"""
sql/load_to_sqlite.py
~~~~~~~~~~~~~~~~~~~~~
Loads the generated ad_events parquet/CSV into a local SQLite database
so the SQL queries in sql/queries.sql can be run directly.

Usage:
    python sql/load_to_sqlite.py
Then open with:
    sqlite3 data/ad_events.db
    .read sql/queries.sql
"""

import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PARQUET  = os.path.join(BASE_DIR, "data", "ad_events.parquet")
CSV      = os.path.join(BASE_DIR, "data", "ad_events.csv")
DB_PATH  = os.path.join(BASE_DIR, "data", "ad_events.db")


def load_to_sqlite():
    if os.path.exists(PARQUET):
        df = pd.read_parquet(PARQUET)
    elif os.path.exists(CSV):
        df = pd.read_csv(CSV, parse_dates=["timestamp"])
    else:
        raise FileNotFoundError(
            "No data found. Run `python data/generate_data.py` first."
        )

    print(f"[sqlite] Loading {len(df):,} rows into {DB_PATH} …")
    con = sqlite3.connect(DB_PATH)
    df.to_sql("ad_events", con, if_exists="replace", index=False, chunksize=10_000)
    con.execute("CREATE INDEX IF NOT EXISTS idx_user ON ad_events(user_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ts   ON ad_events(timestamp)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_strat ON ad_events(targeting_strategy)")
    con.commit()

    # quick sanity check
    count = pd.read_sql("SELECT COUNT(*) AS n FROM ad_events", con).iloc[0, 0]
    print(f"[sqlite] Verified: {count:,} rows in ad_events table.")
    con.close()
    print(f"[sqlite] Done. Run:  sqlite3 {DB_PATH}  then  .read sql/queries.sql")


if __name__ == "__main__":
    load_to_sqlite()
