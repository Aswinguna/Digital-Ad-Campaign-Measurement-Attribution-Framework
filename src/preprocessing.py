"""
src/preprocessing.py
~~~~~~~~~~~~~~~~~~~~
Loads the raw ad-event data, cleans it, and produces a model-ready
feature matrix.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ── constants ─────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "targeting_strategy", "ad_format", "device_type",
    "publisher_id", "vertical", "ab_group",
]

NUMERIC_COLS = [
    "recency_days", "frequency", "prior_clicks", "age_bracket",
    "cohort_size", "context_score", "bid_price_cpm",
    "hour_of_day", "day_of_week", "campaign_day",
]

TARGET_COL = "converted"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── loader ────────────────────────────────────────────────────────────────────

def load_raw(path: str | None = None) -> pd.DataFrame:
    """Load the parquet file (falls back to CSV)."""
    if path is None:
        parquet = os.path.join(DATA_DIR, "ad_events.parquet")
        csv     = os.path.join(DATA_DIR, "ad_events.csv")
        path    = parquet if os.path.exists(parquet) else csv

    ext = os.path.splitext(path)[-1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["timestamp"])

    print(f"[preprocessing] Loaded {len(df):,} rows from {path}")
    return df


# ── cleaning ──────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # drop duplicate event IDs
    df = df.drop_duplicates(subset="event_id")

    # fill any unexpected nulls
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("unknown")

    # remove rows where spend is zero or negative (data quality)
    df = df[df["ad_spend_usd"] > 0].copy()

    print(f"[preprocessing] After cleaning: {len(df):,} rows")
    return df


# ── feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # time-of-day buckets
    df["is_evening"]   = (df["hour_of_day"] >= 18).astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour_of_day"].between(12, 20).astype(int)

    # audience quality signals
    df["freq_x_prior"]      = df["frequency"] * df["prior_clicks"]
    df["log_cohort_size"]   = np.log1p(df["cohort_size"])
    df["recency_bucket"]    = pd.cut(
        df["recency_days"], bins=[0, 1, 7, 30, 60], labels=[3, 2, 1, 0]
    ).astype(int)

    # campaign phase flag
    df["is_post_ab"] = (df["campaign_day"] >= 45).astype(int)

    # normalised bid
    df["log_bid"] = np.log1p(df["bid_price_cpm"])

    return df


# ── encoding ──────────────────────────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    encoders: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    One-hot encode low-cardinality categoricals; label-encode publisher_id.
    Returns (encoded_df, encoders_dict).
    """
    df = df.copy()
    encoders = encoders or {}

    low_card = ["targeting_strategy", "ad_format", "device_type", "vertical", "ab_group"]
    df = pd.get_dummies(df, columns=low_card, drop_first=False)

    # label-encode publisher
    if "publisher_id" in df.columns:
        if fit:
            le = LabelEncoder()
            df["publisher_id"] = le.fit_transform(df["publisher_id"].astype(str))
            encoders["publisher_id"] = le
        else:
            le = encoders["publisher_id"]
            known = set(le.classes_)
            df["publisher_id"] = df["publisher_id"].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df["publisher_id"] = le.transform(df["publisher_id"])

    return df, encoders


# ── scaler ────────────────────────────────────────────────────────────────────

def scale_numerics(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = True,
    cols: list[str] | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    df = df.copy()
    cols = cols or [c for c in df.columns if df[c].dtype in [np.float64, np.int64]
                    and c not in {"event_id", "user_id", "converted", "clicked",
                                  "click_through", "view_through", "campaign_day"}]
    if fit:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])
    return df, scaler


# ── full pipeline ─────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame | None = None,
    path: str | None = None,
    encoders: dict | None = None,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict, StandardScaler]:
    """
    End-to-end: load → clean → engineer → encode → scale.
    Returns (X_df_with_target, encoders, scaler).
    """
    if df is None:
        df = load_raw(path)
    df = clean(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df, encoders=encoders, fit=fit)
    # don't scale target / id columns
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in {"event_id", "user_id", "converted", "clicked",
                             "click_through", "view_through"}]
    df, scaler = scale_numerics(df, scaler=scaler, fit=fit, cols=num_cols)
    return df, encoders, scaler
