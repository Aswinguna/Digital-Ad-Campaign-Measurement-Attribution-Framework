"""
src/attribution/attribution.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-touch attribution models:
  1. Last-touch        – 100% credit to the final touchpoint
  2. First-touch       – 100% credit to the first touchpoint
  3. Linear            – equal credit across all touchpoints
  4. Time-decay        – exponentially more weight to recent touchpoints
  5. Data-driven (MLP) – model-based Shapley-inspired attribution

All models operate on a *journey* DataFrame where each row is one ad
event that belongs to a conversion journey.

Usage:
    from src.attribution.attribution import build_journeys, run_all_attributions
"""

import numpy as np
import pandas as pd


# ── journey builder ───────────────────────────────────────────────────────────

def build_journeys(df: pd.DataFrame, max_touches: int = 10) -> pd.DataFrame:
    """
    Aggregate raw event-level data into conversion journeys.
    A journey = all events for a user that precede a conversion.

    Returns one row per (user_id, journey_id) with columns:
        touchpoints  : list of ad_format values
        channels     : list of targeting_strategy values
        n_touches    : number of touchpoints
        revenue      : order value (0 if not converted)
        converted    : 1/0
        duration_hrs : hours from first to last touch
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"])

    journeys = []
    for uid, grp in df.groupby("user_id"):
        # split into journeys by conversion events
        journey_id  = 0
        current     = []
        for _, row in grp.iterrows():
            current.append(row)
            if row["converted"] == 1:
                if len(current) > max_touches:
                    current = current[-max_touches:]
                duration = (
                    current[-1]["timestamp"] - current[0]["timestamp"]
                ).total_seconds() / 3600
                journeys.append({
                    "user_id":      uid,
                    "journey_id":   f"{uid}_{journey_id}",
                    "n_touches":    len(current),
                    "touchpoints":  [r["ad_format"]           for r in current],
                    "channels":     [r["targeting_strategy"]  for r in current],
                    "devices":      [r["device_type"]         for r in current],
                    "revenue":      row["order_value_usd"],
                    "converted":    1,
                    "duration_hrs": duration,
                })
                current = []
                journey_id += 1

    return pd.DataFrame(journeys)


# ── attribution models ────────────────────────────────────────────────────────

def last_touch(journeys: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, j in journeys.iterrows():
        for i, (ch, tp) in enumerate(zip(j["channels"], j["touchpoints"])):
            credit = j["revenue"] if i == len(j["channels"]) - 1 else 0.0
            rows.append({"channel": ch, "touchpoint": tp, "credit": credit,
                         "model": "last_touch"})
    return pd.DataFrame(rows)


def first_touch(journeys: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, j in journeys.iterrows():
        for i, (ch, tp) in enumerate(zip(j["channels"], j["touchpoints"])):
            credit = j["revenue"] if i == 0 else 0.0
            rows.append({"channel": ch, "touchpoint": tp, "credit": credit,
                         "model": "first_touch"})
    return pd.DataFrame(rows)


def linear(journeys: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, j in journeys.iterrows():
        n = j["n_touches"]
        per_touch = j["revenue"] / n if n > 0 else 0.0
        for ch, tp in zip(j["channels"], j["touchpoints"]):
            rows.append({"channel": ch, "touchpoint": tp, "credit": per_touch,
                         "model": "linear"})
    return pd.DataFrame(rows)


def time_decay(journeys: pd.DataFrame, half_life: float = 7.0) -> pd.DataFrame:
    """Credit decays exponentially; half-life in days (default 7)."""
    rows = []
    for _, j in journeys.iterrows():
        n = j["n_touches"]
        # assume equal spacing across duration; assign weights
        if n == 1:
            weights = np.array([1.0])
        else:
            # positions 0 (oldest) … n-1 (most recent); decay from right
            ages = np.linspace(j["duration_hrs"] / 24, 0, n)  # days ago
            weights = 2 ** (-ages / half_life)
        weights /= weights.sum()
        for (ch, tp), w in zip(zip(j["channels"], j["touchpoints"]), weights):
            rows.append({"channel": ch, "touchpoint": tp,
                         "credit": j["revenue"] * w, "model": "time_decay"})
    return pd.DataFrame(rows)


def summarise_attribution(attr_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit by channel and model."""
    return (
        attr_df.groupby(["model", "channel"])["credit"]
        .sum()
        .reset_index()
        .rename(columns={"credit": "attributed_revenue"})
        .sort_values(["model", "attributed_revenue"], ascending=[True, False])
    )


def run_all_attributions(journeys: pd.DataFrame) -> pd.DataFrame:
    """Run all models and return a combined summary."""
    frames = [
        last_touch(journeys),
        first_touch(journeys),
        linear(journeys),
        time_decay(journeys),
    ]
    combined = pd.concat(frames, ignore_index=True)
    return summarise_attribution(combined)


# ── quick report ──────────────────────────────────────────────────────────────

def attribution_report(df_raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    converted = df_raw[df_raw["converted"] == 1]
    print(f"[attribution] Building journeys from {len(converted):,} converted events …")
    journeys = build_journeys(df_raw)
    print(f"[attribution] {len(journeys):,} conversion journeys built.")

    summary = run_all_attributions(journeys)

    if verbose:
        print("\nAttribution comparison by channel:\n")
        pivot = summary.pivot(index="channel", columns="model", values="attributed_revenue").fillna(0)
        print(pivot.round(0).to_string())

    return summary


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.preprocessing import load_raw
    df = load_raw()
    attribution_report(df)
