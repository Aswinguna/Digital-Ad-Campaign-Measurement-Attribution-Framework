"""
generate_data.py
~~~~~~~~~~~~~~~~
Generates a synthetic but realistic ad event dataset with 200 000+ records
covering impressions, clicks, and conversions across three targeting strategies:
  - Addressable  (user-level / cookie-based)
  - Cohort-based (aggregated audience segments)
  - Contextual   (page-content signals, no user identity)

Run:
    python data/generate_data.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
N_EVENTS = 210_000          # total impression events
OUTPUT_DIR = os.path.join(os.path.dirname(__file__))


# ── helpers ──────────────────────────────────────────────────────────────────

def _logistic(x):
    return 1 / (1 + np.exp(-x))


def _clamp(arr, lo=0.0, hi=1.0):
    return np.clip(arr, lo, hi)


# ── main generator ────────────────────────────────────────────────────────────

def generate_ad_events(n: int = N_EVENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    print(f"[generate_data] Generating {n:,} ad event records …")

    # ── time range: 90-day campaign (pre/post split at day 45) ───────────────
    start_date = datetime(2024, 1, 1)
    timestamps = [
        start_date + timedelta(
            days=float(rng.uniform(0, 90)),
            hours=float(rng.uniform(0, 24))
        )
        for _ in range(n)
    ]
    timestamps = sorted(timestamps)

    # ── campaign / targeting metadata ────────────────────────────────────────
    targeting_strategies = rng.choice(
        ["addressable", "cohort", "contextual"],
        size=n,
        p=[0.45, 0.30, 0.25]
    )

    ad_formats = rng.choice(
        ["display_banner", "video_pre_roll", "native", "carousel", "interstitial"],
        size=n,
        p=[0.30, 0.20, 0.20, 0.15, 0.15]
    )

    devices = rng.choice(
        ["mobile", "desktop", "tablet", "ctv"],
        size=n,
        p=[0.50, 0.35, 0.10, 0.05]
    )

    publishers = rng.choice(
        [f"pub_{i:03d}" for i in range(1, 41)],
        size=n
    )

    verticals = rng.choice(
        ["e-commerce", "travel", "finance", "automotive", "fashion", "tech"],
        size=n,
        p=[0.30, 0.15, 0.20, 0.10, 0.15, 0.10]
    )

    hour_of_day = np.array([t.hour for t in timestamps])
    day_of_week = np.array([t.weekday() for t in timestamps])
    campaign_day = np.array([(t - start_date).days for t in timestamps])

    # ── user / cohort features ────────────────────────────────────────────────
    user_ids = rng.integers(1, 60_001, size=n)          # ~60 k unique users

    # recency: days since last site visit (lower → warmer audience)
    recency_days = rng.exponential(scale=7, size=n).astype(int).clip(0, 60)

    # frequency: number of prior impressions in the last 30 days
    frequency = rng.integers(1, 16, size=n)

    # prior_clicks: number of prior ad clicks (signals intent)
    prior_clicks = rng.integers(0, 6, size=n)

    # cohort_size: size of the audience cohort (smaller → more targeted)
    cohort_sizes = rng.integers(500, 50_001, size=n)

    # bid_price (CPM in USD)
    bid_price = rng.uniform(0.5, 15.0, size=n).round(3)

    # page context score (0–1: how relevant the page content is to the ad)
    context_score = rng.beta(2, 5, size=n).round(4)

    # user age bracket (ordinal: 0=18-24 … 4=55+)
    age_bracket = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.18, 0.27, 0.25, 0.18, 0.12])

    # ── A/B test assignment ───────────────────────────────────────────────────
    # Test starts at campaign day 45 (post-period)
    # Group A = control (broad targeting), Group B = treatment (optimised targeting)
    ab_group = np.where(campaign_day < 45, "pre", rng.choice(["control", "treatment"], size=n))

    # ── CTR model ─────────────────────────────────────────────────────────────
    ctr_logit = (
        -3.5
        + 0.4  * (targeting_strategies == "addressable").astype(float)
        + 0.2  * (targeting_strategies == "cohort").astype(float)
        - 0.05 * recency_days / 10
        + 0.3  * prior_clicks
        + 0.15 * context_score
        + 0.1  * (ad_formats == "video_pre_roll").astype(float)
        + 0.08 * (ad_formats == "carousel").astype(float)
        - 0.1  * (devices == "tablet").astype(float)
        + 0.05 * (hour_of_day >= 18).astype(float)       # evening lift
        - 0.05 * (day_of_week >= 5).astype(float)        # weekend dip
        + 0.25 * (ab_group == "treatment").astype(float) # A/B treatment lift
        + rng.normal(0, 0.3, size=n)
    )
    ctr_prob = _clamp(_logistic(ctr_logit), 0.001, 0.15)
    clicked = rng.binomial(1, ctr_prob).astype(bool)

    # ── CVR model (conditional on click) ─────────────────────────────────────
    cvr_logit = (
        -4.0
        + 0.6  * (targeting_strategies == "addressable").astype(float)
        + 0.3  * (targeting_strategies == "cohort").astype(float)
        + 0.5  * prior_clicks
        - 0.04 * recency_days / 10
        - 0.03 * np.log1p(cohort_sizes) / 10
        + 0.2  * context_score
        + 0.15 * (verticals == "e-commerce").astype(float)
        + 0.1  * (verticals == "finance").astype(float)
        + 0.3  * (ab_group == "treatment").astype(float)
        + rng.normal(0, 0.4, size=n)
    )
    cvr_prob = _clamp(_logistic(cvr_logit), 0.001, 0.40)
    converted = np.where(clicked, rng.binomial(1, cvr_prob).astype(bool), False)

    # ── revenue / ROAS ────────────────────────────────────────────────────────
    order_value = np.where(
        converted,
        rng.lognormal(mean=4.0, sigma=0.8, size=n).round(2),
        0.0
    )
    ad_spend = (bid_price / 1000).round(4)     # CPM → cost per impression
    roas = np.where(ad_spend > 0, order_value / ad_spend, 0.0).round(2)

    # ── attribution windows ───────────────────────────────────────────────────
    view_through = converted & ~clicked        # converted without clicking
    click_through = converted & clicked        # converted after clicking

    # ── assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "event_id":            range(1, n + 1),
        "timestamp":           timestamps,
        "campaign_day":        campaign_day,
        "user_id":             user_ids,
        "ab_group":            ab_group,

        # targeting
        "targeting_strategy":  targeting_strategies,
        "ad_format":           ad_formats,
        "device_type":         devices,
        "publisher_id":        publishers,
        "vertical":            verticals,

        # user features
        "recency_days":        recency_days,
        "frequency":           frequency,
        "prior_clicks":        prior_clicks,
        "age_bracket":         age_bracket,
        "cohort_size":         cohort_sizes,
        "context_score":       context_score.round(4),

        # pricing
        "bid_price_cpm":       bid_price,
        "ad_spend_usd":        ad_spend,

        # outcomes
        "ctr_prob":            ctr_prob.round(4),
        "clicked":             clicked.astype(int),
        "cvr_prob":            cvr_prob.round(4),
        "converted":           converted.astype(int),
        "order_value_usd":     order_value,
        "roas":                roas,

        # attribution
        "click_through":       click_through.astype(int),
        "view_through":        view_through.astype(int),

        "hour_of_day":         hour_of_day,
        "day_of_week":         day_of_week,
    })

    return df


def save_data(df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> None:
    csv_path = os.path.join(out_dir, "ad_events.csv")
    parquet_path = os.path.join(out_dir, "ad_events.parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"[generate_data] Saved {len(df):,} rows → {csv_path}")
    print(f"[generate_data] Saved {len(df):,} rows → {parquet_path}")
    _print_summary(df)


def _print_summary(df: pd.DataFrame) -> None:
    print("\n── Dataset summary ─────────────────────────────────────────────")
    print(f"  Total events      : {len(df):>10,}")
    print(f"  Total clicks      : {df['clicked'].sum():>10,}  ({df['clicked'].mean()*100:.2f}%)")
    print(f"  Total conversions : {df['converted'].sum():>10,}  ({df['converted'].mean()*100:.2f}%)")
    print(f"  Total revenue     : ${df['order_value_usd'].sum():>12,.2f}")
    print(f"  Total ad spend    : ${df['ad_spend_usd'].sum():>12,.2f}")
    print(f"  Overall ROAS      : {df['order_value_usd'].sum()/df['ad_spend_usd'].sum():.2f}x")
    print(f"  Date range        : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print("────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    df = generate_ad_events()
    save_data(df)
