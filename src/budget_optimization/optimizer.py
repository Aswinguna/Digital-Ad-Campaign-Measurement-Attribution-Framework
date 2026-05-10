"""
src/budget_optimization/optimizer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Budget allocation optimizer using model-predicted ROAS per channel.

Two approaches:
  1. Greedy allocation  – rank channels by predicted ROAS, allocate greedily
  2. Scipy-based LP     – maximize expected revenue subject to budget constraint

The optimizer simulates the 12% revenue uplift claimed in the resume by
re-allocating budget from low-ROAS to high-ROAS targeting strategies.

Usage:
    from src.budget_optimization.optimizer import run_optimization
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
import warnings
warnings.filterwarnings("ignore")


# ── channel-level ROAS summary ────────────────────────────────────────────────

def channel_roas_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute observed ROAS, CTR, CVR and spend per (targeting_strategy, ad_format).
    """
    grp = df.groupby(["targeting_strategy", "ad_format"]).agg(
        impressions   = ("event_id",         "count"),
        clicks        = ("clicked",           "sum"),
        conversions   = ("converted",         "sum"),
        revenue       = ("order_value_usd",   "sum"),
        spend         = ("ad_spend_usd",      "sum"),
    ).reset_index()

    grp["CTR"]  = grp["clicks"]      / grp["impressions"]
    grp["CVR"]  = grp["conversions"] / grp["clicks"].clip(lower=1)
    grp["ROAS"] = grp["revenue"]     / grp["spend"].clip(lower=1e-6)
    grp["CPA"]  = grp["spend"]       / grp["conversions"].clip(lower=1)

    return grp.sort_values("ROAS", ascending=False).round(4)


# ── predicted ROAS helper ─────────────────────────────────────────────────────

def predict_channel_roas(model, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Use the fitted model to predict conversion probability per row, then
    aggregate to channel-level predicted ROAS.
    """
    df = df.copy()
    X = df[feature_cols]
    df["pred_cvr"] = model.predict_proba(X)[:, 1]

    # predicted revenue = pred_cvr × avg order value per channel
    avg_order = (
        df[df["converted"] == 1]
        .groupby("targeting_strategy")["order_value_usd"]
        .mean()
        .rename("avg_order_value")
    )
    df = df.merge(avg_order, on="targeting_strategy", how="left")
    df["avg_order_value"] = df["avg_order_value"].fillna(df["order_value_usd"].mean())
    df["pred_revenue"] = df["pred_cvr"] * df["avg_order_value"]

    summary = df.groupby("targeting_strategy").agg(
        pred_revenue = ("pred_revenue", "sum"),
        actual_spend = ("ad_spend_usd", "sum"),
        n            = ("event_id",     "count"),
    ).reset_index()
    summary["pred_ROAS"] = summary["pred_revenue"] / summary["actual_spend"].clip(lower=1e-6)
    return summary.sort_values("pred_ROAS", ascending=False)


# ── greedy budget allocator ────────────────────────────────────────────────────

def greedy_allocation(
    channel_stats: pd.DataFrame,
    total_budget: float,
    roas_col: str = "ROAS",
    channel_col: str = "targeting_strategy",
    min_pct: float = 0.05,      # minimum 5% per channel
) -> pd.DataFrame:
    """
    Allocate budget proportionally to ROAS, with a minimum floor per channel.
    Returns a DataFrame with allocated_budget and projected_revenue per channel.
    """
    channels = channel_stats[[channel_col, roas_col]].copy()
    channels = channels.groupby(channel_col)[roas_col].mean().reset_index()

    n = len(channels)
    min_budget = total_budget * min_pct
    remaining  = total_budget - min_budget * n

    total_roas = channels[roas_col].sum()
    channels["alloc_extra"] = channels[roas_col] / total_roas * remaining
    channels["allocated_budget"] = min_budget + channels["alloc_extra"]

    channels["projected_revenue"]  = channels["allocated_budget"] * channels[roas_col]
    channels["projected_ROAS"]     = channels[roas_col]
    channels = channels.drop(columns=["alloc_extra"])
    return channels.sort_values("allocated_budget", ascending=False).round(2)


# ── scipy nonlinear optimizer ─────────────────────────────────────────────────

def scipy_optimize(
    channel_stats: pd.DataFrame,
    total_budget: float,
    roas_col: str = "ROAS",
    channel_col: str = "targeting_strategy",
    min_pct: float = 0.05,
    max_pct: float = 0.70,
) -> pd.DataFrame:
    """
    Maximise expected revenue:
        max  Σ  ROAS_i × budget_i
        s.t. Σ  budget_i = total_budget
             min_pct × B ≤ budget_i ≤ max_pct × B  ∀i
    """
    channels = channel_stats.groupby(channel_col)[roas_col].mean().reset_index()
    roas_arr = channels[roas_col].values
    n        = len(roas_arr)

    # minimise negative revenue
    obj    = lambda x: -np.dot(roas_arr, x)
    grad   = lambda x: -roas_arr

    bounds      = [(total_budget * min_pct, total_budget * max_pct)] * n
    constraints = {"type": "eq", "fun": lambda x: x.sum() - total_budget}
    x0          = np.full(n, total_budget / n)

    res = minimize(obj, x0, jac=grad, bounds=bounds, constraints=constraints,
                   method="SLSQP")

    channels["allocated_budget"]   = res.x.round(2)
    channels["projected_revenue"]  = (res.x * roas_arr).round(2)
    channels["projected_ROAS"]     = roas_arr.round(4)
    return channels.sort_values("allocated_budget", ascending=False)


# ── uplift comparison ─────────────────────────────────────────────────────────

def compute_uplift(
    df: pd.DataFrame,
    total_budget: float = 50_000.0,
) -> dict:
    """
    Compare:
      - Current (even-split) allocation
      - Greedy optimised allocation
      - Scipy optimised allocation

    Returns a dict with baseline and optimised revenue + uplift %.
    """
    stats = channel_roas_summary(df)

    # baseline: even split
    n_channels     = stats["targeting_strategy"].nunique()
    even_budget    = total_budget / n_channels
    channel_roas   = stats.groupby("targeting_strategy")["ROAS"].mean()
    baseline_rev   = (channel_roas * even_budget).sum()

    greedy = greedy_allocation(stats, total_budget)
    greedy_rev = greedy["projected_revenue"].sum()

    opt    = scipy_optimize(stats, total_budget)
    opt_rev = opt["projected_revenue"].sum()

    return {
        "baseline_revenue":  round(baseline_rev,  2),
        "greedy_revenue":    round(greedy_rev,     2),
        "optimized_revenue": round(opt_rev,        2),
        "greedy_uplift_pct": round((greedy_rev - baseline_rev) / baseline_rev * 100, 2),
        "opt_uplift_pct":    round((opt_rev    - baseline_rev) / baseline_rev * 100, 2),
        "greedy_allocation": greedy,
        "scipy_allocation":  opt,
        "channel_stats":     stats,
    }


# ── full optimization run ─────────────────────────────────────────────────────

def run_optimization(df: pd.DataFrame, total_budget: float = 50_000.0) -> dict:
    print(f"\n[optimizer] Running budget optimization (budget=${total_budget:,.0f}) …")
    results = compute_uplift(df, total_budget=total_budget)

    print(f"\n  Baseline revenue (even split) : ${results['baseline_revenue']:>12,.2f}")
    print(f"  Greedy optimised revenue      : ${results['greedy_revenue']:>12,.2f}"
          f"  ({results['greedy_uplift_pct']:+.1f}%)")
    print(f"  Scipy optimised revenue       : ${results['optimized_revenue']:>12,.2f}"
          f"  ({results['opt_uplift_pct']:+.1f}%)")
    print()
    print("  Greedy allocation:")
    print(results["greedy_allocation"][["targeting_strategy", "ROAS",
                                        "allocated_budget", "projected_revenue"]].to_string(index=False))
    return results


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.preprocessing import load_raw
    df = load_raw()
    run_optimization(df)
