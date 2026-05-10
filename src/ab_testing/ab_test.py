"""
src/ab_testing/ab_test.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Pre/post and concurrent A/B test analysis for audience targeting strategies.

Metrics computed:
  - CTR   (Click-Through Rate)
  - CVR   (Conversion Rate)
  - ROAS  (Return On Ad Spend)
  - CPA   (Cost Per Acquisition)

Statistical tests:
  - Two-proportion z-test for CTR / CVR
  - Mann–Whitney U for ROAS / CPA (non-normal)
  - Bootstrap confidence intervals

Usage:
    python -m src.ab_testing.ab_test
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple


# ── KPI helpers ───────────────────────────────────────────────────────────────

def compute_kpis(group: pd.DataFrame) -> dict:
    n         = len(group)
    clicks    = group["clicked"].sum()
    convs     = group["converted"].sum()
    revenue   = group["order_value_usd"].sum()
    spend     = group["ad_spend_usd"].sum()

    return {
        "n_impressions": n,
        "n_clicks":      int(clicks),
        "n_conversions": int(convs),
        "CTR":           clicks / n if n else 0.0,
        "CVR":           convs / clicks if clicks else 0.0,
        "ROAS":          revenue / spend if spend else 0.0,
        "CPA":           spend / convs if convs else 0.0,
        "total_revenue": revenue,
        "total_spend":   spend,
    }


# ── two-proportion z-test ─────────────────────────────────────────────────────

def z_test_proportions(
    n1: int, x1: int, n2: int, x2: int, alpha: float = 0.05
) -> dict:
    """
    H0: p1 == p2
    Returns z-stat, p-value, uplift %, and significance flag.
    """
    p1 = x1 / n1 if n1 else 0
    p2 = x2 / n2 if n2 else 0
    p_pool = (x1 + x2) / (n1 + n2)

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z  = (p2 - p1) / se if se > 0 else 0.0
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "p_control":    round(p1, 6),
        "p_treatment":  round(p2, 6),
        "uplift_pct":   round((p2 - p1) / p1 * 100, 2) if p1 else None,
        "z_stat":       round(z, 4),
        "p_value":      round(p_val, 4),
        "significant":  p_val < alpha,
    }


# ── Mann–Whitney U ────────────────────────────────────────────────────────────

def mannwhitney_test(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "median_control":   round(float(np.median(a)), 4),
        "median_treatment": round(float(np.median(b)), 4),
        "uplift_pct":       round((np.median(b) - np.median(a)) / np.median(a) * 100, 2)
                            if np.median(a) != 0 else None,
        "u_stat":           round(float(stat), 2),
        "p_value":          round(float(p), 4),
        "significant":      p < alpha,
    }


# ── bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    arr: np.ndarray, stat_fn=np.mean,
    n_boot: int = 2000, ci: float = 0.95, seed: int = 42
) -> Tuple[float, float]:
    rng      = np.random.default_rng(seed)
    boot     = [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi   = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return round(float(lo), 6), round(float(hi), 6)


# ── pre/post A/B analysis ─────────────────────────────────────────────────────

def prepost_analysis(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Compare the 'pre' period (before campaign day 45) against
    'control' and 'treatment' groups in the post period.
    """
    pre       = df[df["ab_group"] == "pre"]
    control   = df[df["ab_group"] == "control"]
    treatment = df[df["ab_group"] == "treatment"]

    pre_kpis       = compute_kpis(pre)
    control_kpis   = compute_kpis(control)
    treatment_kpis = compute_kpis(treatment)

    # CTR tests
    ctr_pre_vs_ctrl = z_test_proportions(
        pre_kpis["n_impressions"], pre_kpis["n_clicks"],
        control_kpis["n_impressions"], control_kpis["n_clicks"],
        alpha=alpha
    )
    ctr_ctrl_vs_trt = z_test_proportions(
        control_kpis["n_impressions"], control_kpis["n_clicks"],
        treatment_kpis["n_impressions"], treatment_kpis["n_clicks"],
        alpha=alpha
    )

    # CVR tests (among clicked impressions)
    cvr_pre_vs_ctrl = z_test_proportions(
        pre_kpis["n_clicks"], pre_kpis["n_conversions"],
        control_kpis["n_clicks"], control_kpis["n_conversions"],
        alpha=alpha
    )
    cvr_ctrl_vs_trt = z_test_proportions(
        control_kpis["n_clicks"], control_kpis["n_conversions"],
        treatment_kpis["n_clicks"], treatment_kpis["n_conversions"],
        alpha=alpha
    )

    # ROAS Mann–Whitney
    roas_ctrl = df.loc[df["ab_group"] == "control",   "roas"].values
    roas_trt  = df.loc[df["ab_group"] == "treatment", "roas"].values
    roas_mw   = mannwhitney_test(roas_ctrl[roas_ctrl > 0], roas_trt[roas_trt > 0], alpha)

    # Bootstrap CI for CTR uplift (treatment vs control)
    ctr_ctrl_arr = df.loc[df["ab_group"] == "control",   "clicked"].values.astype(float)
    ctr_trt_arr  = df.loc[df["ab_group"] == "treatment", "clicked"].values.astype(float)
    ctr_ci_ctrl  = bootstrap_ci(ctr_ctrl_arr)
    ctr_ci_trt   = bootstrap_ci(ctr_trt_arr)

    return {
        "kpis": {
            "pre":       pre_kpis,
            "control":   control_kpis,
            "treatment": treatment_kpis,
        },
        "tests": {
            "ctr_pre_vs_control":      ctr_pre_vs_ctrl,
            "ctr_control_vs_treatment": ctr_ctrl_vs_trt,
            "cvr_pre_vs_control":      cvr_pre_vs_ctrl,
            "cvr_control_vs_treatment": cvr_ctrl_vs_trt,
            "roas_control_vs_treatment": roas_mw,
        },
        "bootstrap_ci": {
            "ctr_control_95ci":   ctr_ci_ctrl,
            "ctr_treatment_95ci": ctr_ci_trt,
        },
    }


# ── strategy-level comparison ─────────────────────────────────────────────────

def strategy_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """KPI table by targeting strategy × ab_group."""
    rows = []
    for (strategy, group), subset in df.groupby(["targeting_strategy", "ab_group"]):
        kpis = compute_kpis(subset)
        kpis["targeting_strategy"] = strategy
        kpis["ab_group"]           = group
        rows.append(kpis)
    return pd.DataFrame(rows).round(4)


# ── revenue uplift simulation ─────────────────────────────────────────────────

def simulate_uplift(
    df: pd.DataFrame,
    treatment_ctr_lift: float = 0.25,    # +25% CTR from model
    treatment_cvr_lift: float = 0.30,    # +30% CVR from model
    budget_usd: float = 50_000.0,
) -> dict:
    """
    Simulates the revenue uplift from rolling out the optimised treatment
    targeting strategy across the full budget.

    Returns a dict with baseline and projected revenue and ROAS.
    """
    baseline = compute_kpis(df[df["ab_group"] == "control"])
    baseline_revenue = baseline["total_revenue"]
    baseline_roas    = baseline["ROAS"]

    # apply lifts
    proj_ctr  = baseline["CTR"] * (1 + treatment_ctr_lift)
    proj_cvr  = baseline["CVR"] * (1 + treatment_cvr_lift)
    proj_conv = baseline["n_impressions"] * proj_ctr * proj_cvr
    avg_order = (
        baseline["total_revenue"] / baseline["n_conversions"]
        if baseline["n_conversions"] > 0 else 50.0
    )
    proj_revenue = proj_conv * avg_order
    proj_roas    = proj_revenue / baseline["total_spend"] if baseline["total_spend"] else 0

    uplift_pct = (proj_revenue - baseline_revenue) / baseline_revenue * 100 if baseline_revenue else 0

    return {
        "baseline_revenue": round(baseline_revenue, 2),
        "projected_revenue": round(proj_revenue, 2),
        "uplift_pct":        round(uplift_pct, 2),
        "baseline_roas":     round(baseline_roas, 2),
        "projected_roas":    round(proj_roas, 2),
        "budget_usd":        budget_usd,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_ab_report(results: dict) -> None:
    kpis  = results["kpis"]
    tests = results["tests"]

    print("\n" + "="*65)
    print("  A/B TEST REPORT – AUDIENCE TARGETING EXPERIMENT")
    print("="*65)

    for label, k in kpis.items():
        print(f"\n  [{label.upper()}]")
        print(f"    Impressions : {k['n_impressions']:>10,}")
        print(f"    Clicks      : {k['n_clicks']:>10,}  CTR={k['CTR']*100:.2f}%")
        print(f"    Conversions : {k['n_conversions']:>10,}  CVR={k['CVR']*100:.2f}%")
        print(f"    Revenue     : ${k['total_revenue']:>12,.2f}")
        print(f"    ROAS        : {k['ROAS']:>10.2f}x")

    print("\n  STATISTICAL TESTS")
    for test_name, t in tests.items():
        sig = "✓ SIGNIFICANT" if t.get("significant") else "✗ not significant"
        uplift = t.get("uplift_pct")
        uplift_str = f"uplift={uplift:+.1f}%" if uplift is not None else ""
        p = t.get("p_value", "—")
        print(f"  {test_name:<40} p={p:.4f}  {uplift_str:>14}  {sig}")
    print()


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.preprocessing import load_raw
    df = load_raw()
    results = prepost_analysis(df)
    print_ab_report(results)

    uplift = simulate_uplift(df)
    print(f"  Simulated revenue uplift: {uplift['uplift_pct']:+.1f}%")
    print(f"  Baseline ROAS : {uplift['baseline_roas']:.2f}x  →  Projected ROAS : {uplift['projected_roas']:.2f}x")
