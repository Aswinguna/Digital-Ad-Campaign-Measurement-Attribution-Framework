"""
run_pipeline.py
~~~~~~~~~~~~~~~
End-to-end pipeline runner for the Digital Ad Campaign Measurement &
Attribution Framework.

Steps:
  1. Generate synthetic data (200K+ ad events)
  2. Preprocess & build feature matrix
  3. Train XGBoost / RF / MLP with MLflow tracking
  4. Evaluate models & generate plots
  5. Run A/B test analysis
  6. Run multi-touch attribution
  7. Run budget optimisation
  8. Compute SHAP explanations

Usage:
    python run_pipeline.py [--skip-data] [--skip-shap]
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(__file__)
DATA_P  = os.path.join(ROOT, "data", "ad_events.parquet")
DATA_C  = os.path.join(ROOT, "data", "ad_events.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "models"),  exist_ok=True)

sys.path.insert(0, ROOT)


def banner(msg: str) -> None:
    print("\n" + "="*65)
    print(f"  {msg}")
    print("="*65)


def main(skip_data: bool = False, skip_shap: bool = False):

    # ── STEP 1: Data generation ───────────────────────────────────────────────
    banner("STEP 1 / 8 — Data Generation")
    if skip_data and (os.path.exists(DATA_P) or os.path.exists(DATA_C)):
        print("[pipeline] Skipping data generation (data found on disk).")
        from src.preprocessing import load_raw
        df_raw = load_raw()
    else:
        from data.generate_data import generate_ad_events, save_data
        df_raw = generate_ad_events()
        save_data(df_raw)

    # ── STEP 2: Preprocessing ─────────────────────────────────────────────────
    banner("STEP 2 / 8 — Preprocessing & Feature Engineering")
    from src.preprocessing import build_feature_matrix
    df_enc, encoders, scaler = build_feature_matrix(df=df_raw)
    print(f"[pipeline] Encoded feature matrix shape: {df_enc.shape}")

    # ── STEP 3: Model training ────────────────────────────────────────────────
    banner("STEP 3 / 8 — Model Training (XGBoost · RF · MLP) + MLflow")
    from src.models.train import train_all_models, prepare_xy
    models = train_all_models(df_enc)

    # ── STEP 4: Evaluation ────────────────────────────────────────────────────
    banner("STEP 4 / 8 — Model Evaluation & Plots")
    from src.models.evaluate import (
        plot_roc, plot_pr, plot_calibration, plot_lift,
        plot_feature_importance, comparison_table,
    )
    X, y = prepare_xy(df_enc)
    split_idx   = int(len(df_enc) * 0.80)
    X_test, y_test = X.iloc[split_idx:], y[split_idx:]

    plot_roc(models, X_test, y_test)
    plot_pr(models, X_test, y_test)
    plot_calibration(models, X_test, y_test)
    plot_lift(models["XGBoost"], X_test, y_test, model_name="XGBoost")
    plot_feature_importance(models["XGBoost"], list(X.columns), model_name="XGBoost")
    plot_feature_importance(models["RandomForest"], list(X.columns), model_name="RandomForest")

    comp = comparison_table(models, X_test, y_test)
    print("\n  Model comparison:\n")
    print(comp.to_string(index=False))
    comp.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)

    # ── STEP 5: A/B testing ───────────────────────────────────────────────────
    banner("STEP 5 / 8 — A/B Test Analysis")
    from src.ab_testing.ab_test import prepost_analysis, simulate_uplift, print_ab_report
    ab_results = prepost_analysis(df_raw)
    print_ab_report(ab_results)

    uplift = simulate_uplift(df_raw)
    print(f"  Simulated revenue uplift: {uplift['uplift_pct']:+.1f}%")
    print(f"  Baseline ROAS → {uplift['baseline_roas']:.2f}x | Projected → {uplift['projected_roas']:.2f}x")

    # ── STEP 6: Attribution ───────────────────────────────────────────────────
    banner("STEP 6 / 8 — Multi-Touch Attribution")
    from src.attribution.attribution import attribution_report
    attr_df = attribution_report(df_raw)
    attr_path = os.path.join(OUT_DIR, "attribution_summary.csv")
    attr_df.to_csv(attr_path, index=False)
    print(f"[pipeline] Attribution saved → {attr_path}")

    # ── STEP 7: Budget optimisation ───────────────────────────────────────────
    banner("STEP 7 / 8 — Budget Optimisation")
    from src.budget_optimization.optimizer import run_optimization
    opt = run_optimization(df_raw)
    greedy_path = os.path.join(OUT_DIR, "budget_allocation_greedy.csv")
    opt["greedy_allocation"].to_csv(greedy_path, index=False)
    print(f"[pipeline] Budget allocation saved → {greedy_path}")

    # ── STEP 8: SHAP ─────────────────────────────────────────────────────────
    if not skip_shap:
        banner("STEP 8 / 8 — SHAP Explainability")
        from src.explainability.shap_analysis import explain_model

        key_features = ["prior_clicks", "recency_days", "context_score",
                        "frequency", "log_bid"]
        dep_features = [f for f in key_features if f in X.columns]

        explain_model(
            models["XGBoost"], X,
            model_name="XGBoost",
            sample_size=3000,
            dependence_features=dep_features,
        )
        explain_model(
            models["RandomForest"], X,
            model_name="RandomForest",
            sample_size=2000,
        )
    else:
        print("[pipeline] SHAP step skipped (--skip-shap).")

    # ── Done ──────────────────────────────────────────────────────────────────
    banner("✓ Pipeline Complete")
    print(f"  Outputs  → {OUT_DIR}/")
    print(f"  MLflow   → mlflow ui  (to view experiment runs)")
    print(f"  Dashboard→ python dashboard/app.py")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full ad attribution pipeline.")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data generation if data files exist.")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP computation (slow for MLP).")
    args = parser.parse_args()
    main(skip_data=args.skip_data, skip_shap=args.skip_shap)
