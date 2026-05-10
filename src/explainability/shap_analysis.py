"""
src/explainability/shap_analysis.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SHAP-based model explainability for the conversion prediction models.
Generates summary plots, waterfall charts, and dependency plots.

Usage:
    from src.explainability.shap_analysis import explain_model
"""

import os
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ── SHAP explainer factory ────────────────────────────────────────────────────

def get_explainer(model, X_background: pd.DataFrame):
    """
    Return the appropriate SHAP explainer:
      - TreeExplainer  for XGBoost / RF
      - KernelExplainer for MLP (slower; uses a small background sample)
    """
    model_type = type(model).__name__
    if model_type in ("XGBClassifier", "RandomForestClassifier"):
        return shap.TreeExplainer(model)
    else:
        bg = shap.sample(X_background, 100)
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
        return shap.KernelExplainer(predict_fn, bg)


# ── compute SHAP values ───────────────────────────────────────────────────────

def compute_shap_values(
    model,
    X: pd.DataFrame,
    sample_size: int = 3000,
    seed: int = 42,
) -> shap.Explanation:
    """
    Compute SHAP values for a random sample of X.
    Returns a shap.Explanation object.
    """
    rng = np.random.default_rng(seed)
    if len(X) > sample_size:
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[idx].reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    explainer   = get_explainer(model, X_sample)
    shap_values = explainer(X_sample)
    return shap_values


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_summary(
    shap_values: shap.Explanation,
    model_name: str = "XGBoost",
    max_display: int = 20,
    save: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, show=False, max_display=max_display, plot_type="dot")
    plt.title(f"SHAP Feature Importance – {model_name}", fontsize=13)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, f"shap_summary_{model_name.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[shap] Saved → {path}")
    plt.close()


def plot_bar_importance(
    shap_values: shap.Explanation,
    model_name: str = "XGBoost",
    max_display: int = 20,
    save: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, show=False, max_display=max_display, plot_type="bar")
    plt.title(f"SHAP Mean |value| – {model_name}", fontsize=13)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, f"shap_bar_{model_name.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[shap] Saved → {path}")
    plt.close()


def plot_waterfall(
    shap_values: shap.Explanation,
    sample_idx: int = 0,
    model_name: str = "XGBoost",
    save: bool = True,
) -> None:
    """Waterfall plot for a single prediction (stakeholder explainability)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(shap_values[sample_idx], show=False)
    plt.title(f"SHAP Waterfall – Single Prediction ({model_name})", fontsize=13)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, f"shap_waterfall_{model_name.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[shap] Saved → {path}")
    plt.close()


def plot_dependence(
    shap_values: shap.Explanation,
    feature: str,
    model_name: str = "XGBoost",
    save: bool = True,
) -> None:
    """Dependence plot: feature value vs SHAP value (interaction-coloured)."""
    if feature not in shap_values.feature_names:
        print(f"[shap] Feature '{feature}' not found; skipping dependence plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(feature, shap_values.values, shap_values.data,
                         feature_names=shap_values.feature_names, show=False, ax=ax)
    plt.title(f"SHAP Dependence: {feature} ({model_name})", fontsize=12)
    plt.tight_layout()
    if save:
        fname = feature.replace(" ", "_")[:30]
        path  = os.path.join(FIG_DIR, f"shap_dep_{fname}_{model_name.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[shap] Saved → {path}")
    plt.close()


# ── top features table ────────────────────────────────────────────────────────

def top_features_table(shap_values: shap.Explanation, top_n: int = 15) -> pd.DataFrame:
    """Return a DataFrame of mean |SHAP| per feature, ranked."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": shap_values.feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
        .round(6)
    )


# ── full explain pipeline ─────────────────────────────────────────────────────

def explain_model(
    model,
    X: pd.DataFrame,
    model_name: str = "XGBoost",
    sample_size: int = 3000,
    dependence_features: list[str] | None = None,
) -> dict:
    """
    Run the full SHAP explainability pipeline for one model.
    Saves all plots and returns the shap_values and top features table.
    """
    print(f"\n[shap] Computing SHAP values for {model_name} (n={min(sample_size, len(X))}) …")
    sv = compute_shap_values(model, X, sample_size=sample_size)

    plot_summary(sv, model_name=model_name)
    plot_bar_importance(sv, model_name=model_name)
    plot_waterfall(sv, model_name=model_name)

    dep_features = dependence_features or []
    for feat in dep_features:
        plot_dependence(sv, feat, model_name=model_name)

    table = top_features_table(sv)
    print(f"\n  Top features for {model_name}:")
    print(table.head(10).to_string(index=False))

    return {"shap_values": sv, "top_features": table}


if __name__ == "__main__":
    print("[shap] Run via run_pipeline.py to access fitted models and feature matrix.")
