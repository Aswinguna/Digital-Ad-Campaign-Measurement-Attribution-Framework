"""
src/models/evaluate.py
~~~~~~~~~~~~~~~~~~~~~~
Post-training evaluation helpers: calibration curves, lift charts,
feature importance, and a model comparison report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ── calibration curve ─────────────────────────────────────────────────────────

def plot_calibration(models: dict, X_test, y_test, save: bool = True):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, "s-", label=name)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "calibration_curves.png")
        fig.savefig(path, dpi=150)
        print(f"[evaluate] Saved → {path}")
    plt.close(fig)


# ── ROC curve ─────────────────────────────────────────────────────────────────

def plot_roc(models: dict, X_test, y_test, save: bool = True):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Conversion Prediction")
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "roc_curves.png")
        fig.savefig(path, dpi=150)
        print(f"[evaluate] Saved → {path}")
    plt.close(fig)


# ── Precision–Recall curve ────────────────────────────────────────────────────

def plot_pr(models: dict, X_test, y_test, save: bool = True):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves")
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "pr_curves.png")
        fig.savefig(path, dpi=150)
        print(f"[evaluate] Saved → {path}")
    plt.close(fig)


# ── gain / lift chart ─────────────────────────────────────────────────────────

def plot_lift(model, X_test, y_test, model_name: str = "XGBoost", save: bool = True):
    prob = model.predict_proba(X_test)[:, 1]
    df = pd.DataFrame({"prob": prob, "label": y_test})
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    df["cumulative_positives"] = df["label"].cumsum()
    df["cumulative_pct"]       = (df.index + 1) / len(df)
    df["lift"] = (df["cumulative_positives"] / (df.index + 1)) / df["label"].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["cumulative_pct"], df["lift"], label=model_name)
    ax.axhline(1.0, color="k", linestyle="--", label="Baseline")
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Lift")
    ax.set_title(f"Lift Chart – {model_name}")
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, f"lift_chart_{model_name.lower()}.png")
        fig.savefig(path, dpi=150)
        print(f"[evaluate] Saved → {path}")
    plt.close(fig)


# ── feature importance ─────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, top_n: int = 20,
                            model_name: str = "XGBoost", save: bool = True):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print(f"[evaluate] {model_name} has no feature_importances_; skipping.")
        return

    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [feature_names[i] for i in reversed(idx)],
        [importances[i] for i in reversed(idx)],
        color="#4C72B0",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{top_n} Feature Importances – {model_name}")
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, f"feature_importance_{model_name.lower()}.png")
        fig.savefig(path, dpi=150)
        print(f"[evaluate] Saved → {path}")
    plt.close(fig)


# ── summary table ─────────────────────────────────────────────────────────────

def comparison_table(models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        from sklearn.metrics import f1_score, brier_score_loss
        rows.append({
            "Model":   name,
            "ROC-AUC": round(roc_auc_score(y_test, prob), 4),
            "PR-AUC":  round(average_precision_score(y_test, prob), 4),
            "F1":      round(f1_score(y_test, pred, zero_division=0), 4),
            "Brier":   round(brier_score_loss(y_test, prob), 4),
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)


if __name__ == "__main__":
    print("[evaluate] Run via run_pipeline.py to access fitted models.")
