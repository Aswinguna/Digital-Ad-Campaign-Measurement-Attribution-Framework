"""
src/models/train.py
~~~~~~~~~~~~~~~~~~~
Trains XGBoost, Random Forest, and MLP classifiers to predict ad conversion
probability.  Every experiment is tracked with MLflow.

Usage:
    python -m src.models.train
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, classification_report, brier_score_loss,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

MLFLOW_EXPERIMENT = "ad-conversion-prediction"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── feature columns (after preprocessing) ─────────────────────────────────────
LABEL_COL = "converted"
DROP_COLS  = ["event_id", "user_id", "timestamp", "converted", "clicked",
              "click_through", "view_through", "ctr_prob", "cvr_prob",
              "order_value_usd", "roas", "ad_spend_usd"]


# ── model configs ─────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "XGBoost": {
        "cls": XGBClassifier,
        "params": {
            "n_estimators":     400,
            "max_depth":        6,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 9,       # imbalance correction
            "eval_metric":      "logloss",
            "use_label_encoder": False,
            "random_state":     42,
            "n_jobs":           -1,
        },
    },
    "RandomForest": {
        "cls": RandomForestClassifier,
        "params": {
            "n_estimators":  300,
            "max_depth":     12,
            "class_weight":  "balanced",
            "random_state":  42,
            "n_jobs":        -1,
        },
    },
    "MLP": {
        "cls": MLPClassifier,
        "params": {
            "hidden_layer_sizes": (256, 128, 64),
            "activation":         "relu",
            "solver":             "adam",
            "alpha":              1e-3,
            "batch_size":         512,
            "max_iter":           200,
            "early_stopping":     True,
            "random_state":       42,
        },
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def prepare_xy(df: pd.DataFrame):
    """Split into feature matrix X and target y."""
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop)
    # ensure all bool → int
    X = X.astype({c: int for c in X.select_dtypes("bool").columns})
    y = df[LABEL_COL].values
    return X, y


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """Return a dict of classification metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc":    round(average_precision_score(y_test, y_prob), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "brier":     round(brier_score_loss(y_test, y_prob), 4),
        "n_pos":     int(y_test.sum()),
        "n_total":   int(len(y_test)),
    }


# ── training loop ─────────────────────────────────────────────────────────────

def train_all_models(
    df: pd.DataFrame,
    targeting_filter: str | None = None,
    experiment_name: str = MLFLOW_EXPERIMENT,
) -> dict:
    """
    Train XGBoost, RF, and MLP models. Optionally filter by targeting strategy.
    Returns a dict of {model_name: fitted_model}.
    """
    mlflow.set_experiment(experiment_name)

    if targeting_filter:
        col = "targeting_strategy" if "targeting_strategy" in df.columns else None
        if col:
            # after one-hot encoding the column is gone; check raw
            pass
        print(f"[train] Filtering to targeting strategy: {targeting_filter}")

    X, y = prepare_xy(df)
    print(f"[train] Feature matrix: {X.shape}  |  positives: {y.sum():,} ({y.mean()*100:.2f}%)")

    # stratified 80/20 split preserving temporal order
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    fitted_models = {}

    for name, cfg in MODEL_CONFIGS.items():
        print(f"\n[train] Training {name} …")
        model = cfg["cls"](**cfg["params"])

        with mlflow.start_run(run_name=f"{name}_{targeting_filter or 'all'}"):
            mlflow.log_params(cfg["params"])
            mlflow.log_param("targeting_filter", targeting_filter or "all")
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test",  len(X_test))
            mlflow.log_param("pos_rate", float(y_train.mean()))

            # cross-validation on training set
            cv = StratifiedKFold(n_splits=5, shuffle=False)
            cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
            mlflow.log_metric("cv_roc_auc_mean", float(cv_auc.mean()))
            mlflow.log_metric("cv_roc_auc_std",  float(cv_auc.std()))

            # final fit
            model.fit(X_train, y_train)

            # test evaluation
            metrics = evaluate_model(model, X_test, y_test)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # log model artefact
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"  CV AUC  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
            print(f"  Test AUC: {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC  : {metrics['pr_auc']:.4f}")
            print(f"  F1      : {metrics['f1']:.4f}")

        fitted_models[name] = model

    return fitted_models


# ── per-strategy training ─────────────────────────────────────────────────────

def train_per_strategy(df_raw: pd.DataFrame) -> dict:
    """
    Train separate XGBoost models for each targeting strategy using the
    raw (pre-encoded) DataFrame so we can filter cleanly.

    Returns nested dict: {strategy: {model_name: model}}.
    """
    from src.preprocessing import build_feature_matrix

    results = {}
    strategies = df_raw["targeting_strategy"].unique()

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"  Strategy: {strategy.upper()}")
        print('='*60)
        subset = df_raw[df_raw["targeting_strategy"] == strategy].copy()
        df_enc, _, _ = build_feature_matrix(df=subset)
        models = train_all_models(df_enc, targeting_filter=strategy)
        results[strategy] = models

    return results


if __name__ == "__main__":
    from src.preprocessing import load_raw, build_feature_matrix

    raw = load_raw()
    df_enc, encoders, scaler = build_feature_matrix(df=raw)
    models = train_all_models(df_enc)
    print("\n[train] All models trained successfully.")
