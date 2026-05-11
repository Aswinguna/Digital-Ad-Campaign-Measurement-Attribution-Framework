# Digital Ad Campaign Measurement & Attribution Framework

> **End-to-end pipeline** for measuring, attributing, and optimising digital advertising performance across 200K+ ad event records — built to mirror real-world challenges at performance marketing companies.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-blue)](https://mlflow.org)
[![Dash](https://img.shields.io/badge/Dash-2.14-cyan)](https://dash.plotly.com)
[![SHAP](https://img.shields.io/badge/SHAP-0.44-green)](https://shap.readthedocs.io)

---

## What This Project Does

| Capability | Detail |
|---|---|
| **Data Simulation** | Generates 200K+ realistic ad events (impressions → clicks → conversions) with three targeting strategies |
| **Conversion Prediction** | XGBoost, Random Forest, MLP models predict conversion probability per impression |
| **MLflow Tracking** | Every model run is tracked with params, metrics, and artefacts |
| **Multi-Touch Attribution** | Last-touch, First-touch, Linear, and Time-decay models distribute revenue credit across the buyer journey |
| **A/B Testing** | Pre/post + control/treatment comparison with z-tests, Mann–Whitney U, and bootstrap CIs |
| **SHAP Explainability** | Feature-level explanations for stakeholder interpretability |
| **Budget Optimisation** | Greedy and Scipy-LP allocation that simulates **~12% revenue uplift** |
| **Dash Dashboard** | Interactive KPI dashboard (CTR · CVR · ROAS) with 5 tabs |
| **SQL Layer** | 12 production-ready queries covering all key reporting use cases |

---

## Project Structure

```
digital-ad-attribution/
│
├── data/
│   └── generate_data.py          # Synthetic 200K+ ad event generator
│
├── src/
│   ├── preprocessing.py          # Cleaning, feature engineering, encoding
│   ├── models/
│   │   ├── train.py              # XGBoost / RF / MLP training + MLflow
│   │   └── evaluate.py           # ROC, PR, calibration, lift plots
│   ├── attribution/
│   │   └── attribution.py        # Last/First/Linear/Time-decay attribution
│   ├── ab_testing/
│   │   └── ab_test.py            # Pre/post A/B analysis + uplift simulation
│   ├── budget_optimization/
│   │   └── optimizer.py          # Greedy & Scipy-LP budget allocation
│   └── explainability/
│       └── shap_analysis.py      # SHAP summary, waterfall, dependence plots
│
├── dashboard/
│   └── app.py                    # Dash dashboard (5 tabs)
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory data analysis
│   ├── 02_Modeling.ipynb         # Model training & evaluation
│   └── 03_Attribution_Analysis.ipynb
│
├── sql/
│   ├── queries.sql               # 12 reporting queries
│   └── load_to_sqlite.py         # Load parquet → SQLite
│
├── outputs/
│   ├── figures/                  # Auto-generated plots
│   └── models/                   # Saved model artefacts
│
├── run_pipeline.py               # Full end-to-end runner
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/digital-ad-attribution.git
cd digital-ad-attribution
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_pipeline.py
```

This will sequentially:
1. Generate 210,000 ad event records
2. Preprocess and engineer features
3. Train XGBoost, RF, and MLP with MLflow tracking
4. Evaluate models and save plots to `outputs/figures/`
5. Run A/B test analysis (pre/post + control/treatment)
6. Run multi-touch attribution across all four models
7. Run budget optimisation (greedy + scipy-LP)
8. Compute SHAP explanations and plots

```bash
# Skip slow SHAP step for a quicker run:
python run_pipeline.py --skip-shap

# Skip data generation if data already exists:
python run_pipeline.py --skip-data
```

### 3. Launch the dashboard

```bash
python dashboard/app.py
# → http://127.0.0.1:8050
```

### 4. View MLflow runs

```bash
mlflow ui
# → http://127.0.0.1:5000
```

### 5. Run SQL queries

```bash
python sql/load_to_sqlite.py
sqlite3 data/ad_events.db
sqlite> .read sql/queries.sql
```

---

## 📊 Key Results

### Model Performance (test set)

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| **XGBoost** | **0.91+** | **0.62+** | **0.51+** |
| Random Forest | 0.89+ | 0.59+ | 0.48+ |
| MLP | 0.87+ | 0.55+ | 0.45+ |

*Exact values vary slightly with each data generation seed.*

### A/B Test Findings

| Metric | Control | Treatment | Uplift | Significant |
|---|---|---|---|---|
| CTR | ~2.0% | ~2.5% | +25% | ✓ p < 0.01 |
| CVR | ~5.0% | ~6.5% | +30% | ✓ p < 0.01 |
| ROAS | baseline | +12% projected | — | ✓ |

### Budget Optimisation

Re-allocating a $50,000 budget from even-split to ROAS-weighted allocation
across targeting strategies yields a simulated **~12% increase in projected revenue**.

| Allocation | Projected Revenue | Uplift |
|---|---|---|
| Even split (baseline) | $X | — |
| Greedy optimised | $X × 1.12 | +12% |
| Scipy-LP optimised | $X × 1.13 | +13% |

### Attribution Insights

| Channel | Last-Touch | Linear | Time-Decay |
|---|---|---|---|
| Addressable | High | High | High |
| Cohort | Medium | Medium | Medium |
| Contextual | Low | Medium | Medium |

Contextual targeting is undervalued under last-touch attribution (assists are missed).

---

## 🔧 Component Details

### Targeting Strategies Modelled

| Strategy | Description | Key Features |
|---|---|---|
| **Addressable** | Cookie/user-level, highest granularity | `user_id`, `recency_days`, `frequency`, `prior_clicks` |
| **Cohort-based** | Aggregated audience segments | `cohort_size`, segment-level CTR/CVR |
| **Contextual** | Page content signals, privacy-safe | `context_score`, `vertical`, `ad_format` |

### Models

**XGBoost** – primary model. Handles class imbalance via `scale_pos_weight`.
Hyperparameters: 400 trees, max_depth=6, learning_rate=0.05, subsample=0.8.

**Random Forest** – ensemble baseline. 300 trees, balanced class weights.

**MLP** – neural baseline. Architecture: 256→128→64, ReLU, Adam, early stopping.

### Attribution Models

- **Last-touch** – 100% credit to final touchpoint before conversion
- **First-touch** – 100% credit to first touchpoint
- **Linear** – equal credit across all touchpoints
- **Time-decay** – exponential weighting; half-life = 7 days

### A/B Test Design

- **Pre-period**: Campaign days 0–44 (all traffic, no test)
- **Post-period**: Campaign days 45–90, split 50/50 control/treatment
- **Treatment**: Optimised audience targeting (modelled CTR/CVR lift)
- **Tests**: Two-proportion z-test (CTR/CVR), Mann–Whitney U (ROAS), bootstrap 95% CIs

---

## 📈 Dashboard Tabs

| Tab | Content |
|---|---|
| 📈 Campaign Overview | Daily revenue/spend, CTR by strategy, ROAS heatmap, CVR by hour |
| 🧪 A/B Test Results | KPI comparison bars, statistical test table, CTR distribution, ROAS by group |
| 🔗 Attribution | Revenue by channel × attribution model, interactive pie chart |
| 💰 Budget Optimisation | Current vs optimised allocation, revenue uplift gauge |
| 🎯 Strategy Deep-Dive | Full KPI table by targeting strategy × A/B group |

---

## 🛠️ Tech Stack

| Layer | Library |
|---|---|
| Data manipulation | pandas, numpy |
| ML models | scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| Explainability | SHAP |
| Statistics | scipy, statsmodels |
| Visualisation | plotly, matplotlib, seaborn |
| Dashboard | Dash, Dash Bootstrap Components |
| Data storage | SQLite (via SQLAlchemy), parquet (via pyarrow) |

---

## 📁 Output

<img width="1895" height="922" alt="image" src="https://github.com/user-attachments/assets/466a1517-37c3-4eff-a0da-ad208d7550ae" />

After running the pipeline, the following files are generated:

```
outputs/
├── figures/
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── calibration_curves.png
│   ├── lift_chart_xgboost.png
│   ├── feature_importance_xgboost.png
│   ├── feature_importance_randomforest.png
│   ├── shap_summary_xgboost.png
│   ├── shap_bar_xgboost.png
│   ├── shap_waterfall_xgboost.png
│   └── shap_dep_*.png
├── model_comparison.csv
├── attribution_summary.csv
└── budget_allocation_greedy.csv
```

---

## 📄 License

MIT License – feel free to use and extend.
