"""
dashboard/app.py
~~~~~~~~~~~~~~~~
Interactive Dash dashboard for the Digital Ad Campaign Measurement &
Attribution Framework.

Tabs:
  1. Campaign Overview   – KPI cards, daily trend, spend vs revenue
  2. A/B Test Results    – pre/post and control/treatment comparison
  3. Attribution         – channel credit by model
  4. Model Performance   – ROC-AUC, PR-AUC per targeting strategy
  5. Budget Optimisation – current vs optimised allocation

Run:
    python dashboard/app.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from src.preprocessing import load_raw, build_feature_matrix
from src.ab_testing.ab_test import compute_kpis, prepost_analysis, simulate_uplift, strategy_comparison
from src.attribution.attribution import attribution_report
from src.budget_optimization.optimizer import channel_roas_summary, compute_uplift

# ── load data ─────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ad_events.parquet")
CSV_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "ad_events.csv")

print("[dashboard] Loading data …")
try:
    df_raw = load_raw(DATA_PATH if os.path.exists(DATA_PATH) else CSV_PATH)
except FileNotFoundError:
    print("[dashboard] Data not found. Generating …")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.generate_data import generate_ad_events, save_data
    df_raw = generate_ad_events()
    save_data(df_raw)

df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
df_raw["date"]      = df_raw["timestamp"].dt.date

# precompute
ab_results  = prepost_analysis(df_raw)
uplift_sim  = simulate_uplift(df_raw)
channel_s   = channel_roas_summary(df_raw)
opt_results = compute_uplift(df_raw)
strat_comp  = strategy_comparison(df_raw)
attr_summary = attribution_report(df_raw, verbose=False)

# ── colour palette ─────────────────────────────────────────────────────────────
COLORS = {
    "addressable": "#2563EB",
    "cohort":      "#10B981",
    "contextual":  "#F59E0B",
    "control":     "#6B7280",
    "treatment":   "#EF4444",
    "pre":         "#9CA3AF",
    "bg":          "#F8FAFC",
    "card":        "#FFFFFF",
}

CARD_STYLE = {
    "borderRadius": "12px",
    "boxShadow":    "0 2px 8px rgba(0,0,0,0.08)",
    "padding":      "1.2rem",
    "background":   "#FFFFFF",
    "height":       "100%",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def kpi_card(title: str, value: str, delta: str = "", color: str = "#2563EB") -> dbc.Col:
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.P(title, className="text-muted mb-1", style={"fontSize": "0.82rem", "fontWeight": 600}),
                html.H4(value, style={"color": color, "fontWeight": 700, "marginBottom": "0"}),
                html.Small(delta, style={"color": "#10B981" if "+" in delta else "#6B7280"}),
            ])
        ], style=CARD_STYLE),
        width=3
    )


def empty_fig(msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font={"size": 16, "color": "#9CA3AF"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ── global KPIs ───────────────────────────────────────────────────────────────

overall  = compute_kpis(df_raw)
treat_k  = ab_results["kpis"]["treatment"]
ctrl_k   = ab_results["kpis"]["control"]

ctr_delta   = f"+{(treat_k['CTR']-ctrl_k['CTR'])/ctrl_k['CTR']*100:.1f}% vs control"
cvr_delta   = f"+{(treat_k['CVR']-ctrl_k['CVR'])/ctrl_k['CVR']*100:.1f}% vs control"
roas_delta  = f"{uplift_sim['uplift_pct']:+.1f}% simulated uplift"


# ── app layout ────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Ad Attribution Dashboard",
)

app.layout = dbc.Container([

    # header
    dbc.Row([
        dbc.Col(html.Div([
            html.H2("📊 Digital Ad Campaign Measurement & Attribution Framework",
                    style={"fontWeight": 700, "color": "#1E293B"}),
            html.P("Buyer-journey analysis · 200K+ events · XGBoost / RF / MLP · SHAP · A/B Testing",
                   style={"color": "#64748B", "marginBottom": 0}),
        ]), width=12)
    ], className="my-4"),

    # KPI row
    dbc.Row([
        kpi_card("Total Impressions",  f"{overall['n_impressions']:,}"),
        kpi_card("Overall CTR",        f"{overall['CTR']*100:.2f}%",     ctr_delta,  "#2563EB"),
        kpi_card("Overall CVR",        f"{overall['CVR']*100:.2f}%",     cvr_delta,  "#10B981"),
        kpi_card("Overall ROAS",       f"{overall['ROAS']:.2f}×",        roas_delta, "#F59E0B"),
    ], className="mb-4 g-3"),

    # tabs
    dbc.Tabs([

        # ── TAB 1: Campaign Overview ──────────────────────────────────────────
        dbc.Tab(label="📈 Campaign Overview", tab_id="tab-overview", children=[
            dbc.Row([
                dbc.Col([
                    html.H6("Daily Revenue & Spend", className="mt-3 mb-2 text-muted"),
                    dcc.Graph(id="daily-trend"),
                ], width=8),
                dbc.Col([
                    html.H6("CTR by Targeting Strategy", className="mt-3 mb-2 text-muted"),
                    dcc.Graph(id="ctr-by-strategy"),
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H6("ROAS by Ad Format × Device", className="mt-3 mb-2 text-muted"),
                    dcc.Graph(id="roas-heatmap"),
                ], width=6),
                dbc.Col([
                    html.H6("Conversion Rate by Hour of Day", className="mt-3 mb-2 text-muted"),
                    dcc.Graph(id="cvr-hour"),
                ], width=6),
            ]),
        ]),

        # ── TAB 2: A/B Test Results ───────────────────────────────────────────
        dbc.Tab(label="🧪 A/B Test Results", tab_id="tab-ab", children=[
            dbc.Row([
                dbc.Col([
                    html.H6("KPI Comparison: Pre / Control / Treatment", className="mt-3 mb-2 text-muted"),
                    dcc.Graph(id="ab-kpi-bar"),
                ], width=7),
                dbc.Col([
                    html.H6("Statistical Test Results", className="mt-3 mb-2 text-muted"),
                    html.Div(id="ab-stat-table"),
                ], width=5),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H6("CTR Distribution (control vs treatment)", className="mt-3 mb-2 text-muted"),
                    dcc.Graph(id="ctr-dist"),
                ], width=6),
                dbc.Col([
                    html.H6("ROAS by Strategy & Group", className="mt-3 mb-2 text-muted"),
                    dcc.Graph(id="roas-group"),
                ], width=6),
            ]),
        ]),

        # ── TAB 3: Attribution ────────────────────────────────────────────────
        dbc.Tab(label="🔗 Attribution", tab_id="tab-attr", children=[
            dbc.Row([
                dbc.Col([
                    html.H6("Attributed Revenue by Channel × Model", className="mt-3 mb-2 text-muted"),
                    dcc.Graph(id="attr-grouped"),
                ], width=8),
                dbc.Col([
                    html.H6("Attribution Model", className="mt-3 mb-2 text-muted"),
                    dcc.Dropdown(
                        id="attr-model-select",
                        options=[{"label": m, "value": m}
                                 for m in attr_summary["model"].unique()],
                        value="linear",
                        clearable=False,
                    ),
                    dcc.Graph(id="attr-pie"),
                ], width=4),
            ]),
        ]),

        # ── TAB 4: Budget Optimisation ────────────────────────────────────────
        dbc.Tab(label="💰 Budget Optimisation", tab_id="tab-budget", children=[
            dbc.Row([
                dbc.Col([
                    html.H6("Current vs Optimised Budget Allocation", className="mt-3 mb-2 text-muted"),
                    dcc.Graph(id="budget-compare"),
                ], width=7),
                dbc.Col([
                    html.H6("Projected Revenue Uplift", className="mt-3 mb-2 text-muted"),
                    dcc.Graph(id="uplift-gauge"),
                ], width=5),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H6("ROAS per Channel", className="mt-3 mb-2 text-muted"),
                    dcc.Graph(id="roas-channel"),
                ], width=12),
            ]),
        ]),

        # ── TAB 5: Strategy Deep-Dive ─────────────────────────────────────────
        dbc.Tab(label="🎯 Strategy Deep-Dive", tab_id="tab-strategy", children=[
            dbc.Row([
                dbc.Col([
                    html.H6("KPI Table by Targeting Strategy × A/B Group",
                            className="mt-3 mb-2 text-muted"),
                    dash_table.DataTable(
                        data=strat_comp.round(4).to_dict("records"),
                        columns=[{"name": c, "id": c} for c in strat_comp.columns],
                        style_table={"overflowX": "auto"},
                        style_cell={"fontSize": "0.82rem", "padding": "6px"},
                        style_header={"fontWeight": 700, "background": "#F1F5F9"},
                        style_data_conditional=[
                            {"if": {"filter_query": "{targeting_strategy} = addressable"},
                             "backgroundColor": "#EFF6FF"},
                        ],
                        page_size=15,
                    ),
                ], width=12),
            ]),
        ]),

    ], id="main-tabs", active_tab="tab-overview"),

], fluid=True, style={"background": COLORS["bg"], "minHeight": "100vh"})


# ── callbacks ──────────────────────────────────────────────────────────────────

# --- daily trend ---
@app.callback(Output("daily-trend", "figure"), Input("main-tabs", "active_tab"))
def update_daily_trend(_):
    daily = (
        df_raw.groupby("date")
        .agg(revenue=("order_value_usd", "sum"), spend=("ad_spend_usd", "sum"))
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["revenue"],
                             name="Revenue", line=dict(color="#2563EB", width=2)))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["spend"],
                             name="Spend", line=dict(color="#F59E0B", width=2, dash="dash")))
    # A/B split line
    from datetime import date
    split_date = date(2024, 2, 15)
    fig.add_vline(x=str(split_date), line_dash="dot", line_color="#EF4444",
                  annotation_text="A/B Start", annotation_position="top right")
    fig.update_layout(margin=dict(t=20), legend=dict(orientation="h"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(gridcolor="#F1F5F9"), yaxis=dict(gridcolor="#F1F5F9"))
    return fig


# --- CTR by strategy ---
@app.callback(Output("ctr-by-strategy", "figure"), Input("main-tabs", "active_tab"))
def update_ctr_strategy(_):
    d = (df_raw.groupby("targeting_strategy")
         .apply(lambda g: g["clicked"].sum() / len(g), include_groups=False)
         .reset_index(name="CTR"))
    fig = px.bar(d, x="targeting_strategy", y="CTR",
                 color="targeting_strategy",
                 color_discrete_map=COLORS,
                 labels={"CTR": "Click-Through Rate"})
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10),
                      yaxis=dict(tickformat=".2%", gridcolor="#F1F5F9"))
    return fig


# --- ROAS heatmap ---
@app.callback(Output("roas-heatmap", "figure"), Input("main-tabs", "active_tab"))
def update_roas_heatmap(_):
    pivot = (
        df_raw.groupby(["ad_format", "device_type"])
        .apply(lambda g: g["order_value_usd"].sum() / g["ad_spend_usd"].sum(), include_groups=False)
        .reset_index(name="ROAS")
        .pivot(index="ad_format", columns="device_type", values="ROAS")
        .fillna(0)
    )
    fig = px.imshow(pivot, color_continuous_scale="Blues",
                    labels=dict(color="ROAS"), aspect="auto")
    fig.update_layout(margin=dict(t=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# --- CVR by hour ---
@app.callback(Output("cvr-hour", "figure"), Input("main-tabs", "active_tab"))
def update_cvr_hour(_):
    d = (df_raw.groupby("hour_of_day")
         .apply(lambda g: g["converted"].sum() / g["clicked"].clip(lower=1).sum(), include_groups=False)
         .reset_index(name="CVR"))
    fig = px.line(d, x="hour_of_day", y="CVR",
                  labels={"hour_of_day": "Hour of Day", "CVR": "Conversion Rate"},
                  markers=True, color_discrete_sequence=["#10B981"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=10), yaxis=dict(tickformat=".2%", gridcolor="#F1F5F9"))
    return fig


# --- A/B KPI bar ---
@app.callback(Output("ab-kpi-bar", "figure"), Input("main-tabs", "active_tab"))
def update_ab_kpi(_):
    kpis = ab_results["kpis"]
    metrics = ["CTR", "CVR", "ROAS"]
    groups  = ["pre", "control", "treatment"]
    fig = go.Figure()
    for g in groups:
        vals = [kpis[g][m] for m in metrics]
        fig.add_trace(go.Bar(name=g.capitalize(), x=metrics, y=vals,
                             marker_color=COLORS.get(g, "#888")))
    fig.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10),
                      legend=dict(orientation="h"),
                      yaxis=dict(gridcolor="#F1F5F9"))
    return fig


# --- statistical test table ---
@app.callback(Output("ab-stat-table", "children"), Input("main-tabs", "active_tab"))
def update_ab_stat_table(_):
    tests = ab_results["tests"]
    rows = []
    for name, t in tests.items():
        rows.append({
            "Test":      name.replace("_", " ").title(),
            "Uplift %":  f"{t.get('uplift_pct', 0):+.1f}%" if t.get('uplift_pct') else "—",
            "p-value":   f"{t.get('p_value', 1):.4f}",
            "Sig.":      "✓" if t.get("significant") else "✗",
        })
    df_t = pd.DataFrame(rows)
    return dash_table.DataTable(
        data=df_t.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_t.columns],
        style_cell={"fontSize": "0.78rem", "padding": "5px"},
        style_header={"fontWeight": 700, "background": "#F1F5F9"},
        style_data_conditional=[
            {"if": {"filter_query": '{Sig.} = "✓"', "column_id": "Sig."},
             "color": "#10B981", "fontWeight": 700},
        ],
    )


# --- CTR distribution ---
@app.callback(Output("ctr-dist", "figure"), Input("main-tabs", "active_tab"))
def update_ctr_dist(_):
    sample = df_raw[df_raw["ab_group"].isin(["control", "treatment"])].sample(
        min(5000, len(df_raw)), random_state=42)
    fig = px.histogram(sample, x="ctr_prob", color="ab_group",
                       nbins=50, barmode="overlay", opacity=0.7,
                       color_discrete_map=COLORS,
                       labels={"ctr_prob": "Predicted CTR probability", "ab_group": "Group"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=10), yaxis=dict(gridcolor="#F1F5F9"))
    return fig


# --- ROAS by group ---
@app.callback(Output("roas-group", "figure"), Input("main-tabs", "active_tab"))
def update_roas_group(_):
    d = (df_raw[df_raw["ab_group"].isin(["control", "treatment"])]
         .groupby(["targeting_strategy", "ab_group"])
         .apply(lambda g: g["order_value_usd"].sum() / g["ad_spend_usd"].sum(), include_groups=False)
         .reset_index(name="ROAS"))
    fig = px.bar(d, x="targeting_strategy", y="ROAS", color="ab_group",
                 barmode="group", color_discrete_map=COLORS)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=10), yaxis=dict(gridcolor="#F1F5F9"))
    return fig


# --- attribution grouped ---
@app.callback(Output("attr-grouped", "figure"), Input("main-tabs", "active_tab"))
def update_attr_grouped(_):
    fig = px.bar(attr_summary, x="channel", y="attributed_revenue",
                 color="model", barmode="group",
                 labels={"attributed_revenue": "Attributed Revenue (USD)", "channel": "Channel"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=10), yaxis=dict(gridcolor="#F1F5F9"))
    return fig


# --- attribution pie ---
@app.callback(Output("attr-pie", "figure"),
              Input("attr-model-select", "value"))
def update_attr_pie(model_val):
    d = attr_summary[attr_summary["model"] == model_val]
    fig = px.pie(d, names="channel", values="attributed_revenue",
                 color="channel", color_discrete_map=COLORS,
                 hole=0.4)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10))
    return fig


# --- budget comparison ---
@app.callback(Output("budget-compare", "figure"), Input("main-tabs", "active_tab"))
def update_budget_compare(_):
    greedy = opt_results["greedy_allocation"].copy()
    n_ch   = greedy["targeting_strategy"].nunique()
    even   = 50_000 / n_ch

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Even Split",
                         x=greedy["targeting_strategy"], y=[even]*len(greedy),
                         marker_color="#9CA3AF"))
    fig.add_trace(go.Bar(name="Optimised",
                         x=greedy["targeting_strategy"], y=greedy["allocated_budget"],
                         marker_color="#2563EB"))
    fig.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10),
                      yaxis=dict(tickprefix="$", gridcolor="#F1F5F9"))
    return fig


# --- uplift gauge ---
@app.callback(Output("uplift-gauge", "figure"), Input("main-tabs", "active_tab"))
def update_uplift_gauge(_):
    uplift_val = opt_results.get("greedy_uplift_pct", 12.0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=uplift_val,
        delta={"reference": 0, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [-5, 25]},
            "bar": {"color": "#2563EB"},
            "steps": [
                {"range": [-5, 0],  "color": "#FEE2E2"},
                {"range": [0, 10],  "color": "#FEF3C7"},
                {"range": [10, 25], "color": "#D1FAE5"},
            ],
            "threshold": {"line": {"color": "#10B981", "width": 3}, "value": 12},
        },
        title={"text": "Revenue Uplift %"},
        number={"suffix": "%", "valueformat": ".1f"},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, b=10))
    return fig


# --- ROAS per channel ---
@app.callback(Output("roas-channel", "figure"), Input("main-tabs", "active_tab"))
def update_roas_channel(_):
    d = channel_s.groupby("targeting_strategy")["ROAS"].mean().reset_index()
    fig = px.bar(d, x="targeting_strategy", y="ROAS",
                 color="targeting_strategy", color_discrete_map=COLORS,
                 text=d["ROAS"].round(2),
                 labels={"ROAS": "Return on Ad Spend (×)"})
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10),
                      yaxis=dict(gridcolor="#F1F5F9"))
    return fig


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[dashboard] Starting on http://127.0.0.1:8050")
    app.run(debug=True, host="0.0.0.0", port=8050)
