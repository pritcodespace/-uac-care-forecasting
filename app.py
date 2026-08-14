"""
Predictive Forecasting of Care Load & Placement Demand
--------------------------------------------------------
Streamlit dashboard for the HHS Unaccompanied Alien Children (UAC) Program.

Run with:  streamlit run app/app.py
"""
import sys
import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from data_pipeline import build_dataset  # noqa: E402
from models import (naive_forecast, moving_average_forecast, exp_smoothing_forecast,
                     sarima_forecast, ml_recursive_forecast, mae, rmse, mape,
                     STATSMODELS_AVAILABLE)  # noqa: E402

try:
    import plotly.graph_objects as go
    PLOTLY = True
except ImportError:
    PLOTLY = False

RAW_DATA_PATH = ROOT / 'data' / 'HHS_Unaccompanied_Alien_Children_Program.csv'
FEATURED_PATH = ROOT / 'data' / 'featured_dataset.csv'
RESULTS_PATH = ROOT / 'outputs' / 'evaluation_results.json'
MODELS_DIR = ROOT / 'models'

TARGET_LABELS = {'hhs_care': 'Children in HHS Care (Care Load)',
                  'discharged': 'Children Discharged (Placement / Discharge Demand)'}

MODEL_NAMES = ['Naive Persistence', 'Moving Average (7d)', 'Exponential Smoothing',
               'SARIMA', 'Random Forest', 'Gradient Boosting']

st.set_page_config(page_title="UAC Care Load & Placement Demand Forecasting",
                    layout="wide", page_icon="📈")


# ---------------------------------------------------------------------------
# Data / model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading and preparing data...")
def load_data():
    if FEATURED_PATH.exists():
        df = pd.read_csv(FEATURED_PATH, parse_dates=['date'])
    else:
        df = build_dataset(str(RAW_DATA_PATH))
        df.to_csv(FEATURED_PATH, index=False)
    return df


@st.cache_resource(show_spinner=False)
def load_ml_model(target, mtype):
    path = MODELS_DIR / f'{mtype}_{target}.pkl'
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_eval_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def get_forecast(name, df, target_col, h):
    series = df[target_col]
    if name == 'Naive Persistence':
        return naive_forecast(series, h)
    if name == 'Moving Average (7d)':
        return moving_average_forecast(series, h)
    if name == 'Exponential Smoothing':
        return exp_smoothing_forecast(series, h)
    if name == 'SARIMA':
        return sarima_forecast(series, h)
    if name == 'Random Forest':
        art = load_ml_model(target_col, 'rf')
        return ml_recursive_forecast(art['model'], art['feat_cols'], df, target_col, h, art['resid_std'])
    if name == 'Gradient Boosting':
        art = load_ml_model(target_col, 'gb')
        return ml_recursive_forecast(art['model'], art['feat_cols'], df, target_col, h, art['resid_std'])
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.title("🧭 Predictive Forecasting of Care Load & Placement Demand")
st.caption("HHS Unaccompanied Alien Children (UAC) Program — decision-support dashboard")

if not STATSMODELS_AVAILABLE:
    st.warning("`statsmodels` is not installed in this environment — SARIMA falls back to "
               "an Exponential Smoothing approximation. Run `pip install -r requirements.txt` "
               "for full SARIMA support.", icon="⚠️")

df = load_data()
eval_results = load_eval_results()

with st.sidebar:
    st.header("Forecast Controls")

    horizon = st.select_slider("Forecast horizon (days)", options=[1, 7, 14, 30], value=14)

    selected_models = st.multiselect("Model toggle", MODEL_NAMES,
                                      default=['SARIMA', 'Random Forest'])

    target = st.radio("Target variable", list(TARGET_LABELS.keys()),
                       format_func=lambda k: TARGET_LABELS[k])

    st.divider()
    st.subheader("Scenario comparison")
    scenario_on = st.checkbox("Enable side-by-side scenario view", value=False)
    if scenario_on:
        scenario_a = st.selectbox("Scenario A model", MODEL_NAMES, index=3, key='sa')
        scenario_b = st.selectbox("Scenario B model", MODEL_NAMES, index=4, key='sb')

    st.divider()
    st.subheader("Capacity assumption")
    default_capacity = int(df['hhs_care'].quantile(0.95))
    capacity_threshold = st.number_input("Facility capacity threshold (children)",
                                          min_value=100, max_value=20000,
                                          value=default_capacity, step=50)

st.divider()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------


def compute_kpis(target_col, horizon):
    res = eval_results.get(target_col, {})
    accs, stabilities = [], []
    for model, hz in res.items():
        if str(horizon) in hz:
            m = hz[str(horizon)]
            accs.append(max(0, 100 - m['MAPE']))
            stabilities.append(m['RMSE'])
    forecast_accuracy = np.mean(accs) if accs else np.nan
    stability_index = (1 - (np.std(stabilities) / np.mean(stabilities))) * 100 if stabilities else np.nan

    point, lo, hi = get_forecast('SARIMA' if 'SARIMA' in selected_models else selected_models[0]
                                  if selected_models else 'Naive Persistence',
                                  df, target_col, horizon)
    breach_prob = float(np.mean(hi > capacity_threshold) * 100) if target_col == 'hhs_care' else None

    surge_lead_time = None
    if target_col == 'hhs_care':
        breach_days = np.where(hi > capacity_threshold)[0]
        surge_lead_time = int(breach_days[0] + 1) if len(breach_days) > 0 else None

    return forecast_accuracy, stability_index, breach_prob, surge_lead_time


f_acc, f_stab, breach_prob, surge_lead = compute_kpis(target, horizon)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Forecast Accuracy", f"{f_acc:.1f}%" if not np.isnan(f_acc) else "N/A",
          help="Average (100 - MAPE) across all evaluated models at this horizon.")
k2.metric("Forecast Stability Index", f"{f_stab:.1f}" if not np.isnan(f_stab) else "N/A",
          help="Higher = more consistent RMSE across models (model robustness/agreement).")
if breach_prob is not None:
    k3.metric("Capacity Breach Probability", f"{breach_prob:.0f}%",
              help="Share of the forecast horizon where the upper 95% CI exceeds the capacity threshold.")
else:
    k3.metric("Capacity Breach Probability", "N/A (discharge target)")
if surge_lead is not None:
    k4.metric("Surge Lead Time", f"{surge_lead} day(s)",
              help="Days of advance warning before the forecast is expected to breach capacity.")
else:
    k4.metric("Surge Lead Time", "No breach expected")

st.divider()

# ---------------------------------------------------------------------------
# Main forecast chart
# ---------------------------------------------------------------------------

st.subheader(f"📈 {TARGET_LABELS[target]} — Forecast Chart")

history_window = st.slider("History shown (days)", 30, 365, 120, step=30)
hist_df = df.tail(history_window)
last_date = df['date'].iloc[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

if not selected_models:
    st.info("Select at least one model from the sidebar to see forecasts.")
else:
    if PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df[target], mode='lines',
                                  name='Actual', line=dict(color='#1f2937', width=2)))
        colors = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed', '#0891b2']
        for i, name in enumerate(selected_models):
            point, lo, hi = get_forecast(name, df, target, horizon)
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=future_dates, y=point, mode='lines+markers',
                                      name=name, line=dict(color=color, width=2, dash='dash')))
            fig.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(hi) + list(lo[::-1]),
                fill='toself', fillcolor=color, opacity=0.12,
                line=dict(width=0), showlegend=False, name=f'{name} 95% CI',
                hoverinfo='skip'))
        if target == 'hhs_care':
            fig.add_hline(y=capacity_threshold, line_dash="dot", line_color="red",
                           annotation_text="Capacity threshold")
        fig.update_layout(height=480, hovermode='x unified',
                           legend=dict(orientation='h', y=1.08),
                           margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        plot_df = hist_df[['date', target]].set_index('date')
        st.line_chart(plot_df)
        st.caption("Install `plotly` for confidence-interval bands and multi-model overlays.")

st.divider()

# ---------------------------------------------------------------------------
# Discharge Demand Forecast Panel
# ---------------------------------------------------------------------------

st.subheader("🚪 Discharge Demand Forecast Panel")
dc1, dc2 = st.columns([2, 1])
with dc1:
    dc_model = st.selectbox("Model for discharge panel", MODEL_NAMES, index=3)
    point, lo, hi = get_forecast(dc_model, df, 'discharged', horizon)
    disc_df = pd.DataFrame({'date': future_dates, 'forecast': point, 'lower': lo, 'upper': hi})
    if PLOTLY:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=disc_df['date'], y=disc_df['forecast'], name='Forecast discharges',
                               marker_color='#059669'))
        fig2.add_trace(go.Scatter(x=disc_df['date'], y=disc_df['upper'], line=dict(width=0),
                                   showlegend=False, hoverinfo='skip'))
        fig2.add_trace(go.Scatter(x=disc_df['date'], y=disc_df['lower'], line=dict(width=0),
                                   fill='tonexty', fillcolor='rgba(5,150,105,0.15)',
                                   showlegend=False, hoverinfo='skip'))
        fig2.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.bar_chart(disc_df.set_index('date')['forecast'])
with dc2:
    st.metric("Total discharges forecast (horizon)", f"{point.sum():.0f}")
    intake_point, _, _ = get_forecast(dc_model, df, 'hhs_care', 1)
    net_flow = df['transferred_to_hhs'].tail(7).mean() - point.mean()
    st.metric("Avg. net daily pressure (inflow - discharge)", f"{net_flow:+.1f}",
              help="Positive = care load likely rising; negative = capacity easing.")
    sufficiency = "✅ Likely sufficient" if net_flow <= 0 else "⚠️ May be insufficient"
    st.write(f"**Discharge capacity vs. incoming transfers:** {sufficiency}")

st.divider()

# ---------------------------------------------------------------------------
# Model Selection & Comparison
# ---------------------------------------------------------------------------

st.subheader("🧪 Model Selection & Comparison")
st.caption("Walk-forward, multi-horizon evaluation on held-out historical periods "
           "(strict time-based split — no random sampling).")

res = eval_results.get(target, {})
rows = []
for model, hz in res.items():
    if str(horizon) in hz:
        m = hz[str(horizon)]
        rows.append({'Model': model, 'MAE': m['MAE'], 'RMSE': m['RMSE'],
                     'MAPE (%)': m['MAPE'], 'Folds': m['n_folds']})
if rows:
    comp_df = pd.DataFrame(rows).sort_values('RMSE')
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    best = comp_df.iloc[0]['Model']
    st.success(f"Best performing model at h={horizon} days (lowest RMSE): **{best}**")
else:
    st.info("No evaluation results found for this horizon. Run `python src/evaluate.py` first.")

st.divider()

# ---------------------------------------------------------------------------
# Scenario comparison view
# ---------------------------------------------------------------------------

if scenario_on:
    st.subheader("🔀 Scenario Comparison")
    pa, loa, hia = get_forecast(scenario_a, df, target, horizon)
    pb, lob, hib = get_forecast(scenario_b, df, target, horizon)
    sc_df = pd.DataFrame({'date': future_dates, scenario_a: pa, scenario_b: pb})
    if PLOTLY:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=sc_df['date'], y=sc_df[scenario_a], name=scenario_a,
                                   line=dict(color='#2563eb', width=3)))
        fig3.add_trace(go.Scatter(x=sc_df['date'], y=sc_df[scenario_b], name=scenario_b,
                                   line=dict(color='#dc2626', width=3)))
        fig3.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.line_chart(sc_df.set_index('date'))
    diff = (pa - pb).mean()
    st.write(f"Average difference ({scenario_a} − {scenario_b}) over horizon: **{diff:+.1f} children/day**")

st.divider()
st.caption("Data source: HHS UAC Program daily reporting. Forecasts are model estimates for "
           "planning support only and should be interpreted alongside operational judgment.")
