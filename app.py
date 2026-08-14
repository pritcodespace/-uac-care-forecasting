"""
Predictive Forecasting of Care Load & Placement Demand
--------------------------------------------------------
Streamlit dashboard for the HHS Unaccompanied Alien Children (UAC) Program.

SELF-CONTAINED VERSION: everything (data pipeline, models, dashboard) lives
in this one file, and it looks for the CSV in the same folder as this
script — so it works no matter how flat or nested your repo layout is.

Run with:  streamlit run app.py
"""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import Holt
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    import plotly.graph_objects as go
    PLOTLY = True
except ImportError:
    PLOTLY = False

HERE = Path(__file__).resolve().parent
CSV_NAME = 'HHS_Unaccompanied_Alien_Children_Program.csv'

TARGET_LABELS = {'hhs_care': 'Children in HHS Care (Care Load)',
                  'discharged': 'Children Discharged (Placement / Discharge Demand)'}
MODEL_NAMES = ['Naive Persistence', 'Moving Average (7d)', 'Exponential Smoothing',
               'SARIMA', 'Random Forest', 'Gradient Boosting']
HORIZONS = [1, 7, 14, 30]

st.set_page_config(page_title="UAC Care Load & Placement Demand Forecasting",
                    layout="wide", page_icon="📈")


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def find_csv() -> Path:
    """Search this folder, and one level up/down, for the raw CSV."""
    candidates = [HERE / CSV_NAME, HERE / 'data' / CSV_NAME,
                  HERE.parent / CSV_NAME, HERE.parent / 'data' / CSV_NAME]
    for c in candidates:
        if c.exists():
            return c
    # last resort: search recursively (shallow) under HERE
    for p in HERE.rglob(CSV_NAME):
        return p
    return None


@st.cache_data(show_spinner="Loading and preparing data...")
def load_data():
    csv_path = find_csv()
    if csv_path is None:
        st.error(
            f"Could not find `{CSV_NAME}` anywhere near `{HERE}`.\n\n"
            f"Make sure the CSV file is uploaded to the same GitHub repo as app.py."
        )
        st.stop()

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[df.columns[0]]).copy()
    df.columns = ['date', 'intake', 'cbp_custody', 'transferred_to_hhs', 'hhs_care', 'discharged']

    for c in ['intake', 'cbp_custody', 'transferred_to_hhs', 'hhs_care', 'discharged']:
        df[c] = df[c].astype(str).str.replace(',', '', regex=False)
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df['date'] = pd.to_datetime(df['date'], format='%B %d, %Y', errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    df = df.drop_duplicates(subset='date', keep='last')

    # reindex to continuous daily calendar + interpolate gaps
    df = df.set_index('date').sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(full_idx)
    df.index.name = 'date'
    for c in ['intake', 'transferred_to_hhs', 'discharged']:
        df[c] = df[c].interpolate(method='linear').round()
    for c in ['cbp_custody', 'hhs_care']:
        df[c] = df[c].interpolate(method='linear')
    df = df.reset_index()

    # feature engineering
    df['net_pressure'] = df['transferred_to_hhs'] - df['discharged']
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    for col in ['hhs_care', 'discharged']:
        for lag in [1, 7, 14]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        for win in [7, 14]:
            df[f'{col}_roll_mean{win}'] = df[col].shift(1).rolling(win).mean()
            df[f'{col}_roll_std{win}'] = df[col].shift(1).rolling(win).std()
    df['net_pressure_roll7'] = df['net_pressure'].shift(1).rolling(7).mean()

    return df


# ---------------------------------------------------------------------------
# Forecasting models
# ---------------------------------------------------------------------------

def naive_forecast(series, h):
    last = series.iloc[-1]
    point = np.repeat(last, h)
    resid_std = series.diff().dropna().std()
    ci = 1.96 * resid_std * np.sqrt(np.arange(1, h + 1))
    return point, point - ci, point + ci


def moving_average_forecast(series, h, window=7):
    ma = series.iloc[-window:].mean()
    point = np.repeat(ma, h)
    resid_std = series.diff().dropna().std()
    ci = 1.96 * resid_std * np.sqrt(np.arange(1, h + 1))
    return point, point - ci, point + ci


def exp_smoothing_forecast(series, h):
    if STATSMODELS_AVAILABLE:
        model = Holt(series.values, initialization_method='estimated').fit(optimized=True)
        point = model.forecast(h)
        resid_std = np.std(model.resid)
        ci = 1.96 * resid_std * np.sqrt(np.arange(1, h + 1))
        return point, point - ci, point + ci
    alpha, beta = 0.3, 0.1
    level, trend = series.iloc[0], series.iloc[1] - series.iloc[0]
    for y in series.iloc[1:]:
        last_level = level
        level = alpha * y + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
    point = np.array([level + (i + 1) * trend for i in range(h)])
    resid_std = series.diff().dropna().std()
    ci = 1.96 * resid_std * np.sqrt(np.arange(1, h + 1))
    return point, point - ci, point + ci


def sarima_forecast(series, h, order=(2, 1, 2), seasonal_order=(1, 0, 1, 7)):
    if not STATSMODELS_AVAILABLE:
        return exp_smoothing_forecast(series, h)
    try:
        model = ARIMA(series.values, order=order, seasonal_order=seasonal_order).fit()
    except Exception:
        model = ARIMA(series.values, order=(1, 1, 1)).fit()
    fc = model.get_forecast(h)
    point = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    return point, ci[:, 0], ci[:, 1]


def _feat_cols(target_col):
    return [f'{target_col}_lag1', f'{target_col}_lag7', f'{target_col}_lag14',
            f'{target_col}_roll_mean7', f'{target_col}_roll_std7',
            f'{target_col}_roll_mean14', f'{target_col}_roll_std14',
            'day_of_week', 'month', 'net_pressure_roll7']


@st.cache_resource(show_spinner="Training model...")
def train_ml_model(target_col, model_type):
    df = load_data()
    feat_cols = _feat_cols(target_col)
    data = df.dropna(subset=feat_cols + [target_col])
    X, y = data[feat_cols], data[target_col]
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    resid_std = np.std(y - model.predict(X))
    return model, feat_cols, resid_std


def ml_recursive_forecast(model, feat_cols, df_history, target_col, h, resid_std):
    hist = df_history.copy().reset_index(drop=True)
    preds = []
    last_date = hist['date'].iloc[-1]
    for step in range(h):
        next_date = last_date + pd.Timedelta(days=step + 1)
        series = hist[target_col]
        row = {
            f'{target_col}_lag1': series.iloc[-1],
            f'{target_col}_lag7': series.iloc[-7] if len(series) >= 7 else series.iloc[0],
            f'{target_col}_lag14': series.iloc[-14] if len(series) >= 14 else series.iloc[0],
            f'{target_col}_roll_mean7': series.iloc[-7:].mean(),
            f'{target_col}_roll_std7': series.iloc[-7:].std(),
            f'{target_col}_roll_mean14': series.iloc[-14:].mean(),
            f'{target_col}_roll_std14': series.iloc[-14:].std(),
            'day_of_week': next_date.dayofweek,
            'month': next_date.month,
            'net_pressure_roll7': hist['net_pressure'].iloc[-7:].mean() if 'net_pressure' in hist else 0,
        }
        X_next = pd.DataFrame([row])[feat_cols].ffill(axis=0)
        yhat = model.predict(X_next)[0]
        preds.append(yhat)
        new_row = hist.iloc[-1:].copy()
        new_row['date'] = next_date
        new_row[target_col] = yhat
        hist = pd.concat([hist, new_row], ignore_index=True)
    preds = np.array(preds)
    ci = 1.96 * resid_std * np.sqrt(np.arange(1, h + 1))
    return preds, preds - ci, preds + ci


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
    if name in ('Random Forest', 'Gradient Boosting'):
        mtype = 'rf' if name == 'Random Forest' else 'gb'
        model, feat_cols, resid_std = train_ml_model(target_col, mtype)
        return ml_recursive_forecast(model, feat_cols, df, target_col, h, resid_std)
    raise ValueError(name)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# Evaluation (walk-forward, multi-horizon) — cached, computed once per session
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running walk-forward model evaluation (first load only)...")
def run_evaluation():
    df = load_data()
    n_test_origins, spacing = 5, 40
    max_h = max(HORIZONS)
    n = len(df)
    last_origin = n - max_h - 1
    origins = sorted([o for o in [last_origin - i * spacing for i in range(n_test_origins)] if o > 150])

    all_results = {}
    for target in ['hhs_care', 'discharged']:
        results = {m: {h: [] for h in HORIZONS} for m in MODEL_NAMES}
        for origin in origins:
            train_df = df.iloc[:origin + 1].copy()
            actual_future = df[target].iloc[origin + 1: origin + 1 + max_h].values
            for name in MODEL_NAMES:
                try:
                    point, lo, hi = get_forecast(name, train_df, target, max_h)
                except Exception:
                    continue
                for h in HORIZONS:
                    if len(actual_future) < h:
                        continue
                    y_true, y_pred = actual_future[:h], point[:h]
                    results[name][h].append({'mae': mae(y_true, y_pred), 'rmse': rmse(y_true, y_pred),
                                              'mape': mape(y_true, y_pred)})
        summary = {}
        for name in MODEL_NAMES:
            summary[name] = {}
            for h in HORIZONS:
                runs = results[name][h]
                if not runs:
                    continue
                summary[name][h] = {
                    'MAE': round(np.mean([r['mae'] for r in runs]), 2),
                    'RMSE': round(np.mean([r['rmse'] for r in runs]), 2),
                    'MAPE': round(np.mean([r['mape'] for r in runs]), 2),
                    'n_folds': len(runs),
                }
        all_results[target] = summary
    return all_results


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🧭 Predictive Forecasting of Care Load & Placement Demand")
st.caption("HHS Unaccompanied Alien Children (UAC) Program — decision-support dashboard")

if not STATSMODELS_AVAILABLE:
    st.warning("`statsmodels` is not installed — SARIMA falls back to an Exponential "
               "Smoothing approximation. Add `statsmodels` to requirements.txt for full SARIMA.",
               icon="⚠️")

df = load_data()
eval_results = run_evaluation()

with st.sidebar:
    st.header("Forecast Controls")
    horizon = st.select_slider("Forecast horizon (days)", options=HORIZONS, value=14)
    selected_models = st.multiselect("Model toggle", MODEL_NAMES, default=['SARIMA', 'Random Forest'])
    target = st.radio("Target variable", list(TARGET_LABELS.keys()), format_func=lambda k: TARGET_LABELS[k])

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
                                          min_value=100, max_value=20000, value=default_capacity, step=50)

st.divider()


def compute_kpis(target_col, horizon):
    res = eval_results.get(target_col, {})
    accs, stabilities = [], []
    for model, hz in res.items():
        if horizon in hz:
            m = hz[horizon]
            accs.append(max(0, 100 - m['MAPE']))
            stabilities.append(m['RMSE'])
    forecast_accuracy = np.mean(accs) if accs else np.nan
    stability_index = (1 - (np.std(stabilities) / np.mean(stabilities))) * 100 if stabilities else np.nan

    ref_model = 'SARIMA' if 'SARIMA' in selected_models else (selected_models[0] if selected_models else 'Naive Persistence')
    point, lo, hi = get_forecast(ref_model, df, target_col, horizon)
    breach_prob = float(np.mean(hi > capacity_threshold) * 100) if target_col == 'hhs_care' else None
    surge_lead_time = None
    if target_col == 'hhs_care':
        breach_days = np.where(hi > capacity_threshold)[0]
        surge_lead_time = int(breach_days[0] + 1) if len(breach_days) > 0 else None
    return forecast_accuracy, stability_index, breach_prob, surge_lead_time


f_acc, f_stab, breach_prob, surge_lead = compute_kpis(target, horizon)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Forecast Accuracy", f"{f_acc:.1f}%" if not np.isnan(f_acc) else "N/A")
k2.metric("Forecast Stability Index", f"{f_stab:.1f}" if not np.isnan(f_stab) else "N/A")
if breach_prob is not None:
    k3.metric("Capacity Breach Probability", f"{breach_prob:.0f}%")
else:
    k3.metric("Capacity Breach Probability", "N/A (discharge target)")
if surge_lead is not None:
    k4.metric("Surge Lead Time", f"{surge_lead} day(s)")
else:
    k4.metric("Surge Lead Time", "No breach expected")

st.divider()
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
                x=list(future_dates) + list(future_dates[::-1]), y=list(hi) + list(lo[::-1]),
                fill='toself', fillcolor=color, opacity=0.12, line=dict(width=0),
                showlegend=False, hoverinfo='skip'))
        if target == 'hhs_care':
            fig.add_hline(y=capacity_threshold, line_dash="dot", line_color="red",
                           annotation_text="Capacity threshold")
        fig.update_layout(height=480, hovermode='x unified', legend=dict(orientation='h', y=1.08),
                           margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(hist_df[['date', target]].set_index('date'))

st.divider()
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
    net_flow = df['transferred_to_hhs'].tail(7).mean() - point.mean()
    st.metric("Avg. net daily pressure (inflow - discharge)", f"{net_flow:+.1f}")
    sufficiency = "✅ Likely sufficient" if net_flow <= 0 else "⚠️ May be insufficient"
    st.write(f"**Discharge capacity vs. incoming transfers:** {sufficiency}")

st.divider()
st.subheader("🧪 Model Selection & Comparison")
st.caption("Walk-forward, multi-horizon evaluation on held-out historical periods "
           "(strict time-based split — no random sampling).")
res = eval_results.get(target, {})
rows = []
for model, hz in res.items():
    if horizon in hz:
        m = hz[horizon]
        rows.append({'Model': model, 'MAE': m['MAE'], 'RMSE': m['RMSE'], 'MAPE (%)': m['MAPE'], 'Folds': m['n_folds']})
if rows:
    comp_df = pd.DataFrame(rows).sort_values('RMSE')
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.success(f"Best performing model at h={horizon} days (lowest RMSE): **{comp_df.iloc[0]['Model']}**")
else:
    st.info("Not enough history for evaluation at this horizon yet.")

st.divider()
if scenario_on:
    st.subheader("🔀 Scenario Comparison")
    pa, loa, hia = get_forecast(scenario_a, df, target, horizon)
    pb, lob, hib = get_forecast(scenario_b, df, target, horizon)
    sc_df = pd.DataFrame({'date': future_dates, scenario_a: pa, scenario_b: pb})
    if PLOTLY:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=sc_df['date'], y=sc_df[scenario_a], name=scenario_a, line=dict(color='#2563eb', width=3)))
        fig3.add_trace(go.Scatter(x=sc_df['date'], y=sc_df[scenario_b], name=scenario_b, line=dict(color='#dc2626', width=3)))
        fig3.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.line_chart(sc_df.set_index('date'))
    diff = (pa - pb).mean()
    st.write(f"Average difference ({scenario_a} − {scenario_b}) over horizon: **{diff:+.1f} children/day**")

st.divider()
st.caption("Data source: HHS UAC Program daily reporting. Forecasts are model estimates for "
           "planning support only and should be interpreted alongside operational judgment.")
