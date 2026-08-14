"""
Forecasting models + evaluation utilities for the UAC care-load project.

Includes:
  - Baseline: naive persistence, moving average
  - Statistical: simple exponential smoothing, Holt's linear trend,
                 Holt-Winters seasonal, ARIMA/SARIMA (via statsmodels if available)
  - ML: RandomForestRegressor, GradientBoostingRegressor (recursive multi-step)

All models expose a common interface:
    fit(train_series)  -> fitted object
    forecast(fitted, h) -> np.array of length h  (point forecast)
    forecast_with_ci(fitted, h) -> (point, lower, upper)
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

def naive_forecast(series: pd.Series, h: int):
    last = series.iloc[-1]
    point = np.repeat(last, h)
    resid_std = series.diff().dropna().std()
    ci = 1.96 * resid_std * np.sqrt(np.arange(1, h + 1))
    return point, point - ci, point + ci


def moving_average_forecast(series: pd.Series, h: int, window: int = 7):
    ma = series.iloc[-window:].mean()
    point = np.repeat(ma, h)
    resid_std = series.diff().dropna().std()
    ci = 1.96 * resid_std * np.sqrt(np.arange(1, h + 1))
    return point, point - ci, point + ci


# ---------------------------------------------------------------------------
# Statistical models
# ---------------------------------------------------------------------------

def exp_smoothing_forecast(series: pd.Series, h: int):
    """Holt's linear trend exponential smoothing (handles trend, no seasonality)."""
    if STATSMODELS_AVAILABLE:
        model = Holt(series.values, initialization_method='estimated').fit(optimized=True)
        point = model.forecast(h)
        resid_std = np.std(model.resid)
        ci = 1.96 * resid_std * np.sqrt(np.arange(1, h + 1))
        return point, point - ci, point + ci
    else:
        # manual double exponential smoothing (Holt's method) fallback
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


def sarima_forecast(series: pd.Series, h: int, order=(2, 1, 2), seasonal_order=(1, 0, 1, 7)):
    """SARIMA with weekly seasonality (period=7). Falls back to ARIMA if statsmodels missing."""
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


# ---------------------------------------------------------------------------
# ML models (recursive multi-step forecasting using lag/rolling features)
# ---------------------------------------------------------------------------

FEATURE_COLS = None  # set dynamically by caller


def _make_ml_features(df, target_col):
    cols = [f'{target_col}_lag1', f'{target_col}_lag7', f'{target_col}_lag14',
            f'{target_col}_roll_mean7', f'{target_col}_roll_std7',
            f'{target_col}_roll_mean14', f'{target_col}_roll_std14',
            'day_of_week', 'month', 'net_pressure_roll7']
    cols = [c for c in cols if c in df.columns]
    return cols


def train_ml_model(df: pd.DataFrame, target_col: str, model_type='rf'):
    feat_cols = _make_ml_features(df, target_col)
    data = df.dropna(subset=feat_cols + [target_col])
    X, y = data[feat_cols], data[target_col]

    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                           learning_rate=0.05, random_state=42)
    model.fit(X, y)

    # residual std for CI (in-sample)
    resid_std = np.std(y - model.predict(X))
    return model, feat_cols, resid_std


def ml_recursive_forecast(model, feat_cols, df_history: pd.DataFrame, target_col: str, h: int, resid_std: float):
    """
    Recursive multi-step forecast: at each step, recompute lag/rolling features
    from the (growing) history including previous forecasts, then predict next day.
    """
    hist = df_history.copy().reset_index(drop=True)
    preds = []
    last_date = hist['date'].iloc[-1]

    for step in range(h):
        next_date = last_date + pd.Timedelta(days=step + 1)
        row = {}
        series = hist[target_col]
        row[f'{target_col}_lag1'] = series.iloc[-1]
        row[f'{target_col}_lag7'] = series.iloc[-7] if len(series) >= 7 else series.iloc[0]
        row[f'{target_col}_lag14'] = series.iloc[-14] if len(series) >= 14 else series.iloc[0]
        row[f'{target_col}_roll_mean7'] = series.iloc[-7:].mean()
        row[f'{target_col}_roll_std7'] = series.iloc[-7:].std()
        row[f'{target_col}_roll_mean14'] = series.iloc[-14:].mean()
        row[f'{target_col}_roll_std14'] = series.iloc[-14:].std()
        row['day_of_week'] = next_date.dayofweek
        row['month'] = next_date.month
        row['net_pressure_roll7'] = hist['net_pressure'].iloc[-7:].mean() if 'net_pressure' in hist else 0

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


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
