"""
Strict time-based train/test split + walk-forward, multi-horizon evaluation
of all forecasting models for both targets: hhs_care (care load) and
discharged (discharge/placement demand).
"""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'src')
from data_pipeline import build_dataset
from models import (naive_forecast, moving_average_forecast, exp_smoothing_forecast,
                     sarima_forecast, train_ml_model, ml_recursive_forecast,
                     mae, rmse, mape, STATSMODELS_AVAILABLE)

HORIZONS = [1, 7, 14, 30]
N_TEST_ORIGINS = 6          # number of walk-forward evaluation windows
TEST_SPACING = 30           # days between successive walk-forward origins


def get_model_forecast(name, train_df, target_col, h):
    series = train_df[target_col]
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
        model, feat_cols, resid_std = train_ml_model(train_df, target_col, mtype)
        return ml_recursive_forecast(model, feat_cols, train_df, target_col, h, resid_std)
    raise ValueError(name)


def run_evaluation(df, target_col):
    model_names = ['Naive Persistence', 'Moving Average (7d)', 'Exponential Smoothing',
                    'SARIMA', 'Random Forest', 'Gradient Boosting']

    max_h = max(HORIZONS)
    n = len(df)
    # walk-forward origins: spaced out, leaving room for max horizon at the end
    last_origin = n - max_h - 1
    origins = [last_origin - i * TEST_SPACING for i in range(N_TEST_ORIGINS)]
    origins = [o for o in origins if o > 100]  # need enough history to train
    origins = sorted(origins)

    results = {m: {h: [] for h in HORIZONS} for m in model_names}

    for origin in origins:
        train_df = df.iloc[:origin + 1].copy()
        actual_future = df[target_col].iloc[origin + 1: origin + 1 + max_h].values

        for name in model_names:
            try:
                point, lo, hi = get_model_forecast(name, train_df, target_col, max_h)
            except Exception as e:
                print(f"  [WARN] {name} failed at origin {origin}: {e}")
                continue
            for h in HORIZONS:
                y_true = actual_future[:h]
                y_pred = point[:h]
                if len(y_true) < h:
                    continue
                results[name][h].append({
                    'mae': mae(y_true, y_pred),
                    'rmse': rmse(y_true, y_pred),
                    'mape': mape(y_true, y_pred),
                })

    # aggregate
    summary = {}
    for name in model_names:
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
    return summary


if __name__ == '__main__':
    print("statsmodels available:", STATSMODELS_AVAILABLE)
    df = build_dataset('/mnt/user-data/uploads/HHS_Unaccompanied_Alien_Children_Program.csv')

    all_results = {}
    for target in ['hhs_care', 'discharged']:
        print(f"\n=== Evaluating target: {target} ===")
        summary = run_evaluation(df, target)
        all_results[target] = summary
        for model, hz in summary.items():
            print(f"\n{model}")
            for h, m in hz.items():
                print(f"  h={h:>2}: MAE={m['MAE']:>8} RMSE={m['RMSE']:>8} MAPE={m['MAPE']:>6}%  (n={m['n_folds']})")

    with open('outputs/evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved outputs/evaluation_results.json")
