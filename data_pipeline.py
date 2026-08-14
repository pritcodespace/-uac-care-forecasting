"""
Data loading, cleaning, resampling and feature engineering pipeline
for the UAC (Unaccompanied Alien Children) HHS care-load forecasting project.
"""
import pandas as pd
import numpy as np


def load_and_clean(raw_path: str) -> pd.DataFrame:
    """Load the raw HHS UAC CSV and return a clean, sorted dataframe."""
    df = pd.read_csv(raw_path)
    df = df.dropna(subset=[df.columns[0]]).copy()
    df.columns = ['date', 'intake', 'cbp_custody', 'transferred_to_hhs',
                  'hhs_care', 'discharged']

    for c in ['intake', 'cbp_custody', 'transferred_to_hhs', 'hhs_care', 'discharged']:
        df[c] = df[c].astype(str).str.replace(',', '', regex=False)
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df['date'] = pd.to_datetime(df['date'], format='%B %d, %Y', errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    df = df.drop_duplicates(subset='date', keep='last')
    return df


def make_continuous_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    HHS only reports on ~5-6 days a week (gaps on Fridays/Saturdays and
    occasional holidays / reporting lapses). Reindex to a full daily
    calendar and interpolate the gaps so the series is suitable for
    time-series decomposition and lag/rolling features.
    """
    reported_dates = set(df['date'])
    df = df.set_index('date').sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(full_idx)
    df.index.name = 'date'

    # Flow variables: interpolate then round (activity between reports)
    for c in ['intake', 'transferred_to_hhs', 'discharged']:
        df[c] = df[c].interpolate(method='linear').round()

    # Stock/level variables: interpolate (level persists smoothly between reports)
    for c in ['cbp_custody', 'hhs_care']:
        df[c] = df[c].interpolate(method='linear')

    df = df.reset_index()
    df['is_reported'] = df['date'].isin(reported_dates)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering per the project's analytical methodology."""
    df = df.copy()

    # net pressure indicator: inflow to HHS minus outflow (discharge)
    df['net_pressure'] = df['transferred_to_hhs'] - df['discharged']

    # calendar effects
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    target_cols = ['hhs_care', 'discharged']
    for col in target_cols:
        for lag in [1, 7, 14]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        for win in [7, 14]:
            df[f'{col}_roll_mean{win}'] = df[col].shift(1).rolling(win).mean()
            df[f'{col}_roll_std{win}'] = df[col].shift(1).rolling(win).std()

    df['net_pressure_roll7'] = df['net_pressure'].shift(1).rolling(7).mean()

    return df


def build_dataset(raw_path: str) -> pd.DataFrame:
    df = load_and_clean(raw_path)
    df = make_continuous_daily(df)
    df = add_features(df)
    return df


if __name__ == '__main__':
    out = build_dataset('/mnt/user-data/uploads/HHS_Unaccompanied_Alien_Children_Program.csv')
    out.to_csv('data/featured_dataset.csv', index=False)
    print(out.shape)
    print(out.tail())
    print("Nulls per col:\n", out.isnull().sum())
