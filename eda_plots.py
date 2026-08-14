import sys
sys.path.insert(0, 'src')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import json
from data_pipeline import build_dataset

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

df = build_dataset('/mnt/user-data/uploads/HHS_Unaccompanied_Alien_Children_Program.csv')

# 1. Time series overview
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(df['date'], df['hhs_care'], color='#1f2937', lw=1.2)
axes[0].set_title('Children in HHS Care (Care Load) — Jan 2023 to Dec 2025')
axes[0].set_ylabel('Children in care')
axes[0].grid(alpha=0.3)

axes[1].plot(df['date'], df['discharged'], color='#059669', lw=0.8)
axes[1].set_title('Children Discharged from HHS Care (Daily Placements)')
axes[1].set_ylabel('Discharges/day')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/01_timeseries_overview.png')
plt.close()

# 2. Decomposition (classical additive, manual: rolling trend + day-of-week seasonality)
series = df.set_index('date')['hhs_care']
trend = series.rolling(30, center=True, min_periods=1).mean()
detrended = (series - trend).reset_index(drop=True)
dow_key = df['date'].dt.dayofweek.reset_index(drop=True)
dow_seasonal = detrended.groupby(dow_key).transform('mean')
residual = detrended - dow_seasonal
dow_seasonal.index = series.index
residual.index = series.index

fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
axes[0].plot(series.index, series.values, color='#1f2937'); axes[0].set_ylabel('Observed'); axes[0].grid(alpha=0.3)
axes[1].plot(series.index, trend.values, color='#2563eb'); axes[1].set_ylabel('Trend (30d MA)'); axes[1].grid(alpha=0.3)
axes[2].plot(series.index, dow_seasonal.values, color='#d97706'); axes[2].set_ylabel('Weekly seasonal'); axes[2].grid(alpha=0.3)
axes[3].plot(series.index, residual.values, color='#dc2626', lw=0.6); axes[3].set_ylabel('Residual'); axes[3].grid(alpha=0.3)
axes[0].set_title('Time-Series Decomposition — Children in HHS Care')
plt.tight_layout()
plt.savefig('outputs/figures/02_decomposition.png')
plt.close()

# 3. Correlation heatmap
cols = ['intake', 'cbp_custody', 'transferred_to_hhs', 'hhs_care', 'discharged', 'net_pressure']
corr = df[cols].corr()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right')
ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=8,
                 color='white' if abs(corr.iloc[i,j])>0.5 else 'black')
plt.colorbar(im, fraction=0.046, pad=0.04)
ax.set_title('Correlation Matrix — Flow & Stock Variables')
plt.tight_layout()
plt.savefig('outputs/figures/03_correlation.png')
plt.close()

# 4. Day-of-week seasonality bar chart
dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
dow_avg_intake = df.groupby(df['date'].dt.dayofweek)['intake'].mean()
fig, ax = plt.subplots(figsize=(7,4))
ax.bar(dow_names, dow_avg_intake.reindex(range(7)).values, color='#2563eb')
ax.set_title('Average Daily Intake by Day of Week')
ax.set_ylabel('Avg. children apprehended/placed')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('outputs/figures/04_dow_seasonality.png')
plt.close()

# 5. Model comparison chart from evaluation results
with open('outputs/evaluation_results.json') as f:
    results = json.load(f)

for target, fname, title in [('hhs_care', '05_model_comparison_hhs_care.png', 'Care Load'),
                               ('discharged', '06_model_comparison_discharged.png', 'Discharge Demand')]:
    models = list(results[target].keys())
    horizons = [1, 7, 14, 30]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.13
    x = np.arange(len(horizons))
    colors = ['#94a3b8', '#64748b', '#2563eb', '#059669', '#d97706', '#dc2626']
    for i, m in enumerate(models):
        vals = [results[target][m].get(str(h), {}).get('RMSE', np.nan) for h in horizons]
        ax.bar(x + i*width, vals, width, label=m, color=colors[i % len(colors)])
    ax.set_xticks(x + width*2.5)
    ax.set_xticklabels([f'h={h}' for h in horizons])
    ax.set_ylabel('RMSE')
    ax.set_title(f'Model Comparison by Forecast Horizon — {title}')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'outputs/figures/{fname}')
    plt.close()

print("All EDA figures saved to outputs/figures/")
import os
print(os.listdir('outputs/figures'))
