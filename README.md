# Predictive Forecasting of Care Load & Placement Demand

Forecasting system for the HHS Unaccompanied Alien Children (UAC) Program —
predicts future **care load** (children in HHS care) and **discharge/placement
demand**, with baseline, statistical, and machine-learning models plus a
Streamlit decision-support dashboard.

## Project structure

```
uac_forecast/
├── data/
│   ├── HHS_Unaccompanied_Alien_Children_Program.csv   # raw dataset
│   └── featured_dataset.csv                           # cleaned + engineered (generated)
├── src/
│   ├── data_pipeline.py   # cleaning, daily resampling, feature engineering
│   ├── models.py          # all forecasting models + metrics
│   ├── evaluate.py        # walk-forward, multi-horizon evaluation
│   └── train_final.py     # trains & saves production ML models
├── models/                 # trained model artifacts (.pkl, generated)
├── outputs/
│   └── evaluation_results.json  # model comparison metrics (generated)
├── app/
│   └── app.py              # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Build features, evaluate models, train final models

```bash
python src/evaluate.py      # walk-forward evaluation -> outputs/evaluation_results.json
python src/train_final.py   # trains + saves RF/GB models -> models/*.pkl, data/featured_dataset.csv
```

## 2. Run the dashboard

```bash
streamlit run app/app.py
```

Then open the local URL Streamlit prints (typically http://localhost:8501).

## Notes

- `statsmodels` powers SARIMA / Holt exponential smoothing. If it isn't
  installed, the app automatically falls back to a manual exponential-
  smoothing approximation so the dashboard still runs — install it
  (`pip install statsmodels`) for true SARIMA forecasts.
- The raw HHS report is only published ~5-6 days a week (gaps on Fridays/
  Saturdays and some holidays). The pipeline reindexes to a continuous daily
  calendar and linearly interpolates gaps, per the required methodology.
- Re-run `src/evaluate.py` / `src/train_final.py` whenever you refresh the
  raw CSV with newer HHS data.
