# SmartGrid Monitor

Web-based electricity **load forecasting + anomaly detection + cost decision-support**, powered by AI.

Built for: BEPRC RA application (DIU EEE) — mirrors the funded project
*"Web-Based Remote Monitoring, Electricity Cost Modeling, and Decision Support System Powered by AI."*

## What it does
1. **Forecast** — predict future electricity load from past usage (LSTM / N-BEATS).
2. **Anomaly detection** — flag abnormal consumption (spikes, theft, faults).
3. **Cost model** — convert kWh to bill via Bangladesh tariff slabs + what-if simulation.
4. **Web app** — Streamlit dashboard + FastAPI endpoint, deployed on Hugging Face Spaces.

## Dataset
UCI *Individual Household Electric Power Consumption* (Kaggle: `uciml/electric-power-consumption-data-set`).
No paid APIs.

## Structure
```
notebooks/   Kaggle training notebooks (forecast, anomaly, cost)
data/        raw + processed data (gitignored, not committed)
models/      saved artifacts (.pt, .pkl)
app/         Streamlit + FastAPI web app for HF Spaces
reports/     plots, metrics, figures for the writeup
```

## Stack
Python, PyTorch, scikit-learn, pandas, Streamlit, FastAPI, Hugging Face Spaces.

## Workflow
Train on Kaggle (GPU) → export model artifacts → serve via app on HF Spaces.
