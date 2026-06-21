# SmartGrid Monitor — Project Writeup

**AI-powered electricity load forecasting, anomaly detection, and cost decision-support, deployed as a live web service.**

🔗 Live demo: https://stubbornbutterfly-smartgrid-monitor.hf.space
🔗 Code: https://github.com/stuBBornButterfly/Smartgrid-Monitor

---

## Summary (cover-letter ready)

I built **SmartGrid Monitor**, an end-to-end machine-learning system for electricity grids that
forecasts demand, detects abnormal consumption, and models cost — then serves it all through a live
web platform. Working from a public hourly dataset of electricity load, real market prices, and
weather, I engineered a full pipeline: data cleaning, feature engineering, deep-learning forecasting,
probabilistic (uncertainty-aware) prediction, model benchmarking, interpretability analysis, and
deployment. The trained models are served by a **FastAPI** REST service and visualised in a
**Streamlit** dashboard, containerised with Docker and deployed live on Hugging Face Spaces. The
project directly mirrors the responsibilities of the advertised position — time-series forecasting,
anomaly detection, cost modelling, decision support, and integrating AI models into web platforms via
APIs.

---

## Key results

- **Short-term load forecasting** — an LSTM network predicts next-hour demand at **0.97% MAPE**
  (MAE 273 MW on ~28,700 MW average load).
- **Model benchmarking** — the LSTM (0.97%) and an XGBoost baseline (0.98%) outperform persistence
  (3.71%) and seasonal-naive (8.60%) baselines by **4–9×**, justifying the model choice with evidence.
- **Interpretability (SHAP)** — feature-attribution analysis confirms the forecast is driven by
  recent load, solar generation (a daylight proxy), and time-of-day — i.e. the model learned real
  grid behaviour rather than spurious noise.
- **Day-ahead probabilistic forecasting** — a quantile LSTM predicts the full **next 24 hours** with
  P10–P90 uncertainty bands at **4.66% MAPE** and **76.9% interval coverage** (target 80%), giving
  calibrated, decision-useful risk ranges. Overfitting was diagnosed and corrected with a held-out
  validation set, early stopping, and weight decay.
- **Anomaly detection** — a forecast-residual detector (3σ) flags abnormal hours (e.g. a sudden
  12,286 MW unexpected drop), surfacing faults/events for operator attention.
- **Cost modelling & decision support** — using real prices, the system estimates the bill to within
  **0.16%** and runs what-if peak-shaving scenarios (e.g. trimming 10% of load during the priciest
  hours yields ~€180M / 1.36% savings over the test period).
- **Deployment** — FastAPI REST API + Streamlit dashboard, Dockerised and live on Hugging Face Spaces.

---

## How it maps to the job requirements

| Requirement in the call | What I delivered |
| --- | --- |
| Time-series forecasting | LSTM next-hour (0.97% MAPE) + quantile LSTM day-ahead with uncertainty |
| Anomaly detection | 3σ forecast-residual detector |
| Electricity cost modelling | Real-price bill estimation (0.16% error) |
| Decision support | What-if peak-shaving simulator |
| Data collection & preprocessing | Cleaning, gap-filling, weather merge, feature engineering |
| Integrate AI into web platforms via APIs | FastAPI REST service consumed by a Streamlit dashboard |
| Model evaluation & rigour | Benchmark table, SHAP interpretability, calibration, overfit control |
| Technical reporting | This writeup + reproducible notebooks and figures |

---

## Technical stack

Python · PyTorch · scikit-learn · XGBoost · SHAP · pandas · NumPy · FastAPI · Streamlit · Plotly ·
Docker · Hugging Face Spaces. Trained on Kaggle GPU. No paid APIs.

## Data

Hourly electricity load, day-ahead/actual market prices, and multi-city weather
(`nicholasjhana/energy-consumption-generation-prices-and-weather`).
