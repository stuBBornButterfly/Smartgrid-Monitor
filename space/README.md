---
title: SmartGrid Monitor
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# ⚡ SmartGrid Monitor

Electricity **load forecasting + anomaly detection + cost decision-support**, powered by AI.

The dashboard (Streamlit, public) calls a **FastAPI** service running inside the same
container — demonstrating live model-serving over an API.

- **Forecast** — LSTM predicts next-hour electricity load (~0.98% MAPE on test).
- **Anomaly detection** — flags abnormal hours via 3σ forecast residuals.
- **Cost + decision-support** — real prices → bill estimate + peak-shaving what-if.

Built on the UCI / Spain hourly energy dataset. No paid APIs.

## Architecture
```
Streamlit dashboard (7860, public)
        │  HTTP
        ▼
FastAPI model service (8000, internal)
        │
        ▼
LSTM + scalers (models/)
```
