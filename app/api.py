"""
FastAPI REST service for SmartGrid Monitor.
Run:  uvicorn api:app --app-dir app --port 8000
Docs: http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, Query

import model_utils as m

app = FastAPI(title="SmartGrid Monitor API", version="1.0")

# Load the model + data ONCE when the server starts (not per request)
model, feat_scaler, target_scaler, cfg = m.load_artifacts()
recent = m.load_recent_data()
_cache = {}

# Day-ahead model is optional — API still runs if its files aren't present yet
try:
    qmodel_da, qcfg = m.load_quantile_artifacts()
except Exception as e:
    qmodel_da, qcfg = None, None
    print("Day-ahead model not loaded:", e)


def get_processed():
    """Run forecast + anomaly + cost over the recent data once, then reuse."""
    if "df" not in _cache:
        out = m.predict_series(recent, model, feat_scaler, target_scaler, cfg)
        out = m.add_anomaly(out)
        out = m.add_cost(out)
        _cache["df"] = out
    return _cache["df"]


@app.get("/")
def root():
    return {"service": "SmartGrid Monitor API",
            "docs": "/docs",
            "endpoints": ["/health", "/forecast/next", "/forecast/series",
                          "/forecast/dayahead", "/anomalies", "/cost/whatif"]}


@app.get("/health")
def health():
    return {"status": "ok", "rows": len(recent), "window": cfg["window"]}


@app.get("/forecast/next")
def forecast_next():
    val = m.predict_next(recent, model, feat_scaler, target_scaler, cfg)
    return {"from_time": str(recent.index[-1]),
            "next_hour_load_mw": round(val, 1)}


@app.get("/forecast/series")
def forecast_series():
    out = get_processed().dropna(subset=["predicted"])
    return {"time": [str(t) for t in out.index],
            "actual": out["load"].round(1).tolist(),
            "predicted": out["predicted"].round(1).tolist()}


@app.get("/anomalies")
def anomalies():
    a = get_processed()
    a = a[a["anomaly"]]
    return {"count": int(len(a)),
            "anomalies": [{"time": str(t),
                           "actual": round(r.load, 1),
                           "predicted": round(r.predicted, 1),
                           "residual": round(r.residual, 1)}
                          for t, r in a.iterrows()]}


@app.get("/cost/whatif")
def cost_whatif(shave: float = Query(0.10, ge=0, le=1),
                price_q: float = Query(0.90, ge=0, le=1)):
    return m.whatif_saving(get_processed(), shave=shave, price_q=price_q)


@app.get("/forecast/dayahead")
def forecast_dayahead():
    if qmodel_da is None:
        return {"error": "day-ahead model not available"}
    return m.predict_dayahead(recent, qmodel_da, qcfg, feat_scaler, target_scaler)
