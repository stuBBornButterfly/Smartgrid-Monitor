"""
Shared 'brain' for the SmartGrid Monitor app.
Loads the trained LSTM + scalers and runs forecast / anomaly / cost.
Both the FastAPI (api.py) and the Streamlit dashboard (streamlit_app.py) import this.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _find_artifacts_dir():
    """Locate the models/ folder in both local (app/../models) and
    flat Space (./models) layouts. ART_DIR env var overrides."""
    here = Path(__file__).resolve().parent
    candidates = [here / "models", here.parent / "models"]
    if os.environ.get("ART_DIR"):
        candidates.insert(0, Path(os.environ["ART_DIR"]))
    for c in candidates:
        if (c / "config.json").exists():
            return c
    return here.parent / "models"


# Folder that holds the artifacts downloaded from Kaggle
ART = _find_artifacts_dir()


# ── Same model structure as the training notebook (Cell 6) ───────────────────
class LoadLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class QuantileLSTM(nn.Module):
    """Day-ahead model: past window -> next `horizon` hours at `n_quantiles` levels."""
    def __init__(self, n_features, horizon, n_quantiles,
                 hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, horizon * n_quantiles)
        self.horizon, self.nq = horizon, n_quantiles

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.view(-1, self.horizon, self.nq)


# ── Load everything once ─────────────────────────────────────────────────────
def load_artifacts(art_dir=ART):
    art_dir = Path(art_dir)
    with open(art_dir / "config.json") as f:
        cfg = json.load(f)
    model = LoadLSTM(cfg["n_features"], cfg["hidden"], cfg["layers"])
    model.load_state_dict(torch.load(art_dir / "lstm_load.pt", map_location="cpu"))
    model.eval()
    feat_scaler = joblib.load(art_dir / "feat_scaler.pkl")
    target_scaler = joblib.load(art_dir / "target_scaler.pkl")
    return model, feat_scaler, target_scaler, cfg


def load_recent_data(art_dir=ART):
    df = pd.read_csv(Path(art_dir) / "recent_data.csv",
                     index_col=0, parse_dates=[0])
    return df


# ── Core operations ──────────────────────────────────────────────────────────
def predict_series(df, model, feat_scaler, target_scaler, cfg):
    """For each hour that has 24h of history, predict its load (MW)."""
    feats, W = cfg["feature_cols"], cfg["window"]
    X = feat_scaler.transform(df[feats])
    preds = np.full(len(df), np.nan)
    with torch.no_grad():
        for i in range(W, len(df)):
            win = torch.tensor(X[i - W:i], dtype=torch.float32).unsqueeze(0)
            preds[i] = target_scaler.inverse_transform(model(win).numpy())[0, 0]
    out = df.copy()
    out["predicted"] = preds
    return out


def predict_next(df, model, feat_scaler, target_scaler, cfg):
    """Predict the single next hour after the last row of df."""
    feats, W = cfg["feature_cols"], cfg["window"]
    X = feat_scaler.transform(df[feats].tail(W))
    win = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return float(target_scaler.inverse_transform(model(win).numpy())[0, 0])


def add_anomaly(out, sigma=3):
    res = out["load"] - out["predicted"]
    mu, sd = res.mean(), res.std()
    out = out.copy()
    out["residual"] = res
    out["anomaly"] = (res > mu + sigma * sd) | (res < mu - sigma * sd)
    return out


def add_cost(out):
    out = out.copy()
    out["actual_cost"] = out["load"] * out["price"]
    out["predicted_cost"] = out["predicted"] * out["price"]
    return out


def whatif_saving(out, shave=0.10, price_q=0.90):
    """Money saved by shaving `shave` of load during the priciest hours."""
    cut = out["price"].quantile(price_q)
    mask = out["price"] >= cut
    saving = float((out.loc[mask, "load"] * shave * out.loc[mask, "price"]).sum())
    total = float((out["load"] * out["price"]).sum())
    return {"hours_affected": int(mask.sum()),
            "saving": saving,
            "saving_pct": 100 * saving / total}


# ── Day-ahead (24h) probabilistic forecast ───────────────────────────────────
def load_quantile_artifacts(art_dir=ART):
    art_dir = Path(art_dir)
    with open(art_dir / "quantile_config.json") as f:
        cfg = json.load(f)
    model = QuantileLSTM(cfg["n_features"], cfg["output_h"], len(cfg["quantiles"]),
                         hidden=cfg["hidden"], layers=cfg["layers"])
    model.load_state_dict(torch.load(art_dir / "quantile_lstm.pt", map_location="cpu"))
    model.eval()
    return model, cfg


def predict_dayahead(df, qmodel, qcfg, feat_scaler, target_scaler):
    """Predict the next `output_h` hours with a calibrated P10-P90 band."""
    feats, W, c = qcfg["feature_cols"], qcfg["input_w"], qcfg["calib_c"]
    X = feat_scaler.transform(df[feats].tail(W))
    x = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = qmodel(x).numpy()[0]          # (horizon, 3), scaled
    inv = lambda a: target_scaler.inverse_transform(a.reshape(-1, 1)).flatten()
    p10, p50, p90 = inv(out[:, 0]), inv(out[:, 1]), inv(out[:, 2])
    low  = p50 - c * (p50 - p10)            # apply calibration width factor
    high = p50 + c * (p90 - p50)
    return {"hours_ahead": list(range(1, qcfg["output_h"] + 1)),
            "median": p50.round(1).tolist(),
            "low":  low.round(1).tolist(),
            "high": high.round(1).tolist()}
