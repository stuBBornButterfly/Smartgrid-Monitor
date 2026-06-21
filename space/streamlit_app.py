"""
SmartGrid Monitor — Streamlit dashboard.
Talks to the FastAPI service over HTTP (true API integration).
Run:  streamlit run app/streamlit_app.py     (API must be running too)
"""
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="SmartGrid Monitor", page_icon="⚡", layout="wide")

API = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")


@st.cache_data(ttl=300)
def get(path, params=None):
    r = requests.get(f"{API}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


st.title("⚡ SmartGrid Monitor")
st.caption("Electricity load forecasting · anomaly detection · cost decision-support")

# ── Fetch core data from the API ─────────────────────────────────────────────
try:
    health = get("/health")
    nxt = get("/forecast/next")
    series = get("/forecast/series")
    anom = get("/anomalies")
except Exception as e:
    st.error(f"Cannot reach the API at {API}. Is it running?\n\n{e}")
    st.stop()

# ── Top metrics ──────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Next-hour load", f"{nxt['next_hour_load_mw']:,.0f} MW")
c2.metric("Anomalies found", anom["count"])
c3.metric("History window", f"{health['window']} h")

# ── Forecast chart with anomalies ────────────────────────────────────────────
ts = pd.to_datetime(series["time"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts, y=series["actual"], name="Actual", line=dict(width=1)))
fig.add_trace(go.Scatter(x=ts, y=series["predicted"], name="Predicted", line=dict(width=1)))
if anom["count"]:
    adf = pd.DataFrame(anom["anomalies"])
    fig.add_trace(go.Scatter(x=pd.to_datetime(adf["time"]), y=adf["actual"],
                             mode="markers", name="Anomaly",
                             marker=dict(color="red", size=8)))
fig.update_layout(title="Load — actual vs predicted (red = anomaly)",
                  xaxis_title="Time", yaxis_title="MW", height=450,
                  legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# ── Anomaly table ────────────────────────────────────────────────────────────
if anom["count"]:
    with st.expander(f"See {anom['count']} flagged hours"):
        st.dataframe(pd.DataFrame(anom["anomalies"]), use_container_width=True)

# ── Cost decision-support ────────────────────────────────────────────────────
st.subheader("💸 Cost decision-support — what-if peak shaving")
col1, col2 = st.columns(2)
shave = col1.slider("Shave load (%)", 0, 30, 10)
top = col2.slider("During the priciest hours (top %)", 5, 50, 10)

res = get("/cost/whatif", {"shave": shave / 100, "price_q": 1 - top / 100})
m1, m2, m3 = st.columns(3)
m1.metric("Hours affected", res["hours_affected"])
m2.metric("Estimated saving", f"€{res['saving']:,.0f}")
m3.metric("Share of total bill", f"{res['saving_pct']:.2f}%")
