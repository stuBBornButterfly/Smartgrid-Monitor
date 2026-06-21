"""
SmartGrid Monitor — Streamlit dashboard (professional dark UI).
Talks to the FastAPI service over HTTP. Run: streamlit run app/streamlit_app.py
"""
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="SmartGrid Monitor", page_icon="⚡", layout="wide")

# ── Brand palette ────────────────────────────────────────────────────────────
C_ACTUAL, C_PRED, C_ANOM, C_COST = "#38bdf8", "#10b981", "#f43f5e", "#f59e0b"

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden;}
.block-container {padding-top:1.2rem; padding-bottom:2rem; max-width:1250px;}
.hero {background:linear-gradient(135deg,#0f766e 0%,#0e7490 55%,#1e3a8a 100%);
       border-radius:18px; padding:26px 32px; margin-bottom:20px;
       box-shadow:0 10px 30px rgba(8,47,73,.45);}
.hero h1 {margin:0; font-size:32px; font-weight:800; color:#f0fdfa; letter-spacing:-.5px;}
.hero p  {margin:6px 0 0; color:#a5f3fc; font-size:15px;}
.kpi {background:linear-gradient(160deg,#1e293b,#0f172a); border:1px solid rgba(148,163,184,.16);
      border-radius:16px; padding:18px 20px; box-shadow:0 6px 22px rgba(0,0,0,.28); height:100%;}
.kpi .lbl {font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#94a3b8;}
.kpi .val {font-size:30px; font-weight:800; margin-top:6px; line-height:1.1;}
.kpi .sub {font-size:12px; color:#64748b; margin-top:4px;}
.stTabs [data-baseweb="tab-list"] {gap:6px;}
.stTabs [data-baseweb="tab"] {background:#1e293b; border-radius:10px 10px 0 0; padding:8px 18px;}
.stTabs [aria-selected="true"] {background:#0e7490 !important; color:#ecfeff !important;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ SmartGrid Monitor")
    API = st.text_input("API endpoint", "http://127.0.0.1:8000")
    st.markdown("---")
    st.markdown("**Models**")
    st.caption("• LSTM next-hour — 0.97% MAPE\n\n"
               "• Quantile LSTM — 24h + uncertainty\n\n"
               "• 3σ residual anomaly detector\n\n"
               "• Real-price cost + what-if")
    st.markdown("---")
    st.caption("FastAPI · PyTorch · Streamlit · HF Spaces")


@st.cache_data(ttl=300)
def get(path, params=None):
    r = requests.get(f"{API}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def kpi(col, lbl, val, sub, color):
    col.markdown(f'<div class="kpi"><div class="lbl">{lbl}</div>'
                 f'<div class="val" style="color:{color}">{val}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def style_fig(fig, height=430):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#cbd5e1"),
                      legend=dict(orientation="h", y=1.12, x=0),
                      xaxis=dict(gridcolor="rgba(148,163,184,.12)"),
                      yaxis=dict(gridcolor="rgba(148,163,184,.12)"))
    return fig


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero"><h1>⚡ SmartGrid Monitor</h1>'
            '<p>AI-powered electricity load forecasting · anomaly detection · cost decision-support</p>'
            '</div>', unsafe_allow_html=True)

# ── Fetch ────────────────────────────────────────────────────────────────────
try:
    health = get("/health")
    nxt    = get("/forecast/next")
    series = get("/forecast/series")
    anom   = get("/anomalies")
    da     = get("/forecast/dayahead")
except Exception as e:
    st.error(f"Cannot reach the API at {API}. Is it running?\n\n{e}")
    st.stop()

has_da = isinstance(da, dict) and "error" not in da

# ── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Next-hour load", f"{nxt['next_hour_load_mw']:,.0f}", "MW · live", C_PRED)
peak = f"{max(da['median']):,.0f}" if has_da else "—"
kpi(k2, "Day-ahead peak", peak, "MW · next 24h", C_ACTUAL)
kpi(k3, "Anomalies", f"{anom['count']}", "flagged in window", C_ANOM)
kpi(k4, "History window", f"{health['window']}", "hours of context", C_COST)

st.write("")

# ── Tabs ─────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["📈  Recent forecast", "🔮  Day-ahead", "⚠️  Anomalies", "💸  Cost"])

with t1:
    ts = pd.to_datetime(series["time"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=series["actual"], name="Actual",
                             line=dict(color=C_ACTUAL, width=1.4)))
    fig.add_trace(go.Scatter(x=ts, y=series["predicted"], name="Predicted",
                             line=dict(color=C_PRED, width=1.4)))
    if anom["count"]:
        adf = pd.DataFrame(anom["anomalies"])
        fig.add_trace(go.Scatter(x=pd.to_datetime(adf["time"]), y=adf["actual"],
                                 mode="markers", name="Anomaly",
                                 marker=dict(color=C_ANOM, size=7, line=dict(width=1, color="#fff"))))
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption("Hourly actual vs LSTM prediction. Red points = abnormal hours (3σ residual).")

with t2:
    st.markdown("##### Next 24 hours with calibrated uncertainty band")
    if not has_da:
        st.info("Day-ahead model not loaded on the server.")
    else:
        h = da["hours_ahead"]
        figd = go.Figure()
        figd.add_trace(go.Scatter(x=h, y=da["high"], line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
        figd.add_trace(go.Scatter(x=h, y=da["low"], fill="tonexty",
                                  fillcolor="rgba(16,185,129,.18)", line=dict(width=0),
                                  name="P10–P90 band"))
        figd.add_trace(go.Scatter(x=h, y=da["median"], name="Median forecast",
                                  line=dict(color=C_PRED, width=2.6)))
        figd.update_xaxes(title="Hours ahead"); figd.update_yaxes(title="Load (MW)")
        st.plotly_chart(style_fig(figd), use_container_width=True)
        st.caption("Shaded band ≈ 80% confidence (conformally calibrated). Wider band = less certain.")

with t3:
    c1, c2 = st.columns([1, 2])
    c1.metric("Hours flagged", anom["count"])
    c1.metric("Flag rate", f"{100*anom['count']/max(len(series['time']),1):.2f}%")
    if anom["count"]:
        c2.dataframe(pd.DataFrame(anom["anomalies"]), use_container_width=True, height=320)
    else:
        c2.success("No anomalies in the current window.")

with t4:
    st.markdown("##### What-if peak shaving")
    col1, col2 = st.columns(2)
    shave = col1.slider("Shave load (%)", 0, 30, 10)
    top   = col2.slider("During the priciest hours (top %)", 5, 50, 10)
    res = get("/cost/whatif", {"shave": shave / 100, "price_q": 1 - top / 100})
    m1, m2, m3 = st.columns(3)
    kpi(m1, "Hours affected", f"{res['hours_affected']}", "priciest hours", C_COST)
    kpi(m2, "Estimated saving", f"€{res['saving']:,.0f}", "over test period", C_PRED)
    kpi(m3, "Share of bill", f"{res['saving_pct']:.2f}%", "total cost", C_ACTUAL)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<div style="text-align:center;color:#64748b;font-size:12px;margin-top:26px;">'
            'SmartGrid Monitor · FastAPI + Streamlit · PyTorch LSTM · deployed on Hugging Face Spaces'
            '</div>', unsafe_allow_html=True)
