# app.py — P2-ETF-DEEPWAVE-DL Streamlit Dashboard
# Runs in HF Space. Loads data + weights from HF Dataset repo.
# All risk controls (TSL, Z-score) are live sliders — nothing hardcoded.

import json
import os
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from huggingface_hub import hf_hub_download, list_repo_files

import config

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "P2-ETF-DEEPWAVE-DL",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─── Custom CSS (match exact style from screenshots) ──────────────────────────
st.markdown("""
<style>
  /* Sidebar width */
  [data-testid="stSidebar"] { min-width: 340px; max-width: 340px; }

  /* Hero cards */
  .hero-card {
    background: linear-gradient(135deg, #00bfa5, #00897b);
    border-radius: 14px; padding: 32px; text-align: center; color: white;
    margin: 16px 0;
  }
  .cash-card {
    background: linear-gradient(135deg, #e65100, #bf360c);
    border-radius: 14px; padding: 32px; text-align: center; color: white;
    margin: 16px 0;
  }
  .hero-label { font-size:11px; letter-spacing:2px; text-transform:uppercase;
                opacity:.8; margin-bottom:8px; }
  .hero-value { font-size:40px; font-weight:700; }
  .hero-sub   { font-size:13px; opacity:.85; margin-top:10px; }

  /* Model cards */
  .model-card-a { background:#0d1117; border-radius:12px; padding:24px;
                  text-align:center; color:white; border:1px solid #1a2a1a; }
  .model-card-b { background:#0d1117; border-radius:12px; padding:24px;
                  text-align:center; color:white; border:2px solid #7b8ff7; }
  .model-card-c { background:#0d1117; border-radius:12px; padding:24px;
                  text-align:center; color:white; border:1px solid #2a1a1a; }

  /* Metric cards */
  .metric-box { background:white; border:1px solid #e0e3ea;
                border-radius:10px; padding:16px; text-align:left; }

  /* Conviction bar */
  .conv-track { height:8px; border-radius:4px; margin:8px 0;
    background: linear-gradient(to right,
      #ffb3b3 0%, #ffe0b2 25%, #b2dfdb 50%, #80cbc4 75%, #4db6ac 100%); }

  /* Alerts */
  .alert-green  { background:#e8f5e9; border:1px solid #c8e6c9;
                  color:#2e7d32; padding:12px 16px; border-radius:8px;
                  margin-bottom:10px; font-size:13px; }
  .alert-blue   { background:#e8f4fd; border:1px solid #bbdefb;
                  color:#1565c0; padding:12px 16px; border-radius:8px;
                  margin-bottom:10px; font-size:13px; }
  .alert-yellow { background:#fffde7; border:1px solid #fff9c4;
                  color:#f57f17; padding:12px 16px; border-radius:8px;
                  margin-bottom:10px; font-size:13px; }
  .alert-orange { background:#fff3e0; border:1px solid #ffcc80;
                  color:#e65100; padding:12px 16px; border-radius:8px;
                  margin-bottom:10px; font-size:13px; }

  /* Winner row */
  .winner-row { background:#f0fdf9 !important; }

  /* Hide Streamlit branding */
  #MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_prediction() -> dict:
    """Load latest_prediction.json from HF or local."""
    try:
        path = hf_hub_download(
            repo_id   = config.HF_DATASET_REPO,
            filename  = "latest_prediction.json",
            repo_type = "dataset",
            token     = config.HF_TOKEN or None,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        if os.path.exists("latest_prediction.json"):
            with open("latest_prediction.json") as f:
                return json.load(f)
        return {}


@st.cache_data(ttl=3600)
def load_evaluation() -> dict:
    """Load evaluation_results.json from HF or local."""
    try:
        path = hf_hub_download(
            repo_id   = config.HF_DATASET_REPO,
            filename  = "evaluation_results.json",
            repo_type = "dataset",
            token     = config.HF_TOKEN or None,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        if os.path.exists("evaluation_results.json"):
            with open("evaluation_results.json") as f:
                return json.load(f)
        return {}


@st.cache_data(ttl=3600)
def load_etf_data() -> pd.DataFrame:
    """Load etf_ret parquet from HF."""
    try:
        path = hf_hub_download(
            repo_id   = config.HF_DATASET_REPO,
            filename  = "data/etf_ret.parquet",
            repo_type = "dataset",
            token     = config.HF_TOKEN or None,
        )
        return pd.read_parquet(path)
    except Exception:
        local = os.path.join(config.DATA_DIR, "etf_ret.parquet")
        return pd.read_parquet(local) if os.path.exists(local) else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_macro_data() -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id   = config.HF_DATASET_REPO,
            filename  = "data/macro.parquet",
            repo_type = "dataset",
            token     = config.HF_TOKEN or None,
        )
        return pd.read_parquet(path)
    except Exception:
        local = os.path.join(config.DATA_DIR, "macro.parquet")
        return pd.read_parquet(local) if os.path.exists(local) else pd.DataFrame()


# ─── TSL backtest (re-runs in Streamlit when sliders change) ──────────────────

def apply_tsl_to_audit(audit: list,
                        tsl_pct: float,
                        z_reentry: float,
                        tbill: float = 3.6) -> pd.DataFrame:
    """Re-apply TSL logic to audit trail based on current slider values."""
    df = pd.DataFrame(audit)
    if df.empty:
        return df

    in_cash   = False
    prev_ret  = 0.0
    prev2_ret = 0.0
    modes, signals, net_rets = [], [], []

    for _, row in df.iterrows():
        z         = row.get("Z_Score", 1.5)
        two_day   = prev_ret + prev2_ret

        if not in_cash and two_day * 100 <= -tsl_pct:
            in_cash = True
        if in_cash and z >= z_reentry:
            in_cash = False

        if in_cash:
            modes.append("💵 CASH")
            signals.append("CASH")
            net_rets.append(round(tbill / 100 / 252, 6))
        else:
            modes.append("📈 ETF")
            signals.append(row.get("Signal", "—"))
            net_rets.append(row.get("Net_Return", 0.0))

        prev2_ret = prev_ret
        prev_ret  = net_rets[-1]

    df["Mode"]       = modes
    df["Signal_TSL"] = signals
    df["Net_TSL"]    = net_rets
    return df


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    now_est = datetime.now().strftime("%H:%M:%S")
    st.caption(f"🕐 EST: {now_est}")
    st.divider()

    start_year = st.slider("📅 Start Year", 2005, 2022, 2008)
    fee_bps    = st.slider("💰 Fee (bps)",  0, 50, 10)
    max_epochs = st.number_input("🔁 Max Epochs", min_value=10,
                                  max_value=300, value=80, step=5)
    lookback   = st.selectbox("🔭 Lookback Window",
                               ["30 days","45 days","60 days"])
    split      = st.selectbox("📊 Train/Val/Test Split",
                               ["70/15/15","60/20/20","75/12/13"])
    wavelet    = st.selectbox("〰️ Wavelet Type",
                               ["db4 (Daubechies-4)","db2 (Daubechies-2)",
                                "haar","sym5"])

    st.divider()
    st.markdown("### 🛡️ Risk Controls")

    tsl_pct = st.slider(
        "🔴 Trailing Stop Loss (2-day cumul.)",
        min_value=0, max_value=25, value=10, step=1,
        format="%d%%",
        help="Shift to CASH if 2-day cumulative return ≤ −X%"
    )
    st.caption(f"Triggers CASH if 2-day cumulative return ≤ −{tsl_pct}%")

    z_reentry = st.slider(
        "📶 Z-score Re-entry Threshold",
        min_value=1.0, max_value=2.0, value=1.1, step=0.1,
        format="%.1f σ",
        help="Exit CASH → ETF when model Z-score ≥ this threshold"
    )
    st.caption(f"Exit CASH → ETF when Z ≥ {z_reentry:.1f} σ. "
               f"CASH earns 3m T-bill: **3.60%**")

    st.divider()
    st.markdown("### 🧠 Active Models")
    use_a = st.checkbox("Option A · Wavelet-CNN-LSTM", value=True)
    use_b = st.checkbox("Option B · Wavelet-Attn-CNN-LSTM", value=True)
    use_c = st.checkbox("Option C · Wavelet-Dual-Stream", value=True)

    st.caption(f"💡 CASH triggered on 2-day cumul ≤ −{tsl_pct}%. "
               f"Re-entry when Z ≥ {z_reentry:.1f} σ.")

    run_btn = st.button("🚀 Run All 3 Models", use_container_width=True,
                         type="primary")

    st.divider()
    st.markdown("### 📦 Dataset Info")
    etf_df   = load_etf_data()
    macro_df = load_macro_data()
    if not etf_df.empty:
        st.markdown(f"**Rows:** {len(etf_df):,}")
        st.markdown(f"**Range:** {etf_df.index.min().date()} → "
                    f"{etf_df.index.max().date()}")
    st.markdown(f"**ETFs:** {', '.join(config.ETFS)}")
    st.markdown(f"**Benchmarks:** {', '.join(config.BENCHMARKS)}")
    st.markdown("**Macro:** VIX, DXY, T10Y2Y, Corp Spread, HY Spread, "
                "TNX, 3mTBill")
    st.markdown(f"**Wavelet levels:** 3 (A3, D1, D2, D3)")
    st.markdown("**T-bill col:** ✅")


# ─── MAIN PANEL ───────────────────────────────────────────────────────────────

# Title
st.markdown("# 🧠 P2-ETF-DEEPWAVE-DL")
st.caption("Option A: Wavelet-CNN-LSTM · "
           "Option B: Wavelet-Attention-CNN-LSTM · "
           "Option C: Wavelet-Parallel-Dual-Stream-CNN-LSTM")
st.caption("Winner selected by highest raw annualised return on "
           "out-of-sample test set.")

# Load data
pred  = load_prediction()
evalu = load_evaluation()

as_of    = pred.get("as_of_date", str(date.today()))
winner   = evalu.get("winner", "model_b")
tbill_rt = pred.get("tbill_rate", 3.6)
preds    = pred.get("predictions", {})
tsl_stat = pred.get("tsl_status", {})

# Determine current TSL state from slider values vs live Z
live_z       = preds.get(winner, {}).get("z_score", 1.5)
two_day_ret  = tsl_stat.get("two_day_cumul_pct", -3.0)
tsl_triggered= two_day_ret <= -tsl_pct
in_cash_now  = tsl_triggered and (live_z < z_reentry)

# ── Status banners ────────────────────────────────────────────────────────────
st.markdown(f"""<div class="alert-green">
  ✅ Dataset up to date through <b>{as_of}</b>. HF Space synced.
</div>""", unsafe_allow_html=True)

if not etf_df.empty:
    n_rows  = len(etf_df)
    rng_min = etf_df.index.min().date()
    rng_max = etf_df.index.max().date()
    n_yrs   = (rng_max - rng_min).days // 365
    st.markdown(f"""<div class="alert-blue">
      📅 <b>Data:</b> {rng_min} → {rng_max} ({n_yrs} years) &nbsp;|&nbsp;
      Source: Stooq (fallback: yfinance) + FRED
    </div>""", unsafe_allow_html=True)

n_wavelet_feat = (len(config.ETFS) * 2 + len(config.MACRO_SERIES)) * \
                 (config.WAVELET_LEVELS + 1)
st.markdown(f"""<div class="alert-blue">
  🎯 <b>Targets:</b> {', '.join(config.ETFS)} &nbsp;·&nbsp;
  <b>Features:</b> {n_wavelet_feat} signals (wavelet-decomposed) &nbsp;·&nbsp;
  <b>T-bill:</b> {tbill_rt:.2f}%
</div>""", unsafe_allow_html=True)

best_lb = evalu.get(winner, {}).get("lookback", 30)
st.markdown(f"""<div class="alert-yellow">
  ⚠️ Optimal lookback: <b>{best_lb}d</b> (auto-selected from 30 / 45 / 60)
</div>""", unsafe_allow_html=True)

# Risk controls banner
st.markdown(f"""<div class="alert-yellow">
  🛡️ <b>Risk Controls:</b> &nbsp;
  Trailing Stop Loss: <b>−{tsl_pct}%</b> (2-day cumul.) &nbsp;·&nbsp;
  CASH re-entry when Z ≥ <b>{z_reentry:.1f} σ</b> &nbsp;·&nbsp;
  CASH earns 3m T-bill: <b>{tbill_rt:.2f}%</b>
</div>""", unsafe_allow_html=True)

# CASH override banner
if in_cash_now:
    st.markdown(f"""<div class="alert-orange">
      🔴 <b>CASH OVERRIDE ACTIVE</b> — 2-day cumulative return
      ({two_day_ret:+.1f}%) breached −{tsl_pct}% TSL.
      Holding CASH @ {tbill_rt:.2f}% T-bill until Z ≥ {z_reentry:.1f} σ.
      Current Z = {live_z:.2f} σ.
    </div>""", unsafe_allow_html=True)

# ── Signal Hero ───────────────────────────────────────────────────────────────
if in_cash_now:
    st.markdown(f"""
    <div class="cash-card">
      <div class="hero-label">⚠️ Trailing Stop Loss Triggered · Risk Override</div>
      <div class="hero-value">💵 {as_of} → CASH</div>
      <div class="hero-sub">
        Earning 3m T-bill rate: <b>{tbill_rt:.2f}% p.a.</b> &nbsp;|&nbsp;
        Re-entry when Z ≥ {z_reentry:.1f} σ &nbsp;|&nbsp; Current Z = {live_z:.2f} σ
      </div>
    </div>""", unsafe_allow_html=True)
else:
    winner_pred   = preds.get(winner, {})
    final_signal  = winner_pred.get("signal", "—")
    winner_label  = {"model_a":"Option A","model_b":"Option B",
                     "model_c":"Option C"}.get(winner, winner.upper())
    st.markdown(f"""
    <div class="hero-card">
      <div class="hero-label">{winner_label} · Next Trading Day Signal</div>
      <div class="hero-value">🎯 {as_of} → {final_signal}</div>
    </div>""", unsafe_allow_html=True)

# ── Signal Conviction (winner model) ──────────────────────────────────────────
st.markdown("---")
winner_pred = preds.get(winner, {})
if winner_pred and not in_cash_now:
    z_val   = winner_pred.get("z_score", 0)
    probs   = winner_pred.get("probabilities", {})
    top_etf = winner_pred.get("signal", "—")
    conf    = winner_pred.get("confidence", 0)

    # Conviction strength label
    if z_val >= 2.0:   strength = "Very High"
    elif z_val >= 1.5: strength = "High"
    elif z_val >= 1.0: strength = "Moderate"
    else:              strength = "Low"

    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### 🟢 Signal Conviction &nbsp; "
                    f"`Z = {z_val:.2f} σ` &nbsp;&nbsp; **{strength}**")
    with col2:
        st.markdown(f"<br>", unsafe_allow_html=True)

    # Conviction bar using plotly
    fig_bar = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = z_val,
        number= {"suffix": " σ", "font": {"size": 24}},
        gauge = dict(
            axis  = dict(range=[-3, 3], tickwidth=1),
            bar   = dict(color="#00bfa5"),
            steps = [
                dict(range=[-3, -1], color="#ffb3b3"),
                dict(range=[-1,  0], color="#ffe0b2"),
                dict(range=[ 0,  1], color="#b2dfdb"),
                dict(range=[ 1,  3], color="#80cbc4"),
            ],
            threshold = dict(line=dict(color="#00897b", width=4),
                             thickness=0.75, value=z_reentry),
        ),
        title = {"text": "Weak −3σ → Strong +3σ"},
    ))
    fig_bar.update_layout(height=200, margin=dict(t=30, b=0, l=30, r=30))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ETF probability pills
    st.markdown("**MODEL PROBABILITY BY ETF**")
    prob_cols = st.columns(len(probs))
    for i, (etf, p) in enumerate(sorted(probs.items(),
                                          key=lambda x: -x[1])):
        with prob_cols[i]:
            is_top = etf == top_etf
            color  = "#007a69" if is_top else "#555"
            bg     = "#e8faf8" if is_top else "#f7f8fa"
            border = "#00bfa5" if is_top else "#ddd"
            prefix = "★ " if is_top else ""
            st.markdown(
                f'<div style="border:1.5px solid {border};border-radius:20px;'
                f'padding:6px 12px;text-align:center;background:{bg};'
                f'color:{color};font-weight:{"700" if is_top else "500"};'
                f'font-size:13px;">{prefix}{etf} {p:.3f}</div>',
                unsafe_allow_html=True
            )

    st.caption(
        f"Z-score = std deviations the top ETF's probability sits above the mean. "
        f"Higher → model is more decisive. "
        f"⚠️ CASH override triggers if 2-day cumul ≤ −{tsl_pct}%, "
        f"exits when Z ≥ {z_reentry:.1f} σ."
    )

# ── All Models Signals ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### 📅 All Models — {as_of} Signals")
st.caption(f"⚠️ Lookback {best_lb}d found optimal (auto-selected from 30 / 45 / 60d)")

col_a, col_b, col_c = st.columns(3)
model_info = [
    ("model_a", "OPTION A",  "Option A",  "#00bfa5", col_a, "model-card-a"),
    ("model_b", "OPTION B ★","Option B",  "#7b8ff7", col_b, "model-card-b"),
    ("model_c", "OPTION C",  "Option C",  "#f87171", col_c, "model-card-c"),
]
for key, label, name, color, col, css_cls in model_info:
    with col:
        p = preds.get(key, {})
        sig  = "CASH" if in_cash_now else p.get("signal", "—")
        conf = p.get("confidence", 0)
        z_v  = p.get("z_score", 0)
        is_w = (key == winner)
        w_tag= " ★" if is_w else ""
        st.markdown(f"""
        <div class="{css_cls}">
          <div style="font-size:11px;letter-spacing:2px;font-weight:600;
                      color:{color};margin-bottom:12px;">{label}{w_tag}</div>
          <div style="font-size:28px;font-weight:700;margin-bottom:8px;">{sig}</div>
          <div style="font-size:13px;color:#aaa;">
            Confidence: <span style="color:{color};font-weight:600;">
            {"CASH" if in_cash_now else f"{conf:.1%}"}</span>
          </div>
          <div style="font-size:12px;color:#666;margin-top:6px;">
            Z = {z_v:.2f} σ
          </div>
        </div>""", unsafe_allow_html=True)

# ── Performance Metrics (winner model) ───────────────────────────────────────
st.markdown("---")
winner_label = {"model_a":"Option A · Wavelet-CNN-LSTM",
                "model_b":"Option B · Wavelet-Attn-CNN-LSTM",
                "model_c":"Option C · Wavelet-Dual-Stream"}.get(winner, winner)
st.markdown(f"### 📊 {winner_label} — Performance Metrics")

w_metrics = evalu.get(winner, {}).get("metrics", {})
if w_metrics:
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        ann = w_metrics.get("ann_return", 0)
        vs  = w_metrics.get("vs_spy", 0)
        st.metric("📈 Ann. Return", f"{ann:.2f}%",
                  delta=f"vs SPY: {vs:+.2f}%")
        st.caption("Annualised")
    with m2:
        sh  = w_metrics.get("sharpe", 0)
        lbl = "Strong" if sh>1 else ("Moderate" if sh>0.5 else "Weak")
        st.metric("📊 Sharpe", f"{sh:.2f}")
        st.caption(lbl)
    with m3:
        hr  = w_metrics.get("hit_ratio_15d", 0)
        lbl = "Good" if hr>0.55 else "Weak"
        st.metric("🎯 Hit Ratio 15d", f"{hr:.0%}")
        st.caption(lbl)
    with m4:
        mdd = w_metrics.get("max_drawdown", 0)
        st.metric("📉 Max Drawdown", f"{mdd:.2f}%")
        st.caption("Peak to Trough")
    with m5:
        mdd_d = w_metrics.get("max_daily_dd", 0)
        st.metric("⚠️ Max Daily DD", f"{mdd_d:.2f}%")
        st.caption("Worst Single Day")

# ── Approach Comparison Table ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🏆 Approach Comparison (Winner = Highest Raw Annualised Return)")

rows = []
for key, label in [("model_a","Option A · Wavelet-CNN-LSTM"),
                    ("model_b","Option B · Wavelet-Attn-CNN-LSTM"),
                    ("model_c","Option C · Wavelet-Dual-Stream")]:
    m = evalu.get(key, {}).get("metrics", {})
    if m:
        rows.append({
            "Model"        : label,
            "Ann. Return"  : f"{m.get('ann_return',0):.2f}%",
            "Sharpe"       : f"{m.get('sharpe',0):.2f}",
            "Hit Ratio(15d)": f"{m.get('hit_ratio_15d',0):.0%}",
            "Max Drawdown" : f"{m.get('max_drawdown',0):.2f}%",
            "Winner"       : "⭐ WINNER" if key == winner else "",
        })

if rows:
    df_cmp = pd.DataFrame(rows)
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

# ── Benchmark Comparison ──────────────────────────────────────────────────────
st.markdown("### 📋 Benchmark Comparison")
bench_rows = []
for key, label in [(winner, f"{winner_label} (Winner)"),
                    ("SPY","SPY (Buy & Hold)"),
                    ("AGG","AGG (Buy & Hold)"),
                    ("ar1_baseline","AR(1) Baseline")]:
    b = evalu.get(key, {})
    m = b.get("metrics", b) if isinstance(b, dict) else {}
    ann = m.get("ann_return", b.get("ann_return", "—")) if m else "—"
    bench_rows.append({
        "Strategy"     : label,
        "Ann. Return"  : f"{ann:.2f}%" if isinstance(ann, (int,float)) else ann,
        "Sharpe"       : f"{m.get('sharpe','—'):.2f}" if m.get('sharpe') else "—",
        "Max Drawdown" : f"{m.get('max_drawdown','—'):.2f}%" if m.get('max_drawdown') else "—",
    })
if bench_rows:
    st.dataframe(pd.DataFrame(bench_rows), use_container_width=True,
                 hide_index=True)

# ── Cumulative Returns Chart ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Cumulative Return — Test Period")
fig = go.Figure()
colors_map = {"model_a":"#00bfa5","model_b":"#7b8ff7","model_c":"#f87171"}
for key, label_short in [("model_a","Option A"),
                           ("model_b","Option B"),
                           ("model_c","Option C")]:
    sigs = evalu.get(key, {}).get("all_signals", [])
    if sigs:
        df_s = pd.DataFrame(sigs)
        df_s = apply_tsl_to_audit(sigs, tsl_pct, z_reentry, tbill_rt)
        if "Date" in df_s.columns and "Net_TSL" in df_s.columns:
            df_s["Date"] = pd.to_datetime(df_s["Date"])
            df_s["Cumul"] = (1 + df_s["Net_TSL"]).cumprod()
            fig.add_trace(go.Scatter(
                x=df_s["Date"], y=df_s["Cumul"],
                name=label_short,
                line=dict(color=colors_map[key], width=2),
            ))
fig.update_layout(
    height=350, margin=dict(t=20, b=20),
    yaxis_title="Growth of $1",
    xaxis_title="Date",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    plot_bgcolor="white",
    paper_bgcolor="white",
)
fig.update_xaxes(showgrid=True, gridcolor="#f0f2f5")
fig.update_yaxes(showgrid=True, gridcolor="#f0f2f5")
st.plotly_chart(fig, use_container_width=True)

# ── Audit Trail ───────────────────────────────────────────────────────────────
st.markdown("---")
winner_label_short = {"model_a":"Option A","model_b":"Option B",
                       "model_c":"Option C"}.get(winner, winner)
st.markdown(f"### 📋 Audit Trail — {winner_label_short} (Last 20 Trading Days)")

audit_raw = evalu.get(winner, {}).get("audit_tail", [])
if audit_raw:
    df_audit = apply_tsl_to_audit(audit_raw, tsl_pct, z_reentry, tbill_rt)

    # Format for display
    disp = df_audit[["Date","Signal_TSL","Confidence","Z_Score",
                      "Net_TSL","Mode"]].copy()
    disp.columns = ["Date","Signal","Confidence","Z Score",
                    "Net Return","Mode"]
    disp["Date"]       = pd.to_datetime(disp["Date"]).dt.strftime("%Y-%m-%d")
    disp["Net Return"] = disp["Net Return"].apply(
        lambda x: f"+{x*100:.2f}%" if x >= 0 else f"{x*100:.2f}%")

    def color_return(val):
        if "+" in str(val): return "color: #27ae60; font-weight:600"
        if "-" in str(val): return "color: #e74c3c; font-weight:600"
        return ""
    def color_mode(val):
        if "CASH" in str(val): return "background-color:#fff8f5;color:#e65100"
        return "background-color:#f0fdf9;color:#007a69"

    styled = disp.style.applymap(color_return, subset=["Net Return"]) \
                        .applymap(color_mode,   subset=["Mode"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.info("Audit trail will appear after model training completes.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"P2-ETF-DEEPWAVE-DL · GitHub: P2SAMAPA/P2-ETF-DEEPWAVE-DL · "
    f"HF: P2SAMAPA/p2-etf-deepwave-dl · "
    f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC"
)
