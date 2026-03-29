# app.py — P2-ETF-DEEPWAVE-DL Streamlit Dashboard
# Tab 1: Single-Year FI Results (original)
# Tab 2: Multi-Year Consensus Sweep (original)
# Tab 3: Equity ETFs Signal (new)

import json
import os
import requests
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download

import config

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P2-ETF-DEEPWAVE-DL",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { min-width:340px; max-width:340px; }
  .hero-card  { background:linear-gradient(135deg,#00bfa5,#00897b);
                border-radius:14px;padding:32px;text-align:center;
                color:white;margin:16px 0; }
  .hero-card-eq { background:linear-gradient(135deg,#2563eb,#1d4ed8);
                  border-radius:14px;padding:32px;text-align:center;
                  color:white;margin:16px 0; }
  .cash-card  { background:linear-gradient(135deg,#e65100,#bf360c);
                border-radius:14px;padding:32px;text-align:center;
                color:white;margin:16px 0; }
  .hero-label { font-size:11px;letter-spacing:2px;text-transform:uppercase;
                opacity:.8;margin-bottom:8px; }
  .hero-value { font-size:40px;font-weight:700; }
  .hero-sub   { font-size:13px;opacity:.85;margin-top:10px; }
  .model-card-a { background:#0d1117;border-radius:12px;padding:24px;
                  text-align:center;color:white;border:1px solid #1a2a1a; }
  .model-card-b { background:#0d1117;border-radius:12px;padding:24px;
                  text-align:center;color:white;border:2px solid #7b8ff7; }
  .model-card-c { background:#0d1117;border-radius:12px;padding:24px;
                  text-align:center;color:white;border:1px solid #2a1a1a; }
  .alert-green  { background:#e8f5e9;border:1px solid #c8e6c9;color:#2e7d32;
                  padding:12px 16px;border-radius:8px;margin-bottom:10px;font-size:13px; }
  .alert-blue   { background:#e8f4fd;border:1px solid #bbdefb;color:#1565c0;
                  padding:12px 16px;border-radius:8px;margin-bottom:10px;font-size:13px; }
  .alert-yellow { background:#fffde7;border:1px solid #fff9c4;color:#f57f17;
                  padding:12px 16px;border-radius:8px;margin-bottom:10px;font-size:13px; }
  .alert-orange { background:#fff3e0;border:1px solid #ffcc80;color:#e65100;
                  padding:12px 16px;border-radius:8px;margin-bottom:10px;font-size:13px; }
  .alert-indigo { background:#eef2ff;border:1px solid #c7d2fe;color:#3730a3;
                  padding:12px 16px;border-radius:8px;margin-bottom:10px;font-size:13px; }
  #MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Next US trading day ───────────────────────────────────────────────────────
US_HOLIDAYS_2025_2026 = {
    date(2025,1,1),  date(2025,1,20),  date(2025,2,17), date(2025,4,18),
    date(2025,5,26), date(2025,6,19),  date(2025,7,4),  date(2025,9,1),
    date(2025,11,27),date(2025,12,25),
    date(2026,1,1),  date(2026,1,19),  date(2026,2,16), date(2026,4,3),
    date(2026,5,25), date(2026,6,19),  date(2026,7,3),  date(2026,9,7),
    date(2026,11,26),date(2026,12,25),
}
MARKET_CLOSE_HOUR_EST = 16

def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in US_HOLIDAYS_2025_2026

def current_signal_date() -> date:
    now_est = datetime.utcnow() - timedelta(hours=5)
    today, hour = now_est.date(), now_est.hour
    if is_trading_day(today) and hour < MARKET_CLOSE_HOUR_EST:
        return today
    d = today + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d

def next_trading_day(from_date: date = None) -> date:
    d = from_date or date.today()
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d

def last_trading_day(from_date: date = None) -> date:
    d = from_date or date.today()
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


# ─── GitHub Actions trigger ───────────────────────────────────────────────────
def trigger_github_training(start_year, wavelet, tsl_pct, z_reentry,
                             epochs=80, fee_bps=10, sweep_mode="",
                             workflow="train_models.yml") -> bool:
    token = os.getenv("P2SAMAPA_GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", config.GITHUB_TOKEN))
    if not token:
        return False
    url = (f"https://api.github.com/repos/{config.GITHUB_REPO}"
           f"/actions/workflows/{workflow}/dispatches")
    payload = {"ref": "main", "inputs": {
        "model": "all", "epochs": str(epochs),
        "start_year": str(start_year), "wavelet": wavelet,
        "tsl_pct": str(tsl_pct), "z_reentry": str(z_reentry),
        "fee_bps": str(fee_bps), "sweep_mode": sweep_mode,
    }}
    resp = requests.post(url, json=payload, headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    })
    return resp.status_code == 204


# ─── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_prediction() -> dict:
    try:
        path = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                               filename="latest_prediction.json",
                               repo_type="dataset", token=config.HF_TOKEN or None)
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data(ttl=1800)
def load_prediction_equity() -> dict:
    """Load equity-specific prediction output."""
    try:
        path = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                               filename="latest_prediction_equity.json",
                               repo_type="dataset", token=config.HF_TOKEN or None)
        with open(path) as f:
            return json.load(f)
    except Exception:
        if os.path.exists("latest_prediction_equity.json"):
            with open("latest_prediction_equity.json") as f:
                return json.load(f)
        return {}

@st.cache_data(ttl=1800)
def load_evaluation() -> dict:
    try:
        path = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                               filename="evaluation_results.json",
                               repo_type="dataset", token=config.HF_TOKEN or None)
        with open(path) as f:
            return json.load(f)
    except Exception:
        if os.path.exists("evaluation_results.json"):
            with open("evaluation_results.json") as f:
                return json.load(f)
        return {}

@st.cache_data(ttl=1800)
def load_evaluation_equity() -> dict:
    """Load equity evaluation results."""
    try:
        path = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                               filename="evaluation_results_equity.json",
                               repo_type="dataset", token=config.HF_TOKEN or None)
        with open(path) as f:
            return json.load(f)
    except Exception:
        if os.path.exists("evaluation_results_equity.json"):
            with open("evaluation_results_equity.json") as f:
                return json.load(f)
        return {}

@st.cache_data(ttl=300)
def load_etf_ret_fresh() -> pd.DataFrame:
    try:
        path = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                               filename="data/etf_ret.parquet", repo_type="dataset",
                               token=config.HF_TOKEN or None, force_download=True)
        df = pd.read_parquet(path)
    except Exception:
        local = os.path.join(config.DATA_DIR, "etf_ret.parquet")
        df = pd.read_parquet(local) if os.path.exists(local) else pd.DataFrame()
    if df.empty:
        return df
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    elif not isinstance(df.index, pd.DatetimeIndex):
        unit = 'ms' if df.index.dtype == 'int64' and df.index.max() > 1e12 else 's'
        df.index = pd.to_datetime(df.index, unit=unit) if df.index.dtype == 'int64' \
                   else pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    for col in list(df.columns):
        if isinstance(col, str) and col.lower() in ('date', 'index', 'level_0'):
            df = df.drop(columns=[col])
    return df


def extend_audit_with_live_dates(audit_rows, winner_pred):
    if not audit_rows:
        return audit_rows
    etf_ret = load_etf_ret_fresh()
    if etf_ret.empty:
        return audit_rows
    etf_ret.columns = [c.split("_")[0] if "_" in c else c for c in etf_ret.columns]
    etf_ret.index   = pd.to_datetime(etf_ret.index)
    last_audit_date = pd.to_datetime(max(r["Date"] for r in audit_rows)).date()
    new_dates       = [d for d in etf_ret.index if d.date() > last_audit_date]
    if not new_dates:
        return audit_rows
    live_signal = winner_pred.get("signal", audit_rows[-1].get("Signal", "GLD"))
    live_conf   = winner_pred.get("confidence", 0.2)
    live_z      = winner_pred.get("z_score", 1.0)
    extra = []
    for dt in sorted(new_dates):
        etf = live_signal if live_signal != "CASH" else audit_rows[-1].get("Signal","GLD")
        ret = float(etf_ret.loc[dt, etf]) if etf in etf_ret.columns else 0.0
        extra.append(dict(Date=str(dt.date()), Signal=etf,
                          Confidence=round(live_conf,4), Z_Score=round(live_z,4),
                          Two_Day_Cumul_Pct=0.0, Mode="📈 ETF",
                          Net_Return=round(ret,6), TSL_Triggered=False))
    return audit_rows + extra


# ─── TSL re-apply ─────────────────────────────────────────────────────────────
def apply_tsl_to_audit(audit, tsl_pct, z_reentry, tbill=3.6):
    df = pd.DataFrame(audit)
    if df.empty:
        return df
    in_cash, tsl_days, prev_ret, prev2_ret = False, 0, 0.0, 0.0
    modes, signals, net_rets = [], [], []
    for _, row in df.iterrows():
        z       = float(row.get("Z_Score", 1.5))
        two_day = (prev_ret + prev2_ret) * 100
        if not in_cash and two_day <= -tsl_pct:
            in_cash, tsl_days = True, 0
        if in_cash and tsl_days >= 1 and z >= z_reentry:
            in_cash = False
        if in_cash:
            tsl_days += 1
            modes.append("💵 CASH"); signals.append("CASH")
            net_rets.append(round(tbill / 100 / 252, 6))
        else:
            modes.append("📈 ETF"); signals.append(row.get("Signal","—"))
            net_rets.append(float(row.get("Net_Return", 0.0)))
        prev2_ret, prev_ret = prev_ret, net_rets[-1]
    df = df.copy()
    df["Mode"]       = modes
    df["Signal_TSL"] = signals
    df["Net_TSL"]    = net_rets
    return df


# ─── Weights + data on startup ────────────────────────────────────────────────
@st.cache_resource
def ensure_weights_and_data():
    import shutil
    from huggingface_hub import HfApi, hf_hub_download
    token      = config.HF_TOKEN or None
    downloaded = {"weights": 0, "data": 0}
    os.makedirs(config.DATA_DIR, exist_ok=True)
    for f in ["etf_price","etf_ret","etf_vol","bench_price","bench_ret","bench_vol","macro"]:
        local = os.path.join(config.DATA_DIR, f"{f}.parquet")
        if not os.path.exists(local):
            try:
                dl = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                                     filename=f"data/{f}.parquet",
                                     repo_type="dataset", token=token)
                shutil.copy(dl, local); downloaded["data"] += 1
            except Exception:
                pass
    try:
        api   = HfApi(token=token)
        files = api.list_repo_files(repo_id=config.HF_DATASET_REPO,
                                    repo_type="dataset", token=token)
        for f in files:
            if f.startswith("models/") and f.endswith((".keras",".pkl",".json")):
                local = f
                if not os.path.exists(local):
                    os.makedirs(os.path.dirname(local), exist_ok=True)
                    try:
                        dl = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                                             filename=f, repo_type="dataset", token=token)
                        shutil.copy(dl, local); downloaded["weights"] += 1
                    except Exception:
                        pass
    except Exception:
        pass
    return downloaded


# ─── Persistent preferences ───────────────────────────────────────────────────
PREFS_FILE = "user_prefs.json"

def load_prefs() -> dict:
    defaults = {"start_year":2008,"fee_bps":10,"max_epochs":80,
                "wavelet":"db4 (Daubechies-4)","tsl_pct":10,"z_reentry":1.1}
    try:
        path = hf_hub_download(repo_id=config.HF_DATASET_REPO, filename=PREFS_FILE,
                               repo_type="dataset", token=config.HF_TOKEN or None)
        defaults.update(json.load(open(path)))
    except Exception:
        pass
    return defaults

def save_prefs(prefs):
    try:
        from huggingface_hub import HfApi
        import tempfile
        api = HfApi(token=config.HF_TOKEN or None)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(prefs, f, indent=2); tmp = f.name
        api.upload_file(path_or_fileobj=tmp, path_in_repo=PREFS_FILE,
                        repo_id=config.HF_DATASET_REPO, repo_type="dataset")
    except Exception:
        pass


# ─── Session state ────────────────────────────────────────────────────────────
if "prefs_loaded" not in st.session_state:
    _prefs = load_prefs()
    st.session_state.tsl_pct      = _prefs["tsl_pct"]
    st.session_state.z_reentry    = _prefs["z_reentry"]
    st.session_state.start_year   = _prefs["start_year"]
    st.session_state.fee_bps      = _prefs["fee_bps"]
    st.session_state.max_epochs   = _prefs["max_epochs"]
    st.session_state.wavelet      = _prefs["wavelet"]
    st.session_state.prefs_loaded = True
if "last_eval" not in st.session_state:
    st.session_state.last_eval = {}


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.caption(f"🕐 EST: {datetime.now().strftime('%H:%M:%S')}")
    st.divider()

    start_year = st.slider("📅 Start Year", min_value=config.START_YEAR_MIN,
                            max_value=config.START_YEAR_MAX,
                            value=st.session_state.start_year)
    st.session_state.start_year = start_year
    st.caption("↑ Changing this requires 🚀 Retrain. TSL/Z sliders are instant.")

    fee_bps = st.slider("💰 Fee (bps)", 0, 50, value=st.session_state.fee_bps)
    st.session_state.fee_bps = fee_bps

    max_epochs = st.number_input("🔁 Max Epochs", min_value=10, max_value=300,
                                  value=st.session_state.max_epochs, step=5)
    st.session_state.max_epochs = int(max_epochs)

    _wavelet_options = ["db4 (Daubechies-4)","db2 (Daubechies-2)","haar","sym5"]
    _wavelet_idx     = _wavelet_options.index(st.session_state.wavelet) \
                       if st.session_state.wavelet in _wavelet_options else 0
    wavelet_choice   = st.selectbox("〰️ Wavelet Type", options=_wavelet_options,
                                     index=_wavelet_idx)
    st.session_state.wavelet = wavelet_choice
    wavelet_key = wavelet_choice.split(" ")[0]

    st.divider()
    st.markdown("### 🛡️ Risk Controls")
    tsl_pct = st.slider("🔴 Trailing Stop Loss (2-day cumul.)", 0, 25,
                         value=st.session_state.tsl_pct, step=1, format="%d%%",
                         key="tsl_slider")
    st.session_state.tsl_pct = tsl_pct
    z_reentry = st.slider("📶 Z-score Re-entry", 1.0, 2.0,
                           value=st.session_state.z_reentry,
                           step=0.1, format="%.1f σ", key="z_slider")
    st.session_state.z_reentry = z_reentry
    st.caption(f"CASH earns 3m T-bill. Re-entry when Z ≥ {z_reentry:.1f} σ.")

    st.divider()
    st.markdown("### 🧠 Active Models")
    use_a = st.checkbox("Option A · Wavelet-CNN-LSTM",      value=True)
    use_b = st.checkbox("Option B · Wavelet-Attn-CNN-LSTM", value=True)
    use_c = st.checkbox("Option C · Wavelet-Dual-Stream",   value=True)
    st.caption("💡 Lookback auto-selected (30/45/60d). Split fixed 80/10/10.")

    if st.button("🔄 Recalculate Risk Controls", use_container_width=True):
        st.cache_data.clear(); st.rerun()

    st.divider()
    has_gh_token = bool(os.getenv("P2SAMAPA_GITHUB_TOKEN", os.getenv("GITHUB_TOKEN","")))

    st.markdown("**Retrain FI Models:**")
    if st.button("🚀 Retrain FI (A/B/C)", use_container_width=True, type="primary"):
        save_prefs({"start_year":start_year,"fee_bps":fee_bps,
                    "max_epochs":int(max_epochs),"wavelet":wavelet_choice,
                    "tsl_pct":tsl_pct,"z_reentry":z_reentry})
        if has_gh_token:
            ok = trigger_github_training(start_year, wavelet_key, tsl_pct,
                                          z_reentry, int(max_epochs), fee_bps)
            st.success("✅ FI retraining triggered!") if ok \
                else st.error("❌ Trigger failed.")
        else:
            st.info(f"Run train_models.yml manually with start_year={start_year} "
                    f"wavelet={wavelet_key} epochs={max_epochs}")

    st.markdown("**Retrain Equity Models:**")
    if st.button("🚀 Retrain Equity (A/B/C)", use_container_width=True):
        if has_gh_token:
            ok = trigger_github_training(start_year, wavelet_key, tsl_pct,
                                          z_reentry, int(max_epochs), fee_bps,
                                          workflow="train_equity_models.yml")
            st.success("✅ Equity retraining triggered!") if ok \
                else st.error("❌ Trigger failed.")
        else:
            st.info("Run train_equity_models.yml manually.")

    st.divider()
    st.markdown("### 📦 Dataset Info")
    etf_df_info = load_etf_ret_fresh()
    if not etf_df_info.empty:
        st.markdown(f"**Rows:** {len(etf_df_info):,}")
        st.markdown(f"**Range:** {etf_df_info.index.min().date()} → {etf_df_info.index.max().date()}")
    st.markdown(f"**FI ETFs:** {', '.join(config.FI_ETFS)}")
    st.markdown(f"**Equity ETFs:** {', '.join(config.EQUITY_ETFS)}")
    st.markdown(f"**Benchmarks:** {', '.join(config.BENCHMARKS)}")


# ─── STARTUP ──────────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading model weights from HF Dataset..."):
    dl_status = ensure_weights_and_data()
if dl_status["weights"] > 0 or dl_status["data"] > 0:
    st.cache_data.clear()

st.markdown("# 🧠 P2-ETF-DEEPWAVE-DL")
st.caption("Option A: Wavelet-CNN-LSTM · Option B: Wavelet-Attention-CNN-LSTM · "
           "Option C: Wavelet-Parallel-Dual-Stream-CNN-LSTM")

SWEEP_YEARS = [2008, 2013, 2015, 2017, 2019, 2021]

def _today_est():
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    return (_dt.now(_tz.utc) - _td(hours=5)).date()

def _sweep_fname(year, for_date):
    return f"sweep_{year}_{for_date.strftime('%Y%m%d')}.json"

@st.cache_data(ttl=120)
def _load_sweep_hf(date_str):
    from datetime import date as _d
    for_date = _d.fromisoformat(date_str)
    cache = {}
    try:
        token = os.getenv("HF_TOKEN", config.HF_TOKEN)
        for yr in SWEEP_YEARS:
            fname = _sweep_fname(yr, for_date)
            try:
                path = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                                       filename=f"sweep/{fname}",
                                       repo_type="dataset", token=token,
                                       force_download=True)
                with open(path) as f:
                    cache[yr] = json.load(f)
            except Exception:
                pass
    except Exception:
        pass
    return cache

@st.cache_data(ttl=120)
def _load_sweep_any():
    from datetime import datetime as _dt2
    from huggingface_hub import HfApi
    found, best_date = {}, None
    try:
        token = os.getenv("HF_TOKEN", config.HF_TOKEN)
        api   = HfApi()
        files = list(api.list_repo_files(repo_id=config.HF_DATASET_REPO,
                                          repo_type="dataset", token=token))
        for fpath in files:
            fname = os.path.basename(fpath)
            if fname.startswith("sweep_") and fname.endswith(".json"):
                parts = fname.replace(".json","").split("_")
                if len(parts) == 3:
                    try:
                        dt = _dt2.strptime(parts[2], "%Y%m%d").date()
                        if best_date is None or dt > best_date:
                            best_date = dt
                    except Exception:
                        pass
        if best_date:
            for yr in SWEEP_YEARS:
                fname = _sweep_fname(yr, best_date)
                try:
                    path = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                                           filename=f"sweep/{fname}",
                                           repo_type="dataset", token=token,
                                           force_download=True)
                    with open(path) as f:
                        found[yr] = json.load(f)
                except Exception:
                    pass
    except Exception:
        pass
    return found, best_date

def _compute_consensus(sweep_data):
    rows = []
    for yr, sig in sweep_data.items():
        rows.append({"year":yr,"signal":sig.get("signal","?"),
                     "ann_return":sig.get("ann_return",0.0),"z_score":sig.get("z_score",0.0),
                     "sharpe":sig.get("sharpe",0.0),"max_dd":sig.get("max_dd",0.0)})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    def _mm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)
    df["wtd"] = (0.40*_mm(df["ann_return"]) + 0.20*_mm(df["z_score"]) +
                 0.20*_mm(df["sharpe"])      + 0.20*_mm(-df["max_dd"]))
    etf_agg = {}
    for _, row in df.iterrows():
        e = row["signal"]
        etf_agg.setdefault(e, {"years":[],"scores":[],"returns":[],"zs":[],"sharpes":[],"dds":[]})
        etf_agg[e]["years"].append(row["year"]); etf_agg[e]["scores"].append(row["wtd"])
        etf_agg[e]["returns"].append(row["ann_return"]); etf_agg[e]["zs"].append(row["z_score"])
        etf_agg[e]["sharpes"].append(row["sharpe"]); etf_agg[e]["dds"].append(row["max_dd"])
    total = sum(sum(v["scores"]) for v in etf_agg.values()) + 1e-9
    summary = {}
    for e, v in etf_agg.items():
        cs = sum(v["scores"])
        summary[e] = {"cum_score":round(cs,4),"score_share":round(cs/total,3),
                      "n_years":len(v["years"]),"years":v["years"],
                      "avg_return":round(float(np.mean(v["returns"])),4),
                      "avg_z":round(float(np.mean(v["zs"])),3),
                      "avg_sharpe":round(float(np.mean(v["sharpes"])),3),
                      "avg_max_dd":round(float(np.mean(v["dds"])),4)}
    winner_etf = max(summary, key=lambda e: summary[e]["cum_score"])
    return {"winner":winner_etf,"etf_summary":summary,
            "per_year":df.to_dict("records"),"n_years":len(rows)}


# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 FI — Single-Year Results",
    "🔄 FI — Multi-Year Consensus",
    "🚀 Equity ETF Signal",
])

ETF_COLORS_SW = {"TLT":"#4e79a7","VCIT":"#f28e2b","LQD":"#59a14f","HYG":"#e15759",
                 "VNQ":"#76b7b2","SLV":"#edc948","GLD":"#b07aa1","CASH":"#aaaaaa"}

EQ_ETF_COLORS = {"SPY":"#2563eb","QQQ":"#7c3aed","XLK":"#0891b2","XLF":"#059669",
                 "XLE":"#d97706","XLV":"#dc2626","XLI":"#9333ea","XLY":"#db2777",
                 "XLP":"#16a34a","XLU":"#ca8a04","XME":"#64748b","GDX":"#b45309",
                 "IWM":"#0f766e","CASH":"#aaaaaa"}


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FI Single-Year Results  (original, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    pred  = load_prediction()
    evalu = load_evaluation()

    next_td  = current_signal_date()
    last_td  = last_trading_day()
    as_of    = pred.get("as_of_date", str(next_td))
    winner   = evalu.get("winner", "model_a")
    tbill_rt = pred.get("tbill_rate", 3.6)
    preds    = pred.get("predictions", {})
    tsl_stat = pred.get("tsl_status", {})
    trained_from_year = pred.get("trained_from_year")
    trained_wavelet   = pred.get("trained_wavelet")
    trained_at        = pred.get("trained_at")

    live_z        = preds.get(winner, {}).get("z_score", 1.5)
    two_day_ret   = tsl_stat.get("two_day_cumul_pct", 0.0)
    tsl_triggered = float(two_day_ret) <= -tsl_pct
    in_cash_now   = tsl_triggered and (live_z < z_reentry)
    best_lb       = evalu.get(winner, {}).get("lookback", 30)
    etf_df        = load_etf_ret_fresh()

    st.markdown(f'<div class="alert-green">✅ FI dataset through <b>{as_of}</b>.</div>',
                unsafe_allow_html=True)
    if not etf_df.empty:
        n_yrs = (etf_df.index.max() - etf_df.index.min()).days // 365
        st.markdown(f'<div class="alert-blue">📅 <b>Data:</b> {etf_df.index.min().date()} → {etf_df.index.max().date()} ({n_yrs} years)</div>',
                    unsafe_allow_html=True)

    n_feat = (len(config.FI_ETFS)*2 + len(config.MACRO_SERIES)) * (config.WAVELET_LEVELS+1)
    st.markdown(f'<div class="alert-blue">🎯 <b>FI ETFs:</b> {", ".join(config.FI_ETFS)} &nbsp;·&nbsp; <b>Features:</b> {n_feat} &nbsp;·&nbsp; <b>Wavelet:</b> {wavelet_key}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="alert-yellow">⚠️ Optimal lookback: <b>{best_lb}d</b></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="alert-yellow">🛡️ TSL: <b>−{tsl_pct}%</b> · Re-entry Z ≥ <b>{z_reentry:.1f} σ</b> · CASH earns <b>{tbill_rt:.2f}%</b></div>',
                unsafe_allow_html=True)
    if in_cash_now:
        st.markdown(f'<div class="alert-orange">🔴 <b>CASH OVERRIDE</b> — 2-day cumul ({float(two_day_ret):+.1f}%) ≤ −{tsl_pct}%</div>',
                    unsafe_allow_html=True)

    winner_label = {"model_a":"Option A","model_b":"Option B","model_c":"Option C"}.get(winner,"Option A")
    if in_cash_now:
        st.markdown(f"""<div class="cash-card">
          <div class="hero-label">⚠️ Trailing Stop Loss Triggered</div>
          <div class="hero-value">💵 {next_td} → CASH</div>
          <div class="hero-sub">Earning 3m T-bill: <b>{tbill_rt:.2f}% p.a.</b> · Re-entry Z ≥ {z_reentry:.1f} σ</div>
        </div>""", unsafe_allow_html=True)
    else:
        wp           = preds.get(winner, {})
        final_signal = wp.get("signal", "—")
        provenance   = (f"Trained from {trained_from_year} · {trained_wavelet} wavelet"
                        if trained_from_year and trained_wavelet
                        else "Training metadata unavailable")
        st.markdown(f"""<div class="hero-card">
          <div class="hero-label">{winner_label} · FI SIGNAL</div>
          <div class="hero-value">🎯 {next_td} → {final_signal}</div>
          <div class="hero-sub">📋 {provenance}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    wp = preds.get(winner, {})
    if wp and not in_cash_now:
        z_val = float(wp.get("z_score", 0))
        probs = wp.get("probabilities", {})
        top_etf = wp.get("signal", "—")
        if z_val >= 2.0:    strength = "Very High"
        elif z_val >= 1.5:  strength = "High"
        elif z_val >= 1.0:  strength = "Moderate"
        else:               strength = "Low"
        st.markdown(f"### 🟢 Signal Conviction &nbsp; `Z = {z_val:.2f} σ` &nbsp; **{strength}**")
        prob_cols = st.columns(len(probs))
        for i, (etf, p) in enumerate(sorted(probs.items(), key=lambda x: -x[1])):
            with prob_cols[i]:
                is_top = etf == top_etf
                color  = "#007a69" if is_top else "#555"
                bg     = "#e8faf8" if is_top else "#f7f8fa"
                border = "#00bfa5" if is_top else "#ddd"
                prefix = "★ " if is_top else ""
                st.markdown(f'<div style="border:1.5px solid {border};border-radius:20px;padding:6px 12px;text-align:center;background:{bg};color:{color};font-weight:{"700" if is_top else "500"};font-size:13px;">{prefix}{etf} {p:.3f}</div>',
                            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"### 📅 All FI Models — {next_td}")
    col_a, col_b, col_c = st.columns(3)
    for key, label, color, col, css in [
        ("model_a","OPTION A","#00bfa5",col_a,"model-card-a"),
        ("model_b","OPTION B","#7b8ff7",col_b,"model-card-b"),
        ("model_c","OPTION C","#f87171",col_c,"model-card-c"),
    ]:
        with col:
            p   = preds.get(key, {})
            sig = "CASH" if in_cash_now else p.get("signal","—")
            conf= float(p.get("confidence",0)); z_v = float(p.get("z_score",0))
            w_tag = " ★" if key == winner else ""
            st.markdown(f"""<div class="{css}">
              <div style="font-size:11px;letter-spacing:2px;font-weight:600;color:{color};margin-bottom:12px;">{label}{w_tag}</div>
              <div style="font-size:28px;font-weight:700;margin-bottom:8px;">{sig}</div>
              <div style="font-size:13px;color:#aaa;">Conf: <span style="color:{color};font-weight:600;">{"CASH" if in_cash_now else f"{conf:.1%}"}</span></div>
              <div style="font-size:12px;color:#666;margin-top:6px;">Z = {z_v:.2f} σ</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    winner_full = {"model_a":"Option A · Wavelet-CNN-LSTM","model_b":"Option B · Wavelet-Attn-CNN-LSTM",
                   "model_c":"Option C · Wavelet-Dual-Stream"}.get(winner, winner)
    st.markdown(f"### 📊 {winner_full} — Performance Metrics")
    w_met = evalu.get(winner, {}).get("metrics", {})
    if w_met:
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("📈 Ann. Return", f"{w_met.get('ann_return',0):.2f}%",
                  delta=f"vs SPY: {w_met.get('vs_spy',0):+.2f}%")
        m2.metric("📊 Sharpe", f"{w_met.get('sharpe',0):.2f}")
        m3.metric("🎯 Hit Ratio 15d", f"{w_met.get('hit_ratio_15d',0):.0%}")
        m4.metric("📉 Max Drawdown", f"{w_met.get('max_drawdown',0):.2f}%")
        m5.metric("⚠️ Max Daily DD", f"{w_met.get('max_daily_dd',0):.2f}%")

    st.markdown("---")
    st.markdown("### 📈 Cumulative Return — FI Test Period")
    fig = go.Figure()
    for key, lbl, clr in [("model_a","Option A","#00bfa5"),
                           ("model_b","Option B","#7b8ff7"),
                           ("model_c","Option C","#f87171")]:
        sigs = evalu.get(key,{}).get("all_signals",[])
        if sigs:
            df_s = apply_tsl_to_audit(sigs, tsl_pct, z_reentry, tbill_rt)
            if "Date" in df_s.columns and "Net_TSL" in df_s.columns:
                df_s["Date"]  = pd.to_datetime(df_s["Date"])
                df_s["Cumul"] = (1 + df_s["Net_TSL"]).cumprod()
                fig.add_trace(go.Scatter(x=df_s["Date"], y=df_s["Cumul"],
                                         name=lbl, line=dict(color=clr, width=2)))
    fig.update_layout(height=350, margin=dict(t=20,b=20),
                      yaxis_title="Growth of $1", xaxis_title="Date",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(f"### 📋 FI Audit Trail — {winner_full} (Last 30 Trading Days)")
    audit_raw = evalu.get(winner, {}).get("audit_tail", [])
    if audit_raw:
        audit_raw = extend_audit_with_live_dates(audit_raw, preds.get(winner, {}))
    if audit_raw:
        df_audit = apply_tsl_to_audit(audit_raw, tsl_pct, z_reentry, tbill_rt)
        disp = df_audit[["Date","Signal_TSL","Confidence","Z_Score","Net_TSL","Mode"]].copy()
        disp.columns = ["Date","Signal","Confidence","Z Score","Net Return","Mode"]
        disp["Date"]       = pd.to_datetime(disp["Date"], format="mixed").dt.strftime("%Y-%m-%d")
        disp["Confidence"] = disp["Confidence"].apply(lambda x: f"{float(x):.1%}" if isinstance(x,(int,float)) else x)
        disp["Z Score"]    = disp["Z Score"].apply(lambda x: f"{float(x):.2f}" if isinstance(x,(int,float)) else x)
        disp["Net Return"] = disp["Net Return"].apply(lambda x: f"+{x*100:.2f}%" if float(x)>=0 else f"{x*100:.2f}%")
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Multi-Year Consensus Sweep  (original, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔄 FI Multi-Year Consensus Sweep")
    today_sw    = _today_est()
    today_str   = str(today_sw)
    today_cache = _load_sweep_hf(today_str)
    prev_cache, prev_date = _load_sweep_any()
    if prev_date == today_sw:
        prev_cache, prev_date = {}, None

    sweep_complete = len(today_cache) == len(SWEEP_YEARS)
    display_cache  = today_cache if today_cache else prev_cache
    display_date   = today_sw    if today_cache else prev_date

    if display_cache and display_date and display_date < today_sw:
        st.warning(f"⚠️ Showing results from **{display_date}**. Today's sweep hasn't run yet.")

    cols = st.columns(len(SWEEP_YEARS))
    for i, yr in enumerate(SWEEP_YEARS):
        with cols[i]:
            if yr in today_cache:
                st.success(f"**{yr}**\n✅ {today_cache[yr].get('signal','?')}")
            elif yr in prev_cache:
                st.warning(f"**{yr}**\n📅 {prev_cache[yr].get('signal','?')}")
            else:
                st.error(f"**{yr}**\n⏳ Not run")

    st.divider()
    missing_today = [yr for yr in SWEEP_YEARS if yr not in today_cache]
    force_rerun   = st.checkbox("🔄 Force re-run all years", value=False)
    trigger_years = SWEEP_YEARS if force_rerun else missing_today

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        sweep_btn = st.button("🚀 Run Consensus Sweep", type="primary",
                               use_container_width=True,
                               disabled=(sweep_complete and not force_rerun))
    with col_info:
        if sweep_complete and not force_rerun:
            st.success(f"✅ Today's sweep complete — all {len(SWEEP_YEARS)} years ready")
        else:
            st.info(f"**{len(today_cache)}/{len(SWEEP_YEARS)}** years done today.")

    if sweep_btn and trigger_years:
        sweep_str = ",".join(str(y) for y in trigger_years)
        with st.spinner(f"Triggering sweep for {sweep_str}..."):
            ok = trigger_github_training(trigger_years[0], wavelet_key, tsl_pct,
                                          z_reentry, fee_bps=fee_bps,
                                          sweep_mode=sweep_str)
        st.success(f"✅ Triggered: {sweep_str}") if ok \
            else st.error("❌ Failed. Check GITHUB_TOKEN.")

    if not display_cache:
        st.info("👆 No sweep results yet.")
        st.stop()

    consensus = _compute_consensus(display_cache)
    if not consensus:
        st.warning("Could not compute consensus.")
        st.stop()

    winner_sw = consensus["winner"]
    w_info    = consensus["etf_summary"][winner_sw]
    win_color = ETF_COLORS_SW.get(winner_sw, "#0066cc")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border:2px solid {win_color};
                border-radius:16px;padding:32px;text-align:center;margin:16px 0;">
      <div style="font-size:11px;letter-spacing:3px;color:#aaa;">CONSENSUS · {len(display_cache)} YEARS · {display_date}</div>
      <div style="font-size:72px;font-weight:900;color:{win_color};">{winner_sw}</div>
      <div style="font-size:14px;color:#ccc;">Score share {w_info['score_share']*100:.0f}% · {w_info['n_years']}/{len(SWEEP_YEARS)} years</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Weighted Score per ETF**")
        es          = consensus["etf_summary"]
        sorted_etfs = sorted(es.keys(), key=lambda e: -es[e]["cum_score"])
        fig_bar = go.Figure(go.Bar(
            x=sorted_etfs, y=[es[e]["cum_score"] for e in sorted_etfs],
            marker_color=[ETF_COLORS_SW.get(e,"#888") for e in sorted_etfs],
            text=[f"{es[e]['n_years']}yr · {es[e]['score_share']*100:.0f}%" for e in sorted_etfs],
            textposition="outside"))
        fig_bar.update_layout(template="plotly_dark", height=360,
                              yaxis_title="Cumulative Score", showlegend=False,
                              margin=dict(t=20,b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.markdown("**Z-Score Conviction by Start Year**")
        fig_sc = go.Figure()
        for row in consensus["per_year"]:
            etf = row["signal"]
            col = ETF_COLORS_SW.get(etf, "#888")
            fig_sc.add_trace(go.Scatter(
                x=[row["year"]], y=[row["z_score"]], mode="markers+text",
                marker=dict(size=18, color=col, line=dict(color="white",width=1)),
                text=[etf], textposition="top center", name=etf, showlegend=False))
        fig_sc.update_layout(template="plotly_dark", height=360,
                             xaxis_title="Start Year", yaxis_title="Z-Score (σ)",
                             margin=dict(t=20,b=20))
        st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Equity ETF Signal (NEW)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🚀 Equity ETF Signal")
    st.markdown(
        f"*{' · '.join(config.EQUITY_ETFS)}*  \n"
        "Same wavelet A/B/C architecture as FI — trained independently on equity returns."
    )

    eq_pred  = load_prediction_equity()
    eq_evalu = load_evaluation_equity()

    next_td_eq = current_signal_date()

    if not eq_pred and not eq_evalu:
        st.warning("⏳ No equity model data yet.")
        st.info("**To get started:**\n\n"
                "1. Confirm `config.py` has `EQUITY_ETFS` defined ✅\n"
                "2. Run **Seed Data** workflow to include equity tickers in the parquet\n"
                "3. Trigger **🚀 Retrain Equity (A/B/C)** from the sidebar\n"
                "4. Once complete, equity predictions will appear here automatically")
        st.stop()

    eq_winner   = eq_evalu.get("winner", "model_a")
    eq_tbill    = eq_pred.get("tbill_rate", 3.6)
    eq_preds    = eq_pred.get("predictions", {})
    eq_tsl_stat = eq_pred.get("tsl_status", {})
    eq_trained_from = eq_pred.get("trained_from_year")
    eq_trained_wav  = eq_pred.get("trained_wavelet")

    eq_live_z       = eq_preds.get(eq_winner, {}).get("z_score", 1.5)
    eq_two_day_ret  = eq_tsl_stat.get("two_day_cumul_pct", 0.0)
    eq_tsl_triggered= float(eq_two_day_ret) <= -tsl_pct
    eq_in_cash      = eq_tsl_triggered and (eq_live_z < z_reentry)
    eq_best_lb      = eq_evalu.get(eq_winner, {}).get("lookback", 30)

    st.markdown(f'<div class="alert-indigo">🚀 <b>Equity ETFs:</b> {", ".join(config.EQUITY_ETFS)}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="alert-yellow">⚠️ Optimal lookback: <b>{eq_best_lb}d</b> · Wavelet: <b>{eq_trained_wav or wavelet_key}</b></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="alert-yellow">🛡️ TSL: <b>−{tsl_pct}%</b> · Re-entry Z ≥ <b>{z_reentry:.1f} σ</b> · CASH earns <b>{eq_tbill:.2f}%</b></div>',
                unsafe_allow_html=True)
    if eq_in_cash:
        st.markdown(f'<div class="alert-orange">🔴 <b>EQUITY CASH OVERRIDE</b> — 2-day cumul ({float(eq_two_day_ret):+.1f}%) ≤ −{tsl_pct}%</div>',
                    unsafe_allow_html=True)

    eq_winner_label = {"model_a":"Option A","model_b":"Option B","model_c":"Option C"}.get(eq_winner,"Option A")

    if eq_in_cash:
        st.markdown(f"""<div class="cash-card">
          <div class="hero-label">⚠️ Equity TSL Triggered</div>
          <div class="hero-value">💵 {next_td_eq} → CASH</div>
          <div class="hero-sub">Earning 3m T-bill: <b>{eq_tbill:.2f}% p.a.</b> · Re-entry when Z ≥ {z_reentry:.1f} σ</div>
        </div>""", unsafe_allow_html=True)
    else:
        eq_wp     = eq_preds.get(eq_winner, {})
        eq_signal = eq_wp.get("signal", "—")
        eq_conf   = float(eq_wp.get("confidence", 0))
        eq_z      = float(eq_wp.get("z_score", 0))
        provenance = (f"Trained from {eq_trained_from} · {eq_trained_wav} wavelet"
                      if eq_trained_from and eq_trained_wav
                      else "Training metadata unavailable")
        st.markdown(f"""<div class="hero-card-eq">
          <div class="hero-label">{eq_winner_label} · EQUITY SIGNAL</div>
          <div class="hero-value">🎯 {next_td_eq} → {eq_signal}</div>
          <div class="hero-sub">Conf: {eq_conf:.1%} · Z: {eq_z:.2f} σ · {provenance}</div>
        </div>""", unsafe_allow_html=True)

    # ── Probability bar across all 13 equity ETFs ──────────────────────────
    st.markdown("---")
    eq_wp   = eq_preds.get(eq_winner, {})
    probs   = eq_wp.get("probabilities", {})
    top_etf = eq_wp.get("signal", "—")

    if probs and not eq_in_cash:
        st.markdown("### 📊 Model Probability — All Equity ETFs")
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        tickers = [e for e, _ in sorted_probs]
        vals    = [p for _, p in sorted_probs]
        colors  = [EQ_ETF_COLORS.get(t, "#888") if t == top_etf else "#d1d5db"
                   for t in tickers]
        fig_eq = go.Figure(go.Bar(
            x=tickers, y=vals, marker_color=colors,
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            textfont=dict(size=11)
        ))
        fig_eq.add_hline(y=1/len(probs), line_dash="dot",
                         line_color="#6b7280", annotation_text="Uniform")
        fig_eq.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0),
                             yaxis_title="Softmax probability",
                             plot_bgcolor="white", paper_bgcolor="white",
                             showlegend=False)
        st.plotly_chart(fig_eq, use_container_width=True)

    # ── All 3 model cards ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 📅 All Equity Models — {next_td_eq}")
    col_a, col_b, col_c = st.columns(3)
    for key, label, color, col, css in [
        ("model_a","OPTION A","#2563eb",col_a,"model-card-a"),
        ("model_b","OPTION B","#7c3aed",col_b,"model-card-b"),
        ("model_c","OPTION C","#0891b2",col_c,"model-card-c"),
    ]:
        with col:
            p    = eq_preds.get(key, {})
            sig  = "CASH" if eq_in_cash else p.get("signal","—")
            conf = float(p.get("confidence",0))
            z_v  = float(p.get("z_score",0))
            w_tag = " ★" if key == eq_winner else ""
            st.markdown(f"""<div class="{css}">
              <div style="font-size:11px;letter-spacing:2px;font-weight:600;color:{color};margin-bottom:12px;">{label}{w_tag}</div>
              <div style="font-size:28px;font-weight:700;margin-bottom:8px;">{sig}</div>
              <div style="font-size:13px;color:#aaa;">Conf: <span style="color:{color};font-weight:600;">{"CASH" if eq_in_cash else f"{conf:.1%}"}</span></div>
              <div style="font-size:12px;color:#666;margin-top:6px;">Z = {z_v:.2f} σ</div>
            </div>""", unsafe_allow_html=True)

    # ── Equity performance metrics ─────────────────────────────────────────
    st.markdown("---")
    eq_winner_full = {"model_a":"Option A · Wavelet-CNN-LSTM","model_b":"Option B · Wavelet-Attn-CNN-LSTM",
                      "model_c":"Option C · Wavelet-Dual-Stream"}.get(eq_winner, eq_winner)
    st.markdown(f"### 📊 {eq_winner_full} — Equity Performance Metrics")
    eq_w_met = eq_evalu.get(eq_winner, {}).get("metrics", {})
    if eq_w_met:
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("📈 Ann. Return",   f"{eq_w_met.get('ann_return',0):.2f}%",
                  delta=f"vs SPY: {eq_w_met.get('vs_spy',0):+.2f}%")
        m2.metric("📊 Sharpe",        f"{eq_w_met.get('sharpe',0):.2f}")
        m3.metric("🎯 Hit Ratio 15d", f"{eq_w_met.get('hit_ratio_15d',0):.0%}")
        m4.metric("📉 Max Drawdown",  f"{eq_w_met.get('max_drawdown',0):.2f}%")
        m5.metric("⚠️ Max Daily DD",  f"{eq_w_met.get('max_daily_dd',0):.2f}%")
    else:
        st.info("Equity performance metrics will appear here after the first training run.")

    # ── Equity cumulative return chart ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Cumulative Return — Equity Test Period")
    fig_eq_cumul = go.Figure()
    for key, lbl, clr in [("model_a","Option A","#2563eb"),
                           ("model_b","Option B","#7c3aed"),
                           ("model_c","Option C","#0891b2")]:
        sigs = eq_evalu.get(key,{}).get("all_signals",[])
        if sigs:
            df_s = apply_tsl_to_audit(sigs, tsl_pct, z_reentry, eq_tbill)
            if "Date" in df_s.columns and "Net_TSL" in df_s.columns:
                df_s["Date"]  = pd.to_datetime(df_s["Date"])
                df_s["Cumul"] = (1 + df_s["Net_TSL"]).cumprod()
                fig_eq_cumul.add_trace(go.Scatter(x=df_s["Date"], y=df_s["Cumul"],
                                                   name=lbl, line=dict(color=clr, width=2)))
    if fig_eq_cumul.data:
        fig_eq_cumul.update_layout(height=350, margin=dict(t=20,b=20),
                                   yaxis_title="Growth of $1", xaxis_title="Date",
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02),
                                   plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_eq_cumul, use_container_width=True)
    else:
        st.info("Equity cumulative chart will appear after the first evaluation run.")

    # ── Equity audit trail ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 📋 Equity Audit Trail — {eq_winner_full}")
    eq_audit_raw = eq_evalu.get(eq_winner, {}).get("audit_tail", [])
    if eq_audit_raw:
        df_eq_audit = apply_tsl_to_audit(eq_audit_raw, tsl_pct, z_reentry, eq_tbill)
        disp = df_eq_audit[["Date","Signal_TSL","Confidence","Z_Score","Net_TSL","Mode"]].copy()
        disp.columns = ["Date","Signal","Confidence","Z Score","Net Return","Mode"]
        disp["Date"]       = pd.to_datetime(disp["Date"], format="mixed").dt.strftime("%Y-%m-%d")
        disp["Confidence"] = disp["Confidence"].apply(lambda x: f"{float(x):.1%}" if isinstance(x,(int,float)) else x)
        disp["Z Score"]    = disp["Z Score"].apply(lambda x: f"{float(x):.2f}" if isinstance(x,(int,float)) else x)
        disp["Net Return"] = disp["Net Return"].apply(lambda x: f"+{x*100:.2f}%" if float(x)>=0 else f"{x*100:.2f}%")
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("Equity audit trail will appear after the first training + evaluation run.")

st.markdown("---")
st.caption(f"P2-ETF-DEEPWAVE-DL · HF: {config.HF_DATASET_REPO} · "
           f"Wavelet: {wavelet_key} · Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC")
