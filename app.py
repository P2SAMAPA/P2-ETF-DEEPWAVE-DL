# app.py — P2-ETF-DEEPWAVE-DL Streamlit Dashboard
# Fixes:
#  a) Lookback auto-selected by model (removed from UI, shown in output only)
#  b) Next trading day computed correctly (skips weekends + US holidays)
#  c) Train/Val/Test hardcoded 80/10/10 (removed from UI)
#  d) AR(1) removed from benchmark comparison table
#  e) Start year slider 2008-2024
#  f) Wavelet type feeds into GitHub Actions trigger via API
#  g) Run All 3 Models triggers GitHub Actions workflow_dispatch

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
  #MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Next US trading day ───────────────────────────────────────────────────────

US_HOLIDAYS_2025_2026 = {
    date(2025,1,1), date(2025,1,20), date(2025,2,17), date(2025,4,18),
    date(2025,5,26), date(2025,6,19), date(2025,7,4), date(2025,9,1),
    date(2025,11,27), date(2025,12,25),
    date(2026,1,1), date(2026,1,19), date(2026,2,16), date(2026,4,3),
    date(2026,5,25), date(2026,6,19), date(2026,7,3), date(2026,9,7),
    date(2026,11,26), date(2026,12,25),
}

MARKET_CLOSE_HOUR_EST = 16

def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in US_HOLIDAYS_2025_2026

def current_signal_date() -> date:
    now_est = datetime.utcnow() - timedelta(hours=5)
    today   = now_est.date()
    hour    = now_est.hour
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

def trigger_github_training(start_year: int, wavelet: str,
                             tsl_pct: float, z_reentry: float,
                             epochs: int = 80, fee_bps: int = 10,
                             sweep_mode: str = "") -> bool:
    token = os.getenv("P2SAMAPA_GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", config.GITHUB_TOKEN))
    if not token:
        return False

    url = (f"https://api.github.com/repos/{config.GITHUB_REPO}"
           f"/actions/workflows/train_models.yml/dispatches")

    payload = {
        "ref": "main",
        "inputs": {
            "model":      "all",
            "epochs":     str(epochs),
            "start_year": str(start_year),
            "wavelet":    wavelet,
            "tsl_pct":    str(tsl_pct),
            "z_reentry":  str(z_reentry),
            "fee_bps":    str(fee_bps),
            "sweep_mode": sweep_mode,
        }
    }

    resp = requests.post(url,
                         json=payload,
                         headers={
                             "Authorization": f"token {token}",
                             "Accept": "application/vnd.github+json",
                         })
    return resp.status_code == 204


# ─── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def load_prediction() -> dict:
    try:
        path = hf_hub_download(
            repo_id=config.HF_DATASET_REPO,
            filename="latest_prediction.json",
            repo_type="dataset",
            token=config.HF_TOKEN or None,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=1800)
def load_evaluation() -> dict:
    try:
        path = hf_hub_download(
            repo_id=config.HF_DATASET_REPO,
            filename="evaluation_results.json",
            repo_type="dataset",
            token=config.HF_TOKEN or None,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        if os.path.exists("evaluation_results.json"):
            with open("evaluation_results.json") as f:
                return json.load(f)
        return {}


@st.cache_data(ttl=300)
def load_etf_ret_fresh() -> pd.DataFrame:
    """Load latest etf_ret parquet directly — used to extend audit trail."""
    try:
        path = hf_hub_download(
            repo_id   = config.HF_DATASET_REPO,
            filename  = "data/etf_ret.parquet",
            repo_type = "dataset",
            token     = config.HF_TOKEN or None,
            force_download = True,
        )
        df = pd.read_parquet(path)
    except Exception:
        local = os.path.join(config.DATA_DIR, "etf_ret.parquet")
        df = pd.read_parquet(local) if os.path.exists(local) else pd.DataFrame()

    if df.empty:
        return df

    # FIX: Ensure proper DatetimeIndex regardless of how parquet was saved
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    elif not isinstance(df.index, pd.DatetimeIndex):
        if df.index.dtype == 'int64':
            unit = 'ms' if df.index.max() > 1e12 else 's'
            df.index = pd.to_datetime(df.index, unit=unit)
        else:
            df.index = pd.to_datetime(df.index)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"

    # Drop any residual Date/index columns that leaked in
    for col in list(df.columns):
        if isinstance(col, str) and col.lower() in ('date', 'index', 'level_0'):
            df = df.drop(columns=[col])

    return df


def extend_audit_with_live_dates(audit_rows: list, winner_pred: dict) -> list:
    if not audit_rows:
        return audit_rows

    etf_ret = load_etf_ret_fresh()
    if etf_ret.empty:
        return audit_rows

    etf_ret.columns = [c.split("_")[0] if "_" in c else c for c in etf_ret.columns]
    etf_ret.index   = pd.to_datetime(etf_ret.index)

    last_audit_date = pd.to_datetime(max(r["Date"] for r in audit_rows)).date()

    new_dates = [d for d in etf_ret.index if d.date() > last_audit_date]
    if not new_dates:
        return audit_rows

    live_signal = winner_pred.get("signal", audit_rows[-1].get("Signal", "GLD"))
    live_conf   = winner_pred.get("confidence", 0.2)
    live_z      = winner_pred.get("z_score", 1.0)

    extra = []
    for dt in sorted(new_dates):
        etf = live_signal if live_signal != "CASH" else audit_rows[-1].get("Signal", "GLD")
        ret = float(etf_ret.loc[dt, etf]) if etf in etf_ret.columns else 0.0
        extra.append(dict(
            Date              = str(dt.date()),
            Signal            = etf,
            Confidence        = round(live_conf, 4),
            Z_Score           = round(live_z, 4),
            Two_Day_Cumul_Pct = 0.0,
            Mode              = "📈 ETF",
            Net_Return        = round(ret, 6),
            TSL_Triggered     = False,
        ))

    return audit_rows + extra


# ─── TSL re-apply ─────────────────────────────────────────────────────────────

def apply_tsl_to_audit(audit: list, tsl_pct: float,
                        z_reentry: float, tbill: float = 3.6) -> pd.DataFrame:
    df = pd.DataFrame(audit)
    if df.empty:
        return df

    in_cash   = False
    tsl_days  = 0
    prev_ret  = 0.0
    prev2_ret = 0.0
    modes, signals, net_rets = [], [], []

    for _, row in df.iterrows():
        z       = float(row.get("Z_Score", 1.5))
        two_day = (prev_ret + prev2_ret) * 100

        if not in_cash and two_day <= -tsl_pct:
            in_cash  = True
            tsl_days = 0

        if in_cash and tsl_days >= 1 and z >= z_reentry:
            in_cash = False

        if in_cash:
            tsl_days += 1
            modes.append("💵 CASH")
            signals.append("CASH")
            net_rets.append(round(tbill / 100 / 252, 6))
        else:
            modes.append("📈 ETF")
            signals.append(row.get("Signal", "—"))
            net_rets.append(float(row.get("Net_Return", 0.0)))

        prev2_ret = prev_ret
        prev_ret  = net_rets[-1]

    df = df.copy()
    df["Mode"]       = modes
    df["Signal_TSL"] = signals
    df["Net_TSL"]    = net_rets
    return df


# ─── Download weights + data from HF Dataset on startup ─────────────────────

@st.cache_resource
def ensure_weights_and_data():
    import shutil
    from huggingface_hub import HfApi, hf_hub_download

    token = config.HF_TOKEN or None
    downloaded = {"weights": 0, "data": 0}

    os.makedirs(config.DATA_DIR, exist_ok=True)
    for f in ["etf_price","etf_ret","etf_vol",
              "bench_price","bench_ret","bench_vol","macro"]:
        local = os.path.join(config.DATA_DIR, f"{f}.parquet")
        if not os.path.exists(local):
            try:
                dl = hf_hub_download(
                    repo_id=config.HF_DATASET_REPO,
                    filename=f"data/{f}.parquet",
                    repo_type="dataset", token=token)
                shutil.copy(dl, local)
                downloaded["data"] += 1
            except Exception:
                pass

    try:
        api   = HfApi(token=token)
        files = api.list_repo_files(
            repo_id=config.HF_DATASET_REPO,
            repo_type="dataset", token=token)
        for f in files:
            if f.startswith("models/") and f.endswith((".keras", ".pkl", ".json")):
                local = f
                if not os.path.exists(local):
                    os.makedirs(os.path.dirname(local), exist_ok=True)
                    try:
                        dl = hf_hub_download(
                            repo_id=config.HF_DATASET_REPO,
                            filename=f, repo_type="dataset", token=token)
                        shutil.copy(dl, local)
                        downloaded["weights"] += 1
                    except Exception:
                        pass
    except Exception:
        pass

    return downloaded


# ─── Persistent user preferences ───────────────────────────────────────────────

PREFS_FILE = "user_prefs.json"

def load_prefs() -> dict:
    defaults = {
        "start_year": 2008, "fee_bps": 10,
        "max_epochs": 80,   "wavelet": "db4 (Daubechies-4)",
        "tsl_pct": 10,      "z_reentry": 1.1,
    }
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=config.HF_DATASET_REPO,
            filename=PREFS_FILE,
            repo_type="dataset",
            token=config.HF_TOKEN or None,
        )
        saved = json.load(open(path))
        defaults.update(saved)
    except Exception:
        pass
    return defaults


def save_prefs(prefs: dict):
    try:
        from huggingface_hub import HfApi
        import tempfile
        api = HfApi(token=config.HF_TOKEN or None)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(prefs, f, indent=2)
            tmp = f.name
        api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo=PREFS_FILE,
            repo_id=config.HF_DATASET_REPO,
            repo_type="dataset",
        )
    except Exception:
        pass


# ─── Session state ───────────────────────────────────────────────────────────
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

    start_year = st.slider("📅 Start Year",
                            min_value=config.START_YEAR_MIN,
                            max_value=config.START_YEAR_MAX,
                            value=st.session_state.start_year)
    st.session_state.start_year = start_year
    st.caption("↑ Changing this requires 🚀 Retrain. "
               "TSL/Z-score sliders are instant (no retrain needed).")

    fee_bps = st.slider("💰 Fee (bps)", 0, 50, value=st.session_state.fee_bps)
    st.session_state.fee_bps = fee_bps

    max_epochs = st.number_input("🔁 Max Epochs",
                                  min_value=10, max_value=300,
                                  value=st.session_state.max_epochs, step=5)
    st.session_state.max_epochs = int(max_epochs)

    _wavelet_options = ["db4 (Daubechies-4)", "db2 (Daubechies-2)", "haar", "sym5"]
    _wavelet_idx = _wavelet_options.index(st.session_state.wavelet) \
                   if st.session_state.wavelet in _wavelet_options else 0
    wavelet_choice = st.selectbox("〰️ Wavelet Type", options=_wavelet_options,
                                   index=_wavelet_idx,
                                   help="Selected wavelet is passed to GitHub Actions training job")
    st.session_state.wavelet = wavelet_choice
    wavelet_key = wavelet_choice.split(" ")[0]

    st.divider()
    st.markdown("### 🛡️ Risk Controls")

    tsl_pct = st.slider("🔴 Trailing Stop Loss (2-day cumul.)",
                         min_value=0, max_value=25,
                         value=st.session_state.tsl_pct,
                         step=1, format="%d%%",
                         help="Shift to CASH if 2-day cumulative return ≤ −X%",
                         key="tsl_slider")
    st.session_state.tsl_pct = tsl_pct
    st.caption(f"Triggers CASH if 2-day cumulative return ≤ −{tsl_pct}%")

    z_reentry = st.slider("📶 Z-score Re-entry Threshold",
                           min_value=1.0, max_value=2.0,
                           value=st.session_state.z_reentry,
                           step=0.1, format="%.1f σ", key="z_slider")
    st.session_state.z_reentry = z_reentry
    st.caption(f"Exit CASH → ETF when Z ≥ {z_reentry:.1f} σ. CASH earns 3m T-bill.")

    st.divider()
    st.markdown("### 🧠 Active Models")
    use_a = st.checkbox("Option A · Wavelet-CNN-LSTM",        value=True)
    use_b = st.checkbox("Option B · Wavelet-Attn-CNN-LSTM",   value=True)
    use_c = st.checkbox("Option C · Wavelet-Dual-Stream",     value=True)
    st.caption("💡 Lookback auto-selected (30/45/60d) by lowest val MSE. Split fixed at 80/10/10.")

    st.markdown("<div style='font-size:11px;color:#888;margin-bottom:4px;'>TSL / Z-score changes apply instantly:</div>",
                unsafe_allow_html=True)
    recalc_btn = st.button("🔄 Recalculate Risk Controls", use_container_width=True)
    if recalc_btn:
        st.cache_data.clear()
        st.success(f"✅ Risk recalculated — TSL={tsl_pct}%  Z={z_reentry:.1f}σ")
        st.rerun()
    st.caption("↑ Instant. No retraining needed.")

    st.markdown("<br>", unsafe_allow_html=True)

    has_gh_token = bool(os.getenv("P2SAMAPA_GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", "")))
    st.markdown("<div style='font-size:11px;color:#888;margin-bottom:4px;'>Changing Start Year / Wavelet / Epochs requires retraining:</div>",
                unsafe_allow_html=True)
    run_btn = st.button("🚀 Retrain All 3 Models", use_container_width=True, type="primary")
    if run_btn:
        save_prefs({
            "start_year": start_year, "fee_bps": fee_bps,
            "max_epochs": int(max_epochs), "wavelet": wavelet_choice,
            "tsl_pct": tsl_pct, "z_reentry": z_reentry,
        })
        if has_gh_token:
            with st.spinner("Triggering GitHub Actions training pipeline..."):
                ok = trigger_github_training(
                    start_year=start_year, wavelet=wavelet_key,
                    tsl_pct=tsl_pct, z_reentry=z_reentry,
                    epochs=max_epochs, fee_bps=fee_bps)
            if ok:
                st.success(f"✅ Retraining triggered! Training from {start_year} with {wavelet_key} wavelet.")
            else:
                st.error("❌ GitHub Actions trigger failed. Check P2SAMAPA_GITHUB_TOKEN secret.")
        else:
            st.info(f"**To retrain manually:**\n\nGitHub → Actions → Train Models → Run workflow\n"
                    f"- start_year = **{start_year}**\n- wavelet = **{wavelet_key}**\n- epochs = **{max_epochs}**")
    st.caption(f"↑ Retrains on data from {start_year} onwards using {wavelet_key} wavelet. ~1-2 hrs.")

    st.divider()
    st.markdown("### 📦 Dataset Info")
    etf_df = load_etf_ret_fresh()
    if not etf_df.empty:
        st.markdown(f"**Rows:** {len(etf_df):,}")
        rng_min = etf_df.index.min().date()
        rng_max = etf_df.index.max().date()
        st.markdown(f"**Range:** {rng_min} → {rng_max}")
    st.markdown(f"**ETFs:** {', '.join(config.ETFS)}")
    st.markdown(f"**Benchmarks:** {', '.join(config.BENCHMARKS)}")
    st.markdown("**Macro:** VIX, DXY, T10Y2Y, Corp Spread, HY Spread, TNX, 3mTBill")
    st.markdown("**Wavelet levels:** 3 (A3, D1, D2, D3)")
    st.markdown("**Split:** 80 / 10 / 10")
    st.markdown("**T-bill col:** ✅")


# ─── MAIN PANEL ───────────────────────────────────────────────────────────────

with st.spinner("🔄 Loading model weights from HF Dataset..."):
    dl_status = ensure_weights_and_data()
if dl_status["weights"] > 0 or dl_status["data"] > 0:
    st.cache_data.clear()

st.markdown("# 🧠 P2-ETF-DEEPWAVE-DL")
st.caption("Option A: Wavelet-CNN-LSTM · Option B: Wavelet-Attention-CNN-LSTM · Option C: Wavelet-Parallel-Dual-Stream-CNN-LSTM")
st.caption("Winner selected by highest raw annualised return on out-of-sample test set.")

SWEEP_YEARS = [2008, 2013, 2015, 2017, 2019, 2021]

def _today_est():
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    return (_dt.now(_tz.utc) - _td(hours=5)).date()

def _sweep_fname(year: int, for_date) -> str:
    return f"sweep_{year}_{for_date.strftime('%Y%m%d')}.json"

@st.cache_data(ttl=120)
def _load_sweep_hf(date_str: str) -> dict:
    from datetime import date as _d
    for_date = _d.fromisoformat(date_str)
    cache = {}
    try:
        token   = os.getenv("HF_TOKEN", config.HF_TOKEN)
        repo_id = config.HF_DATASET_REPO
        for yr in SWEEP_YEARS:
            fname = _sweep_fname(yr, for_date)
            try:
                path = hf_hub_download(repo_id=repo_id, filename=f"sweep/{fname}",
                                       repo_type="dataset", token=token, force_download=True)
                with open(path) as f:
                    cache[yr] = json.load(f)
            except Exception:
                pass
    except Exception:
        pass
    return cache

@st.cache_data(ttl=120)
def _load_sweep_any() -> tuple:
    from datetime import datetime as _dt2
    from huggingface_hub import HfApi
    found, best_date = {}, None
    try:
        token   = os.getenv("HF_TOKEN", config.HF_TOKEN)
        repo_id = config.HF_DATASET_REPO
        api     = HfApi()
        files   = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token))
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
                    path = hf_hub_download(repo_id=repo_id, filename=f"sweep/{fname}",
                                           repo_type="dataset", token=token, force_download=True)
                    with open(path) as f:
                        found[yr] = json.load(f)
                except Exception:
                    pass
    except Exception:
        pass
    return found, best_date

def _compute_consensus(sweep_data: dict) -> dict:
    rows = []
    for yr, sig in sweep_data.items():
        rows.append({
            "year": yr, "signal": sig.get("signal","?"),
            "ann_return": sig.get("ann_return",0.0), "z_score": sig.get("z_score",0.0),
            "sharpe": sig.get("sharpe",0.0), "max_dd": sig.get("max_dd",0.0),
        })
    if not rows:
        return {}
    import pandas as _pd
    df = _pd.DataFrame(rows)
    def _mm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)
    df["wtd"] = (0.40*_mm(df["ann_return"]) + 0.20*_mm(df["z_score"]) +
                 0.20*_mm(df["sharpe"])      + 0.20*_mm(-df["max_dd"]))
    etf_agg = {}
    for _, row in df.iterrows():
        e = row["signal"]
        etf_agg.setdefault(e, {"years":[],"scores":[],"returns":[],"zs":[],"sharpes":[],"dds":[]})
        etf_agg[e]["years"].append(row["year"])
        etf_agg[e]["scores"].append(row["wtd"])
        etf_agg[e]["returns"].append(row["ann_return"])
        etf_agg[e]["zs"].append(row["z_score"])
        etf_agg[e]["sharpes"].append(row["sharpe"])
        etf_agg[e]["dds"].append(row["max_dd"])
    total = sum(sum(v["scores"]) for v in etf_agg.values()) + 1e-9
    summary = {}
    for e, v in etf_agg.items():
        cs = sum(v["scores"])
        summary[e] = {
            "cum_score": round(cs,4), "score_share": round(cs/total,3),
            "n_years": len(v["years"]), "years": v["years"],
            "avg_return": round(float(np.mean(v["returns"])),4),
            "avg_z": round(float(np.mean(v["zs"])),3),
            "avg_sharpe": round(float(np.mean(v["sharpes"])),3),
            "avg_max_dd": round(float(np.mean(v["dds"])),4),
        }
    winner_etf = max(summary, key=lambda e: summary[e]["cum_score"])
    return {"winner": winner_etf, "etf_summary": summary,
            "per_year": df.to_dict("records"), "n_years": len(rows)}

tab1, tab2 = st.tabs(["📊 Single-Year Results", "🔄 Multi-Year Consensus Sweep"])

with tab1:
    pred  = load_prediction()
    evalu = load_evaluation()

    next_td   = current_signal_date()
    last_td   = last_trading_day()
    as_of     = pred.get("as_of_date", str(next_td))
    winner    = evalu.get("winner", "model_a")
    tbill_rt  = pred.get("tbill_rate", 3.6)
    preds     = pred.get("predictions", {})
    tsl_stat  = pred.get("tsl_status", {})
    trained_from_year = pred.get("trained_from_year")
    trained_wavelet   = pred.get("trained_wavelet")
    trained_at        = pred.get("trained_at")

    live_z       = preds.get(winner, {}).get("z_score", 1.5)
    two_day_ret  = tsl_stat.get("two_day_cumul_pct", 0.0)
    tsl_triggered= float(two_day_ret) <= -tsl_pct
    in_cash_now  = tsl_triggered and (live_z < z_reentry)
    best_lb      = evalu.get(winner, {}).get("lookback", 30)

    etf_df = load_etf_ret_fresh()
    st.markdown(f'<div class="alert-green">✅ Dataset up to date through <b>{as_of}</b>. HF Space synced.</div>',
                unsafe_allow_html=True)

    if not etf_df.empty:
        n_yrs = (etf_df.index.max() - etf_df.index.min()).days // 365
        st.markdown(f'<div class="alert-blue">📅 <b>Data:</b> {etf_df.index.min().date()} → {etf_df.index.max().date()} ({n_yrs} years) &nbsp;|&nbsp; Source: yfinance + FRED</div>',
                    unsafe_allow_html=True)

    n_feat = (len(config.ETFS)*2 + len(config.MACRO_SERIES)) * (config.WAVELET_LEVELS+1)
    st.markdown(f'<div class="alert-blue">🎯 <b>Targets:</b> {", ".join(config.ETFS)} &nbsp;·&nbsp; <b>Features:</b> {n_feat} signals &nbsp;·&nbsp; <b>T-bill:</b> {tbill_rt:.2f}% &nbsp;·&nbsp; <b>Wavelet:</b> {wavelet_key} &nbsp;·&nbsp; <b>Split:</b> 80/10/10</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="alert-yellow">⚠️ Optimal lookback: <b>{best_lb}d</b> (auto-selected from 30 / 45 / 60 by lowest validation MSE)</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="alert-yellow">🛡️ <b>Risk Controls:</b> TSL: <b>−{tsl_pct}%</b> &nbsp;·&nbsp; Re-entry Z ≥ <b>{z_reentry:.1f} σ</b> &nbsp;·&nbsp; CASH earns <b>{tbill_rt:.2f}%</b></div>',
                unsafe_allow_html=True)

    if in_cash_now:
        st.markdown(f'<div class="alert-orange">🔴 <b>CASH OVERRIDE ACTIVE</b> — 2-day cumulative return ({float(two_day_ret):+.1f}%) breached −{tsl_pct}% TSL. Current Z = {live_z:.2f} σ.</div>',
                    unsafe_allow_html=True)

    winner_label = {"model_a":"Option A","model_b":"Option B","model_c":"Option C"}.get(winner,"Option A")

    if in_cash_now:
        st.markdown(f"""<div class="cash-card">
          <div class="hero-label">⚠️ Trailing Stop Loss Triggered · Risk Override</div>
          <div class="hero-value">💵 {next_td} → CASH</div>
          <div class="hero-sub">Earning 3m T-bill: <b>{tbill_rt:.2f}% p.a.</b> &nbsp;|&nbsp; Re-entry when Z ≥ {z_reentry:.1f} σ &nbsp;|&nbsp; Current Z = {live_z:.2f} σ</div>
        </div>""", unsafe_allow_html=True)
    else:
        wp           = preds.get(winner, {})
        final_signal = wp.get("signal", "—")
        now_est      = datetime.utcnow() - timedelta(hours=5)
        is_today     = (next_td == now_est.date())
        td_label     = "TODAY'S SIGNAL" if is_today else "NEXT TRADING DAY SIGNAL"
        if trained_from_year and trained_wavelet:
            trained_at_str = f" · Generated {trained_at[:10]}" if trained_at else ""
            provenance = f"Trained from {trained_from_year} · {trained_wavelet} wavelet{trained_at_str}"
        else:
            provenance = "Training metadata unavailable — retrain to stamp results"
        st.markdown(f"""<div class="hero-card">
          <div class="hero-label">{winner_label} · {td_label}</div>
          <div class="hero-value">🎯 {next_td} → {final_signal}</div>
          <div class="hero-sub" style="margin-top:8px;font-size:13px;opacity:0.8;">📋 {provenance}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    wp = preds.get(winner, {})
    if wp and not in_cash_now:
        z_val   = float(wp.get("z_score", 0))
        probs   = wp.get("probabilities", {})
        top_etf = wp.get("signal", "—")
        conf    = float(wp.get("confidence", 0))
        if z_val >= 2.0:   strength = "Very High"
        elif z_val >= 1.5: strength = "High"
        elif z_val >= 1.0: strength = "Moderate"
        else:              strength = "Low"
        st.markdown(f"### 🟢 Signal Conviction &nbsp; `Z = {z_val:.2f} σ` &nbsp; **{strength}**")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=z_val,
            number={"suffix":" σ","font":{"size":24}},
            gauge=dict(axis=dict(range=[-3,3]), bar=dict(color="#00bfa5"),
                       steps=[dict(range=[-3,-1],color="#ffb3b3"),dict(range=[-1,0],color="#ffe0b2"),
                               dict(range=[0,1],color="#b2dfdb"),dict(range=[1,3],color="#80cbc4")],
                       threshold=dict(line=dict(color="#00897b",width=4),thickness=0.75,value=z_reentry)),
            title={"text":"Weak −3σ → Strong +3σ"},
        ))
        fig_gauge.update_layout(height=200, margin=dict(t=30,b=0,l=30,r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("**MODEL PROBABILITY BY ETF**")
        prob_cols = st.columns(len(probs))
        for i, (etf, p) in enumerate(sorted(probs.items(), key=lambda x:-x[1])):
            with prob_cols[i]:
                is_top = etf == top_etf
                color  = "#007a69" if is_top else "#555"
                bg     = "#e8faf8" if is_top else "#f7f8fa"
                border = "#00bfa5" if is_top else "#ddd"
                prefix = "★ " if is_top else ""
                st.markdown(f'<div style="border:1.5px solid {border};border-radius:20px;padding:6px 12px;text-align:center;background:{bg};color:{color};font-weight:{"700" if is_top else "500"};font-size:13px;">{prefix}{etf} {p:.3f}</div>',
                            unsafe_allow_html=True)
        st.caption(f"Z-score = std deviations the top ETF probability sits above the mean. ⚠️ CASH override triggers if 2-day cumul ≤ −{tsl_pct}%, exits when Z ≥ {z_reentry:.1f} σ.")

    st.markdown("---")
    st.markdown(f"### 📅 All Models — {next_td} Signals")
    st.caption(f"⚠️ Lookback {best_lb}d found optimal (auto-selected from 30 / 45 / 60d by lowest val MSE)")
    col_a, col_b, col_c = st.columns(3)
    model_info = [("model_a","OPTION A","#00bfa5",col_a,"model-card-a"),
                  ("model_b","OPTION B","#7b8ff7",col_b,"model-card-b"),
                  ("model_c","OPTION C","#f87171",col_c,"model-card-c")]
    for key, label, color, col, css in model_info:
        with col:
            p    = preds.get(key, {})
            sig  = "CASH" if in_cash_now else p.get("signal","—")
            conf = float(p.get("confidence", 0))
            z_v  = float(p.get("z_score", 0))
            is_w = (key == winner)
            w_tag= " ★" if is_w else ""
            st.markdown(f"""<div class="{css}">
              <div style="font-size:11px;letter-spacing:2px;font-weight:600;color:{color};margin-bottom:12px;">{label}{w_tag}</div>
              <div style="font-size:28px;font-weight:700;margin-bottom:8px;">{sig}</div>
              <div style="font-size:13px;color:#aaa;">Confidence: <span style="color:{color};font-weight:600;">{"CASH" if in_cash_now else f"{conf:.1%}"}</span></div>
              <div style="font-size:12px;color:#666;margin-top:6px;">Z = {z_v:.2f} σ</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    winner_full = {"model_a":"Option A · Wavelet-CNN-LSTM","model_b":"Option B · Wavelet-Attn-CNN-LSTM",
                   "model_c":"Option C · Wavelet-Dual-Stream"}.get(winner, winner)
    st.markdown(f"### 📊 {winner_full} — Performance Metrics")
    w_met = evalu.get(winner, {}).get("metrics", {})
    if w_met:
        m1,m2,m3,m4,m5 = st.columns(5)
        with m1:
            st.metric("📈 Ann. Return", f"{w_met.get('ann_return',0):.2f}%", delta=f"vs SPY: {w_met.get('vs_spy',0):+.2f}%")
            st.caption("Annualised")
        with m2:
            sh = w_met.get("sharpe",0)
            st.metric("📊 Sharpe", f"{sh:.2f}")
            st.caption("Strong" if sh>1 else ("Moderate" if sh>0.5 else "Weak"))
        with m3:
            hr = w_met.get("hit_ratio_15d",0)
            st.metric("🎯 Hit Ratio 15d", f"{hr:.0%}")
            st.caption("Good" if hr>0.55 else "Weak")
        with m4:
            st.metric("📉 Max Drawdown", f"{w_met.get('max_drawdown',0):.2f}%")
            st.caption("Peak to Trough")
        with m5:
            st.metric("⚠️ Max Daily DD", f"{w_met.get('max_daily_dd',0):.2f}%")
            st.caption("Worst Single Day")

    st.markdown("---")
    st.markdown("### 🏆 Approach Comparison (Winner = Highest Raw Annualised Return)")
    rows = []
    for key, lbl in [("model_a","Option A · Wavelet-CNN-LSTM"),
                      ("model_b","Option B · Wavelet-Attn-CNN-LSTM"),
                      ("model_c","Option C · Wavelet-Dual-Stream")]:
        m = evalu.get(key, {}).get("metrics", {})
        if m:
            lb_k   = evalu.get(key, {}).get("lookback", "—")
            p_info = preds.get(key, {})
            sig    = "CASH" if in_cash_now else p_info.get("signal","—")
            conf   = float(p_info.get("confidence", 0))
            rows.append({
                "Model": lbl, "Lookback": f"{lb_k}d",
                f"Signal {next_td}": sig,
                "Confidence": f"{conf:.1%}" if conf > 0 else "—",
                "Ann. Return": f"{m.get('ann_return',0):.2f}%",
                "Sharpe": f"{m.get('sharpe',0):.2f}",
                "Hit Ratio (15d)": f"{m.get('hit_ratio_15d',0):.0%}",
                "Max Drawdown": f"{m.get('max_drawdown',0):.2f}%",
                "Winner": "⭐ WINNER" if key==winner else "",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### 📋 Benchmark Comparison")
    bench_rows = []
    for key, lbl in [(winner, f"{winner_full} (Winner)"),("SPY","SPY (Buy & Hold)"),("AGG","AGG (Buy & Hold)")]:
        b   = evalu.get(key, {})
        m   = b.get("metrics", b) if isinstance(b, dict) else {}
        ann = m.get("ann_return", b.get("ann_return","—"))
        bench_rows.append({
            "Strategy": lbl,
            "Ann. Return": f"{ann:.2f}%" if isinstance(ann,(int,float)) else ann,
            "Sharpe": f"{m.get('sharpe','—'):.2f}" if m.get("sharpe") else "—",
            "Max Drawdown": f"{m.get('max_drawdown','—'):.2f}%" if m.get("max_drawdown") else "—",
        })
    if bench_rows:
        st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📈 Cumulative Return — Test Period")
    fig = go.Figure()
    colors_map = {"model_a":"#00bfa5","model_b":"#7b8ff7","model_c":"#f87171"}
    for key, lbl in [("model_a","Option A"),("model_b","Option B"),("model_c","Option C")]:
        sigs = evalu.get(key,{}).get("all_signals",[])
        if sigs:
            df_s = apply_tsl_to_audit(sigs, tsl_pct, z_reentry, tbill_rt)
            if "Date" in df_s.columns and "Net_TSL" in df_s.columns:
                df_s["Date"]  = pd.to_datetime(df_s["Date"])
                df_s["Cumul"] = (1 + df_s["Net_TSL"]).cumprod()
                fig.add_trace(go.Scatter(x=df_s["Date"], y=df_s["Cumul"],
                                         name=lbl, line=dict(color=colors_map[key], width=2)))
    fig.update_layout(height=350, margin=dict(t=20,b=20),
                      yaxis_title="Growth of $1", xaxis_title="Date",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="#f0f2f5")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f2f5")
    st.plotly_chart(fig, use_container_width=True)

    winner_pred_info = preds.get(winner, {})
    all_probs = list(winner_pred_info.get("probabilities", {}).values())
    if all_probs and max(all_probs) < 0.25:
        st.markdown("""<div class="alert-orange">⚠️ <b>Model probabilities are near-uniform (≈20% each)</b> — retrain needed for meaningful ETF discrimination. Click <b>🚀 Retrain All 3 Models</b> in the sidebar.</div>""",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"### 📋 Audit Trail — {winner_full} (Last 30 Trading Days)")
    audit_raw = evalu.get(winner, {}).get("audit_tail", [])
    if audit_raw:
        audit_raw = extend_audit_with_live_dates(audit_raw, winner_pred_info)
    if audit_raw:
        df_audit = apply_tsl_to_audit(audit_raw, tsl_pct, z_reentry, tbill_rt)
        disp = df_audit[["Date","Signal_TSL","Confidence","Z_Score","Net_TSL","Mode"]].copy()
        disp.columns = ["Date","Signal","Confidence","Z Score","Net Return","Mode"]
        disp["Date"]       = pd.to_datetime(disp["Date"], format="mixed").dt.strftime("%Y-%m-%d")
        disp["Confidence"] = disp["Confidence"].apply(lambda x: f"{float(x):.1%}" if isinstance(x,(int,float)) else x)
        disp["Z Score"]    = disp["Z Score"].apply(lambda x: f"{float(x):.2f}" if isinstance(x,(int,float)) else x)
        disp["Net Return"] = disp["Net Return"].apply(lambda x: f"+{x*100:.2f}%" if float(x)>=0 else f"{x*100:.2f}%")
        def color_ret(val):
            if "+" in str(val): return "color:#27ae60;font-weight:600"
            if "-" in str(val): return "color:#e74c3c;font-weight:600"
            return ""
        def color_mode(val):
            if "CASH" in str(val): return "background-color:#fff8f5;color:#e65100"
            return "background-color:#f0fdf9;color:#007a69"
        styled = disp.style.applymap(color_ret, subset=["Net Return"]).applymap(color_mode, subset=["Mode"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        latest_date = disp["Date"].iloc[-1]
        if latest_date < str(next_td):
            st.caption(f"⚠️ Audit trail through {latest_date}. Rows up to {next_td} will appear after next retrain.")

    st.markdown("---")
    st.caption(f"P2-ETF-DEEPWAVE-DL · GitHub: {config.GITHUB_REPO} · HF: {config.HF_DATASET_REPO} · Wavelet: {wavelet_key} · Split: 80/10/10 · Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Multi-Year Consensus Sweep
# ═══════════════════════════════════════════════════════════════════════════════
ETF_COLORS_SW = {
    "TLT":"#4e79a7","VCIT":"#f28e2b","LQD":"#59a14f","HYG":"#e15759",
    "VNQ":"#76b7b2","SLV":"#edc948","GLD":"#b07aa1","CASH":"#aaaaaa",
}

with tab2:
    st.subheader("🔄 Multi-Year Consensus Sweep")
    st.markdown(
        "Trains the winning wavelet model across **6 start years** and aggregates "
        "signals into a weighted consensus.  \n"
        f"**Sweep years:** {', '.join(str(y) for y in SWEEP_YEARS)}  &nbsp;·&nbsp;  "
        "**Score:** 40% Return · 20% Z · 20% Sharpe · 20% (–MaxDD)  \n"
        "Auto-runs daily at **8pm EST**. Results are date-stamped — stale cache never shown."
    )

    today_sw   = _today_est()
    today_str  = str(today_sw)
    today_cache = _load_sweep_hf(today_str)
    prev_cache, prev_date = _load_sweep_any()
    if prev_date == today_sw:
        prev_cache, prev_date = {}, None

    sweep_complete = len(today_cache) == len(SWEEP_YEARS)
    display_cache  = today_cache if today_cache else prev_cache
    display_date   = today_sw    if today_cache else prev_date

    if display_cache and display_date and display_date < today_sw:
        st.warning(f"⚠️ Showing results from **{display_date}**. Today's sweep hasn't run yet — auto-triggers at 8pm EST.", icon="📅")

    cols = st.columns(len(SWEEP_YEARS))
    for i, yr in enumerate(SWEEP_YEARS):
        with cols[i]:
            if yr in today_cache:
                st.success(f"**{yr}**\n✅ {today_cache[yr].get('signal','?')}")
            elif yr in prev_cache:
                st.warning(f"**{yr}**\n📅 {prev_cache[yr].get('signal','?')}")
            else:
                st.error(f"**{yr}**\n⏳ Not run")
    st.caption("✅ today · 📅 previous day · ⏳ not run")
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
            st.success(f"✅ Today's sweep complete ({today_str}) — all {len(SWEEP_YEARS)} years ready")
        else:
            st.info(f"**{len(today_cache)}/{len(SWEEP_YEARS)}** years done today. Will trigger **{len(trigger_years)}** jobs: {', '.join(str(y) for y in trigger_years)}")

    if sweep_btn and trigger_years:
        sweep_str = ",".join(str(y) for y in trigger_years)
        with st.spinner(f"Triggering sweep for {sweep_str}..."):
            ok = trigger_github_training(start_year=trigger_years[0], wavelet=wavelet_key,
                                          tsl_pct=tsl_pct, z_reentry=z_reentry,
                                          fee_bps=fee_bps, sweep_mode=sweep_str)
        if ok:
            st.success(f"✅ Triggered {len(trigger_years)} parallel jobs: {sweep_str}.")
        else:
            st.error("❌ Failed to trigger GitHub Actions. Check GITHUB_TOKEN secret.")

    if not display_cache:
        st.info("👆 No sweep results yet. Click **🚀 Run Consensus Sweep** or wait for 8pm EST.")
        st.stop()

    consensus = _compute_consensus(display_cache)
    if not consensus:
        st.warning("Could not compute consensus.")
        st.stop()

    winner_sw  = consensus["winner"]
    w_info     = consensus["etf_summary"][winner_sw]
    win_color  = ETF_COLORS_SW.get(winner_sw, "#0066cc")
    score_pct  = w_info["score_share"] * 100
    split_sig  = w_info["score_share"] < 0.40
    sig_label  = "⚠️ Split Signal" if split_sig else "✅ Clear Consensus"
    date_note  = f"Results from: {display_date}"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border:2px solid {win_color};
                border-radius:16px;padding:32px;text-align:center;margin:16px 0;">
      <div style="font-size:11px;letter-spacing:3px;color:#aaa;margin-bottom:8px;">
        WEIGHTED CONSENSUS · DEEPWAVE · {len(display_cache)} START YEARS · {date_note}</div>
      <div style="font-size:72px;font-weight:900;color:{win_color};text-shadow:0 0 30px {win_color}88;">{winner_sw}</div>
      <div style="font-size:14px;color:#ccc;margin-top:8px;">{sig_label} · Score share {score_pct:.0f}% · {w_info['n_years']}/{len(SWEEP_YEARS)} years</div>
      <div style="display:flex;justify-content:center;gap:32px;margin-top:20px;flex-wrap:wrap;">
        <div style="text-align:center;"><div style="font-size:11px;color:#aaa;">Avg Return</div>
          <div style="font-size:22px;font-weight:700;color:{'#00b894' if w_info['avg_return']>0 else '#d63031'};">{w_info['avg_return']*100:.1f}%</div></div>
        <div style="text-align:center;"><div style="font-size:11px;color:#aaa;">Avg Z</div>
          <div style="font-size:22px;font-weight:700;color:#74b9ff;">{w_info['avg_z']:.2f}σ</div></div>
        <div style="text-align:center;"><div style="font-size:11px;color:#aaa;">Avg Sharpe</div>
          <div style="font-size:22px;font-weight:700;color:#a29bfe;">{w_info['avg_sharpe']:.2f}</div></div>
        <div style="text-align:center;"><div style="font-size:11px;color:#aaa;">Avg MaxDD</div>
          <div style="font-size:22px;font-weight:700;color:#fd79a8;">{w_info['avg_max_dd']*100:.1f}%</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    others = sorted([(e,v) for e,v in consensus["etf_summary"].items() if e != winner_sw],
                    key=lambda x: -x[1]["cum_score"])
    parts = [f'<span style="color:{ETF_COLORS_SW.get(e,"#888")};font-weight:600;">{e}</span> <span style="color:#aaa;">({v["cum_score"]:.2f})</span>'
             for e,v in others]
    st.markdown('<div style="text-align:center;font-size:13px;margin-bottom:12px;">Also ranked: ' + ' &nbsp;|&nbsp; '.join(parts) + '</div>', unsafe_allow_html=True)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Weighted Score per ETF**")
        es = consensus["etf_summary"]
        sorted_etfs = sorted(es.keys(), key=lambda e: -es[e]["cum_score"])
        fig_bar = go.Figure(go.Bar(
            x=sorted_etfs, y=[es[e]["cum_score"] for e in sorted_etfs],
            marker_color=[ETF_COLORS_SW.get(e,"#888") for e in sorted_etfs],
            text=[f"{es[e]['n_years']}yr · {es[e]['score_share']*100:.0f}%" for e in sorted_etfs],
            textposition="outside"))
        fig_bar.update_layout(template="plotly_dark", height=360, yaxis_title="Cumulative Score",
                              showlegend=False, margin=dict(t=20,b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.markdown("**Z-Score Conviction by Start Year**")
        per_year = consensus["per_year"]
        fig_sc = go.Figure()
        for row in per_year:
            etf = row["signal"]
            col = ETF_COLORS_SW.get(etf, "#888")
            fig_sc.add_trace(go.Scatter(
                x=[row["year"]], y=[row["z_score"]], mode="markers+text",
                marker=dict(size=18, color=col, line=dict(color="white",width=1)),
                text=[etf], textposition="top center", name=etf, showlegend=False,
                hovertemplate=f"<b>{etf}</b><br>Year: {row['year']}<br>Z: {row['z_score']:.2f}σ<br>Return: {row['ann_return']*100:.1f}%<extra></extra>"))
        fig_sc.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig_sc.update_layout(template="plotly_dark", height=360,
                             xaxis_title="Start Year", yaxis_title="Z-Score (σ)", margin=dict(t=20,b=20))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("📋 Full Per-Year Breakdown")
    st.caption(f"40% Ann. Return + 20% Z-Score + 20% Sharpe + 20% (–MaxDD), min-max normalised · Results: {display_date}")
    tbl_rows = []
    for row in sorted(per_year, key=lambda r: r["year"]):
        tbl_rows.append({
            "Start Year": row["year"], "Signal": row["signal"],
            "Wtd Score": round(row["wtd"],3), "Z-Score": f"{row['z_score']:.2f}σ",
            "Ann. Return": f"{row['ann_return']*100:.2f}%", "Sharpe": f"{row['sharpe']:.2f}",
            "Max Drawdown": f"{row['max_dd']*100:.2f}%",
            "Date": "✅ Today" if row["year"] in today_cache else f"📅 {display_date}",
        })
    tbl_df = pd.DataFrame(tbl_rows)
    def _style_sig_sw(val):
        c = ETF_COLORS_SW.get(val, "#888")
        return f"background-color:{c}22;color:{c};font-weight:700;"
    def _style_ret_sw(val):
        try:
            v = float(str(val).replace("%",""))
            return "color:#00b894;font-weight:600" if v > 0 else "color:#d63031;font-weight:600"
        except Exception:
            return ""
    st.dataframe(
        tbl_df.style.applymap(_style_sig_sw, subset=["Signal"])
                    .applymap(_style_ret_sw, subset=["Ann. Return"])
                    .set_properties(**{"text-align":"center","font-size":"14px"})
                    .hide(axis="index"),
        use_container_width=True, height=280)
