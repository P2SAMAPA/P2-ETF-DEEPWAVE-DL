# app.py — P2-ETF-DEEPWAVE-DL
# Streamlit Cloud deployment (streamlit.io)
# Secrets loaded via st.secrets (set in Streamlit Cloud dashboard)
# Tabs: FI Signal | FI Consensus | Equity Signal | Equity Consensus

import json
import os
import requests
import config
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download, HfApi  # <-- ADDED missing import

# ─── Streamlit Cloud: load secrets into env before importing config ────────────
def _bootstrap_secrets():
    try:
        # Define the keys to sync from st.secrets to os.environ
        keys_to_sync = [
            "HF_TOKEN", 
            "FRED_API_KEY", 
            "HF_DATASET_REPO",
            "P2SAMAPA_GITHUB_TOKEN", 
            "GITHUB_TOKEN"
        ]
        
        for key in keys_to_sync:
            if key in st.secrets:
                # Force update os.environ to ensure it's available globally
                os.environ[key] = str(st.secrets[key])
                
    except Exception as e:
        st.error(f"Error bootstrapping secrets: {e}")

# Execute the bootstrap immediately
_bootstrap_secrets()

# Final safety check for the specific error you are seeing
if not os.environ.get("GITHUB_TOKEN") and not os.environ.get("P2SAMAPA_GITHUB_TOKEN"):
    st.error("❌ Failed. Check GITHUB_TOKEN in Streamlit secrets.")
    st.info("Ensure your Secrets dashboard has: GITHUB_TOKEN = 'your_token'")
    st.stop()

# Ensure config is available
try:
    # Just a sanity check – if config is missing, this will raise an ImportError
    _ = config.DEFAULT_TSL_PCT
except (ImportError, AttributeError):
    st.error("❌ config.py is missing or incomplete. Please ensure it's in the repository.")
    st.stop()

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
  [data-testid="stSidebar"] { min-width:300px; max-width:300px; }
  .hero-card    { background:linear-gradient(135deg,#00bfa5,#00897b);
                  border-radius:14px;padding:32px;text-align:center;color:white;margin:16px 0; }
  .hero-card-eq { background:linear-gradient(135deg,#2563eb,#1d4ed8);
                  border-radius:14px;padding:32px;text-align:center;color:white;margin:16px 0; }
  .cash-card    { background:linear-gradient(135deg,#e65100,#bf360c);
                  border-radius:14px;padding:32px;text-align:center;color:white;margin:16px 0; }
  .hero-label   { font-size:11px;letter-spacing:2px;text-transform:uppercase;opacity:.8;margin-bottom:8px; }
  .hero-value   { font-size:40px;font-weight:700; }
  .hero-sub     { font-size:13px;opacity:.85;margin-top:10px; }
  .model-card-a { background:#0d1117;border-radius:12px;padding:24px;text-align:center;
                  color:white;border:1px solid #1a2a1a; }
  .model-card-b { background:#0d1117;border-radius:12px;padding:24px;text-align:center;
                  color:white;border:2px solid #7b8ff7; }
  .model-card-c { background:#0d1117;border-radius:12px;padding:24px;text-align:center;
                  color:white;border:1px solid #2a1a1a; }
  .param-box    { background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                  padding:14px 18px;font-size:13px;color:#374151;margin-bottom:8px; }
  .param-row    { display:flex;justify-content:space-between;margin-bottom:4px; }
  .param-key    { color:#6b7280;font-weight:500; }
  .param-val    { color:#111827;font-weight:700;font-family:monospace; }
  .wavelet-pill { display:inline-block;background:#dbeafe;border:1px solid #93c5fd;
                  border-radius:20px;padding:2px 12px;font-size:12px;
                  font-weight:700;color:#1d4ed8; }
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

# ─── Constants ────────────────────────────────────────────────────────────────
TSL_PCT   = config.DEFAULT_TSL_PCT    # 12
Z_REENTRY = config.DEFAULT_Z_REENTRY  # 0.9
FEE_BPS   = config.FEE_BPS            # 12

SWEEP_YEARS = [2008, 2013, 2015, 2017, 2019, 2021]

ETF_COLORS_FI = {
    "TLT":"#4e79a7","VCIT":"#f28e2b","LQD":"#59a14f",
    "HYG":"#e15759","VNQ":"#76b7b2","SLV":"#edc948","GLD":"#b07aa1","CASH":"#aaaaaa",
}
ETF_COLORS_EQ = {
    "QQQ":"#7c3aed","XLK":"#0891b2","XLF":"#059669","XLE":"#d97706",
    "XLV":"#dc2626","XLI":"#9333ea","XLY":"#db2777","XLP":"#16a34a",
    "XLU":"#ca8a04","XME":"#64748b","GDX":"#b45309","IWM":"#0f766e","CASH":"#aaaaaa",
}

# ─── Trading calendar helpers ─────────────────────────────────────────────────
US_HOLIDAYS = {
    date(2025,1,1),date(2025,1,20),date(2025,2,17),date(2025,4,18),
    date(2025,5,26),date(2025,6,19),date(2025,7,4),date(2025,9,1),
    date(2025,11,27),date(2025,12,25),
    date(2026,1,1),date(2026,1,19),date(2026,2,16),date(2026,4,3),
    date(2026,5,25),date(2026,6,19),date(2026,7,3),date(2026,9,7),
    date(2026,11,26),date(2026,12,25),
}

def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in US_HOLIDAYS

def current_signal_date() -> date:
    now_est = datetime.utcnow() - timedelta(hours=5)
    today, hour = now_est.date(), now_est.hour
    if is_trading_day(today) and hour < 16:
        return today
    d = today + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d

def last_trading_day() -> date:
    d = date.today()
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d

def _today_est() -> date:
    return (datetime.utcnow() - timedelta(hours=5)).date()


# ─── Token helpers (Streamlit Cloud secrets + env fallback) ──────────────────
def _hf_token() -> str:
    try:
        return st.secrets.get("HF_TOKEN", "") or config.HF_TOKEN
    except Exception:
        return config.HF_TOKEN

def _gh_token() -> str:
    try:
        return (st.secrets.get("P2SAMAPA_GITHUB_TOKEN", "")
                or st.secrets.get("GITHUB_TOKEN", "")
                or config.GITHUB_TOKEN)
    except Exception:
        return config.GITHUB_TOKEN

def _hf_repo() -> str:
    try:
        return st.secrets.get("HF_DATASET_REPO", "") or config.HF_DATASET_REPO
    except Exception:
        return config.HF_DATASET_REPO


# ─── GitHub Actions trigger ───────────────────────────────────────────────────
def trigger_github(start_year: int, workflow: str = "train_models.yml") -> bool:
    token = _gh_token()
    if not token:
        return False
    url = (f"https://api.github.com/repos/{config.GITHUB_REPO}"
           f"/actions/workflows/{workflow}/dispatches")
    payload = {"ref": "main", "inputs": {
        "model": "all", "epochs": str(config.MAX_EPOCHS),
        "start_year": str(start_year),
    }}
    r = requests.post(url, json=payload, headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    })
    return r.status_code == 204


# ─── HF data loaders ─────────────────────────────────────────────────────────
def _hf_download(filename: str, force: bool = False):
    return hf_hub_download(repo_id=_hf_repo(), filename=filename,
                           repo_type="dataset", token=_hf_token() or None,
                           force_download=force)

@st.cache_data(ttl=1800)
def load_prediction() -> dict:
    try:
        with open(_hf_download("latest_prediction.json")) as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data(ttl=1800)
def load_prediction_equity() -> dict:
    try:
        with open(_hf_download("latest_prediction_equity.json")) as f:
            return json.load(f)
    except Exception:
        if os.path.exists("latest_prediction_equity.json"):
            with open("latest_prediction_equity.json") as f:
                return json.load(f)
        return {}

@st.cache_data(ttl=1800)
def load_evaluation() -> dict:
    try:
        with open(_hf_download("evaluation_results.json")) as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data(ttl=1800)
def load_evaluation_equity() -> dict:
    try:
        with open(_hf_download("evaluation_results_equity.json")) as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data(ttl=300)
def load_etf_ret_fresh() -> pd.DataFrame:
    try:
        df = pd.read_parquet(_hf_download("data/etf_ret.parquet", force=True))
    except Exception:
        local = os.path.join(config.DATA_DIR, "etf_ret.parquet")
        df = pd.read_parquet(local) if os.path.exists(local) else pd.DataFrame()
    if df.empty:
        return df
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    for col in list(df.columns):
        if isinstance(col, str) and col.lower() in ("date", "index", "level_0"):
            df = df.drop(columns=[col])
    return df

@st.cache_data(ttl=120)
def load_sweep_for_date(date_str: str, universe: str = "fi") -> dict:
    """Load all sweep JSON files for a given date and universe."""
    from datetime import date as _d
    for_date = _d.fromisoformat(date_str)
    cache = {}
    prefix = "" if universe == "fi" else "eq_"
    for yr in SWEEP_YEARS:
        fname = f"sweep/{prefix}sweep_{yr}_{for_date.strftime('%Y%m%d')}.json"
        try:
            with open(_hf_download(fname, force=True)) as f:
                cache[yr] = json.load(f)
        except Exception:
            pass
    return cache

@st.cache_data(ttl=120)
def load_latest_sweep(universe: str = "fi") -> tuple:
    """Find and load the most recent sweep results for a universe."""
    from datetime import datetime as _dt
    from huggingface_hub import HfApi
    found, best_date = {}, None
    prefix = "" if universe == "fi" else "eq_"
    try:
        api   = HfApi()
        files = list(api.list_repo_files(repo_id=_hf_repo(),
                                          repo_type="dataset",
                                          token=_hf_token() or None))
        for fpath in files:
            fname = os.path.basename(fpath)
            pfx   = f"{prefix}sweep_"
            if fname.startswith(pfx) and fname.endswith(".json"):
                parts = fname.replace(".json","").split("_")
                if len(parts) >= 3:
                    try:
                        dt = _dt.strptime(parts[-1], "%Y%m%d").date()
                        if best_date is None or dt > best_date:
                            best_date = dt
                    except Exception:
                        pass
        if best_date:
            found = load_sweep_for_date(str(best_date), universe=universe)
    except Exception:
        pass
    return found, best_date


# ─── Consensus calculation ────────────────────────────────────────────────────
def compute_consensus(sweep_data: dict) -> dict:
    rows = []
    for yr, sig in sweep_data.items():
        rows.append({"year": yr, "signal": sig.get("signal", "?"),
                     "ann_return": sig.get("ann_return", 0.0),
                     "z_score": sig.get("z_score", 0.0),
                     "sharpe": sig.get("sharpe", 0.0),
                     "max_dd": sig.get("max_dd", 0.0)})
    if not rows:
        return {}
    df = pd.DataFrame(rows)

    def _mm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    df["wtd"] = (0.40 * _mm(df["ann_return"]) + 0.20 * _mm(df["z_score"]) +
                 0.20 * _mm(df["sharpe"])      + 0.20 * _mm(-df["max_dd"]))

    etf_agg = {}
    for _, row in df.iterrows():
        e = row["signal"]
        etf_agg.setdefault(e, {"years":[],"scores":[],"returns":[],
                                "zs":[],"sharpes":[],"dds":[]})
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
            "cum_score":   round(cs, 4),
            "score_share": round(cs / total, 3),
            "n_years":     len(v["years"]),
            "years":       v["years"],
            "avg_return":  round(float(np.mean(v["returns"])), 4),
            "avg_z":       round(float(np.mean(v["zs"])), 3),
            "avg_sharpe":  round(float(np.mean(v["sharpes"])), 3),
            "avg_max_dd":  round(float(np.mean(v["dds"])), 4),
        }
    winner = max(summary, key=lambda e: summary[e]["cum_score"])
    return {"winner": winner, "etf_summary": summary,
            "per_year": df.to_dict("records"), "n_years": len(rows)}


# ─── TSL replay ───────────────────────────────────────────────────────────────
def apply_tsl_to_audit(audit: list, tbill: float = 3.6) -> pd.DataFrame:
    df = pd.DataFrame(audit)
    if df.empty:
        return df
    in_cash, tsl_days, prev, prev2 = False, 0, 0.0, 0.0
    modes, signals, net_rets = [], [], []
    for _, row in df.iterrows():
        z       = float(row.get("Z_Score", 1.5))
        two_day = (prev + prev2) * 100
        if not in_cash and two_day <= -TSL_PCT:
            in_cash, tsl_days = True, 0
        if in_cash and tsl_days >= 1 and z >= Z_REENTRY:
            in_cash = False
        if in_cash:
            tsl_days += 1
            modes.append("💵 CASH"); signals.append("CASH")
            net_rets.append(round(tbill / 100 / 252, 6))
        else:
            modes.append("📈 ETF"); signals.append(row.get("Signal", "—"))
            net_rets.append(float(row.get("Net_Return", 0.0)))
        prev2, prev = prev, net_rets[-1]
    df = df.copy()
    df["Mode"] = modes; df["Signal_TSL"] = signals; df["Net_TSL"] = net_rets
    return df


# ─── Shared render helpers ────────────────────────────────────────────────────
def render_prob_pills(probs: dict, top_etf: str, etf_colors: dict):
    cols = st.columns(len(probs))
    for i, (etf, p) in enumerate(sorted(probs.items(), key=lambda x: -x[1])):
        with cols[i]:
            is_top = etf == top_etf
            color  = etf_colors.get(etf, "#555") if is_top else "#555"
            bg     = "#f0f9ff" if is_top else "#f7f8fa"
            border = color if is_top else "#ddd"
            prefix = "★ " if is_top else ""
            st.markdown(
                f'<div style="border:1.5px solid {border};border-radius:20px;'
                f'padding:6px 10px;text-align:center;background:{bg};color:{color};'
                f'font-weight:{"700" if is_top else "500"};font-size:12px;">'
                f'{prefix}{etf}<br>{p:.3f}</div>',
                unsafe_allow_html=True,
            )


def render_model_cards(preds: dict, winner: str, in_cash: bool,
                       colors: tuple = ("#00bfa5","#7b8ff7","#f87171")):
    col_a, col_b, col_c = st.columns(3)
    for key, label, color, col, css in [
        ("model_a","OPTION A", colors[0], col_a, "model-card-a"),
        ("model_b","OPTION B", colors[1], col_b, "model-card-b"),
        ("model_c","OPTION C", colors[2], col_c, "model-card-c"),
    ]:
        with col:
            p    = preds.get(key, {})
            sig  = "CASH" if in_cash else p.get("signal", "—")
            conf = float(p.get("confidence", 0))
            z_v  = float(p.get("z_score", 0))
            w_tag = " ★" if key == winner else ""
            st.markdown(f"""<div class="{css}">
              <div style="font-size:11px;letter-spacing:2px;font-weight:600;
                          color:{color};margin-bottom:12px;">{label}{w_tag}</div>
              <div style="font-size:28px;font-weight:700;margin-bottom:8px;">{sig}</div>
              <div style="font-size:13px;color:#aaa;">Conf:
                <span style="color:{color};font-weight:600;">
                  {"CASH" if in_cash else f"{conf:.1%}"}
                </span>
              </div>
              <div style="font-size:12px;color:#666;margin-top:6px;">Z = {z_v:.2f} σ</div>
            </div>""", unsafe_allow_html=True)


def render_consensus_tab(universe: str, etf_colors: dict, label: str):
    """Shared consensus tab renderer for both FI and Equity."""
    st.subheader(f"🔄 {label} — Multi-Year Consensus Sweep")
    st.markdown(
        f"Trains the best wavelet model across **{len(SWEEP_YEARS)} start years** "
        f"and aggregates signals into a weighted consensus.  \n"
        f"**Sweep years:** {', '.join(str(y) for y in SWEEP_YEARS)}  "
        f"&nbsp;·&nbsp;  "
        "**Score:** 40% Return · 20% Z · 20% Sharpe · 20% (–MaxDD)"
    )

    today       = _today_est()
    today_cache = load_sweep_for_date(str(today), universe=universe)
    prev_cache, prev_date = load_latest_sweep(universe=universe)
    if prev_date == today:
        prev_cache, prev_date = {}, None

    sweep_complete = len(today_cache) == len(SWEEP_YEARS)
    display_cache  = today_cache if today_cache else prev_cache
    display_date   = today       if today_cache else prev_date

    if display_cache and display_date and display_date < today:
        st.warning(f"⚠️ Showing results from **{display_date}**. "
                   "Today's sweep hasn't run yet.")

    # Status pills
    cols = st.columns(len(SWEEP_YEARS))
    for i, yr in enumerate(SWEEP_YEARS):
        with cols[i]:
            if yr in today_cache:
                st.success(f"**{yr}**\n✅ {today_cache[yr].get('signal','?')}")
            elif yr in prev_cache:
                st.warning(f"**{yr}**\n📅 {prev_cache[yr].get('signal','?')}")
            else:
                st.error(f"**{yr}**\n⏳ Pending")
    st.caption("✅ today · 📅 previous · ⏳ not run")
    st.divider()

    # Trigger button
    missing = [yr for yr in SWEEP_YEARS if yr not in today_cache]
    force   = st.checkbox("🔄 Force re-run all years",
                           value=False, key=f"force_{universe}")
    trigger_years = SWEEP_YEARS if force else missing
    wf = ("train_models.yml" if universe == "fi"
          else "train_equity_models.yml")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        disabled = sweep_complete and not force
        run_btn  = st.button(f"🚀 Run {label} Sweep", type="primary",
                              use_container_width=True, disabled=disabled,
                              key=f"sweep_btn_{universe}")
    with col_info:
        if sweep_complete and not force:
            st.success(f"✅ Today's sweep complete — all {len(SWEEP_YEARS)} years done")
        else:
            st.info(f"**{len(today_cache)}/{len(SWEEP_YEARS)}** years done today. "
                    f"Will trigger **{len(trigger_years)}** jobs.")

    if run_btn and trigger_years:
        has_token = bool(_gh_token())
        if has_token:
            with st.spinner(f"Triggering {label} sweep..."):
                ok = trigger_github(trigger_years[0], workflow=wf)
            if ok:
                st.success(f"✅ Triggered {len(trigger_years)} jobs.")
            else:
                st.error("❌ Failed. Check GITHUB_TOKEN in Streamlit secrets.")
        else:
            st.info(f"Add `P2SAMAPA_GITHUB_TOKEN` to Streamlit Cloud secrets "
                    f"to trigger workflows from the UI.")

    if not display_cache:
        st.info("👆 No sweep results yet. Trigger the sweep to get started.")
        return

    consensus = compute_consensus(display_cache)
    if not consensus:
        st.warning("Could not compute consensus.")
        return

    winner_etf = consensus["winner"]
    w_info     = consensus["etf_summary"][winner_etf]
    win_color  = etf_colors.get(winner_etf, "#0066cc")
    score_pct  = w_info["score_share"] * 100
    split_sig  = w_info["score_share"] < 0.40
    sig_label  = "⚠️ Split Signal" if split_sig else "✅ Clear Consensus"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                border:2px solid {win_color};border-radius:16px;
                padding:32px;text-align:center;margin:16px 0;">
      <div style="font-size:11px;letter-spacing:3px;color:#aaa;margin-bottom:8px;">
        WEIGHTED CONSENSUS · {label.upper()} · {len(display_cache)} YEARS · {display_date}
      </div>
      <div style="font-size:72px;font-weight:900;color:{win_color};
                  text-shadow:0 0 30px {win_color}88;">{winner_etf}</div>
      <div style="font-size:14px;color:#ccc;margin-top:8px;">
        {sig_label} · Score share {score_pct:.0f}% · {w_info['n_years']}/{len(SWEEP_YEARS)} years
      </div>
      <div style="display:flex;justify-content:center;gap:32px;margin-top:20px;flex-wrap:wrap;">
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg Return</div>
          <div style="font-size:22px;font-weight:700;
                      color:{'#00b894' if w_info['avg_return']>0 else '#d63031'};">
            {w_info['avg_return']*100:.1f}%
          </div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg Z</div>
          <div style="font-size:22px;font-weight:700;color:#74b9ff;">
            {w_info['avg_z']:.2f}σ
          </div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg Sharpe</div>
          <div style="font-size:22px;font-weight:700;color:#a29bfe;">
            {w_info['avg_sharpe']:.2f}
          </div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:11px;color:#aaa;">Avg MaxDD</div>
          <div style="font-size:22px;font-weight:700;color:#fd79a8;">
            {w_info['avg_max_dd']*100:.1f}%
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Runner-up badges
    others = sorted([(e, v) for e, v in consensus["etf_summary"].items()
                     if e != winner_etf],
                    key=lambda x: -x[1]["cum_score"])
    parts = [
        f'<span style="color:{etf_colors.get(e,"#888")};font-weight:600;">{e}</span>'
        f' <span style="color:#aaa;">({v["cum_score"]:.2f})</span>'
        for e, v in others
    ]
    st.markdown(
        '<div style="text-align:center;font-size:13px;margin-bottom:12px;">'
        'Also ranked: ' + ' &nbsp;|&nbsp; '.join(parts) + '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Weighted Score per ETF**")
        es   = consensus["etf_summary"]
        etfs = sorted(es.keys(), key=lambda e: -es[e]["cum_score"])
        fig  = go.Figure(go.Bar(
            x=etfs, y=[es[e]["cum_score"] for e in etfs],
            marker_color=[etf_colors.get(e, "#888") for e in etfs],
            text=[f"{es[e]['n_years']}yr · {es[e]['score_share']*100:.0f}%"
                  for e in etfs],
            textposition="outside",
        ))
        fig.update_layout(template="plotly_dark", height=360,
                          yaxis_title="Cumulative Score",
                          showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Z-Score Conviction by Start Year**")
        fig2 = go.Figure()
        for row in consensus["per_year"]:
            etf = row["signal"]
            clr = etf_colors.get(etf, "#888")
            fig2.add_trace(go.Scatter(
                x=[row["year"]], y=[row["z_score"]],
                mode="markers+text",
                marker=dict(size=18, color=clr,
                            line=dict(color="white", width=1)),
                text=[etf], textposition="top center",
                name=etf, showlegend=False,
                hovertemplate=(f"<b>{etf}</b><br>Year: {row['year']}<br>"
                               f"Z: {row['z_score']:.2f}σ<br>"
                               f"Return: {row['ann_return']*100:.1f}%<extra></extra>"),
            ))
        fig2.add_hline(y=0, line_dash="dot",
                       line_color="rgba(255,255,255,0.3)")
        fig2.update_layout(template="plotly_dark", height=360,
                           xaxis_title="Start Year",
                           yaxis_title="Z-Score (σ)",
                           margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    # Per-year breakdown table
    st.subheader("📋 Full Per-Year Breakdown")
    tbl_rows = []
    for row in sorted(consensus["per_year"], key=lambda r: r["year"]):
        tbl_rows.append({
            "Start Year":   row["year"],
            "Signal":       row["signal"],
            "Wtd Score":    round(row["wtd"], 3),
            "Z-Score":      f"{row['z_score']:.2f}σ",
            "Ann. Return":  f"{row['ann_return']*100:.2f}%",
            "Sharpe":       f"{row['sharpe']:.2f}",
            "Max Drawdown": f"{row['max_dd']*100:.2f}%",
            "Date":         "✅ Today" if row["year"] in today_cache
                            else f"📅 {display_date}",
        })
    tbl_df = pd.DataFrame(tbl_rows)

    def _style_sig(val):
        c = etf_colors.get(val, "#888")
        return f"background-color:{c}22;color:{c};font-weight:700;"

    def _style_ret(val):
        try:
            v = float(str(val).replace("%", ""))
            return ("color:#00b894;font-weight:600" if v > 0
                    else "color:#d63031;font-weight:600")
        except Exception:
            return ""

    st.dataframe(
        tbl_df.style
              .applymap(_style_sig, subset=["Signal"])
              .applymap(_style_ret, subset=["Ann. Return"])
              .set_properties(**{"text-align": "center", "font-size": "14px"})
              .hide(axis="index"),
        use_container_width=True, height=280,
    )


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.caption(f"🕐 EST: {(datetime.utcnow()-timedelta(hours=5)).strftime('%H:%M:%S')}")
    st.divider()

    start_year = st.slider("📅 Training Start Year",
                            min_value=config.START_YEAR_MIN,
                            max_value=config.START_YEAR_MAX,
                            value=2008)
    st.caption("↑ Used when triggering retraining. Does not affect live signals.")

    st.divider()
    st.markdown("### 🔒 Hardcoded Parameters")
    st.markdown(f"""
<div class="param-box">
  <div class="param-row">
    <span class="param-key">Transaction cost</span>
    <span class="param-val">{FEE_BPS} bps</span>
  </div>
  <div class="param-row">
    <span class="param-key">Max epochs</span>
    <span class="param-val">{config.MAX_EPOCHS}</span>
  </div>
  <div class="param-row">
    <span class="param-key">Trailing stop loss</span>
    <span class="param-val">−{TSL_PCT}%</span>
  </div>
  <div class="param-row">
    <span class="param-key">Z-score re-entry</span>
    <span class="param-val">≥ {Z_REENTRY} σ</span>
  </div>
  <div class="param-row">
    <span class="param-key">Wavelet</span>
    <span class="param-val">auto-optimised</span>
  </div>
  <div class="param-row">
    <span class="param-key">Lookback</span>
    <span class="param-val">auto (30/45/60d)</span>
  </div>
  <div class="param-row">
    <span class="param-key">Split</span>
    <span class="param-val">80 / 10 / 10</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🚀 Retrain")
    has_token = bool(_gh_token())

    if st.button("🧠 Retrain FI Models", use_container_width=True, type="primary"):
        if has_token:
            ok = trigger_github(start_year, workflow="train_models.yml")
            if ok:
                st.success("✅ FI retraining triggered!")
            else:
                st.error("❌ Failed. Check GITHUB_TOKEN secret.")
        else:
            st.info("Add `P2SAMAPA_GITHUB_TOKEN` to Streamlit Cloud secrets.")

    if st.button("🚀 Retrain Equity Models", use_container_width=True):
        if has_token:
            ok = trigger_github(start_year, workflow="train_equity_models.yml")
            if ok:
                st.success("✅ Equity retraining triggered!")
            else:
                st.error("❌ Failed.")
        else:
            st.info("Add `P2SAMAPA_GITHUB_TOKEN` to Streamlit Cloud secrets.")

    st.divider()
    st.markdown("### 📦 Dataset")
    etf_df_sb = load_etf_ret_fresh()
    if not etf_df_sb.empty:
        st.markdown(f"**Rows:** {len(etf_df_sb):,}")
        st.markdown(f"**Range:** {etf_df_sb.index.min().date()} → "
                    f"{etf_df_sb.index.max().date()}")
    st.markdown(f"**FI ETFs:** {', '.join(config.FI_ETFS)}")
    st.markdown(f"**Equity ETFs:** {', '.join(config.EQUITY_ETFS)}")
    st.markdown(f"**Benchmarks:** {', '.join(config.BENCHMARKS)}")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 P2-ETF-DEEPWAVE-DL")
st.caption("Option A: Wavelet-CNN-LSTM · Option B: Wavelet-Attention-CNN-LSTM · "
           "Option C: Wavelet-Parallel-Dual-Stream-CNN-LSTM")

tab_fi, tab_fi_sweep, tab_eq, tab_eq_sweep = st.tabs([
    "📊 FI Signal",
    "🔄 FI Consensus",
    "🚀 Equity Signal",
    "🔄 Equity Consensus",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FI Signal
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fi:
    pred    = load_prediction()
    evalu   = load_evaluation()
    next_td = current_signal_date()
    winner  = evalu.get("winner", "model_a")
    tbill   = pred.get("tbill_rate", 3.6)
    preds   = pred.get("predictions", {})
    tsl_s   = pred.get("tsl_status", {})

    # Read wavelet used from training summary (show in UI)
    fi_wavelet = pred.get("trained_wavelet") or config.WAVELET
    fi_lb      = evalu.get(winner, {}).get("lookback", 30)

    live_z    = preds.get(winner, {}).get("z_score", 1.5)
    two_day   = float(tsl_s.get("two_day_cumul_pct", 0.0))
    in_cash   = (two_day <= -TSL_PCT) and (live_z < Z_REENTRY)

    st.markdown(f'<div class="alert-blue">🎯 <b>FI ETFs:</b> {", ".join(config.FI_ETFS)} '
                f'&nbsp;·&nbsp; <b>Wavelet:</b> '
                f'<span class="wavelet-pill">{fi_wavelet}</span>'
                f' (auto-optimised) &nbsp;·&nbsp; <b>Lookback:</b> {fi_lb}d'
                f' &nbsp;·&nbsp; <b>T-bill:</b> {tbill:.2f}%</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="alert-yellow">🛡️ TSL: <b>−{TSL_PCT}%</b> '
                f'&nbsp;·&nbsp; Re-entry Z ≥ <b>{Z_REENTRY} σ</b> '
                f'&nbsp;·&nbsp; Fee: <b>{FEE_BPS} bps</b> '
                f'&nbsp;·&nbsp; CASH earns <b>{tbill:.2f}%</b></div>',
                unsafe_allow_html=True)

    if in_cash:
        st.markdown(f'<div class="alert-orange">🔴 <b>CASH OVERRIDE</b> — '
                    f'2-day cumul ({two_day:+.1f}%) ≤ −{TSL_PCT}% · '
                    f'Z={live_z:.2f}σ < {Z_REENTRY}σ</div>',
                    unsafe_allow_html=True)

    winner_label = {"model_a":"Option A","model_b":"Option B",
                    "model_c":"Option C"}.get(winner, "Option A")

    if in_cash:
        st.markdown(f"""<div class="cash-card">
          <div class="hero-label">⚠️ Trailing Stop Loss Triggered</div>
          <div class="hero-value">💵 {next_td} → CASH</div>
          <div class="hero-sub">Earning 3m T-bill: <b>{tbill:.2f}% p.a.</b>
          · Re-entry when Z ≥ {Z_REENTRY} σ</div>
        </div>""", unsafe_allow_html=True)
    else:
        wp     = preds.get(winner, {})
        signal = wp.get("signal", "—")
        conf   = float(wp.get("confidence", 0))
        z_v    = float(wp.get("z_score", 0))
        prov   = (f"Trained from {pred.get('trained_from_year')} · "
                  f"{fi_wavelet} wavelet (auto-selected)"
                  if pred.get("trained_from_year")
                  else "Training metadata unavailable")
        st.markdown(f"""<div class="hero-card">
          <div class="hero-label">{winner_label} · FI SIGNAL</div>
          <div class="hero-value">🎯 {next_td} → {signal}</div>
          <div class="hero-sub">Conf: {conf:.1%} · Z: {z_v:.2f}σ · {prov}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if not in_cash and preds.get(winner, {}).get("probabilities"):
        st.markdown("### 📊 Model Probabilities — FI ETFs")
        render_prob_pills(preds[winner]["probabilities"],
                          preds[winner]["signal"], ETF_COLORS_FI)

    st.markdown("---")
    st.markdown(f"### 📅 All FI Models — {next_td}")
    render_model_cards(preds, winner, in_cash)

    st.markdown("---")
    winner_full = {"model_a":"Option A · Wavelet-CNN-LSTM",
                   "model_b":"Option B · Wavelet-Attn-CNN-LSTM",
                   "model_c":"Option C · Wavelet-Dual-Stream"}.get(winner, winner)
    st.markdown(f"### 📈 Cumulative Return — FI Test Period ({winner_full})")
    fig_fi = go.Figure()
    for key, lbl, clr in [("model_a","Option A","#00bfa5"),
                           ("model_b","Option B","#7b8ff7"),
                           ("model_c","Option C","#f87171")]:
        sigs = evalu.get(key, {}).get("all_signals", [])
        if sigs:
            df_s = apply_tsl_to_audit(sigs, tbill)
            if "Date" in df_s.columns and "Net_TSL" in df_s.columns:
                df_s["Date"]  = pd.to_datetime(df_s["Date"])
                df_s["Cumul"] = (1 + df_s["Net_TSL"]).cumprod()
                fig_fi.add_trace(go.Scatter(x=df_s["Date"], y=df_s["Cumul"],
                                             name=lbl, line=dict(color=clr, width=2)))
    if fig_fi.data:
        fig_fi.update_layout(height=350, margin=dict(t=20, b=20),
                             yaxis_title="Growth of $1",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02),
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_fi, use_container_width=True)

    # Audit trail
    st.markdown("---")
    st.markdown(f"### 📋 FI Audit Trail — {winner_full}")
    audit_raw = evalu.get(winner, {}).get("audit_tail", [])
    if audit_raw:
        df_a = apply_tsl_to_audit(audit_raw, tbill)
        disp = df_a[["Date","Signal_TSL","Confidence","Z_Score","Net_TSL","Mode"]].copy()
        disp.columns = ["Date","Signal","Confidence","Z Score","Net Return","Mode"]
        disp["Date"]       = pd.to_datetime(disp["Date"], format="mixed").dt.strftime("%Y-%m-%d")
        disp["Confidence"] = disp["Confidence"].apply(
            lambda x: f"{float(x):.1%}" if isinstance(x, (int, float)) else x)
        disp["Z Score"]    = disp["Z Score"].apply(
            lambda x: f"{float(x):.2f}" if isinstance(x, (int, float)) else x)
        disp["Net Return"] = disp["Net Return"].apply(
            lambda x: f"+{x*100:.2f}%" if float(x) >= 0 else f"{x*100:.2f}%")
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FI Consensus Sweep
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fi_sweep:
    render_consensus_tab("fi", ETF_COLORS_FI, "Fixed Income")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Equity Signal
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eq:
    eq_pred  = load_prediction_equity()
    eq_evalu = load_evaluation_equity()
    next_td_eq = current_signal_date()

    if not eq_pred and not eq_evalu:
        st.warning("⏳ No equity model data yet.")
        st.info("**Steps:**\n\n"
                "1. `config.py` updated with `EQUITY_ETFS` ✅\n"
                "2. Seed data includes equity tickers ✅\n"
                "3. Trigger **🚀 Retrain Equity Models** from the sidebar\n"
                "4. Equity signals will appear here automatically")
        st.stop()

    eq_winner  = eq_evalu.get("winner", "model_a")
    eq_tbill   = eq_pred.get("tbill_rate", 3.6)
    eq_preds   = eq_pred.get("predictions", {})
    eq_tsl_s   = eq_pred.get("tsl_status", {})

    eq_wavelet = eq_pred.get("trained_wavelet") or config.WAVELET
    eq_lb      = eq_evalu.get(eq_winner, {}).get("lookback", 30)

    eq_live_z  = eq_preds.get(eq_winner, {}).get("z_score", 1.5)
    eq_two_day = float(eq_tsl_s.get("two_day_cumul_pct", 0.0))
    eq_in_cash = (eq_two_day <= -TSL_PCT) and (eq_live_z < Z_REENTRY)

    st.markdown(f'<div class="alert-indigo">🚀 <b>Equity ETFs:</b> '
                f'{", ".join(config.EQUITY_ETFS)}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="alert-blue">🎯 <b>Wavelet:</b> '
                f'<span class="wavelet-pill">{eq_wavelet}</span>'
                f' (auto-optimised) &nbsp;·&nbsp; <b>Lookback:</b> {eq_lb}d'
                f' &nbsp;·&nbsp; <b>T-bill:</b> {eq_tbill:.2f}%</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="alert-yellow">🛡️ TSL: <b>−{TSL_PCT}%</b> '
                f'&nbsp;·&nbsp; Re-entry Z ≥ <b>{Z_REENTRY} σ</b> '
                f'&nbsp;·&nbsp; Fee: <b>{FEE_BPS} bps</b></div>',
                unsafe_allow_html=True)

    if eq_in_cash:
        st.markdown(f'<div class="alert-orange">🔴 <b>EQUITY CASH OVERRIDE</b> — '
                    f'2-day cumul ({eq_two_day:+.1f}%) ≤ −{TSL_PCT}%</div>',
                    unsafe_allow_html=True)

    eq_winner_label = {"model_a":"Option A","model_b":"Option B",
                       "model_c":"Option C"}.get(eq_winner, "Option A")

    if eq_in_cash:
        st.markdown(f"""<div class="cash-card">
          <div class="hero-label">⚠️ Equity TSL Triggered</div>
          <div class="hero-value">💵 {next_td_eq} → CASH</div>
          <div class="hero-sub">Earning 3m T-bill: <b>{eq_tbill:.2f}% p.a.</b>
          · Re-entry when Z ≥ {Z_REENTRY} σ</div>
        </div>""", unsafe_allow_html=True)
    else:
        eq_wp     = eq_preds.get(eq_winner, {})
        eq_signal = eq_wp.get("signal", "—")
        eq_conf   = float(eq_wp.get("confidence", 0))
        eq_z      = float(eq_wp.get("z_score", 0))
        eq_prov   = (f"Trained from {eq_pred.get('trained_from_year')} · "
                     f"{eq_wavelet} wavelet (auto-selected)"
                     if eq_pred.get("trained_from_year")
                     else "Training metadata unavailable")
        st.markdown(f"""<div class="hero-card-eq">
          <div class="hero-label">{eq_winner_label} · EQUITY SIGNAL</div>
          <div class="hero-value">🎯 {next_td_eq} → {eq_signal}</div>
          <div class="hero-sub">Conf: {eq_conf:.1%} · Z: {eq_z:.2f}σ · {eq_prov}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    eq_wp_probs = eq_preds.get(eq_winner, {}).get("probabilities", {})
    if eq_wp_probs and not eq_in_cash:
        st.markdown("### 📊 Model Probabilities — Equity ETFs")
        render_prob_pills(eq_wp_probs,
                          eq_preds[eq_winner].get("signal", "—"),
                          ETF_COLORS_EQ)

    st.markdown("---")
    st.markdown(f"### 📅 All Equity Models — {next_td_eq}")
    render_model_cards(eq_preds, eq_winner, eq_in_cash,
                       colors=("#2563eb", "#7c3aed", "#0891b2"))

    st.markdown("---")
    eq_winner_full = {"model_a":"Option A · Wavelet-CNN-LSTM",
                      "model_b":"Option B · Wavelet-Attn-CNN-LSTM",
                      "model_c":"Option C · Wavelet-Dual-Stream"}.get(eq_winner, eq_winner)
    eq_w_met = eq_evalu.get(eq_winner, {}).get("metrics", {})
    if eq_w_met:
        st.markdown(f"### 📊 {eq_winner_full} — Performance")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("📈 Ann. Return",   f"{eq_w_met.get('ann_return',0):.2f}%",
                  delta=f"vs SPY: {eq_w_met.get('vs_spy',0):+.2f}%")
        m2.metric("📊 Sharpe",        f"{eq_w_met.get('sharpe',0):.2f}")
        m3.metric("🎯 Hit Ratio 15d", f"{eq_w_met.get('hit_ratio_15d',0):.0%}")
        m4.metric("📉 Max Drawdown",  f"{eq_w_met.get('max_drawdown',0):.2f}%")
        m5.metric("⚠️ Max Daily DD",  f"{eq_w_met.get('max_daily_dd',0):.2f}%")

    st.markdown("---")
    st.markdown(f"### 📈 Cumulative Return — Equity Test Period")
    fig_eq = go.Figure()
    for key, lbl, clr in [("model_a","Option A","#2563eb"),
                           ("model_b","Option B","#7c3aed"),
                           ("model_c","Option C","#0891b2")]:
        sigs = eq_evalu.get(key, {}).get("all_signals", [])
        if sigs:
            df_s = apply_tsl_to_audit(sigs, eq_tbill)
            if "Date" in df_s.columns and "Net_TSL" in df_s.columns:
                df_s["Date"]  = pd.to_datetime(df_s["Date"])
                df_s["Cumul"] = (1 + df_s["Net_TSL"]).cumprod()
                fig_eq.add_trace(go.Scatter(x=df_s["Date"], y=df_s["Cumul"],
                                             name=lbl, line=dict(color=clr, width=2)))
    if fig_eq.data:
        fig_eq.update_layout(height=350, margin=dict(t=20, b=20),
                             yaxis_title="Growth of $1",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02),
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("Equity cumulative chart will appear after the first evaluation run.")

    st.markdown("---")
    st.markdown(f"### 📋 Equity Audit Trail — {eq_winner_full}")
    eq_audit = eq_evalu.get(eq_winner, {}).get("audit_tail", [])
    if eq_audit:
        df_ea = apply_tsl_to_audit(eq_audit, eq_tbill)
        disp  = df_ea[["Date","Signal_TSL","Confidence","Z_Score","Net_TSL","Mode"]].copy()
        disp.columns = ["Date","Signal","Confidence","Z Score","Net Return","Mode"]
        disp["Date"]       = pd.to_datetime(disp["Date"], format="mixed").dt.strftime("%Y-%m-%d")
        disp["Confidence"] = disp["Confidence"].apply(
            lambda x: f"{float(x):.1%}" if isinstance(x, (int, float)) else x)
        disp["Z Score"]    = disp["Z Score"].apply(
            lambda x: f"{float(x):.2f}" if isinstance(x, (int, float)) else x)
        disp["Net Return"] = disp["Net Return"].apply(
            lambda x: f"+{x*100:.2f}%" if float(x) >= 0 else f"{x*100:.2f}%")
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("Equity audit trail will appear after the first training + evaluation run.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Equity Consensus Sweep
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eq_sweep:
    render_consensus_tab("eq", ETF_COLORS_EQ, "Equity")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"P2-ETF-DEEPWAVE-DL · HF Dataset: {config.HF_DATASET_REPO} · "
    f"TSL: {TSL_PCT}% · Z-reentry: {Z_REENTRY}σ · Fee: {FEE_BPS}bps · "
    f"Last refresh: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
)
