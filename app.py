# app.py — P2-ETF-DEEPWAVE-DL
# Architecture:
#   - Scans HF Dataset results/ folder for fi_{year}_{date}.json and eq_{year}_{date}.json
#   - Single-year tab: slider over available years → show that year's signal + metrics
#   - Consensus tab: all available years → weighted consensus auto-computed
#   - No separate sweep/ folder needed

import json
import os
import re
import requests
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ─── Streamlit Cloud secrets ──────────────────────────────────────────────────
def _bootstrap_secrets():
    try:
        for key in ["HF_TOKEN","FRED_API_KEY","HF_DATASET_REPO",
                    "P2SAMAPA_GITHUB_TOKEN","GITHUB_TOKEN"]:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = st.secrets[key]
    except Exception:
        pass

_bootstrap_secrets()
import config   # noqa: E402

st.set_page_config(page_title="P2-ETF-DEEPWAVE-DL", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  [data-testid="stSidebar"] { min-width:300px; max-width:300px; }
  .hero-card    { background:linear-gradient(135deg,#00bfa5,#00897b);
                  border-radius:14px;padding:28px;text-align:center;color:white;margin:12px 0; }
  .hero-card-eq { background:linear-gradient(135deg,#2563eb,#1d4ed8);
                  border-radius:14px;padding:28px;text-align:center;color:white;margin:12px 0; }
  .cash-card    { background:linear-gradient(135deg,#e65100,#bf360c);
                  border-radius:14px;padding:28px;text-align:center;color:white;margin:12px 0; }
  .hero-label { font-size:11px;letter-spacing:2px;text-transform:uppercase;opacity:.8;margin-bottom:6px; }
  .hero-value { font-size:38px;font-weight:700; }
  .hero-sub   { font-size:13px;opacity:.85;margin-top:8px; }
  .model-card-a { background:#0d1117;border-radius:12px;padding:20px;text-align:center;
                  color:white;border:1px solid #1a2a1a; }
  .model-card-b { background:#0d1117;border-radius:12px;padding:20px;text-align:center;
                  color:white;border:2px solid #7b8ff7; }
  .model-card-c { background:#0d1117;border-radius:12px;padding:20px;text-align:center;
                  color:white;border:1px solid #2a1a1a; }
  .param-box  { background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                padding:12px 16px;font-size:13px;color:#374151;margin-bottom:8px; }
  .param-row  { display:flex;justify-content:space-between;margin-bottom:3px; }
  .param-key  { color:#6b7280;font-weight:500; }
  .param-val  { color:#111827;font-weight:700;font-family:monospace; }
  .wavelet-pill { display:inline-block;background:#dbeafe;border:1px solid #93c5fd;
                  border-radius:20px;padding:2px 10px;font-size:12px;
                  font-weight:700;color:#1d4ed8; }
  .alert-blue   { background:#e8f4fd;border:1px solid #bbdefb;color:#1565c0;
                  padding:10px 14px;border-radius:8px;margin-bottom:8px;font-size:13px; }
  .alert-yellow { background:#fffde7;border:1px solid #fff9c4;color:#f57f17;
                  padding:10px 14px;border-radius:8px;margin-bottom:8px;font-size:13px; }
  .alert-orange { background:#fff3e0;border:1px solid #ffcc80;color:#e65100;
                  padding:10px 14px;border-radius:8px;margin-bottom:8px;font-size:13px; }
  #MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

TSL_PCT   = config.DEFAULT_TSL_PCT
Z_REENTRY = config.DEFAULT_Z_REENTRY
FEE_BPS   = config.FEE_BPS

ETF_COLORS_FI = {"TLT":"#4e79a7","VCIT":"#f28e2b","LQD":"#59a14f",
                 "HYG":"#e15759","VNQ":"#76b7b2","SLV":"#edc948","GLD":"#b07aa1","CASH":"#aaaaaa"}
ETF_COLORS_EQ = {"QQQ":"#7c3aed","XLK":"#0891b2","XLF":"#059669","XLE":"#d97706",
                 "XLV":"#dc2626","XLI":"#9333ea","XLY":"#db2777","XLP":"#16a34a",
                 "XLU":"#ca8a04","XME":"#64748b","GDX":"#b45309","IWM":"#0f766e","CASH":"#aaaaaa"}

US_HOLIDAYS = {
    date(2025,1,1),date(2025,1,20),date(2025,2,17),date(2025,4,18),
    date(2025,5,26),date(2025,6,19),date(2025,7,4),date(2025,9,1),
    date(2025,11,27),date(2025,12,25),
    date(2026,1,1),date(2026,1,19),date(2026,2,16),date(2026,4,3),
    date(2026,5,25),date(2026,6,19),date(2026,7,3),date(2026,9,7),
    date(2026,11,26),date(2026,12,25),
}

def is_trading_day(d): return d.weekday() < 5 and d not in US_HOLIDAYS

def current_signal_date():
    now_est = datetime.utcnow() - timedelta(hours=5)
    today, hour = now_est.date(), now_est.hour
    if is_trading_day(today) and hour < 16:
        return today
    d = today + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d

def _hf_token():
    try: return st.secrets.get("HF_TOKEN","") or config.HF_TOKEN
    except: return config.HF_TOKEN

def _gh_token():
    try:
        return (st.secrets.get("P2SAMAPA_GITHUB_TOKEN","") or
                st.secrets.get("GITHUB_TOKEN","") or config.GITHUB_TOKEN)
    except: return config.GITHUB_TOKEN

def _hf_repo():
    try: return st.secrets.get("HF_DATASET_REPO","") or config.HF_DATASET_REPO
    except: return config.HF_DATASET_REPO

def trigger_github(start_year, workflow="train_models.yml"):
    token = _gh_token()
    if not token: return False
    url = (f"https://api.github.com/repos/{config.GITHUB_REPO}"
           f"/actions/workflows/{workflow}/dispatches")
    r = requests.post(url, json={"ref":"main","inputs":{"model":"all","start_year":str(start_year)}},
                      headers={"Authorization":f"token {token}",
                               "Accept":"application/vnd.github+json"})
    return r.status_code == 204


# ─── Scan HF results/ folder ─────────────────────────────────────────────────

@st.cache_data(ttl=300)
def scan_available_years(prefix: str) -> dict:
    """
    Scan HF Dataset results/ for files matching {prefix}_{year}_{date}.json.
    Returns {year: (date_str, hf_filename)} keeping only the most recent date per year.
    prefix: 'fi' or 'eq'
    """
    pattern = re.compile(rf"^results/{prefix}_(\d{{4}})_(\d{{8}})\.json$")
    best = {}
    try:
        from huggingface_hub import HfApi
        api   = HfApi()
        files = list(api.list_repo_files(repo_id=_hf_repo(), repo_type="dataset",
                                          token=_hf_token() or None))
        for f in files:
            m = pattern.match(f)
            if m:
                year, date_str = int(m.group(1)), m.group(2)
                if year not in best or date_str > best[year][0]:
                    best[year] = (date_str, f)
    except Exception as e:
        st.sidebar.caption(f"⚠️ Scan error: {e}")
    return best   # {year: (date_str, hf_path)}


@st.cache_data(ttl=300)
def load_year_result(hf_path: str) -> dict:
    """Download and parse one result JSON file from HF."""
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(repo_id=_hf_repo(), filename=hf_path,
                                repo_type="dataset", token=_hf_token() or None,
                                force_download=True)
        with open(local) as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load {hf_path}: {e}")
        return {}


# ─── Consensus calculation ────────────────────────────────────────────────────

def compute_consensus(year_results: dict) -> dict:
    """
    year_results: {year: result_dict}
    Each result_dict must have consensus_signal, consensus_ann_return,
    consensus_z_score, consensus_sharpe, consensus_max_dd.
    """
    rows = []
    for yr, r in year_results.items():
        if not r or "consensus_signal" not in r:
            continue
        rows.append({
            "year":       yr,
            "signal":     r["consensus_signal"],
            "ann_return": r.get("consensus_ann_return", 0.0),
            "z_score":    r.get("consensus_z_score", 0.0),
            "sharpe":     r.get("consensus_sharpe", 0.0),
            "max_dd":     r.get("consensus_max_dd", 0.0),
        })
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
        for k, v in [("years",row["year"]),("scores",row["wtd"]),
                     ("returns",row["ann_return"]),("zs",row["z_score"]),
                     ("sharpes",row["sharpe"]),("dds",row["max_dd"])]:
            etf_agg[e][k].append(v)

    total = sum(sum(v["scores"]) for v in etf_agg.values()) + 1e-9
    summary = {}
    for e, v in etf_agg.items():
        cs = sum(v["scores"])
        summary[e] = dict(cum_score=round(cs,4), score_share=round(cs/total,3),
                          n_years=len(v["years"]), years=v["years"],
                          avg_return=round(float(np.mean(v["returns"])),4),
                          avg_z=round(float(np.mean(v["zs"])),3),
                          avg_sharpe=round(float(np.mean(v["sharpes"])),3),
                          avg_max_dd=round(float(np.mean(v["dds"])),4))

    winner = max(summary, key=lambda e: summary[e]["cum_score"])
    return {"winner":winner, "etf_summary":summary,
            "per_year":df.to_dict("records"), "n_years":len(rows)}


# ─── TSL replay ───────────────────────────────────────────────────────────────

def apply_tsl(audit, tbill=3.6):
    df = pd.DataFrame(audit)
    if df.empty: return df
    in_cash, tsl_days, prev, prev2 = False, 0, 0.0, 0.0
    modes, signals, net_rets = [], [], []
    for _, row in df.iterrows():
        z       = float(row.get("Z_Score", 1.5))
        two_day = (prev + prev2) * 100
        if not in_cash and two_day <= -TSL_PCT: in_cash, tsl_days = True, 0
        if in_cash and tsl_days >= 1 and z >= Z_REENTRY: in_cash = False
        if in_cash: tsl_days += 1
        if in_cash:
            modes.append("💵 CASH"); signals.append("CASH")
            net_rets.append(round(tbill/100/252, 6))
        else:
            modes.append("📈 ETF"); signals.append(row.get("Signal","—"))
            net_rets.append(float(row.get("Net_Return",0.0)))
        prev2, prev = prev, net_rets[-1]
    df = df.copy()
    df["Mode"] = modes; df["Signal_TSL"] = signals; df["Net_TSL"] = net_rets
    return df


# ─── Shared renderers ─────────────────────────────────────────────────────────

def render_prob_pills(probs, top_etf, etf_colors):
    if not probs: return
    cols = st.columns(min(len(probs), 7))
    for i, (etf, p) in enumerate(sorted(probs.items(), key=lambda x: -x[1])):
        if i >= len(cols): break
        with cols[i]:
            is_top = etf == top_etf
            color  = etf_colors.get(etf,"#555") if is_top else "#555"
            bg     = "#f0f9ff" if is_top else "#f7f8fa"
            border = color if is_top else "#ddd"
            st.markdown(
                f'<div style="border:1.5px solid {border};border-radius:18px;'
                f'padding:5px 8px;text-align:center;background:{bg};color:{color};'
                f'font-weight:{"700" if is_top else "500"};font-size:12px;">'
                f'{"★ " if is_top else ""}{etf}<br>{p:.3f}</div>',
                unsafe_allow_html=True)


def render_model_cards(result, winner, in_cash, colors=("#00bfa5","#7b8ff7","#f87171")):
    col_a, col_b, col_c = st.columns(3)
    for key, label, color, col, css in [
        ("model_a","OPTION A",colors[0],col_a,"model-card-a"),
        ("model_b","OPTION B",colors[1],col_b,"model-card-b"),
        ("model_c","OPTION C",colors[2],col_c,"model-card-c"),
    ]:
        with col:
            p    = result.get(key, {})
            sig  = "CASH" if in_cash else p.get("latest_signal","—")
            conf = float(p.get("latest_confidence",0))
            z_v  = float(p.get("latest_z_score",0))
            lb   = p.get("lookback","?")
            wav  = p.get("wavelet","?")
            w_tag = " ★" if key == winner else ""
            st.markdown(f"""<div class="{css}">
              <div style="font-size:11px;letter-spacing:2px;font-weight:600;
                          color:{color};margin-bottom:10px;">{label}{w_tag}</div>
              <div style="font-size:26px;font-weight:700;margin-bottom:6px;">{sig}</div>
              <div style="font-size:12px;color:#aaa;">
                Conf: <span style="color:{color};font-weight:600;">
                  {"CASH" if in_cash else f"{conf:.1%}"}
                </span>
              </div>
              <div style="font-size:11px;color:#666;margin-top:4px;">
                Z={z_v:.2f}σ &nbsp;·&nbsp; lb={lb}d &nbsp;·&nbsp;
                <span class="wavelet-pill">{wav}</span>
              </div>
            </div>""", unsafe_allow_html=True)


def render_consensus_section(year_results, etf_colors, universe_label):
    if not year_results:
        st.info("No results available yet. Trigger training from the sidebar.")
        return

    consensus = compute_consensus(year_results)
    if not consensus:
        st.warning("Could not compute consensus — check that result files contain consensus fields.")
        return

    winner_etf = consensus["winner"]
    w_info     = consensus["etf_summary"][winner_etf]
    win_color  = etf_colors.get(winner_etf, "#0066cc")
    score_pct  = w_info["score_share"] * 100
    split_sig  = w_info["score_share"] < 0.40
    sig_label  = "⚠️ Split Signal" if split_sig else "✅ Clear Consensus"
    n_years    = consensus["n_years"]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                border:2px solid {win_color};border-radius:16px;
                padding:28px;text-align:center;margin:12px 0;">
      <div style="font-size:11px;letter-spacing:3px;color:#aaa;margin-bottom:8px;">
        WEIGHTED CONSENSUS · {universe_label.upper()} · {n_years} YEARS
      </div>
      <div style="font-size:68px;font-weight:900;color:{win_color};
                  text-shadow:0 0 30px {win_color}88;">{winner_etf}</div>
      <div style="font-size:14px;color:#ccc;margin-top:8px;">
        {sig_label} · Score share {score_pct:.0f}% · {w_info['n_years']}/{n_years} years
      </div>
      <div style="display:flex;justify-content:center;gap:28px;margin-top:16px;flex-wrap:wrap;">
        <div><div style="font-size:11px;color:#aaa;">Avg Return</div>
          <div style="font-size:20px;font-weight:700;
                      color:{'#00b894' if w_info['avg_return']>0 else '#d63031'};">
            {w_info['avg_return']*100:.1f}%</div></div>
        <div><div style="font-size:11px;color:#aaa;">Avg Z</div>
          <div style="font-size:20px;font-weight:700;color:#74b9ff;">
            {w_info['avg_z']:.2f}σ</div></div>
        <div><div style="font-size:11px;color:#aaa;">Avg Sharpe</div>
          <div style="font-size:20px;font-weight:700;color:#a29bfe;">
            {w_info['avg_sharpe']:.2f}</div></div>
        <div><div style="font-size:11px;color:#aaa;">Avg MaxDD</div>
          <div style="font-size:20px;font-weight:700;color:#fd79a8;">
            {w_info['avg_max_dd']*100:.1f}%</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Runner-ups
    others = sorted([(e,v) for e,v in consensus["etf_summary"].items()
                     if e != winner_etf], key=lambda x: -x[1]["cum_score"])
    parts  = [f'<span style="color:{etf_colors.get(e,"#888")};font-weight:600;">{e}</span>'
               f' <span style="color:#aaa;">({v["cum_score"]:.2f})</span>'
               for e,v in others]
    st.markdown('<div style="text-align:center;font-size:13px;margin-bottom:10px;">'
                'Also ranked: ' + ' &nbsp;|&nbsp; '.join(parts) + '</div>',
                unsafe_allow_html=True)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Weighted Score per ETF**")
        es   = consensus["etf_summary"]
        etfs = sorted(es.keys(), key=lambda e: -es[e]["cum_score"])
        fig  = go.Figure(go.Bar(
            x=etfs, y=[es[e]["cum_score"] for e in etfs],
            marker_color=[etf_colors.get(e,"#888") for e in etfs],
            text=[f"{es[e]['n_years']}yr · {es[e]['score_share']*100:.0f}%" for e in etfs],
            textposition="outside"))
        fig.update_layout(template="plotly_dark", height=340,
                          yaxis_title="Cumulative Score", showlegend=False,
                          margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Z-Score by Start Year**")
        fig2 = go.Figure()
        for row in consensus["per_year"]:
            etf = row["signal"]
            clr = etf_colors.get(etf,"#888")
            fig2.add_trace(go.Scatter(
                x=[row["year"]], y=[row["z_score"]], mode="markers+text",
                marker=dict(size=16, color=clr, line=dict(color="white",width=1)),
                text=[etf], textposition="top center", name=etf, showlegend=False,
                hovertemplate=(f"<b>{etf}</b><br>Year: {row['year']}<br>"
                               f"Z: {row['z_score']:.2f}σ<br>"
                               f"Return: {row['ann_return']*100:.1f}%<extra></extra>")))
        fig2.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig2.update_layout(template="plotly_dark", height=340,
                           xaxis_title="Start Year", yaxis_title="Z-Score (σ)",
                           margin=dict(t=20,b=20))
        st.plotly_chart(fig2, use_container_width=True)

    # Per-year table
    st.subheader("📋 Per-Year Breakdown")
    tbl = []
    for row in sorted(consensus["per_year"], key=lambda r: r["year"]):
        tbl.append({"Start Year":row["year"], "Signal":row["signal"],
                    "Wtd Score":round(row["wtd"],3),
                    "Z-Score":f"{row['z_score']:.2f}σ",
                    "Ann. Return":f"{row['ann_return']*100:.2f}%",
                    "Sharpe":f"{row['sharpe']:.2f}",
                    "Max Drawdown":f"{row['max_dd']*100:.2f}%"})
    tbl_df = pd.DataFrame(tbl)

    def _style_sig(val):
        c = etf_colors.get(val,"#888")
        return f"background-color:{c}22;color:{c};font-weight:700;"
    def _style_ret(val):
        try:
            v = float(str(val).replace("%",""))
            return "color:#00b894;font-weight:600" if v>0 else "color:#d63031;font-weight:600"
        except: return ""

    st.dataframe(tbl_df.style.applymap(_style_sig, subset=["Signal"])
                            .applymap(_style_ret, subset=["Ann. Return"])
                            .hide(axis="index"),
                 use_container_width=True, height=280)


def render_single_year_tab(year_results, selected_year, universe, etf_colors):
    if not year_results:
        st.info("No results available yet. Trigger training from the sidebar.")
        return

    result = year_results.get(selected_year, {})
    if not result:
        st.warning(f"No data for {selected_year}. Trigger training for this year.")
        return

    winner   = result.get("winner","model_a")
    tbill    = 3.6
    in_cash  = False   # TSL state shown from latest prediction file
    next_td  = current_signal_date()

    w        = result.get(winner, {})
    signal   = w.get("latest_signal","—")
    conf     = float(w.get("latest_confidence",0))
    z_val    = float(w.get("latest_z_score",0))
    wavelet  = w.get("wavelet", config.WAVELET)
    lb       = w.get("lookback", config.DEFAULT_LOOKBACK)
    metrics  = w.get("metrics",{})
    probs    = w.get("latest_probs",{})

    # Check TSL from live prediction file if available
    try:
        pred_file = ("latest_prediction.json" if universe == "fi"
                     else "latest_prediction_equity.json")
        if os.path.exists(pred_file):
            with open(pred_file) as f:
                pred = json.load(f)
            tsl_s   = pred.get("tsl_status",{})
            tbill   = pred.get("tbill_rate", 3.6)
            in_cash = tsl_s.get("in_cash", False)
    except Exception:
        pass

    uni_label = "FI" if universe == "fi" else "Equity"
    card_cls  = "hero-card" if universe == "fi" else "hero-card-eq"

    st.markdown(f'<div class="alert-blue">🎯 <b>{uni_label} — Start Year: {selected_year}</b>'
                f' &nbsp;·&nbsp; <b>Wavelet:</b> <span class="wavelet-pill">{wavelet}</span>'
                f' (auto) &nbsp;·&nbsp; <b>Lookback:</b> {lb}d'
                f' &nbsp;·&nbsp; <b>T-bill:</b> {tbill:.2f}%</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="alert-yellow">🛡️ TSL: <b>−{TSL_PCT}%</b>'
                f' &nbsp;·&nbsp; Re-entry Z ≥ <b>{Z_REENTRY}σ</b>'
                f' &nbsp;·&nbsp; Fee: <b>{FEE_BPS}bps</b></div>',
                unsafe_allow_html=True)

    if in_cash:
        st.markdown(f"""<div class="cash-card">
          <div class="hero-label">⚠️ TSL Triggered</div>
          <div class="hero-value">💵 {next_td} → CASH</div>
          <div class="hero-sub">Earning 3m T-bill: {tbill:.2f}% p.a.</div>
        </div>""", unsafe_allow_html=True)
    else:
        winner_label = {"model_a":"Option A","model_b":"Option B",
                        "model_c":"Option C"}.get(winner,"Option A")
        st.markdown(f"""<div class="{card_cls}">
          <div class="hero-label">{winner_label} · {uni_label} · trained from {selected_year}</div>
          <div class="hero-value">🎯 {next_td} → {signal}</div>
          <div class="hero-sub">Conf: {conf:.1%} · Z: {z_val:.2f}σ · {wavelet} wavelet</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if probs and not in_cash:
        st.markdown("### 📊 Model Probabilities")
        render_prob_pills(probs, signal, etf_colors)

    st.markdown("---")
    st.markdown(f"### 📅 All Models — {next_td} (trained from {selected_year})")
    render_model_cards(result, winner, in_cash,
                       colors=("#00bfa5","#7b8ff7","#f87171") if universe=="fi"
                              else ("#2563eb","#7c3aed","#0891b2"))

    st.markdown("---")
    if metrics:
        st.markdown(f"### 📊 {uni_label} Performance — trained from {selected_year}")
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("📈 Ann. Return",   f"{metrics.get('ann_return',0):.2f}%",
                  delta=f"vs SPY: {metrics.get('vs_spy',0):+.2f}%")
        m2.metric("📊 Sharpe",        f"{metrics.get('sharpe',0):.2f}")
        m3.metric("🎯 Hit Ratio 15d", f"{metrics.get('hit_ratio_15d',0):.0%}")
        m4.metric("📉 Max Drawdown",  f"{metrics.get('max_drawdown',0):.2f}%")
        m5.metric("⚠️ Max Daily DD",  f"{metrics.get('max_daily_dd',0):.2f}%")

    # Cumulative return chart
    st.markdown("---")
    st.markdown(f"### 📈 Cumulative Return (trained from {selected_year})")
    fig = go.Figure()
    clr_map = ({"model_a":"#00bfa5","model_b":"#7b8ff7","model_c":"#f87171"}
               if universe=="fi"
               else {"model_a":"#2563eb","model_b":"#7c3aed","model_c":"#0891b2"})
    for key, lbl in [("model_a","Option A"),("model_b","Option B"),("model_c","Option C")]:
        sigs = result.get(key,{}).get("all_signals",[])
        if sigs:
            df_s = apply_tsl(sigs, tbill)
            if "Date" in df_s.columns and "Net_TSL" in df_s.columns:
                df_s["Date"]  = pd.to_datetime(df_s["Date"])
                df_s["Cumul"] = (1 + df_s["Net_TSL"]).cumprod()
                fig.add_trace(go.Scatter(x=df_s["Date"], y=df_s["Cumul"],
                                         name=lbl, line=dict(color=clr_map[key], width=2)))
    if fig.data:
        fig.update_layout(height=340, margin=dict(t=20,b=20),
                          yaxis_title="Growth of $1",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # Audit trail
    st.markdown("---")
    st.markdown(f"### 📋 Audit Trail (last 30 days)")
    audit = result.get(winner,{}).get("audit_tail",[])
    if audit:
        df_a = apply_tsl(audit, tbill)
        disp = df_a[["Date","Signal_TSL","Confidence","Z_Score","Net_TSL","Mode"]].copy()
        disp.columns = ["Date","Signal","Confidence","Z Score","Net Return","Mode"]
        disp["Date"]       = pd.to_datetime(disp["Date"], format="mixed").dt.strftime("%Y-%m-%d")
        disp["Confidence"] = disp["Confidence"].apply(lambda x: f"{float(x):.1%}" if isinstance(x,(int,float)) else x)
        disp["Z Score"]    = disp["Z Score"].apply(lambda x: f"{float(x):.2f}" if isinstance(x,(int,float)) else x)
        disp["Net Return"] = disp["Net Return"].apply(lambda x: f"+{x*100:.2f}%" if float(x)>=0 else f"{x*100:.2f}%")
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.caption(f"🕐 EST: {(datetime.utcnow()-timedelta(hours=5)).strftime('%H:%M:%S')}")
    st.divider()
    st.markdown("### 🔒 Hardcoded Parameters")
    st.markdown(f"""<div class="param-box">
      <div class="param-row"><span class="param-key">Transaction cost</span>
        <span class="param-val">{FEE_BPS} bps</span></div>
      <div class="param-row"><span class="param-key">Max epochs</span>
        <span class="param-val">{config.MAX_EPOCHS}</span></div>
      <div class="param-row"><span class="param-key">Trailing stop loss</span>
        <span class="param-val">−{TSL_PCT}%</span></div>
      <div class="param-row"><span class="param-key">Z-score re-entry</span>
        <span class="param-val">≥ {Z_REENTRY} σ</span></div>
      <div class="param-row"><span class="param-key">Wavelet</span>
        <span class="param-val">auto-optimised</span></div>
      <div class="param-row"><span class="param-key">Lookback</span>
        <span class="param-val">auto (30/45/60d)</span></div>
      <div class="param-row"><span class="param-key">Split</span>
        <span class="param-val">80 / 10 / 10</span></div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🚀 Retrain")
    retrain_year = st.selectbox("Start year to retrain",
                                options=["All years"] +
                                        list(range(config.START_YEAR_MIN, config.START_YEAR_MAX+1)),
                                index=0)
    has_token = bool(_gh_token())

    if st.button("🧠 Retrain FI", use_container_width=True, type="primary"):
        year_arg = "" if retrain_year == "All years" else str(retrain_year)
        if has_token:
            ok = trigger_github(year_arg or 2008, "train_models.yml")
            if ok: st.success("✅ FI retraining triggered!")
            else:  st.error("❌ Failed. Check GITHUB_TOKEN.")
        else:
            st.info("Add `P2SAMAPA_GITHUB_TOKEN` to Streamlit Cloud secrets.")

    if st.button("🚀 Retrain Equity", use_container_width=True):
        year_arg = "" if retrain_year == "All years" else str(retrain_year)
        if has_token:
            ok = trigger_github(year_arg or 2008, "train_equity_models.yml")
            if ok: st.success("✅ Equity retraining triggered!")
            else:  st.error("❌ Failed.")
        else:
            st.info("Add `P2SAMAPA_GITHUB_TOKEN` to Streamlit Cloud secrets.")

    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear(); st.rerun()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 P2-ETF-DEEPWAVE-DL")
st.caption("Option A: Wavelet-CNN-LSTM · Option B: Wavelet-Attention-CNN-LSTM · "
           "Option C: Wavelet-Parallel-Dual-Stream-CNN-LSTM")

# Scan available years for both universes
with st.spinner("Scanning available results..."):
    fi_years = scan_available_years("fi")   # {year: (date_str, hf_path)}
    eq_years = scan_available_years("eq")

tab_fi, tab_fi_cons, tab_eq, tab_eq_cons = st.tabs([
    "📊 FI Signal",
    "🔄 FI Consensus",
    "🚀 Equity Signal",
    "🔄 Equity Consensus",
])

# ── FI Signal ─────────────────────────────────────────────────────────────────
with tab_fi:
    if not fi_years:
        st.warning("No FI results found on HF Dataset. Trigger training from the sidebar.")
    else:
        available_fi = sorted(fi_years.keys())
        default_idx  = len(available_fi) - 1   # most recent year as default
        selected_fi  = st.select_slider(
            "📅 Training Start Year",
            options=available_fi,
            value=available_fi[default_idx],
            key="fi_year_slider",
        )
        st.caption(f"Showing results from model trained starting **{selected_fi}** "
                   f"· last updated {fi_years[selected_fi][0]}")

        result_fi = load_year_result(fi_years[selected_fi][1])
        render_single_year_tab(
            {selected_fi: result_fi}, selected_fi, "fi", ETF_COLORS_FI)

# ── FI Consensus ──────────────────────────────────────────────────────────────
with tab_fi_cons:
    st.subheader("🔄 FI — Multi-Year Consensus")
    st.markdown(f"Aggregates signals from **all {len(fi_years)} trained start years** "
                f"into a weighted consensus.  \n"
                "**Score:** 40% Return · 20% Z · 20% Sharpe · 20% (–MaxDD)")
    if fi_years:
        st.caption(f"Available years: {', '.join(str(y) for y in sorted(fi_years.keys()))}")
        with st.spinner("Loading all FI year results..."):
            all_fi = {yr: load_year_result(path) for yr, (_, path) in fi_years.items()}
        render_consensus_section(all_fi, ETF_COLORS_FI, "Fixed Income")
    else:
        st.info("No FI results found. Trigger training from the sidebar.")

# ── Equity Signal ─────────────────────────────────────────────────────────────
with tab_eq:
    if not eq_years:
        st.warning("No Equity results found on HF Dataset. Trigger training from the sidebar.")
    else:
        available_eq = sorted(eq_years.keys())
        selected_eq  = st.select_slider(
            "📅 Training Start Year (Equity)",
            options=available_eq,
            value=available_eq[-1],
            key="eq_year_slider",
        )
        st.caption(f"Showing results from equity model trained starting **{selected_eq}** "
                   f"· last updated {eq_years[selected_eq][0]}")

        result_eq = load_year_result(eq_years[selected_eq][1])
        render_single_year_tab(
            {selected_eq: result_eq}, selected_eq, "eq", ETF_COLORS_EQ)

# ── Equity Consensus ──────────────────────────────────────────────────────────
with tab_eq_cons:
    st.subheader("🔄 Equity — Multi-Year Consensus")
    st.markdown(f"Aggregates signals from **all {len(eq_years)} trained start years** "
                "into a weighted consensus.  \n"
                "**Score:** 40% Return · 20% Z · 20% Sharpe · 20% (–MaxDD)")
    if eq_years:
        st.caption(f"Available years: {', '.join(str(y) for y in sorted(eq_years.keys()))}")
        with st.spinner("Loading all Equity year results..."):
            all_eq = {yr: load_year_result(path) for yr, (_, path) in eq_years.items()}
        render_consensus_section(all_eq, ETF_COLORS_EQ, "Equity")
    else:
        st.info("No Equity results found. Trigger equity training from the sidebar.")

st.markdown("---")
st.caption(f"P2-ETF-DEEPWAVE-DL · HF: {config.HF_DATASET_REPO} · "
           f"TSL: {TSL_PCT}% · Z-reentry: {Z_REENTRY}σ · Fee: {FEE_BPS}bps · "
           f"Last refresh: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
