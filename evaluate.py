# evaluate.py
# All risk params read from config.py — no CLI args for tsl/z/fee.
# Wavelet read from training_summary.json (best_wavelet auto-selected during training).

import json
import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime

import config
from data_download import load_local
from preprocess import run_preprocessing, build_features, apply_scaler, load_scaler, \
                       normalize_etf_columns, flatten_columns
import model_a, model_b, model_c

TSL_PCT   = config.DEFAULT_TSL_PCT    # 12
Z_REENTRY = config.DEFAULT_Z_REENTRY  # 0.9
FEE_BPS   = config.FEE_BPS            # 12


# ─── HF download helper ───────────────────────────────────────────────────────

def download_from_hf_if_needed():
    try:
        from huggingface_hub import HfApi, hf_hub_download
        token = config.HF_TOKEN or None
        os.makedirs(config.DATA_DIR, exist_ok=True)
        for f in ["etf_price","etf_ret","etf_vol",
                  "bench_price","bench_ret","bench_vol","macro"]:
            local = os.path.join(config.DATA_DIR, f"{f}.parquet")
            if not os.path.exists(local):
                try:
                    dl = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                                         filename=f"data/{f}.parquet",
                                         repo_type="dataset", token=token)
                    shutil.copy(dl, local)
                    print(f"  Downloaded data/{f}.parquet")
                except Exception as e:
                    print(f"  Warning data/{f}: {e}")
        os.makedirs(config.MODELS_DIR, exist_ok=True)
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
                        shutil.copy(dl, local)
                        print(f"  Downloaded {f}")
                    except Exception as e:
                        print(f"  Warning {f}: {e}")
    except Exception as e:
        print(f"  WARNING: HF download failed: {e}")


# ─── Read best wavelet + lookback from training summary ───────────────────────

def get_training_meta() -> dict:
    """
    Returns per-model best_lookback and best_wavelet from training_summary.json.
    Falls back to config defaults if summary not present.
    """
    summary_path = os.path.join(config.MODELS_DIR, "training_summary.json")
    defaults = {k: {"best_lookback": config.DEFAULT_LOOKBACK,
                    "best_wavelet":  config.WAVELET}
                for k in ["model_a","model_b","model_c"]}
    if not os.path.exists(summary_path):
        return defaults
    try:
        with open(summary_path) as f:
            s = json.load(f)
        for k in defaults:
            if k in s:
                defaults[k]["best_lookback"] = s[k].get("best_lookback",
                                                config.DEFAULT_LOOKBACK)
                defaults[k]["best_wavelet"]  = s[k].get("best_wavelet",
                                                config.WAVELET)
    except Exception as e:
        print(f"  Warning reading training summary: {e}")
    return defaults


# ─── Signal generation ────────────────────────────────────────────────────────

def raw_signals(model, prep, is_dual=False):
    X_te = prep["X_te"]
    if is_dual:
        n = prep["n_etf_features"]
        inputs = [X_te[:, :, :n], X_te[:, :, n:]]
    else:
        inputs = X_te
    preds    = model.predict(inputs, verbose=0)
    pred_std = preds.std(axis=0).mean()
    print(f"  Raw pred std per ETF: {preds.std(axis=0).round(6)}")
    if pred_std < 1e-4:
        print(f"  WARNING: Near-uniform predictions (std={pred_std:.6f})")
    return preds


def softmax_probs(preds, temperature=1.0):
    preds    = np.array(preds)
    row_sums = preds.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=0.01):
        return np.clip(preds, 0, 1)
    scaled = preds / (temperature + 1e-8)
    e      = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def compute_z_scores(probs):
    top   = probs.max(axis=1)
    mu    = probs.mean(axis=1)
    sigma = probs.std(axis=1) + 1e-8
    return (top - mu) / sigma


# ─── TSL backtest ─────────────────────────────────────────────────────────────

def backtest(probs, dates, etf_returns, tbill_series):
    z_scores    = compute_z_scores(probs)
    records     = []
    in_cash     = False
    prev_ret    = 0.0
    prev2_ret   = 0.0
    last_signal = None
    tsl_days    = 0

    for i in range(len(probs)):
        date   = pd.Timestamp(dates[i])
        prob   = probs[i]
        z      = float(z_scores[i])
        top_i  = int(np.argmax(prob))
        etf    = config.FI_ETFS[top_i]
        conf   = float(prob[top_i])

        two_day_cumul_pct = (prev_ret + prev2_ret) * 100

        if not in_cash and two_day_cumul_pct <= -TSL_PCT:
            in_cash  = True
            tsl_days = 0
        if in_cash and tsl_days >= 1 and z >= Z_REENTRY:
            in_cash = False
        if in_cash:
            tsl_days += 1

        if date in etf_returns.index:
            if in_cash:
                tbill_rate = float(tbill_series.get(date, 3.6))
                gross_ret  = (tbill_rate / 100) / 252
                mode       = "💵 CASH"
                signal     = "CASH"
            else:
                gross_ret  = float(etf_returns.loc[date, etf]) \
                             if etf in etf_returns.columns else 0.0
                fee_cost   = (FEE_BPS / 10000) if etf != last_signal else 0.0
                gross_ret -= fee_cost
                mode       = "📈 ETF"
                signal     = etf
                last_signal = etf
        else:
            gross_ret = 0.0
            mode      = "💵 CASH" if in_cash else "📈 ETF"
            signal    = "CASH"    if in_cash else etf

        records.append(dict(
            Date              = str(date.date()),
            Signal            = signal,
            Confidence        = round(conf, 4),
            Z_Score           = round(z, 4),
            Two_Day_Cumul_Pct = round(two_day_cumul_pct, 2),
            Mode              = mode,
            Net_Return        = round(gross_ret, 6),
            TSL_Triggered     = in_cash,
        ))

        prev2_ret = prev_ret
        prev_ret  = gross_ret

    df = pd.DataFrame(records)
    df["Cumulative"] = (1 + df["Net_Return"]).cumprod()
    return df


# ─── Performance metrics ──────────────────────────────────────────────────────

def compute_metrics(bt, bench_ret, tbill_series):
    rets   = bt["Net_Return"].values
    dates  = pd.to_datetime(bt["Date"])
    n_days = len(rets)

    total   = float((1 + pd.Series(rets)).prod())
    ann_ret = (total ** (252 / n_days) - 1) * 100

    tbill_daily = tbill_series.reindex(dates).ffill().fillna(3.6) / 100 / 252
    excess      = rets - tbill_daily.values
    sharpe      = float((excess.mean() / (excess.std() + 1e-8)) * np.sqrt(252))

    cum     = np.cumprod(1 + rets)
    peak    = np.maximum.accumulate(cum)
    dd      = (cum - peak) / peak
    max_dd  = float(dd.min()) * 100
    max_daily_dd = float(rets.min()) * 100

    hit_15  = float(pd.Series(np.sign(rets)).rolling(15).apply(
                lambda x: (x > 0).mean()).mean())

    spy_dates = bench_ret.index.intersection(dates)
    spy_rets  = bench_ret.loc[spy_dates, "SPY"].values \
                if "SPY" in bench_ret.columns else np.zeros(1)
    spy_ann   = ((1 + pd.Series(spy_rets)).prod() **
                 (252 / max(len(spy_rets), 1)) - 1) * 100

    return dict(
        ann_return    = round(ann_ret, 2),
        sharpe        = round(sharpe, 3),
        hit_ratio_15d = round(hit_15, 3),
        max_drawdown  = round(max_dd, 2),
        max_daily_dd  = round(max_daily_dd, 2),
        vs_spy        = round(ann_ret - spy_ann, 2),
        cash_days     = int((bt["Mode"] == "💵 CASH").sum()),
    )


# ─── Full evaluation ──────────────────────────────────────────────────────────

def run_evaluation():
    print(f"\n{'='*60}")
    print(f"  [FI] Evaluation")
    print(f"  TSL={TSL_PCT}%  Z-reentry={Z_REENTRY}σ  Fee={FEE_BPS}bps")
    print(f"{'='*60}")

    download_from_hf_if_needed()

    data = load_local()
    if not data:
        raise RuntimeError("No data. Run data_download.py first.")

    etf_ret  = normalize_etf_columns(data["etf_ret"].copy())
    etf_ret  = etf_ret[[c for c in config.FI_ETFS if c in etf_ret.columns]]
    bench_ret = normalize_etf_columns(data["bench_ret"].copy())
    macro     = flatten_columns(data["macro"].copy())
    tbill     = macro["TBILL_3M"] if "TBILL_3M" in macro.columns \
                else pd.Series(3.6, index=macro.index)

    meta    = get_training_meta()
    results = {}

    for tag, module, is_dual in [
        ("model_a", model_a, False),
        ("model_b", model_b, False),
        ("model_c", model_c, True),
    ]:
        lb      = meta[tag]["best_lookback"]
        wavelet = meta[tag]["best_wavelet"]
        print(f"\n  Evaluating {tag.upper()} (lb={lb}d wavelet={wavelet})...")

        prep = run_preprocessing(data, lb, wavelet=wavelet)

        try:
            m = module.load_model(lb)
        except Exception as e:
            print(f"  Could not load {tag}: {e}"); continue

        preds    = raw_signals(m, prep, is_dual=is_dual)
        probs    = softmax_probs(preds)
        prob_std = probs.std(axis=1).mean()
        print(f"  Mean prob std: {prob_std:.4f}")
        if prob_std < 0.01:
            print(f"  WARNING: Near-uniform probabilities — possible weight issue")

        bt         = backtest(probs, prep["d_te"], etf_ret, tbill)
        cash_count = (bt["Mode"] == "💵 CASH").sum()
        print(f"  CASH days: {cash_count}/{len(bt)}")

        metrics = compute_metrics(bt, bench_ret, tbill)

        # Live extension — last 60 trading days beyond test set
        live_records = []
        try:
            features     = build_features(data, wavelet=wavelet)
            scaler       = load_scaler(lb, wavelet=wavelet)
            recent_dates = features.index[-60:]
            for dt in recent_dates:
                if dt in prep["d_te"]:
                    continue
                idx = features.index.get_loc(dt)
                if idx < lb:
                    continue
                window = features.iloc[idx - lb: idx].values.astype(np.float32)
                X_win  = apply_scaler(window.reshape(1, lb, -1), scaler)
                if is_dual:
                    n_e = prep["n_etf_features"]
                    inp = [X_win[:, :, :n_e], X_win[:, :, n_e:]]
                else:
                    inp = X_win
                raw    = m.predict(inp, verbose=0)
                pr     = softmax_probs(raw)[0]
                zi     = float((pr.max() - pr.mean()) / (pr.std() + 1e-8))
                ei     = int(np.argmax(pr))
                etf_nm = config.FI_ETFS[ei]
                er     = normalize_etf_columns(data["etf_ret"].copy())
                actual = float(er.loc[dt, etf_nm]) \
                         if etf_nm in er.columns and dt in er.index else 0.0
                live_records.append(dict(
                    Date=str(dt.date()), Signal=etf_nm,
                    Confidence=round(float(pr[ei]), 4), Z_Score=round(zi, 4),
                    Two_Day_Cumul_Pct=0.0, Mode="ETF",
                    Net_Return=round(actual, 6), TSL_Triggered=False,
                ))
        except Exception as ex:
            print(f"  Live extension warning: {ex}")

        all_rows = bt.to_dict(orient="records") + live_records
        all_df   = pd.DataFrame(all_rows)
        all_df["Date"] = pd.to_datetime(all_df["Date"])
        all_df   = all_df.sort_values("Date").drop_duplicates("Date")
        audit_30 = all_df.tail(30).to_dict(orient="records")

        results[tag] = dict(
            metrics     = metrics,
            lookback    = lb,
            wavelet     = wavelet,
            audit_tail  = audit_30,
            all_signals = bt.to_dict(orient="records"),
        )
        print(f"    Ann={metrics['ann_return']}%  Sharpe={metrics['sharpe']}  "
              f"MaxDD={metrics['max_drawdown']}%  CashDays={metrics['cash_days']}")

    # AR(1) baseline
    default_wavelet = meta.get("model_a", {}).get("best_wavelet", config.WAVELET)
    prep30   = run_preprocessing(data, 30, wavelet=default_wavelet)
    ar1_rets = []
    etf_ret_full = normalize_etf_columns(data["etf_ret"].copy())
    etf_ret_full = etf_ret_full[[c for c in config.FI_ETFS if c in etf_ret_full.columns]]
    prev_df  = etf_ret_full.shift(1).fillna(0)
    for dt in pd.to_datetime(prep30["d_te"]):
        if dt in etf_ret_full.index and dt in prev_df.index:
            best = prev_df.loc[dt].idxmax()
            ar1_rets.append(float(etf_ret_full.loc[dt, best]))
    ar1_ann = ((1 + pd.Series(ar1_rets)).prod() **
               (252 / max(len(ar1_rets), 1)) - 1) * 100
    results["ar1_baseline"] = dict(ann_return=round(float(ar1_ann), 2))

    # Benchmarks
    for bench in config.BENCHMARKS:
        b_dates = bench_ret.index.intersection(pd.to_datetime(prep30["d_te"]))
        b_rets  = bench_ret.loc[b_dates, bench].values \
                  if bench in bench_ret.columns else np.zeros(1)
        b_ann   = ((1 + pd.Series(b_rets)).prod() **
                   (252 / max(len(b_rets), 1)) - 1) * 100
        b_sh    = (b_rets.mean() / (b_rets.std() + 1e-8)) * np.sqrt(252)
        b_cum   = np.cumprod(1 + b_rets)
        b_mdd   = float(((b_cum - np.maximum.accumulate(b_cum)) /
                          np.maximum.accumulate(b_cum)).min()) * 100
        results[bench] = dict(ann_return=round(float(b_ann), 2),
                               sharpe=round(float(b_sh), 3),
                               max_drawdown=round(float(b_mdd), 2))

    # Winner
    valid = [k for k in ["model_a","model_b","model_c"] if k in results]
    if valid:
        winner = max(valid, key=lambda k: results[k]["metrics"]["ann_return"])
        results["winner"] = winner
        print(f"\n  ⭐ WINNER: {winner.upper()} "
              f"({results[winner]['metrics']['ann_return']}%)")

    results["evaluated_at"] = datetime.now().isoformat()
    results["tsl_pct"]      = TSL_PCT
    results["z_reentry"]    = Z_REENTRY
    results["fee_bps"]      = FEE_BPS

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved → evaluation_results.json")

    # ── Sweep cache ────────────────────────────────────────────────────────────
    SWEEP_YEARS = [2008, 2013, 2015, 2017, 2019, 2021]
    start_yr    = None
    try:
        summ_path = os.path.join(config.MODELS_DIR, "training_summary.json")
        if os.path.exists(summ_path):
            with open(summ_path) as _f:
                start_yr = json.load(_f).get("start_year")
    except Exception:
        pass

    winner = results.get("winner")
    if start_yr in SWEEP_YEARS and winner and winner in results:
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        date_tag   = (_dt.now(_tz.utc) - _td(hours=5)).strftime("%Y%m%d")
        w_metrics  = results[winner].get("metrics", {})

        next_signal = "?"
        try:
            audit = results[winner].get("audit_tail") or \
                    results[winner].get("all_signals", [])
            if audit:
                last = audit[-1]
                next_signal = last.get("Signal_TSL") or last.get("Signal") or "?"
        except Exception:
            pass

        _z = 0.0
        try:
            if os.path.exists("latest_prediction.json"):
                with open("latest_prediction.json") as _pf:
                    _pred = json.load(_pf)
                _z = float(_pred.get("predictions", {}).get(winner, {}).get("z_score", 0.0) or 0.0)
        except Exception:
            pass

        sweep_payload = {
            "signal":       next_signal,
            "ann_return":   round(float(w_metrics.get("ann_return", 0)) / 100, 6),
            "z_score":      round(_z, 4),
            "sharpe":       round(float(w_metrics.get("sharpe", 0)), 4),
            "max_dd":       round(float(w_metrics.get("max_drawdown", 0)) / 100, 6),
            "winner_model": winner,
            "start_year":   start_yr,
            "sweep_date":   date_tag,
        }
        os.makedirs("sweep", exist_ok=True)
        sweep_fname = f"sweep/sweep_{start_yr}_{date_tag}.json"
        with open(sweep_fname, "w") as _sf:
            json.dump(sweep_payload, _sf, indent=2)
        print(f"  Sweep cache → {sweep_fname}  signal={next_signal}  z={_z:.3f}")

    return results


if __name__ == "__main__":
    # No CLI args — all params hardcoded in config.py
    run_evaluation()
