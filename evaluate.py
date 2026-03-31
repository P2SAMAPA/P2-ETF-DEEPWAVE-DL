# evaluate.py — FI evaluation
# Writes: results/fi_{start_year}_{YYYYMMDD}.json
# Today's file is overwritten on re-run; previous days for other years untouched.
# Usage: python evaluate.py --start_year 2015

import argparse
import json
import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

import config
from data_download import load_local
from preprocess import run_preprocessing, build_features, apply_scaler, \
                       load_scaler, normalize_etf_columns, flatten_columns
import model_a, model_b, model_c

TSL_PCT   = config.DEFAULT_TSL_PCT
Z_REENTRY = config.DEFAULT_Z_REENTRY
FEE_BPS   = config.FEE_BPS

RESULTS_DIR = "results"


def today_tag() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y%m%d")


def result_path(start_year: int) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"fi_{start_year}_{today_tag()}.json")


def hf_result_path(start_year: int) -> str:
    return f"results/fi_{start_year}_{today_tag()}.json"


# ─── HF download ──────────────────────────────────────────────────────────────

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
                except Exception as e:
                    print(f"  ✗ data/{f}: {e}")
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        api   = HfApi(token=token)
        files = api.list_repo_files(repo_id=config.HF_DATASET_REPO,
                                    repo_type="dataset", token=token)
        for f in files:
            if f.startswith("models/") and f.endswith((".keras",".pkl",".json")) \
               and "_eq" not in f:
                local = f
                if not os.path.exists(local):
                    os.makedirs(os.path.dirname(local), exist_ok=True)
                    try:
                        dl = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                                             filename=f, repo_type="dataset",
                                             token=token)
                        shutil.copy(dl, local)
                    except Exception as e:
                        print(f"  ✗ {f}: {e}")
    except Exception as e:
        print(f"  WARNING: HF download failed: {e}")


# ─── Training metadata ────────────────────────────────────────────────────────

def get_training_meta() -> dict:
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
                defaults[k]["best_lookback"] = s[k].get("best_lookback", config.DEFAULT_LOOKBACK)
                defaults[k]["best_wavelet"]  = s[k].get("best_wavelet",  config.WAVELET)
    except Exception as e:
        print(f"  Warning: {e}")
    return defaults


# ─── Signal helpers ───────────────────────────────────────────────────────────

def softmax_probs(preds):
    preds    = np.array(preds)
    row_sums = preds.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=0.01):
        return np.clip(preds, 0, 1)
    scaled = preds / 0.1
    e      = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def compute_z_scores(probs):
    top   = probs.max(axis=1)
    mu    = probs.mean(axis=1)
    sigma = probs.std(axis=1) + 1e-8
    return (top - mu) / sigma


# ─── Backtest ─────────────────────────────────────────────────────────────────

def backtest(probs, dates, etf_returns, tbill_series):
    z_scores    = compute_z_scores(probs)
    records     = []
    in_cash     = False
    prev_ret    = 0.0
    prev2_ret   = 0.0
    last_signal = None
    tsl_days    = 0

    for i in range(len(probs)):
        dt      = pd.Timestamp(dates[i])
        prob    = probs[i]
        z       = float(z_scores[i])
        top_i   = int(np.argmax(prob))
        etf     = config.FI_ETFS[top_i]
        conf    = float(prob[top_i])
        two_day = (prev_ret + prev2_ret) * 100

        if not in_cash and two_day <= -TSL_PCT:
            in_cash, tsl_days = True, 0
        if in_cash and tsl_days >= 1 and z >= Z_REENTRY:
            in_cash = False
        if in_cash:
            tsl_days += 1

        if dt in etf_returns.index:
            if in_cash:
                gross_ret          = (float(tbill_series.get(dt, 3.6)) / 100) / 252
                mode, signal       = "💵 CASH", "CASH"
            else:
                gross_ret          = float(etf_returns.loc[dt, etf]) \
                                     if etf in etf_returns.columns else 0.0
                gross_ret         -= (FEE_BPS / 10000) if etf != last_signal else 0.0
                mode, signal       = "📈 ETF", etf
                last_signal        = etf
        else:
            gross_ret = 0.0
            mode      = "💵 CASH" if in_cash else "📈 ETF"
            signal    = "CASH"    if in_cash else etf

        records.append(dict(Date=str(dt.date()), Signal=signal,
                            Confidence=round(conf, 4), Z_Score=round(z, 4),
                            Two_Day_Cumul_Pct=round(two_day, 2), Mode=mode,
                            Net_Return=round(gross_ret, 6), TSL_Triggered=in_cash))
        prev2_ret, prev_ret = prev_ret, gross_ret

    df = pd.DataFrame(records)
    df["Cumulative"] = (1 + df["Net_Return"]).cumprod()
    return df


def compute_metrics(bt, bench_ret, tbill_series):
    rets   = bt["Net_Return"].values
    dates  = pd.to_datetime(bt["Date"])
    n_days = len(rets)
    total  = float((1 + pd.Series(rets)).prod())
    ann_ret = (total ** (252 / n_days) - 1) * 100
    tbill_daily = tbill_series.reindex(dates).ffill().fillna(3.6) / 100 / 252
    excess  = rets - tbill_daily.values
    sharpe  = float((excess.mean() / (excess.std() + 1e-8)) * np.sqrt(252))
    cum    = np.cumprod(1 + rets)
    peak   = np.maximum.accumulate(cum)
    max_dd = float(((cum - peak) / peak).min()) * 100
    hit_15 = float(pd.Series(np.sign(rets)).rolling(15).apply(
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
        max_daily_dd  = round(float(rets.min()) * 100, 2),
        vs_spy        = round(ann_ret - spy_ann, 2),
        cash_days     = int((bt["Mode"] == "💵 CASH").sum()),
    )


# ─── Main evaluation ──────────────────────────────────────────────────────────

def run_evaluation(start_year: int):
    print(f"\n{'='*60}")
    print(f"  [FI] Evaluation  start_year={start_year}")
    print(f"  TSL={TSL_PCT}%  Z={Z_REENTRY}σ  Fee={FEE_BPS}bps")
    print(f"{'='*60}")

    download_from_hf_if_needed()
    data = load_local()
    if not data:
        raise RuntimeError("No data.")

    # Filter to start_year
    for key in data:
        if hasattr(data[key], 'index') and isinstance(data[key].index, pd.DatetimeIndex):
            data[key] = data[key][data[key].index.year >= start_year]

    etf_ret   = normalize_etf_columns(data["etf_ret"].copy())
    etf_ret   = etf_ret[[c for c in config.FI_ETFS if c in etf_ret.columns]]
    bench_ret = normalize_etf_columns(data["bench_ret"].copy())
    macro     = flatten_columns(data["macro"].copy())
    tbill     = macro["TBILL_3M"] if "TBILL_3M" in macro.columns \
                else pd.Series(3.6, index=macro.index)

    meta    = get_training_meta()
    results = {"start_year": start_year}

    for tag, module, is_dual in [
        ("model_a", model_a, False),
        ("model_b", model_b, False),
        ("model_c", model_c, True),
    ]:
        lb      = meta[tag]["best_lookback"]
        wavelet = meta[tag]["best_wavelet"]
        print(f"\n  {tag.upper()} lb={lb}d wavelet={wavelet}")

        prep = run_preprocessing(data, lb, wavelet=wavelet)
        try:
            m = module.load_model(lb)
        except Exception as e:
            print(f"  Could not load {tag}: {e}"); continue

        X_te   = prep["X_te"]
        inputs = ([X_te[:, :, :prep["n_etf_features"]],
                   X_te[:, :, prep["n_etf_features"]:]]
                  if is_dual else X_te)
        preds  = m.predict(inputs, verbose=0)
        probs  = softmax_probs(preds)

        bt      = backtest(probs, prep["d_te"], etf_ret, tbill)
        metrics = compute_metrics(bt, bench_ret, tbill)

        # Live extension
        live_records = []
        try:
            features = build_features(data, wavelet=wavelet)
            scaler   = load_scaler(lb, wavelet=wavelet)
            for dt in features.index[-60:]:
                if dt in prep["d_te"]:
                    continue
                idx = features.index.get_loc(dt)
                if idx < lb:
                    continue
                window = features.iloc[idx - lb: idx].values.astype("float32")
                X_w    = apply_scaler(window.reshape(1, lb, -1), scaler)
                inp    = ([X_w[:, :, :prep["n_etf_features"]],
                           X_w[:, :, prep["n_etf_features"]:]]
                          if is_dual else X_w)
                pr   = softmax_probs(m.predict(inp, verbose=0))[0]
                zi   = float((pr.max() - pr.mean()) / (pr.std() + 1e-8))
                ei   = int(np.argmax(pr))
                enm  = config.FI_ETFS[ei]
                er   = normalize_etf_columns(data["etf_ret"].copy())
                act  = float(er.loc[dt, enm]) \
                       if enm in er.columns and dt in er.index else 0.0
                live_records.append(dict(Date=str(dt.date()), Signal=enm,
                    Confidence=round(float(pr[ei]), 4), Z_Score=round(zi, 4),
                    Two_Day_Cumul_Pct=0.0, Mode="ETF",
                    Net_Return=round(act, 6), TSL_Triggered=False))
        except Exception as ex:
            print(f"  Live ext warning: {ex}")

        all_rows = bt.to_dict(orient="records") + live_records
        all_df   = (pd.DataFrame(all_rows)
                    .assign(Date=lambda d: pd.to_datetime(d["Date"]))
                    .sort_values("Date").drop_duplicates("Date"))
        audit_30 = all_df.tail(30).to_dict(orient="records")

        # Latest signal info
        latest_probs = {config.FI_ETFS[i]: round(float(probs[-1][i]), 4)
                        for i in range(len(config.FI_ETFS))}
        latest_signal = config.FI_ETFS[int(np.argmax(probs[-1]))]
        latest_z      = float((probs[-1].max() - probs[-1].mean()) /
                               (probs[-1].std() + 1e-8))

        results[tag] = dict(
            metrics      = metrics,
            lookback     = lb,
            wavelet      = wavelet,
            audit_tail   = audit_30,
            all_signals  = bt.to_dict(orient="records"),
            latest_signal    = latest_signal,
            latest_probs     = latest_probs,
            latest_z_score   = round(latest_z, 3),
            latest_confidence= round(float(probs[-1].max()), 4),
        )
        print(f"    Ann={metrics['ann_return']}%  Sharpe={metrics['sharpe']}  "
              f"MaxDD={metrics['max_drawdown']}%")

    # Benchmarks
    default_wavelet = meta.get("model_a", {}).get("best_wavelet", config.WAVELET)
    prep30 = run_preprocessing(data, 30, wavelet=default_wavelet)
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

    valid = [k for k in ["model_a","model_b","model_c"] if k in results]
    if valid:
        winner = max(valid, key=lambda k: results[k]["metrics"]["ann_return"])
        results["winner"]       = winner
        results["evaluated_at"] = datetime.now().isoformat()
        results["tsl_pct"]      = TSL_PCT
        results["z_reentry"]    = Z_REENTRY
        results["fee_bps"]      = FEE_BPS

        # Consensus signal: best wavelet winner's latest signal
        w = results[winner]
        results["consensus_signal"]      = w["latest_signal"]
        results["consensus_z_score"]     = w["latest_z_score"]
        results["consensus_confidence"]  = w["latest_confidence"]
        results["consensus_probs"]       = w["latest_probs"]
        results["consensus_ann_return"]  = w["metrics"]["ann_return"]
        results["consensus_sharpe"]      = w["metrics"]["sharpe"]
        results["consensus_max_dd"]      = w["metrics"]["max_drawdown"]

        print(f"\n  ⭐ WINNER: {winner.upper()}  signal={results['consensus_signal']}")

    # Save result file (overwrites today's file for this year)
    out_path = result_path(start_year)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved → {out_path}")

    # Also write evaluation_results.json for backward compat with predict.py
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, required=True)
    args = parser.parse_args()
    run_evaluation(args.start_year)
