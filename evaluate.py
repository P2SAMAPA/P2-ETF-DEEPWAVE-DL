# evaluate.py
import json
import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime

import config
from data_download import load_local
from preprocess import run_preprocessing
import model_a, model_b, model_c


# ─── HF download helper ───────────────────────────────────────────────────────

def download_from_hf_if_needed():
    """Download data + weights from HF Dataset if not present locally."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        token = config.HF_TOKEN or None

        # Data parquets
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
                    print(f"  Downloaded data/{f}.parquet")
                except Exception as e:
                    print(f"  Warning data/{f}: {e}")

        # Model weights
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        api   = HfApi(token=token)
        files = api.list_repo_files(
            repo_id=config.HF_DATASET_REPO,
            repo_type="dataset", token=token)
        for f in files:
            if f.startswith("models/") and f.endswith((".keras",".pkl",".json")):
                local = f
                if not os.path.exists(local):
                    os.makedirs(os.path.dirname(local), exist_ok=True)
                    try:
                        dl = hf_hub_download(
                            repo_id=config.HF_DATASET_REPO,
                            filename=f, repo_type="dataset", token=token)
                        shutil.copy(dl, local)
                        print(f"  Downloaded {f}")
                    except Exception as e:
                        print(f"  Warning {f}: {e}")
    except Exception as e:
        print(f"  WARNING: HF download failed: {e}")


# ─── Signal generation ────────────────────────────────────────────────────────

def raw_signals(model, prep, is_dual=False):
    X_te = prep["X_te"]
    if is_dual:
        n = prep["n_etf_features"]
        inputs = [X_te[:, :, :n], X_te[:, :, n:]]
    else:
        inputs = X_te
    preds = model.predict(inputs, verbose=0)   # (N, 5)
    # Diagnostic
    pred_std = preds.std(axis=0).mean()
    print(f"  Raw pred std per ETF: {preds.std(axis=0).round(6)}")
    if pred_std < 1e-4:
        print(f"  WARNING: Near-uniform predictions (std={pred_std:.6f}) — possible weight/scaler mismatch")
    return preds


def softmax_probs(preds, temperature=1.0):
    """
    Softmax probabilities. Models now output softmax directly (classification).
    temperature=1.0 = pass-through. Left as parameter for legacy compatibility.
    """
    preds = np.array(preds)
    # If model already outputs softmax (sums to 1), return as-is
    row_sums = preds.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=0.01):
        return np.clip(preds, 0, 1)
    # Otherwise apply softmax (legacy regression models)
    scaled = preds / (temperature + 1e-8)
    e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def compute_z_scores(probs):
    """
    Per-row Z-score: how many std devs is the top ETF above the row mean?
    Returns array of shape (N,)
    """
    top   = probs.max(axis=1)                 # (N,)
    mu    = probs.mean(axis=1)                # (N,)
    sigma = probs.std(axis=1) + 1e-8          # (N,)
    return (top - mu) / sigma                 # (N,)


# ─── TSL backtest ─────────────────────────────────────────────────────────────

def backtest(probs, dates, etf_returns, tbill_series,
             fee_bps=10, tsl_pct=10.0, z_reentry=1.1):
    """
    Day-by-day backtest with TSL + Z-score re-entry.
    Key fix: TSL fires end-of-day, re-entry checked next day.
    """
    z_scores   = compute_z_scores(probs)
    records    = []
    in_cash    = False
    prev_ret   = 0.0
    prev2_ret  = 0.0
    last_signal= None
    tsl_days   = 0   # days spent in CASH after TSL

    for i in range(len(probs)):
        date   = pd.Timestamp(dates[i])
        prob   = probs[i]
        z      = float(z_scores[i])
        top_i  = int(np.argmax(prob))
        etf    = config.ETFS[top_i]
        conf   = float(prob[top_i])

        # 2-day cumulative return (previous 2 days)
        two_day_cumul_pct = (prev_ret + prev2_ret) * 100

        # ── TSL trigger (must be out of CASH to trigger) ───────────────────
        if not in_cash and two_day_cumul_pct <= -tsl_pct:
            in_cash  = True
            tsl_days = 0

        # ── Z-score re-entry (only after at least 1 full CASH day) ────────
        if in_cash and tsl_days >= 1 and z >= z_reentry:
            in_cash = False

        if in_cash:
            tsl_days += 1

        # ── Get actual return ─────────────────────────────────────────────
        if date in etf_returns.index:
            if in_cash:
                tbill_rate = float(tbill_series.get(date, 3.6))
                gross_ret  = (tbill_rate / 100) / 252
                mode       = "💵 CASH"
                signal     = "CASH"
            else:
                gross_ret  = float(etf_returns.loc[date, etf]) \
                             if etf in etf_returns.columns else 0.0
                fee_cost   = (fee_bps / 10000) if etf != last_signal else 0.0
                gross_ret -= fee_cost
                mode       = "📈 ETF"
                signal     = etf
                last_signal= etf
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
    excess  = rets - tbill_daily.values
    sharpe  = float((excess.mean() / (excess.std() + 1e-8)) * np.sqrt(252))

    cum    = np.cumprod(1 + rets)
    peak   = np.maximum.accumulate(cum)
    dd     = (cum - peak) / peak
    max_dd = float(dd.min()) * 100
    max_daily_dd = float(rets.min()) * 100

    signs  = np.sign(rets)
    hit_15 = float(pd.Series(signs).rolling(15).apply(
                lambda x: (x > 0).mean()).mean())

    # SPY benchmark
    spy_dates = bench_ret.index.intersection(dates)
    spy_rets  = bench_ret.loc[spy_dates, "SPY"].values \
                if "SPY" in bench_ret.columns else np.zeros(1)
    spy_total = float((1 + pd.Series(spy_rets)).prod())
    spy_ann   = (spy_total ** (252 / max(len(spy_rets), 1)) - 1) * 100

    # CASH days count
    cash_days = int((bt["Mode"] == "CASH").sum())

    return dict(
        ann_return    = round(ann_ret, 2),
        sharpe        = round(sharpe, 3),
        hit_ratio_15d = round(hit_15, 3),
        max_drawdown  = round(max_dd, 2),
        max_daily_dd  = round(max_daily_dd, 2),
        vs_spy        = round(ann_ret - spy_ann, 2),
        cash_days     = cash_days,
    )


# ─── AR(1) baseline ───────────────────────────────────────────────────────────

def ar1_backtest(etf_returns, test_dates):
    records  = []
    dates_dt = pd.to_datetime(test_dates)
    df = etf_returns[etf_returns.index.isin(dates_dt)].copy()
    prev = df.shift(1).fillna(0)
    for date, row in df.iterrows():
        best = prev.loc[date].idxmax()
        records.append(dict(Date=date, Signal=best,
                            Net_Return=float(row[best])))
    out = pd.DataFrame(records)
    out["Cumulative"] = (1 + out["Net_Return"]).cumprod()
    return out


# ─── Full evaluation ──────────────────────────────────────────────────────────

def run_evaluation(tsl_pct=config.DEFAULT_TSL_PCT,
                   z_reentry=config.DEFAULT_Z_REENTRY,
                   fee_bps=10):

    print(f"\n{'='*60}")
    print(f"  Evaluation — TSL={tsl_pct}%  Z-reentry={z_reentry}σ  "
          f"Fee={fee_bps}bps")
    print(f"{'='*60}")

    # Download data + weights from HF if not available locally
    download_from_hf_if_needed()

    data = load_local()
    if not data:
        raise RuntimeError("No data. Run data_download.py first.")

    # Normalize ETF columns
    from preprocess import normalize_etf_columns, flatten_columns
    etf_ret  = normalize_etf_columns(data["etf_ret"].copy())
    etf_ret  = etf_ret[[c for c in config.ETFS if c in etf_ret.columns]]
    bench_ret= normalize_etf_columns(data["bench_ret"].copy())

    # T-bill series
    macro    = flatten_columns(data["macro"].copy())
    tbill    = macro["TBILL_3M"] if "TBILL_3M" in macro.columns \
               else pd.Series(3.6, index=macro.index)

    # Best lookbacks from training summary
    summary_path = os.path.join(config.MODELS_DIR, "training_summary.json")
    lb_map = {"model_a": 30, "model_b": 30, "model_c": 30}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        for k in lb_map:
            lb_map[k] = s.get(k, {}).get("best_lookback", 30)

    results = {}

    for tag, module, is_dual in [
        ("model_a", model_a, False),
        ("model_b", model_b, False),
        ("model_c", model_c, True),
    ]:
        lb = lb_map[tag]
        print(f"\n  Evaluating {tag.upper()} (lb={lb}d)...")
        prep = run_preprocessing(data, lb)

        try:
            m = module.load_model(lb)
        except Exception as e:
            print(f"  Could not load {tag}: {e}")
            continue

        preds = raw_signals(m, prep, is_dual=is_dual)
        probs = softmax_probs(preds)

        # Check if model is producing meaningful predictions
        prob_std = probs.std(axis=1).mean()
        print(f"  probs sample (first 3 rows):\n{probs[:3]}")
        print(f"  z_scores sample: {compute_z_scores(probs[:5])}")
        print(f"  Mean prob std across ETFs: {prob_std:.4f}  "
              f"(>0.05 = model discriminating, ~0 = uniform = weights issue)")
        if prob_std < 0.01:
            print(f"  WARNING: Model {tag} outputting near-uniform probabilities!")
            print(f"  This usually means the model weights are not loaded correctly.")

        bt = backtest(probs, prep["d_te"], etf_ret, tbill,
                      fee_bps=fee_bps, tsl_pct=tsl_pct,
                      z_reentry=z_reentry)

        cash_count = (bt["Mode"] == "CASH").sum()
        print(f"  CASH days triggered: {cash_count} / {len(bt)}")
        print(f"  Signals distribution:\n{bt['Signal'].value_counts()}")

        metrics = compute_metrics(bt, bench_ret, tbill)
        # Extend audit trail with LIVE recent dates beyond test set
        # Test set ends at ~10% of total data from end.
        # We run inference on the most recent 60 trading days too.
        live_records = []
        try:
            from preprocess import build_features, apply_scaler, load_scaler
            features   = build_features(data)
            scaler     = load_scaler(lb)
            recent_dates = features.index[-60:]   # last 60 trading days
            for dt in recent_dates:
                if dt in prep["d_te"]:
                    continue   # already in test set
                idx = features.index.get_loc(dt)
                if idx < lb:
                    continue
                window = features.iloc[idx - lb : idx].values.astype(np.float32)
                X_win  = apply_scaler(window.reshape(1, lb, -1), scaler)
                if is_dual:
                    n_e = prep["n_etf_features"]
                    inp = [X_win[:, :, :n_e], X_win[:, :, n_e:]]
                else:
                    inp = X_win
                raw = m.predict(inp, verbose=0)
                pr  = softmax_probs(raw)[0]
                zi  = float((pr.max() - pr.mean()) / (pr.std() + 1e-8))
                ei  = int(np.argmax(pr))
                etf_name = config.ETFS[ei]
                # Get actual return if available
                if "etf_ret" in data:
                    from preprocess import normalize_etf_columns
                    er = normalize_etf_columns(data["etf_ret"].copy())
                    ec = [c for c in config.ETFS if c in er.columns]
                    if etf_name in ec and dt in er.index:
                        actual_ret = float(er.loc[dt, etf_name])
                    else:
                        actual_ret = 0.0
                else:
                    actual_ret = 0.0
                live_records.append(dict(
                    Date       = str(dt.date()),
                    Signal     = etf_name,
                    Confidence = round(float(pr[ei]), 4),
                    Z_Score    = round(zi, 4),
                    Two_Day_Cumul_Pct = 0.0,
                    Mode       = "ETF",
                    Net_Return = round(actual_ret, 6),
                    TSL_Triggered = False,
                ))
        except Exception as ex:
            print(f"  Live extension warning: {ex}")

        # Merge test set + live records, sort, take last 30
        all_rows  = bt.to_dict(orient="records") + live_records
        all_df    = pd.DataFrame(all_rows)
        all_df["Date"] = pd.to_datetime(all_df["Date"])
        all_df    = all_df.sort_values("Date").drop_duplicates("Date")
        audit_30  = all_df.tail(30).to_dict(orient="records")

        results[tag] = dict(
            metrics     = metrics,
            lookback    = lb,
            audit_tail  = audit_30,
            all_signals = bt.to_dict(orient="records"),
        )
        print(f"    Ann={metrics['ann_return']}%  "
              f"Sharpe={metrics['sharpe']}  "
              f"MaxDD={metrics['max_drawdown']}%  "
              f"CashDays={metrics['cash_days']}")

    # AR(1) baseline
    prep30   = run_preprocessing(data, 30)
    ar1_bt   = ar1_backtest(etf_ret, prep30["d_te"])
    ar1_rets = ar1_bt["Net_Return"].values
    n        = len(ar1_rets)
    ar1_ann  = ((1 + pd.Series(ar1_rets)).prod() ** (252/n) - 1) * 100
    results["ar1_baseline"] = dict(ann_return=round(float(ar1_ann), 2))

    # Benchmarks
    for bench in config.BENCHMARKS:
        test_dates = prep30["d_te"]
        b_dates    = bench_ret.index.intersection(pd.to_datetime(test_dates))
        b_rets     = bench_ret.loc[b_dates, bench].values \
                     if bench in bench_ret.columns else np.zeros(1)
        b_total    = (1 + pd.Series(b_rets)).prod()
        b_ann      = (b_total ** (252 / max(len(b_rets),1)) - 1) * 100
        b_sh       = (b_rets.mean()/(b_rets.std()+1e-8))*np.sqrt(252)
        b_cum      = np.cumprod(1 + b_rets)
        b_peak     = np.maximum.accumulate(b_cum)
        b_mdd      = float(((b_cum-b_peak)/b_peak).min())*100
        results[bench] = dict(
            ann_return   = round(float(b_ann), 2),
            sharpe       = round(float(b_sh), 3),
            max_drawdown = round(float(b_mdd), 2),
        )

    # Winner
    valid = [k for k in ["model_a","model_b","model_c"] if k in results]
    if valid:
        winner = max(valid,
                     key=lambda k: results[k]["metrics"]["ann_return"])
        results["winner"] = winner
        print(f"\n  ⭐ WINNER: {winner.upper()} "
              f"({results[winner]['metrics']['ann_return']}%)")

    results["evaluated_at"] = datetime.now().isoformat()
    results["tsl_pct"]      = tsl_pct
    results["z_reentry"]    = z_reentry

    with open("evaluation_results.json","w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved → evaluation_results.json")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsl",  type=float, default=config.DEFAULT_TSL_PCT)
    parser.add_argument("--z",    type=float, default=config.DEFAULT_Z_REENTRY)
    parser.add_argument("--fee",  type=float, default=10)
    args = parser.parse_args()
    run_evaluation(tsl_pct=args.tsl, z_reentry=args.z, fee_bps=args.fee)
