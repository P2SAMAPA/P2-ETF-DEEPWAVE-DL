# evaluate.py
# Backtests all 3 models on the test set.
# Applies trailing stop loss (TSL) + Z-score re-entry logic.
# Compares vs SPY and AGG buy-and-hold and AR(1) baseline.
# Outputs evaluation_results.json

import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

import config
from data_download import load_local
from preprocess import run_preprocessing
import model_a, model_b, model_c


# ─── Signal generator ─────────────────────────────────────────────────────────

def raw_signals(model, prep: dict, is_dual: bool = False) -> np.ndarray:
    """
    Run model on test set.
    Returns predicted log-return matrix (N_test, 5).
    """
    X_te = prep["X_te"]
    if is_dual:
        n_etf = prep["n_etf_features"]
        inputs = [X_te[:, :, :n_etf], X_te[:, :, n_etf:]]
    else:
        inputs = X_te
    return model.predict(inputs, verbose=0)   # (N, 5)


def softmax_probs(preds: np.ndarray) -> np.ndarray:
    """Convert raw predicted returns to softmax probabilities."""
    e = np.exp(preds - preds.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def z_score(probs: np.ndarray) -> np.ndarray:
    """
    Z-score: how many std devs is the top ETF prob above the mean of all ETF probs?
    Shape: (N,)
    """
    top   = probs.max(axis=1)
    mu    = probs.mean(axis=1)
    sigma = probs.std(axis=1) + 1e-8
    return (top - mu) / sigma


# ─── TSL + Z-score backtest ───────────────────────────────────────────────────

def backtest(probs: np.ndarray,
             dates: np.ndarray,
             etf_returns: pd.DataFrame,
             tbill_series: pd.Series,
             fee_bps: float = 10,
             tsl_pct: float = config.DEFAULT_TSL_PCT,
             z_reentry: float = config.DEFAULT_Z_REENTRY) -> pd.DataFrame:
    """
    Day-by-day backtest with:
      - Top ETF selected by argmax(probs)
      - Trailing stop loss: if 2-day cumulative return <= -tsl_pct → CASH
      - CASH earns daily 3m T-bill rate
      - Re-enter ETF when Z >= z_reentry

    Returns DataFrame with columns:
      Date, Signal, Confidence, Z_Score, Mode,
      Gross_Return, Fee, Net_Return, Cumulative
    """
    n       = len(probs)
    etf_idx = {e: i for i, e in enumerate(config.ETFS)}

    records    = []
    in_cash    = False
    prev_ret   = 0.0        # previous day net return for 2-day rolling
    prev2_ret  = 0.0        # two days ago
    last_signal= None

    for i in range(n):
        date  = pd.Timestamp(dates[i])
        prob  = probs[i]
        z     = float(z_score(probs[i:i+1])[0])
        top_i = int(np.argmax(prob))
        etf   = config.ETFS[top_i]
        conf  = float(prob[top_i])

        # 2-day cumulative return check (prev_ret + prev2_ret)
        two_day_cumul = prev_ret + prev2_ret

        # Check TSL trigger
        if not in_cash and two_day_cumul <= -(tsl_pct / 100):
            in_cash = True

        # Check Z-score re-entry
        if in_cash and z >= z_reentry:
            in_cash = False

        # Get actual return for this date
        if date in etf_returns.index:
            if in_cash:
                # Earn daily T-bill
                tbill_rate = tbill_series.get(date, 3.6) / 100
                gross_ret  = tbill_rate / 252
                fee        = 0.0
                mode       = "CASH"
                signal     = "CASH"
            else:
                gross_ret  = float(etf_returns.loc[date, etf]) \
                             if etf in etf_returns.columns else 0.0
                # Transaction fee on signal change
                fee = (fee_bps / 10000) if etf != last_signal else 0.0
                gross_ret -= fee
                mode       = "ETF"
                signal     = etf
                last_signal= etf
        else:
            gross_ret = 0.0
            fee       = 0.0
            mode      = "CASH" if in_cash else "ETF"
            signal    = "CASH" if in_cash else etf

        records.append(dict(
            Date        = date,
            Signal      = signal,
            Confidence  = round(conf, 4),
            Z_Score     = round(z, 3),
            Mode        = mode,
            Gross_Return= round(gross_ret, 6),
            Fee         = round(fee, 6),
            Net_Return  = round(gross_ret, 6),
        ))

        prev2_ret = prev_ret
        prev_ret  = gross_ret

    df = pd.DataFrame(records)
    df["Cumulative"] = (1 + df["Net_Return"]).cumprod()
    return df


# ─── Performance metrics ──────────────────────────────────────────────────────

def compute_metrics(bt: pd.DataFrame,
                    bench_ret: pd.DataFrame,
                    tbill_series: pd.Series) -> dict:
    rets  = bt["Net_Return"].values
    dates = bt["Date"]

    # Annualised return
    n_days = len(rets)
    total  = (1 + pd.Series(rets)).prod()
    ann_ret= (total ** (252 / n_days) - 1) * 100

    # Sharpe (excess over T-bill)
    tbill_daily = tbill_series.reindex(dates).ffill().fillna(3.6) / 100 / 252
    excess = rets - tbill_daily.values
    sharpe = (excess.mean() / (excess.std() + 1e-8)) * np.sqrt(252)

    # Max drawdown
    cum   = np.cumprod(1 + rets)
    peak  = np.maximum.accumulate(cum)
    dd    = (cum - peak) / peak
    max_dd= float(dd.min()) * 100

    # Max daily DD
    max_daily_dd = float(rets.min()) * 100

    # Hit ratio (15-day rolling direction)
    signs      = np.sign(rets)
    hit_15     = pd.Series(signs).rolling(15).apply(
                    lambda x: (x > 0).mean()).mean()

    # Benchmark SPY ann return (same test period)
    spy_dates   = bench_ret.index.intersection(dates)
    spy_rets    = bench_ret.loc[spy_dates, "SPY"].values if "SPY" in bench_ret.columns else np.zeros(1)
    spy_total   = (1 + pd.Series(spy_rets)).prod()
    spy_ann     = (spy_total ** (252 / max(len(spy_rets), 1)) - 1) * 100

    return dict(
        ann_return    = round(float(ann_ret), 2),
        sharpe        = round(float(sharpe), 3),
        hit_ratio_15d = round(float(hit_15), 3),
        max_drawdown  = round(float(max_dd), 2),
        max_daily_dd  = round(float(max_daily_dd), 2),
        vs_spy        = round(float(ann_ret - spy_ann), 2),
    )


# ─── AR(1) baseline ───────────────────────────────────────────────────────────

def ar1_backtest(etf_returns: pd.DataFrame,
                 test_dates: np.ndarray) -> pd.DataFrame:
    """Naive AR(1): predict next return = lag-1 return, pick best ETF."""
    records = []
    dates_set = set(pd.to_datetime(test_dates))
    df = etf_returns[etf_returns.index.isin(dates_set)].copy()
    prev = df.shift(1).fillna(0)
    for date, row in df.iterrows():
        best_etf = prev.loc[date].idxmax()
        ret = float(row[best_etf])
        records.append(dict(Date=date, Signal=best_etf, Net_Return=ret))
    out = pd.DataFrame(records)
    out["Cumulative"] = (1 + out["Net_Return"]).cumprod()
    return out


# ─── Full evaluation ──────────────────────────────────────────────────────────

def run_evaluation(tsl_pct: float   = config.DEFAULT_TSL_PCT,
                   z_reentry: float = config.DEFAULT_Z_REENTRY,
                   fee_bps: float   = 10) -> dict:

    print(f"\n{'='*60}")
    print(f"  Evaluation — TSL={tsl_pct}%  Z-reentry={z_reentry}σ")
    print(f"{'='*60}")

    data = load_local()
    if not data:
        raise RuntimeError("No data. Run data_download.py first.")

    # Load training summary for best lookbacks
    summary_path = os.path.join(config.MODELS_DIR, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        lb_a = summary.get("model_a", {}).get("best_lookback", 30)
        lb_b = summary.get("model_b", {}).get("best_lookback", 30)
        lb_c = summary.get("model_c", {}).get("best_lookback", 30)
    else:
        lb_a = lb_b = lb_c = config.DEFAULT_LOOKBACK

    tbill = data["macro"]["TBILL_3M"] if "TBILL_3M" in data["macro"].columns \
            else pd.Series(3.6, index=data["macro"].index)

    results = {}

    for tag, module, lb, is_dual in [
        ("model_a", model_a, lb_a, False),
        ("model_b", model_b, lb_b, False),
        ("model_c", model_c, lb_c, True),
    ]:
        print(f"\n  Evaluating {tag.upper()} (lb={lb}d)...")
        prep   = run_preprocessing(data, lb)

        try:
            m      = module.load_model(lb)
        except Exception as e:
            print(f"  Could not load {tag}: {e}")
            continue

        preds  = raw_signals(m, prep, is_dual=is_dual)
        probs  = softmax_probs(preds)

        bt = backtest(probs, prep["d_te"],
                      data["etf_ret"][config.ETFS],
                      tbill,
                      fee_bps=fee_bps,
                      tsl_pct=tsl_pct,
                      z_reentry=z_reentry)

        metrics = compute_metrics(bt, data["bench_ret"], tbill)
        results[tag] = dict(
            metrics    = metrics,
            lookback   = lb,
            audit_tail = bt.tail(20).to_dict(orient="records"),
            all_signals= bt.to_dict(orient="records"),
        )
        print(f"    Ann Return: {metrics['ann_return']}%  "
              f"Sharpe: {metrics['sharpe']}  "
              f"MaxDD: {metrics['max_drawdown']}%")

    # AR(1) baseline
    ar1_bt   = ar1_backtest(data["etf_ret"][config.ETFS],
                             run_preprocessing(data, 30)["d_te"])
    ar1_rets = ar1_bt["Net_Return"].values
    n_days   = len(ar1_rets)
    ar1_ann  = ((1 + pd.Series(ar1_rets)).prod() ** (252/n_days) - 1) * 100
    results["ar1_baseline"] = dict(ann_return=round(float(ar1_ann), 2))

    # Benchmark metrics (buy & hold, same test period)
    for bench in config.BENCHMARKS:
        test_dates = run_preprocessing(data, 30)["d_te"]
        b_dates    = data["bench_ret"].index.intersection(pd.to_datetime(test_dates))
        b_rets     = data["bench_ret"].loc[b_dates, bench].values \
                     if bench in data["bench_ret"].columns else np.zeros(1)
        b_total    = (1 + pd.Series(b_rets)).prod()
        b_ann      = (b_total ** (252 / max(len(b_rets), 1)) - 1) * 100
        b_sharpe   = (b_rets.mean() / (b_rets.std() + 1e-8)) * np.sqrt(252)
        b_cum      = np.cumprod(1 + b_rets)
        b_peak     = np.maximum.accumulate(b_cum)
        b_mdd      = float(((b_cum - b_peak) / b_peak).min()) * 100
        results[bench] = dict(
            ann_return   = round(float(b_ann), 2),
            sharpe       = round(float(b_sharpe), 3),
            max_drawdown = round(float(b_mdd), 2),
        )

    # Winner
    model_keys  = ["model_a", "model_b", "model_c"]
    valid_models= [k for k in model_keys if k in results]
    if valid_models:
        winner = max(valid_models,
                     key=lambda k: results[k]["metrics"]["ann_return"])
        results["winner"] = winner
        print(f"\n  ⭐ WINNER: {winner.upper()} "
              f"({results[winner]['metrics']['ann_return']}% ann. return)")

    results["evaluated_at"] = datetime.now().isoformat()
    results["tsl_pct"]      = tsl_pct
    results["z_reentry"]    = z_reentry

    out_path = "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsl",      type=float, default=config.DEFAULT_TSL_PCT)
    parser.add_argument("--z",        type=float, default=config.DEFAULT_Z_REENTRY)
    parser.add_argument("--fee",      type=float, default=10)
    args = parser.parse_args()
    run_evaluation(tsl_pct=args.tsl, z_reentry=args.z, fee_bps=args.fee)
