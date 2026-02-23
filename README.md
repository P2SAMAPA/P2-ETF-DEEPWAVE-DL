---
title: P2-ETF-DEEPWAVE-DL
emoji: 🧠
colorFrom: green
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: true
license: mit
---

# 🧠 P2-ETF-DEEPWAVE-DL

**Wavelet Deep Learning ETF Signal System**

Three parallel architectures for daily ETF signal generation:

| Model | Architecture |
|---|---|
| **Option A** | Wavelet-CNN-LSTM |
| **Option B** | Wavelet-Attention-CNN-LSTM |
| **Option C** | Wavelet-Parallel-Dual-Stream-CNN-LSTM |

## ETF Universe
`TLT` `TBT` `VNQ` `GLD` `SLV` + benchmarks `SPY` `AGG`

## Macro Signals
`TNX` `DXY` `Corp Spread` `HY Spread` `VIX` `T10Y2Y` `3m T-Bill`

## Risk Controls
- **Trailing Stop Loss**: configurable 0–25% on 2-day cumulative return
- **Z-score Re-entry**: configurable 1.0–2.0σ threshold for CASH → ETF
- **CASH yield**: earns live 3m T-bill rate while sidelined

## Data
- Source: Stooq (fallback: yfinance) + FRED API
- History: 2008-01-01 → daily updated
- Stored: HuggingFace Dataset `P2SAMAPA/p2-etf-deepwave-dl`

## Pipeline
- Training: GitHub Actions (`P2SAMAPA/P2-ETF-DEEPWAVE-DL`)
- Weights: auto-pushed to HF after every training run
- Data: updated daily at 02:00 UTC on weekdays
