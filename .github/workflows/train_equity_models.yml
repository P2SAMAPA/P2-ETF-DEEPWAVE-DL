name: Train Equity Models (A, B, C)

on:
  workflow_dispatch:
    inputs:
      model:
        description: "Model(s) to train: all | a | b | c | a,b etc."
        required: false
        default: "all"
      epochs:
        description: "Max training epochs"
        required: false
        default: "80"
      start_year:
        description: "Training start year (e.g. 2015)"
        required: false
        default: "2008"
      wavelet:
        description: "Wavelet key: db4 | db2 | haar | sym5"
        required: false
        default: "db4"
      tsl_pct:
        description: "Trailing stop loss %"
        required: false
        default: "10"
      z_reentry:
        description: "Z-score re-entry threshold"
        required: false
        default: "1.1"
      fee_bps:
        description: "Trading fee in bps"
        required: false
        default: "10"

jobs:
  train-equity:
    runs-on: ubuntu-latest
    timeout-minutes: 180   # 3 hours — equity has 13 ETFs vs 7 FI

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Download data from HF Dataset
        env:
          HF_TOKEN:        ${{ secrets.HF_TOKEN }}
          HF_DATASET_REPO: ${{ secrets.HF_DATASET_REPO }}
          FRED_API_KEY:    ${{ secrets.FRED_API_KEY }}
        run: |
          python << 'EOF'
          import shutil, os
          from huggingface_hub import hf_hub_download
          import config

          token = os.environ.get("HF_TOKEN")
          os.makedirs(config.DATA_DIR, exist_ok=True)

          for f in ["etf_price","etf_ret","etf_vol",
                    "bench_price","bench_ret","bench_vol","macro"]:
              try:
                  dl = hf_hub_download(
                      repo_id=config.HF_DATASET_REPO,
                      filename=f"data/{f}.parquet",
                      repo_type="dataset", token=token)
                  shutil.copy(dl, os.path.join(config.DATA_DIR, f"{f}.parquet"))
                  print(f"✓ data/{f}.parquet")
              except Exception as e:
                  print(f"✗ {f}: {e}")
          EOF

      - name: Train equity models
        env:
          HF_TOKEN:        ${{ secrets.HF_TOKEN }}
          HF_DATASET_REPO: ${{ secrets.HF_DATASET_REPO }}
          FRED_API_KEY:    ${{ secrets.FRED_API_KEY }}
        run: |
          python train_equity.py \
            --model      "${{ github.event.inputs.model }}" \
            --epochs     ${{ github.event.inputs.epochs }} \
            --start_year ${{ github.event.inputs.start_year }} \
            --wavelet    ${{ github.event.inputs.wavelet }}

      - name: Push equity weights + summary to HF Dataset
        env:
          HF_TOKEN:        ${{ secrets.HF_TOKEN }}
          HF_DATASET_REPO: ${{ secrets.HF_DATASET_REPO }}
        run: |
          python << 'EOF'
          import glob, os
          from huggingface_hub import HfApi, CommitOperationAdd
          import config

          token = os.environ.get("HF_TOKEN")
          api   = HfApi(token=token)

          # Collect all equity weight files
          files = (
              glob.glob(os.path.join(config.MODELS_DIR, "*_eq_*.keras")) +
              glob.glob(os.path.join(config.MODELS_DIR, "scaler_eq_*.pkl")) +
              [os.path.join(config.MODELS_DIR, "training_summary_equity.json")]
          )
          files = [f for f in files if os.path.exists(f)]

          ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=f) for f in files]
          if ops:
              api.create_commit(
                  repo_id=config.HF_DATASET_REPO,
                  repo_type="dataset",
                  operations=ops,
                  commit_message="[auto] Update equity model weights",
              )
              print(f"✓ Pushed {len(ops)} equity file(s) to HF")
          else:
              print("No equity weight files found to push.")
          EOF

      - name: Run equity prediction
        env:
          HF_TOKEN:        ${{ secrets.HF_TOKEN }}
          HF_DATASET_REPO: ${{ secrets.HF_DATASET_REPO }}
          FRED_API_KEY:    ${{ secrets.FRED_API_KEY }}
        run: |
          python predict_equity.py \
            --tsl ${{ github.event.inputs.tsl_pct }} \
            --z   ${{ github.event.inputs.z_reentry }}

      - name: Push equity prediction to HF Dataset
        env:
          HF_TOKEN:        ${{ secrets.HF_TOKEN }}
          HF_DATASET_REPO: ${{ secrets.HF_DATASET_REPO }}
        run: |
          python << 'EOF'
          import os
          from huggingface_hub import HfApi
          import config

          token = os.environ.get("HF_TOKEN")
          api   = HfApi(token=token)

          for filename in ["latest_prediction_equity.json"]:
              if os.path.exists(filename):
                  api.upload_file(
                      path_or_fileobj=filename,
                      path_in_repo=filename,
                      repo_id=config.HF_DATASET_REPO,
                      repo_type="dataset",
                  )
                  print(f"✓ Pushed {filename}")
          EOF

      - name: Summary
        run: |
          echo "=== Equity Training Summary ==="
          python << 'EOF'
          import json, os, config

          summary_path = os.path.join(config.MODELS_DIR, "training_summary_equity.json")
          if os.path.exists(summary_path):
              with open(summary_path) as f:
                  s = json.load(f)
              print(f"Trained at:   {s.get('trained_at','N/A')}")
              print(f"Universe:     {s.get('universe','equity')}")
              print(f"ETFs:         {s.get('etfs', config.EQUITY_ETFS)}")
              print(f"Start year:   {s.get('start_year','N/A')}")
              print(f"Wavelet:      {s.get('wavelet','N/A')}")
              for key in ["model_a","model_b","model_c"]:
                  info = s.get(key, {})
                  if info:
                      lb  = info.get("best_lookback","?")
                      res = info.get("results",[])
                      best_loss = min(r["val_mse"] for r in res) if res else 0
                      print(f"  {key}: best_lookback={lb}d  best_val_loss={best_loss:.6f}")
          else:
              print("training_summary_equity.json not found.")
          EOF
