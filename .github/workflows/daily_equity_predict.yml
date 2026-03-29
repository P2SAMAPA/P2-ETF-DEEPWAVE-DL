name: Daily Equity Prediction

on:
  schedule:
    - cron: "30 2 * * 1-5"   # 02:30 UTC Mon–Fri — 30 min after FI daily update
  workflow_dispatch:          # Manual trigger

jobs:
  predict-equity:
    runs-on: ubuntu-latest
    timeout-minutes: 30

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

      - name: Download latest data from HF Dataset
        env:
          HF_TOKEN:        ${{ secrets.HF_TOKEN }}
          HF_DATASET_REPO: ${{ secrets.HF_DATASET_REPO }}
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
                      repo_type="dataset", token=token,
                      force_download=True)
                  shutil.copy(dl, os.path.join(config.DATA_DIR, f"{f}.parquet"))
                  print(f"✓ data/{f}.parquet")
              except Exception as e:
                  print(f"✗ {f}: {e}")
          EOF

      - name: Download equity model weights from HF Dataset
        env:
          HF_TOKEN:        ${{ secrets.HF_TOKEN }}
          HF_DATASET_REPO: ${{ secrets.HF_DATASET_REPO }}
        run: |
          python << 'EOF'
          import shutil, os
          from huggingface_hub import HfApi, hf_hub_download
          import config

          token = os.environ.get("HF_TOKEN")
          api   = HfApi(token=token)
          os.makedirs(config.MODELS_DIR, exist_ok=True)

          files = api.list_repo_files(repo_id=config.HF_DATASET_REPO,
                                      repo_type="dataset", token=token)
          for f in files:
              is_equity = ("_eq_" in f or "equity" in f)
              is_weight = f.endswith((".keras", ".pkl", ".json"))
              in_models = f.startswith("models/")
              if is_equity and is_weight and in_models:
                  local = f
                  os.makedirs(os.path.dirname(local), exist_ok=True)
                  try:
                      dl = hf_hub_download(repo_id=config.HF_DATASET_REPO,
                                           filename=f, repo_type="dataset",
                                           token=token, force_download=True)
                      shutil.copy(dl, local)
                      print(f"✓ {f}")
                  except Exception as e:
                      print(f"✗ {f}: {e}")
          EOF

      - name: Run equity prediction
        env:
          HF_TOKEN:        ${{ secrets.HF_TOKEN }}
          HF_DATASET_REPO: ${{ secrets.HF_DATASET_REPO }}
          FRED_API_KEY:    ${{ secrets.FRED_API_KEY }}
        run: |
          python predict_equity.py
          echo "Equity prediction complete"

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
              else:
                  print(f"✗ {filename} not found — prediction may have failed")
                  exit(1)
          EOF

      - name: Commit latest_prediction_equity.json to repo
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add latest_prediction_equity.json || true
          git diff --cached --quiet || git commit -m "[auto] Daily equity prediction update $(date -u +%Y-%m-%d)"
          git push || true

      - name: Summary
        run: |
          echo "=== Equity Prediction Summary ==="
          python << 'EOF'
          import json
          try:
              with open("latest_prediction_equity.json") as f:
                  data = json.load(f)
              print(f"As of date:    {data.get('as_of_date','N/A')}")
              print(f"Winner model:  {data.get('winner_model','N/A')}")
              print(f"Final signal:  {data.get('final_signal','N/A')}")
              print(f"Universe:      {data.get('universe','equity')}")
              preds = data.get("predictions", {})
              for tag, p in preds.items():
                  print(f"  [{tag.upper()}] {p['signal']} | "
                        f"conf={p['confidence']:.1%} | z={p['z_score']:.2f}σ")
              tsl = data.get("tsl_status", {})
              print(f"TSL triggered: {tsl.get('tsl_triggered', False)}")
              print(f"In CASH:       {tsl.get('in_cash', False)}")
          except Exception as e:
              print(f"Error reading summary: {e}")
          EOF
