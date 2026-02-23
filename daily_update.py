# daily_update.py
# Run every weekday by GitHub Actions cron.
# 1. Pulls incremental data (last stored date → today)
# 2. Pushes updated parquet files to HF Dataset
# 3. Runs latest prediction and saves latest_prediction.json
# 4. Pushes prediction json to HF

import os
import sys
from datetime import datetime

import config
from data_download import incremental_update
from data_upload_hf import push_data, push_evaluation
from predict import run_predict


def main():
    print(f"\n{'='*60}")
    print(f"  Daily Update — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*60}")

    # 1. Pull new data
    print("\n[1/3] Incremental data download...")
    data = incremental_update()
    if not data:
        print("  No new data. Exiting.")
        sys.exit(0)

    # 2. Push to HF
    print("\n[2/3] Pushing data to Hugging Face...")
    push_data()

    # 3. Run prediction with default risk params
    # (UI sliders override these at runtime in Streamlit)
    print("\n[3/3] Generating latest prediction...")
    result = run_predict(
        tsl_pct   = config.DEFAULT_TSL_PCT,
        z_reentry = config.DEFAULT_Z_REENTRY,
    )

    # Push prediction json
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=config.HF_TOKEN)
    if os.path.exists("latest_prediction.json"):
        api.create_commit(
            repo_id        = config.HF_DATASET_REPO,
            repo_type      = "dataset",
            operations     = [CommitOperationAdd(
                path_in_repo = "latest_prediction.json",
                path_or_fileobj = "latest_prediction.json",
            )],
            commit_message = "[auto] Daily prediction update",
        )
        print("  Pushed latest_prediction.json → HF")

    print(f"\n  Daily update complete.")
    print(f"  Final signal: {result.get('final_signal', '—')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
