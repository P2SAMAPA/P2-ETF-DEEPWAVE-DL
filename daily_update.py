# daily_update.py
# Run every weekday by GitHub Actions cron.
# 1. Downloads existing dataset from Hugging Face
# 2. Pulls incremental data (last stored date → today)
# 3. Pushes updated parquet files to HF Dataset
# 4. Runs latest prediction and saves latest_prediction.json
# 5. Pushes prediction json to HF

import os
import sys
from datetime import datetime

from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd

import config
from data_download import incremental_update
from predict import run_predict


def download_existing_data():
    """Download all parquet files from HF dataset repo into local data/."""
    api = HfApi(token=config.HF_TOKEN)
    repo_id = config.HF_DATASET_REPO
    repo_type = "dataset"

    os.makedirs(config.DATA_DIR, exist_ok=True)

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except Exception as e:
        print(f"Could not list files in HF repo: {e}")
        return

    parquet_files = [f for f in files if f.endswith(".parquet") and f.startswith("data/")]
    if not parquet_files:
        print("No existing parquet files found in HF. Starting fresh.")
        return

    print(f"Downloading {len(parquet_files)} parquet files from HF...")
    for remote_path in parquet_files:
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type=repo_type,
                token=config.HF_TOKEN,
                local_dir=config.DATA_DIR,
                local_dir_use_symlinks=False,
            )
            print(f"  Downloaded {remote_path}")
        except Exception as e:
            print(f"  Failed to download {remote_path}: {e}")


def upload_files(local_paths, repo_paths, commit_msg):
    """Upload multiple files to HF dataset repo."""
    api = HfApi(token=config.HF_TOKEN)
    operations = []
    for local, remote in zip(local_paths, repo_paths):
        if not os.path.exists(local):
            print(f"  Skipping missing: {local}")
            continue
        operations.append(CommitOperationAdd(
            path_in_repo=remote,
            path_or_fileobj=local
        ))
    if not operations:
        print("Nothing to upload.")
        return
    api.create_commit(
        repo_id=config.HF_DATASET_REPO,
        repo_type="dataset",
        operations=operations,
        commit_message=commit_msg,
    )
    print(f"Uploaded {len(operations)} file(s).")


def ensure_sorted_and_log(data_dict):
    """Ensure each DataFrame is sorted ascending by index, and print summary."""
    for key, df in data_dict.items():
        if df.index.name != "Date":
            df.index.name = "Date"
        # Force sort ascending
        df.sort_index(inplace=True)
        print(f"\n[{key}]")
        print(f"  Rows: {len(df)}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  First 3 dates: {df.index[:3].tolist()}")
        print(f"  Last 3 dates: {df.index[-3:].tolist()}")
    return data_dict


def main():
    print(f"\n{'='*60}")
    print(f"  Daily Update — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*60}")

    # 0. Download existing data from HF (if any)
    print("\n[0/4] Downloading existing dataset from Hugging Face...")
    download_existing_data()

    # 1. Pull new data (incremental)
    print("\n[1/4] Incremental data download...")
    data = incremental_update()
    if not data:
        print("  No new data. Exiting.")
        sys.exit(0)

    # Ensure all DataFrames are sorted ascending and log their stats
    print("\n--- Verifying data after incremental update ---")
    data = ensure_sorted_and_log(data)

    # 2. Push updated data to HF
    print("\n[2/4] Pushing data to Hugging Face...")
    data_files = [os.path.join(config.DATA_DIR, f) for f in os.listdir(config.DATA_DIR) if f.endswith(".parquet")]
    repo_paths = [os.path.join("data", os.path.basename(f)) for f in data_files]
    upload_files(data_files, repo_paths, "[auto] Update ETF + macro dataset")

    # 3. Run prediction with default risk params
    print("\n[3/4] Generating latest prediction...")
    result = run_predict(
        tsl_pct   = config.DEFAULT_TSL_PCT,
        z_reentry = config.DEFAULT_Z_REENTRY,
    )

    # 4. Push prediction json
    print("\n[4/4] Pushing prediction JSON...")
    if os.path.exists("latest_prediction.json"):
        upload_files(["latest_prediction.json"], ["latest_prediction.json"],
                     "[auto] Daily prediction update")
    if os.path.exists("evaluation_results.json"):
        upload_files(["evaluation_results.json"], ["evaluation_results.json"],
                     "[auto] Update evaluation results")

    print(f"\n  Daily update complete.")
    print(f"  Final signal: {result.get('final_signal', '—')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
