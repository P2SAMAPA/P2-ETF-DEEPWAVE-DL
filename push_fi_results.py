"""
.github/scripts/push_fi_results.py
Pushes FI training results for one year to HF Dataset.
- Overwrites today's file: results/fi_{year}_{YYYYMMDD}.json
- Previous days' files for other years are untouched.
- Model weights pushed in batches of 10 to avoid 504 timeout.
"""
import argparse
import glob
import os
import time
from datetime import datetime, timezone, timedelta

from huggingface_hub import HfApi, CommitOperationAdd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config


def today_tag() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y%m%d")


def batched_commit(api, repo_id, operations, commit_message,
                   batch_size=10, retries=3, pause=5):
    """Push operations in batches, retrying on 504."""
    for i in range(0, len(operations), batch_size):
        batch = operations[i:i + batch_size]
        for attempt in range(retries):
            try:
                api.create_commit(
                    repo_id=repo_id, repo_type="dataset",
                    operations=batch,
                    commit_message=f"{commit_message} (batch {i//batch_size+1})",
                )
                print(f"  ✓ batch {i//batch_size+1}: {len(batch)} files")
                break
            except Exception as e:
                print(f"  attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(30)
        time.sleep(pause)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    token   = os.environ.get("HF_TOKEN") or config.HF_TOKEN
    repo_id = os.environ.get("HF_DATASET_REPO") or config.HF_DATASET_REPO
    api     = HfApi(token=token)
    today   = today_tag()
    year    = args.year

    # ── Delete today's existing file for this year (overwrite) ───────────────
    target_path = f"results/fi_{year}_{today}.json"
    try:
        existing = [f for f in api.list_repo_files(repo_id=repo_id,
                                                    repo_type="dataset",
                                                    token=token)
                    if f == target_path]
        if existing:
            api.delete_file(path_in_repo=target_path, repo_id=repo_id,
                            repo_type="dataset", token=token,
                            commit_message=f"Overwrite fi_{year}_{today}")
            print(f"  Deleted existing {target_path}")
    except Exception as e:
        print(f"  Note (delete): {e}")

    # ── Push result JSON ──────────────────────────────────────────────────────
    local_result = os.path.join("results", f"fi_{year}_{today}.json")
    ops = []
    if os.path.exists(local_result):
        ops.append(CommitOperationAdd(path_in_repo=target_path,
                                       path_or_fileobj=local_result))
        print(f"  Queuing {target_path}")
    else:
        print(f"  WARNING: {local_result} not found — evaluate.py may have failed")

    # Also push evaluation_results.json and latest_prediction.json
    for f in ["evaluation_results.json", "latest_prediction.json"]:
        if os.path.exists(f):
            ops.append(CommitOperationAdd(path_in_repo=f, path_or_fileobj=f))

    # ── Push model weights (FI, non-equity, batched) ──────────────────────────
    weight_files = [
        f for f in (
            glob.glob(os.path.join(config.MODELS_DIR, "model_a", "**", "*.keras"),
                      recursive=True) +
            glob.glob(os.path.join(config.MODELS_DIR, "model_b", "**", "*.keras"),
                      recursive=True) +
            glob.glob(os.path.join(config.MODELS_DIR, "model_c", "**", "*.keras"),
                      recursive=True) +
            glob.glob(os.path.join(config.MODELS_DIR, "scaler_lb*.pkl")) +
            [os.path.join(config.MODELS_DIR, "training_summary.json")]
        )
        if os.path.exists(f)
    ]

    all_ops = ops + [CommitOperationAdd(path_in_repo=f, path_or_fileobj=f)
                     for f in weight_files]

    print(f"  Pushing {len(all_ops)} files for fi year={year}")
    batched_commit(api, repo_id, all_ops,
                   commit_message=f"[auto] FI results year={year} {today}")
    print(f"  Done — fi_{year}_{today}.json pushed to HF")


if __name__ == "__main__":
    main()
