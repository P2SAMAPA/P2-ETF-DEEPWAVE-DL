"""
.github/scripts/push_eq_results.py
Pushes Equity training results for one year to HF Dataset.
- Overwrites today's file only: results/eq_{year}_{YYYYMMDD}.json
- Batches of 10 files to avoid 504 timeout.
"""
import argparse, glob, os, sys, time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config
from huggingface_hub import HfApi, CommitOperationAdd


def today_tag():
    return (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y%m%d")


def batched_commit(api, repo_id, operations, msg, batch_size=10, retries=3):
    for i in range(0, len(operations), batch_size):
        batch = operations[i:i + batch_size]
        for attempt in range(retries):
            try:
                api.create_commit(repo_id=repo_id, repo_type="dataset",
                                  operations=batch,
                                  commit_message=f"{msg} (batch {i//batch_size+1})")
                print(f"  ✓ batch {i//batch_size+1}: {len(batch)} files")
                break
            except Exception as e:
                print(f"  attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(30)
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    token   = os.environ.get("HF_TOKEN") or config.HF_TOKEN
    repo_id = os.environ.get("HF_DATASET_REPO") or config.HF_DATASET_REPO
    api     = HfApi(token=token)
    today   = today_tag()
    year    = args.year
    target  = f"results/eq_{year}_{today}.json"

    try:
        if target in list(api.list_repo_files(repo_id=repo_id,
                                               repo_type="dataset", token=token)):
            api.delete_file(path_in_repo=target, repo_id=repo_id,
                            repo_type="dataset", token=token,
                            commit_message=f"Overwrite eq_{year}_{today}")
            print(f"  Deleted existing {target}")
    except Exception as e:
        print(f"  Note (delete): {e}")

    ops = []
    local_result = os.path.join("results", f"eq_{year}_{today}.json")
    if os.path.exists(local_result):
        ops.append(CommitOperationAdd(path_in_repo=target, path_or_fileobj=local_result))
    else:
        print(f"  WARNING: {local_result} not found")

    for f in ["evaluation_results_equity.json", "latest_prediction_equity.json"]:
        if os.path.exists(f):
            ops.append(CommitOperationAdd(path_in_repo=f, path_or_fileobj=f))

    weight_files = [
        f for f in (
            glob.glob(os.path.join(config.MODELS_DIR, "model_*_eq", "**", "*.keras"), recursive=True) +
            glob.glob(os.path.join(config.MODELS_DIR, "scaler_eq_*.pkl")) +
            [os.path.join(config.MODELS_DIR, "training_summary_equity.json")]
        ) if os.path.exists(f)
    ]
    ops += [CommitOperationAdd(path_in_repo=f, path_or_fileobj=f) for f in weight_files]

    print(f"  Pushing {len(ops)} files for eq year={year}")
    batched_commit(api, repo_id, ops, f"[auto] Equity results year={year} {today}")
    print(f"  Done — eq_{year}_{today}.json → HF")


if __name__ == "__main__":
    main()
