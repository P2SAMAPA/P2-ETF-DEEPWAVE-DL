# data_upload_hf.py
# Pushes local data/ parquet files (and optionally model weights) to
# HuggingFace Dataset repo: P2SAMAPA/p2-etf-deepwave-dl
# Usage:
#   python data_upload_hf.py               # push data only
#   python data_upload_hf.py --weights     # push data + models/

import argparse
import os
import glob

from huggingface_hub import HfApi, CommitOperationAdd
import config

api = HfApi(token=config.HF_TOKEN)

DATASET_REPO = config.HF_DATASET_REPO
REPO_TYPE    = "dataset"


def upload_files(local_paths: list, repo_paths: list, commit_msg: str):
    """Upload a batch of files to HF dataset repo."""
    operations = []
    for local_path, repo_path in zip(local_paths, repo_paths):
        if not os.path.exists(local_path):
            print(f"  Skipping (not found): {local_path}")
            continue
        operations.append(
            CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=local_path,
            )
        )

    if not operations:
        print("  Nothing to upload.")
        return

    api.create_commit(
        repo_id    = DATASET_REPO,
        repo_type  = REPO_TYPE,
        operations = operations,
        commit_message = commit_msg,
    )
    print(f"  Pushed {len(operations)} file(s) → {DATASET_REPO}")


def push_data():
    """Push all parquet files from data/ to HF."""
    data_files = glob.glob(os.path.join(config.DATA_DIR, "*.parquet"))
    if not data_files:
        print("No parquet files found in data/ — run data_download.py first.")
        return

    local_paths = data_files
    repo_paths  = [os.path.join("data", os.path.basename(f)) for f in data_files]

    print(f"\nPushing {len(data_files)} parquet file(s) to {DATASET_REPO}...")
    upload_files(local_paths, repo_paths,
                 commit_msg="[auto] Update ETF + macro dataset")


def push_weights():
    """Push all model weights from models/ to HF."""
    weight_files = (
        glob.glob(os.path.join(config.MODELS_DIR, "**", "*.keras"), recursive=True) +
        glob.glob(os.path.join(config.MODELS_DIR, "**", "*.h5"),    recursive=True) +
        glob.glob(os.path.join(config.MODELS_DIR, "**", "*.json"),  recursive=True) +
        glob.glob(os.path.join(config.MODELS_DIR, "**", "*.pkl"),   recursive=True)
    )
    if not weight_files:
        print("No weight files found in models/ — run train.py first.")
        return

    local_paths = weight_files
    repo_paths  = [f for f in weight_files]  # mirror directory structure

    print(f"\nPushing {len(weight_files)} weight file(s) to {DATASET_REPO}...")
    upload_files(local_paths, repo_paths,
                 commit_msg="[auto] Update model weights")


def push_evaluation():
    """Push evaluation_results.json if it exists."""
    path = "evaluation_results.json"
    if os.path.exists(path):
        upload_files([path], [path],
                     commit_msg="[auto] Update evaluation results")
        print("  Pushed evaluation_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", action="store_true",
                        help="Also push model weights")
    args = parser.parse_args()

    # Ensure repo exists (no-op if already created)
    api.create_repo(repo_id=DATASET_REPO, repo_type=REPO_TYPE,
                    exist_ok=True, private=False)

    push_data()
    push_evaluation()

    if args.weights:
        push_weights()

    print("\nHF upload complete.")
