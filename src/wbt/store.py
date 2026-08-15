"""HF dataset repo is the store of record for data and results.

Local files under data/ and results/ are working copies; the repo itself stays
small. Pods push results here rather than relying on the Lambda filesystem.
"""

import os
from pathlib import Path

REPO_ID = "ic-org/wellbeing-in-translation"
REPO_TYPE = "dataset"

ROOT = Path(__file__).resolve().parents[2]


def _token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if tok:
        return tok
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, _, val = line.strip().partition("=")
            if key.strip() == "HF_TOKEN":
                return val.strip().strip("\"'")
    return None


def _api():
    from huggingface_hub import HfApi

    return HfApi(token=_token())


def ensure_repo():
    api = _api()
    api.create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True, private=False)
    return REPO_ID


def push(local_path, path_in_repo=None, message=None):
    """Upload a file or directory. Paths default to repo-relative."""
    api = _api()
    local = Path(local_path)
    if path_in_repo is None:
        path_in_repo = str(local.relative_to(ROOT)) if local.is_absolute() else str(local)
    ensure_repo()
    if local.is_dir():
        return api.upload_folder(
            folder_path=str(local), path_in_repo=path_in_repo,
            repo_id=REPO_ID, repo_type=REPO_TYPE,
            commit_message=message or f"upload {path_in_repo}",
        )
    return api.upload_file(
        path_or_fileobj=str(local), path_in_repo=path_in_repo,
        repo_id=REPO_ID, repo_type=REPO_TYPE,
        commit_message=message or f"upload {path_in_repo}",
    )


def pull(path_in_repo, local_dir=None):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        REPO_ID, path_in_repo, repo_type=REPO_TYPE,
        local_dir=str(local_dir or ROOT),
        token=_token(),
    )
