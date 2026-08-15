#!/bin/bash
# Pull results off the pod and analyse them. Pass the pod IP as $1.
set -euo pipefail
IP="${1:?usage: pull_results.sh <pod-ip>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE=/lambda/nfs/ic-fs-2/repos/wellbeing-in-translation/results

mkdir -p "$ROOT/results"
scp -i ~/.ssh/lambda_id_ed25519 -o StrictHostKeyChecking=accept-new \
    "ubuntu@$IP:$REMOTE/*.jsonl" "$ROOT/results/"

wc -l "$ROOT"/results/*.jsonl
cd "$ROOT" && python3 scripts/analyze.py
