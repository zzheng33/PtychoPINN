#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET="${1:-TP1}"
DEVICE="${2:-cuda}"
MODEL_KEY="${3:-}"
BATCH_SIZE="${4:-}"

cd "$REPO_ROOT"

CMD=(
  conda run -n ptychopinn_torch
  python scripts/run_inference.py
  --dataset "$DATASET"
  --device "$DEVICE"
)

if [[ -n "$MODEL_KEY" ]]; then
  CMD+=(--model-key "$MODEL_KEY")
fi

if [[ -n "$BATCH_SIZE" ]]; then
  CMD+=(--batch-size "$BATCH_SIZE")
fi

"${CMD[@]}"
