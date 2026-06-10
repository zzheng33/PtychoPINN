#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Edit these defaults for your experiment sweep.
DATASETS=(TP1 TP2 IC1 IC2 NCM FLY1 LFP W LCLS)
BATCH_SIZES=(1 8 16 32 64 128 256 512 1024)
DEVICE="cuda"
VENDOR="auto"
DEVICES=""
INTERVAL="0.2"
WARMUP_SECONDS="0.5"
CONDA_ENV="ptychopinn_torch"
GPU_LABEL=""
OUTPUT_ROOT="power_experiments"
CONTINUE_ON_ERROR="false"

cd "${REPO_ROOT}"

if [[ "$#" -gt 0 ]]; then
  python scripts/run_inference_experiments.py "$@"
  exit 0
fi

CMD=(
  python scripts/run_inference_experiments.py
  --datasets "${DATASETS[@]}"
  --batch-sizes "${BATCH_SIZES[@]}"
  --device "${DEVICE}"
  --vendor "${VENDOR}"
  --interval "${INTERVAL}"
  --warmup-seconds "${WARMUP_SECONDS}"
  --conda-env "${CONDA_ENV}"
  --output-root "${OUTPUT_ROOT}"
)

if [[ -n "${DEVICES}" ]]; then
  CMD+=(--devices "${DEVICES}")
fi

if [[ -n "${GPU_LABEL}" ]]; then
  CMD+=(--gpu-label "${GPU_LABEL}")
fi

if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
  CMD+=(--continue-on-error)
fi

"${CMD[@]}"
