#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/../ptychopinn-venvs/aurora}"

DATASETS=(TP1 TP2 IC1 IC2 NCM FLY1 LFP W LCLS)
BATCH_SIZES=(32 64 128 256 512 1024)

DEVICE="${DEVICE:-xpu}"
VENDOR="${VENDOR:-intel}"
DEVICES="${DEVICES:-0}"
INTERVAL="${INTERVAL:-0.2}"
WARMUP_SECONDS="${WARMUP_SECONDS:-0.5}"
GPU_LABEL="${GPU_LABEL:-Max}"
OUTPUT_ROOT="${OUTPUT_ROOT:-power_experiments}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"
TEST="${TEST:-false}"

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
fi

module load gcc/13.4.0
module load python/3.12.12
module load py-pip/25.1.1
module load py-numpy/2.3.4
module load py-scipy/1.16.3
module load py-h5py/3.14.0
module load py-matplotlib/3.10.7
module load py-pandas/2.3.3
module load py-torch/2.10.0
module load py-torchvision/0.25.0
module load py-torchaudio/2.10.0
module load xpu-smi/1.3.5

cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export MLFLOW_ALLOW_FILE_STORE=true
export ZE_AFFINITY_MASK="${DEVICES}"

python - <<'PY'
import torch
if not (getattr(torch, "xpu", None) and torch.xpu.is_available()):
    raise SystemExit("Aurora PyTorch XPU is not available. Check loaded modules and allocation.")
if hasattr(torch.xpu, "get_device_name"):
    print(f"Using XPU device 0: {torch.xpu.get_device_name(0)}")
else:
    print("Using XPU device 0")
PY

CMD=(
  python scripts/run_inference_experiments.py
  --datasets "${DATASETS[@]}"
  --batch-sizes "${BATCH_SIZES[@]}"
  --device "${DEVICE}"
  --vendor "${VENDOR}"
  --interval "${INTERVAL}"
  --warmup-seconds "${WARMUP_SECONDS}"
  --output-root "${OUTPUT_ROOT}"
  --devices "${DEVICES}"
  --gpu-label "${GPU_LABEL}"
)

if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
  CMD+=(--continue-on-error)
fi

if [[ "${TEST}" == "true" ]]; then
  CMD+=(--test)
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

"${CMD[@]}"
