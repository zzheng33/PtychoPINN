#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# Edit these defaults for your experiment sweep.
DATASETS=(TP1 TP2 IC1 IC2 NCM FLY1 LFP W LCLS)
# DATASETS=(IC2)
BATCH_SIZES=(32 64 128 256 512 1024)
# BATCH_SIZES=(1024)
DEVICE="${DEVICE:-cuda}"
VENDOR="${VENDOR:-auto}"
DEVICES="${DEVICES:-0}"
INTERVAL="${INTERVAL:-0.2}"
WARMUP_SECONDS="${WARMUP_SECONDS:-0.5}"
CONDA_ENV="${CONDA_ENV:-ptychopinn_torch}"
PYTHON_BIN="${PYTHON_BIN:-}"
GPU_LABEL="${GPU_LABEL:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-power_experiments}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"
TEST="${TEST:-false}"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniforge3}"

cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  set -u
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No Python executable found. Activate the correct env or set PYTHON_BIN." >&2
    exit 1
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  echo "Activate the correct env or set PYTHON_BIN." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1; then
  echo "Python executable is not usable or is too old: ${PYTHON_BIN}" >&2
  echo "On Grace/ARM systems, use a Python/env built for that machine." >&2
  echo "Example: PYTHON_BIN=/path/to/arm/env/bin/python ./scripts/run_inference_experiments.sh" >&2
  exit 1
fi

echo "Using Python: ${PYTHON_BIN}"

CMD=(
  "${PYTHON_BIN}" scripts/run_inference_experiments.py
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

if [[ "${TEST}" == "true" ]]; then
  CMD+=(--test)
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

"${CMD[@]}"
