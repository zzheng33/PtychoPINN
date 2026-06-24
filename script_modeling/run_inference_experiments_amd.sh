#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DEVICE="${DEVICE:-cuda}"
export VENDOR="${VENDOR:-amd}"
export CONDA_ENV="${CONDA_ENV:-ptychopinn_torch_rocm}"
export GPU_LABEL="${GPU_LABEL:-MI300A}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/modeling_exp}"

MODULE_PATH="${MODULE_PATH:-/soft/modulefiles}"
ROCM_MODULE="${ROCM_MODULE:-rocm/7.0.2}"

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  [[ -d "${MODULE_PATH}" ]] && module use "${MODULE_PATH}"
  module load "${ROCM_MODULE}" || true
fi

exec "${SCRIPT_DIR}/run_inference_experiments.sh" "$@"
