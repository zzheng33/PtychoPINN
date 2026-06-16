#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-ptychopinn_torch}"
PYTHON_BIN="${PYTHON_BIN:-}"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniforge3}"

MODULE_PATH="${MODULE_PATH:-/soft/modulefiles}"
CONDA_MODULE="${CONDA_MODULE:-}"
CUDA_MODULE="${CUDA_MODULE:-}"

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
fi

if command -v module >/dev/null 2>&1; then
  module use "${MODULE_PATH}" >/dev/null 2>&1 || true
  if [[ -n "${CUDA_MODULE}" ]]; then
    module load "${CUDA_MODULE}"
  fi
  if [[ -n "${CONDA_MODULE}" ]]; then
    module load "${CONDA_MODULE}"
  fi
fi

cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    set +u
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    set -u
  elif [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    set -u
  fi
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

if ! "${PYTHON_BIN}" -c 'import numpy' >/dev/null 2>&1; then
  echo "Selected Python does not have numpy: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN to the inference env Python or set CONDA_ENV to the correct conda env." >&2
  exit 1
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" scripts/benchmark_data_loading.py "$@"
