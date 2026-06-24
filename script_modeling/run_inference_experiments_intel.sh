#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PBS_O_WORKDIR:-}" && -f "${PBS_O_WORKDIR}/script_modeling/run_inference_experiments.py" ]]; then
  REPO_ROOT="$(cd "${PBS_O_WORKDIR}" && pwd)"
elif [[ -f "${PWD}/script_modeling/run_inference_experiments.py" ]]; then
  REPO_ROOT="$(pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [[ ! -f "${REPO_ROOT}/script_modeling/run_inference_experiments.py" ]]; then
  echo "Could not locate PtychoPINN repo root. Submit from the repo root or set PBS_O_WORKDIR correctly." >&2
  echo "Resolved REPO_ROOT=${REPO_ROOT}" >&2
  exit 1
fi

export DEVICE="${DEVICE:-xpu}"
export VENDOR="${VENDOR:-intel}"
export GPU_LABEL="${GPU_LABEL:-Max}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/modeling_exp}"
export VENV_DIR="${VENV_DIR:-${REPO_ROOT}/../ptychopinn-venvs/aurora}"

if [[ -z "${MEMMAP_ROOT:-}" ]]; then
  LOCAL_TMP="${TMPDIR:-${PBS_TMPDIR:-/tmp}}"
  export MEMMAP_ROOT="${LOCAL_TMP%/}/ptychopinn_memmap_${PBS_JOBID:-$$}"
fi
mkdir -p "${MEMMAP_ROOT}"
echo "Using TensorDict memmap root: ${MEMMAP_ROOT}"

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load gcc/13.4.0 || true
  module load python/3.12.12 || true
  module load py-pip/25.1.1 || true
  module load py-numpy/2.3.4 || true
  module load py-scipy/1.16.3 || true
  module load py-h5py/3.14.0 || true
  module load py-matplotlib/3.10.7 || true
  module load py-pandas/2.3.3 || true
  module load py-torch/2.10.0 || true
  module load py-torchvision/0.25.0 || true
  module load py-torchaudio/2.10.0 || true
  module load xpu-smi/1.3.5 || true
fi

cd "${REPO_ROOT}"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

export MLFLOW_ALLOW_FILE_STORE=true
export ZE_AFFINITY_MASK="${DEVICES:-0}"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

exec "${SCRIPT_DIR}/run_inference_experiments.sh" "$@"
