#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PBS_O_WORKDIR:-}" && -f "${PBS_O_WORKDIR}/requirements-aurora.txt" ]]; then
  REPO_ROOT="$(cd "${PBS_O_WORKDIR}" && pwd)"
elif [[ -f "${PWD}/requirements-aurora.txt" ]]; then
  REPO_ROOT="$(pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [[ ! -f "${REPO_ROOT}/requirements-aurora.txt" ]]; then
  echo "Could not locate PtychoPINN repo root. Run from the repo root or set PBS_O_WORKDIR correctly." >&2
  echo "Resolved REPO_ROOT=${REPO_ROOT}" >&2
  exit 1
fi
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/../ptychopinn-venvs/aurora}"

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
python -m venv --system-site-packages "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-aurora.txt
python -m pip install -e .

python - <<'PY'
import importlib
import torch

mods = [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "h5py",
    "mlflow",
    "lightning",
    "tensordict",
    "skimage",
    "cv2",
    "noise",
    "perlin_noise",
    "ptychopinn_torch",
]
for name in mods:
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'ok')}")
print(f"torch xpu available: {getattr(torch, 'xpu', None) is not None and torch.xpu.is_available()}")
if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
    print(f"torch xpu device count: {torch.xpu.device_count()}")
    if hasattr(torch.xpu, "get_device_name"):
        print(f"torch xpu device 0: {torch.xpu.get_device_name(0)}")
PY

echo "Aurora PtychoPINN environment is ready: ${VENV_DIR}"
