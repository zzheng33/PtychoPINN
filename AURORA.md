# PtychoPINN on Aurora

Aurora provides the PyTorch/XPU stack as modules. Use those modules and install only the Python packages that are missing from the module list.

## One-time environment setup

From the repository root on Aurora:

```bash
bash scripts/setup_aurora_env.sh
```

This creates `../ptychopinn-venvs/aurora` with `--system-site-packages`, so it can see the site modules. Set `VENV_DIR=/path/to/venv` before running the scripts if you want a different location.

- `python/3.12.12`
- `py-torch/2.10.0`
- `py-torchvision/0.25.0`
- `py-torchaudio/2.10.0`
- `py-numpy`, `py-scipy`, `py-h5py`, `py-matplotlib`, `py-pandas`
- `xpu-smi`

The venv adds `lightning`, `mlflow`, `tensordict`, `scikit-image`, `opencv-python-headless`, `noise`, `perlin-noise`, and an editable install of this repo.

## Data and MLflow artifacts

The repo expects:

```text
data/
mlruns/
```

If you copied the Zenodo tarballs, unpack them at the repository root. Then run:

```bash
source ../ptychopinn-venvs/aurora/bin/activate
python other/initialize_data.py --repo-root "$PWD" --no-dry-run
```

## Quick inference smoke test

Inside a compute allocation:

```bash
bash scripts/run_inference_experiments_aurora.sh --datasets TP1 --batch-sizes 32 --test
```

For a larger sweep, set shell variables before launching:

```bash
DATASETS="TP1 TP2 IC1 IC2 NCM FLY1 LFP W LCLS" \
BATCH_SIZES="32 64 128 256 512 1024" \
DEVICES=0 \
bash scripts/run_inference_experiments_aurora.sh
```

The Aurora wrapper uses `--device xpu`, `--vendor intel`, and `ZE_AFFINITY_MASK` for device selection.
