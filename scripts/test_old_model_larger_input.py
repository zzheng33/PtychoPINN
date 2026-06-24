#!/usr/bin/env python
"""Test a released 64x64 PtychoPINN model with larger input data.

This intentionally keeps the old model, old configs, old probe, and old
coordinate tensors unchanged. It resizes only the dataset ``diff3d`` arrays
before PtychoDataset builds its memmap, so the test reflects the larger memmap
and larger CPU/GPU batch tensor caused by a different input image size.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ptychopinn_torch.dataloader import Collate, PtychoDataset


DATASETS = {
    "TP1": "data/TP1",
    "TP2": "data/TP2",
    "IC1": "data/IC1",
    "IC2": "data/IC2",
    "NCM": "data/NCM",
    "FLY1": "data/FLY1",
    "W": "data/W",
    "LFP": "data/LFP",
    "LCLS": "data/LCLS",
}

MODEL_IDS = {
    "PS_TP1": "f637381fd7fe49158bb0ed2e7a28ca45",
    "PS_TP2": "6fb4668f21e44e0b80056f64fdfedf01",
    "PS_IC1": "345aa234e8f34935af11c3ebed167448",
    "PS_IC2": "06822d7239504a93ae0f7a6c4577cdc8",
    "PS_NCM": "0908cd113f774d15802f41e40b3a51e2",
    "PS_FLY1": "3d2ca583357c43baa6ab17519d500355",
    "PS_W": "74ba23396c4042afb1751afe9fa87520",
    "PS_LFP": "1cda8280703748fabba173f747fc4103",
    "Unified": "f6ce8d9583164c84955fc6209c340e04",
}


def resolve_dataset(dataset: str) -> Path:
    path = Path(DATASETS.get(dataset, dataset))
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError("Dataset directory not found: {}".format(path))
    return path


def resolve_run_id(model_key: str | None, run_id: str | None, dataset: str) -> tuple[str, str]:
    if run_id:
        return run_id, run_id
    if model_key is None:
        candidate = "PS_{}".format(dataset)
        model_key = candidate if candidate in MODEL_IDS else "Unified"
    if model_key not in MODEL_IDS:
        raise KeyError("Unknown model key {!r}. Pass --run-id or one of: {}".format(model_key, sorted(MODEL_IDS)))
    return MODEL_IDS[model_key], model_key


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch, "xpu", None) and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    if requested == "xpu" and not (getattr(torch, "xpu", None) and torch.xpu.is_available()):
        raise RuntimeError("XPU requested, but torch.xpu.is_available() is False.")
    return torch.device(requested)


def resolve_local_model_path(run_id: str, mlruns_path: Path) -> Path:
    candidates = sorted(mlruns_path.glob("*/{}/artifacts/model".format(run_id)))
    for candidate in candidates:
        if (candidate / "MLmodel").exists():
            return candidate
    raise FileNotFoundError("Could not find local MLflow model for run_id {} under {}".format(run_id, mlruns_path))


def resize_real_array(array: np.ndarray, resolution: int, chunk_size: int = 512) -> np.ndarray:
    if array.ndim < 2 or array.shape[-2:] == (resolution, resolution):
        return array

    original_dtype = array.dtype
    leading_shape = array.shape[:-2]
    flat = array.reshape((-1, array.shape[-2], array.shape[-1]))
    chunks = []
    for start in range(0, flat.shape[0], chunk_size):
        chunk = torch.from_numpy(np.asarray(flat[start : start + chunk_size], dtype=np.float32)).unsqueeze(1)
        resized = F.interpolate(chunk, size=(resolution, resolution), mode="bilinear", align_corners=False)
        chunks.append(resized.squeeze(1).cpu().numpy())

    resized_flat = np.concatenate(chunks, axis=0)
    resized_array = resized_flat.reshape((*leading_shape, resolution, resolution))
    if np.issubdtype(original_dtype, np.integer):
        return np.rint(np.maximum(resized_array, 0)).astype(original_dtype)
    return resized_array.astype(original_dtype, copy=False)


def prepare_larger_input_dataset(source_dir: Path, prepared_root: Path, resolution: int, remake: bool) -> Path:
    prepared_dir = prepared_root / "N{}".format(resolution) / source_dir.name
    done_marker = prepared_dir / ".prepared"
    if done_marker.exists() and not remake:
        return prepared_dir

    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(source_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError("No .npz files found in {}".format(source_dir))

    for npz_file in npz_files:
        with np.load(npz_file, allow_pickle=True) as data:
            arrays = {}
            for key in data.files:
                value = data[key]
                # Only change the measured diffraction input. Probe, coords,
                # objectGuess, and labels remain exactly as in the old setup.
                arrays[key] = resize_real_array(value, resolution) if key == "diff3d" else value
            np.savez(prepared_dir / npz_file.name, **arrays)

    done_marker.write_text("source={}\nresolution={}\nresized_key=diff3d\n".format(source_dir, resolution), encoding="utf-8")
    return prepared_dir


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="TP1", help="Dataset key or directory path.")
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resolution", type=int, default=128, help="New image H/W for images only.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="cpu")
    parser.add_argument("--mlruns", type=Path, default=Path("mlruns"))
    parser.add_argument("--prepared-root", type=Path, default=Path("/tmp/ptycho_old_model_larger_input_data"))
    parser.add_argument("--memmap-dir", type=Path, default=Path("/tmp/ptycho_old_model_larger_input_memmap"))
    parser.add_argument("--remake-data", action="store_true")
    parser.add_argument("--remake-map", action="store_true")
    args = parser.parse_args()

    source_dataset_dir = resolve_dataset(args.dataset)
    run_id, model_name = resolve_run_id(args.model_key, args.run_id, args.dataset)
    device = resolve_device(args.device)
    prepared_root = args.prepared_root if args.prepared_root.is_absolute() else REPO_ROOT / args.prepared_root

    mlruns_path = args.mlruns if args.mlruns.is_absolute() else REPO_ROOT / args.mlruns
    tracking_uri = "file:{}".format(mlruns_path.resolve())
    mlflow.set_tracking_uri(tracking_uri)
    model_path = resolve_local_model_path(run_id, mlruns_path)

    prepared_dataset_dir = prepare_larger_input_dataset(
        source_dataset_dir,
        prepared_root,
        args.resolution,
        remake=args.remake_data,
    )

    print("source data:  {}".format(source_dataset_dir))
    print("test data:    {}".format(prepared_dataset_dir))
    print("model:        {}".format(model_name))
    print("run_id:       {}".format(run_id))
    print("model_path:   {}".format(model_path))
    print("device:       {}".format(device))
    print("new input N:  {}".format(args.resolution))

    model = mlflow.pytorch.load_model(str(model_path), map_location=device)
    model.eval()
    model.to(device)

    data_config = model.data_config
    model_config = model.model_config
    print("old DataConfig.N:", data_config.N)

    dataset = PtychoDataset(
        str(prepared_dataset_dir),
        model_config,
        data_config,
        data_dir=str(args.memmap_dir),
        remake_map=args.remake_map,
    )
    print("dataset im_shape:", dataset.im_shape)
    print("memmap images shape:", tuple(dataset.mmap_ptycho["images"].shape))
    print("memmap images bytes:", tensor_bytes(dataset.mmap_ptycho["images"]))

    batch_indices = list(range(min(args.batch_size, len(dataset))))
    batch = Collate(device=device)(dataset[batch_indices])
    batch_data, probe, _legacy_scale = batch
    x_new = batch_data["images"]
    positions = batch_data["coords_relative"]
    in_scale = batch_data["rms_scaling_constant"]

    print("batch images shape:", tuple(x_new.shape))
    print("batch images bytes:", tensor_bytes(x_new))
    print("probe shape:     ", tuple(probe.shape))
    print("positions shape: ", tuple(positions.shape))

    try:
        with torch.no_grad():
            output = model.forward_predict(x_new, positions, probe, in_scale)
        print("SUCCESS")
        print("output shape:", tuple(output.shape))
        return 0
    except Exception as exc:
        print("FAILED")
        print("{}: {}".format(type(exc).__name__, exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
