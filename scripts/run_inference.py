#!/usr/bin/env python
"""Run PtychoPINN inference with the released MLflow models."""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ptychopinn_torch.inference import load_and_predict


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
    # Dead Leaves synthetic, Figure 2 single-experiment transfer.
    "PS_TP1": "f637381fd7fe49158bb0ed2e7a28ca45",
    "PS_TP2": "6fb4668f21e44e0b80056f64fdfedf01",
    "PS_IC1": "345aa234e8f34935af11c3ebed167448",
    "PS_IC2": "06822d7239504a93ae0f7a6c4577cdc8",
    "PS_NCM": "0908cd113f774d15802f41e40b3a51e2",
    "PS_FLY1": "3d2ca583357c43baa6ab17519d500355",
    "PS_W": "74ba23396c4042afb1751afe9fa87520",
    "PS_LFP": "1cda8280703748fabba173f747fc4103",
    # Experiment-only, Figure 2 single-experiment transfer.
    "PE_TP1": "c86ba4cc6d424a8fb1370bcfa87d967c",
    "PE_TP2": "3dcc4ce0423c46f6bab529294886f453",
    "PE_IC1": "b1f8f06f9dee41e48e0323d295e0a5d3",
    "PE_IC2": "e09fa3d8b48e406aa7c9ff78e34f7782",
    "PE_NCM": "293d4107954d4d11832fe979e6045229",
    "PE_FLY1": "4911939f91d147348f450ec1d78811dd",
    "PE_W": "3360765d399443d0a758e9667c8455b5",
    "PE_LFP": "aee5ab755d8e4e558c1f328491adb0fb",
    # Multi-experiment models.
    "Single_W": "bec5909f8c7d407dbdcdc24357495239",
    "Single_LFP": "e4a5907f060c4af1adb9005a4fc1c51e",
    "Single_FLY1": "9c87bda4384f42a9ad8f7fa0bcc6ea5b",
    "Single_IC2": "5c668cbdb3244e92acd8cb1d682234df",
    "W_LFP": "d3dc54d3897941e89bc16c85b82fba44",
    "W_FLY1": "cb2e664b3001486ba1047936d0b4a533",
    "W_IC2": "92f5a0518ea641adb8a75e02242c390e",
    "LFP_FLY1": "f7c72e4058994eea9e9c43ffe7e3b278",
    "LFP_IC2": "f7c72e4058994eea9e9c43ffe7e3b278",
    "FLY1_IC2": "40f07429768142c5b4d7b57f5c072862",
    "Unified": "f6ce8d9583164c84955fc6209c340e04",
}


def resolve_dataset(dataset: str) -> Path:
    path = Path(DATASETS.get(dataset, dataset))
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    return path


def resolve_run_id(model_key: str | None, run_id: str | None, dataset: str) -> tuple[str, str]:
    if run_id:
        return run_id, run_id

    if model_key is None:
        candidate = f"PS_{dataset}"
        model_key = candidate if candidate in MODEL_IDS else "Unified"

    if model_key not in MODEL_IDS:
        raise KeyError(
            f"Unknown model key {model_key!r}. Use --list-models to show available keys, "
            "or pass --run-id directly."
        )
    return MODEL_IDS[model_key], model_key


def cleanup_inference_memmap(output_dir: Path, memmap_dir: Path | None = None) -> None:
    memmap_dir = memmap_dir or output_dir / "_memmap"
    state_file = memmap_dir.parent / f"{memmap_dir.name}_state_files.npz"

    gc.collect()
    if memmap_dir.exists():
        shutil.rmtree(memmap_dir)
        print(f"Deleted memmap cache: {memmap_dir}")
    if state_file.exists():
        state_file.unlink()
        print(f"Deleted memmap state: {state_file}")


def xpu_is_available() -> bool:
    return bool(getattr(torch, "xpu", None) and torch.xpu.is_available())


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        if xpu_is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
    if requested_device == "xpu" and not xpu_is_available():
        raise RuntimeError("XPU was requested with --device xpu, but torch.xpu.is_available() is False.")
    return requested_device


def print_device_info(device: str) -> None:
    if device == "cuda":
        torch.cuda.set_device(0)
        print(f"torch visible CUDA GPUs: {torch.cuda.device_count()}")
        print(f"torch current CUDA GPU:  {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")
    elif device == "xpu":
        torch.xpu.set_device(0)
        device_name = torch.xpu.get_device_name(0) if hasattr(torch.xpu, "get_device_name") else "xpu:0"
        print(f"torch visible XPU GPUs: {torch.xpu.device_count()}")
        print(f"torch current XPU GPU:  {torch.xpu.current_device()} ({device_name})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="TP1", help="Dataset key or directory path.")
    parser.add_argument("--model-key", default=None, help="Named model from recreate_results.ipynb.")
    parser.add_argument("--run-id", default=None, help="Raw MLflow run id. Overrides --model-key.")
    parser.add_argument("--config", default=None, help="Optional config override JSON.")
    parser.add_argument("--file-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size override.")
    parser.add_argument("--output-dir", type=Path, default=Path("inference_outputs"))
    parser.add_argument(
        "--memmap-dir",
        type=Path,
        default=None,
        help="Reusable memmap cache directory. Defaults to a sibling _memmap next to output-dir.",
    )
    parser.add_argument(
        "--remake-map",
        action="store_true",
        help="Force recreation of the memmap cache before inference.",
    )
    parser.add_argument(
        "--cleanup-memmap",
        action="store_true",
        help="Delete the generated _memmap cache after saving inference outputs.",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Latency-only mode: skip comparison plot and reconstruction .npz output.",
    )
    parser.add_argument("--power-output", type=Path, default=None)
    parser.add_argument("--power-vendor", choices=("auto", "nvidia", "amd", "intel"), default="auto")
    parser.add_argument("--power-devices", default=None)
    parser.add_argument("--power-interval", type=float, default=0.2)
    parser.add_argument("--power-label", default="")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xpu"),
        default="auto",
        help="Inference device. auto uses XPU, then CUDA, then CPU.",
    )
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        for key, run_id in MODEL_IDS.items():
            print(f"{key:12s} {run_id}")
        return

    dataset_dir = resolve_dataset(args.dataset)
    run_id, model_name = resolve_run_id(args.model_key, args.run_id, args.dataset)
    device = resolve_device(args.device)
    print_device_info(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"{args.dataset}_{model_name}_comparison.svg"
    npz_path = args.output_dir / f"{args.dataset}_{model_name}_recon.npz"

    print(f"dataset:   {args.dataset} -> {dataset_dir}")
    print(f"model:     {model_name}")
    print(f"run_id:    {run_id}")
    print(f"device:    {device}")
    print(f"output:    {npz_path}")

    result = load_and_predict(
        run_id,
        str(dataset_dir),
        relative_mlflow_path="mlruns",
        config_override_path=args.config,
        file_index=args.file_index,
        save_dir=str(args.output_dir),
        plot_name=plot_name,
        verbose=True,
        device=device,
        batch_size=args.batch_size,
        data_dir=str(args.memmap_dir) if args.memmap_dir is not None else None,
        remake_map=args.remake_map,
        save_outputs=not args.skip_save,
        power_output=args.power_output,
        power_vendor=args.power_vendor,
        power_devices=args.power_devices,
        power_interval=args.power_interval,
        power_label=args.power_label,
    )

    if not args.skip_save:
        result_cpu = result.detach().cpu().numpy()
        if result_cpu.ndim == 3 and result_cpu.shape[0] == 1:
            result_cpu = result_cpu[0]
        np.savez(
            npz_path,
            object=result_cpu,
            dataset=args.dataset,
            dataset_dir=str(dataset_dir),
            model_key=model_name,
            run_id=run_id,
        )
        print(f"Saved reconstruction: {npz_path}")
    else:
        print("Latency-only mode: skipped plot and reconstruction save.")
    if args.cleanup_memmap:
        cleanup_inference_memmap(args.output_dir, args.memmap_dir)


if __name__ == "__main__":
    main()
