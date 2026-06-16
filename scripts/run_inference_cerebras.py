#!/usr/bin/env python
"""Inference-only Cerebras porting entry point for PtychoPINN.

This script is a copied/variant path.  It does not replace the normal GPU
inference workflow.  The current purpose is to validate the real-valued neural
network forward path and export patch predictions that can later be assembled on
the user node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ptychopinn_torch.inference_cerebras import (  # noqa: E402
    export_patch_predictions,
    load_cerebras_inference_model,
    make_inference_dataset,
)
from scripts.run_inference import DATASETS, MODEL_IDS, resolve_dataset, resolve_run_id  # noqa: E402


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="TP1", help="Dataset key or directory path.")
    parser.add_argument("--model-key", default=None, help="Named model from scripts/run_inference.py.")
    parser.add_argument("--run-id", default=None, help="Raw MLflow run id. Overrides --model-key.")
    parser.add_argument("--config", default=None, help="Optional config override JSON.")
    parser.add_argument("--file-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--output-format", choices=("realimag", "amp_phase"), default="realimag")
    parser.add_argument("--output-npz", type=Path, default=Path("cerebras_outputs/patch_predictions.npz"))
    parser.add_argument("--memmap-dir", type=Path, default=Path("_memmap_cerebras"))
    parser.add_argument("--remake-map", action="store_true")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help="Limit batches for smoke testing. Use 0 to export all batches.",
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-datasets", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        for key, run_id in MODEL_IDS.items():
            print(f"{key:12s} {run_id}")
        return 0

    if args.list_datasets:
        for key, path in DATASETS.items():
            print(f"{key:8s} {path}")
        return 0

    dataset_dir = resolve_dataset(args.dataset)
    run_id, model_name = resolve_run_id(args.model_key, args.run_id, args.dataset)
    device = resolve_device(args.device)
    max_batches = None if args.max_batches == 0 else args.max_batches

    print("Cerebras inference variant")
    print(f"dataset:       {args.dataset} -> {dataset_dir}")
    print(f"model:         {model_name}")
    print(f"run_id:        {run_id}")
    print(f"device:        {device}")
    print(f"output_format: {args.output_format}")
    print(f"output_npz:    {args.output_npz}")
    print(f"max_batches:   {'all' if max_batches is None else max_batches}")

    model, configs = load_cerebras_inference_model(
        run_id,
        config_override_path=args.config,
        file_index=args.file_index,
        device=device,
        batch_size=args.batch_size,
        output_format=args.output_format,
    )
    data_config, model_config, _training_config, _inference_config, _datagen_config = configs
    dataset = make_inference_dataset(
        dataset_dir,
        model_config,
        data_config,
        data_dir=args.memmap_dir,
        remake_map=args.remake_map,
    )
    stats = export_patch_predictions(
        model,
        dataset,
        configs,
        output_npz=args.output_npz,
        device=device,
        max_batches=max_batches,
    )

    print(f"saved:         {stats['output_npz']}")
    print(f"predictions:   {stats['num_predictions']}")
    print(f"shape:         {stats['prediction_shape']}")
    print(f"elapsed_s:     {stats['elapsed_s']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
