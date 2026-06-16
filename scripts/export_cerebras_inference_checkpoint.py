#!/usr/bin/env python
"""Export a minimal PtychoPINN inference checkpoint for Cerebras experiments.

The normal inference path loads an MLflow-wrapped Lightning model.  That is
fine on CPU/GPU, but it is too much baggage for the Cerebras wafer launcher
environment.  This exporter creates a plain PyTorch bundle containing only the
autoencoder weights and JSON-serializable configs needed for an inference-only
wafer port.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import mlflow
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ptychopinn_torch.inference import load_all_configs  # noqa: E402
from ptychopinn_torch.utils import load_all_configs_from_mlflow  # noqa: E402
from scripts.run_inference import MODEL_IDS, resolve_run_id  # noqa: E402


def dataclass_to_clean_dict(value):
    if not is_dataclass(value):
        raise TypeError(f"Expected dataclass config, got {type(value)}")
    result = {}
    for key, item in asdict(value).items():
        if isinstance(item, torch.Tensor):
            continue
        result[key] = item
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="TP1", help="Dataset key used for default model-key resolution.")
    parser.add_argument("--model-key", default=None, help="Named model from scripts/run_inference.py.")
    parser.add_argument("--run-id", default=None, help="Raw MLflow run id. Overrides --model-key.")
    parser.add_argument("--config", default=None, help="Optional config override JSON.")
    parser.add_argument("--file-index", type=int, default=0)
    parser.add_argument("--mlruns", default="mlruns", help="Relative or absolute MLflow tracking directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("cerebras_bundle"))
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        for key, run_id in MODEL_IDS.items():
            print(f"{key:12s} {run_id}")
        return 0

    run_id, model_name = resolve_run_id(args.model_key, args.run_id, args.dataset)
    tracking_dir = Path(args.mlruns)
    tracking_uri = f"file:{tracking_dir.resolve()}"
    mlflow.set_tracking_uri(tracking_uri)

    if args.config is None:
        configs = load_all_configs_from_mlflow(run_id, tracking_uri)
    else:
        configs = load_all_configs(args.config, args.file_index)
    data_config, model_config, training_config, inference_config, datagen_config = configs

    print(f"Loading MLflow model runs:/{run_id}/model")
    loaded_model = mlflow.pytorch.load_model(f"runs:/{run_id}/model", map_location=torch.device("cpu"))
    loaded_model.eval()

    if not hasattr(loaded_model, "model") or not hasattr(loaded_model.model, "autoencoder"):
        raise TypeError("Expected MLflow model to contain loaded_model.model.autoencoder")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "autoencoder_state.pt"
    config_path = args.output_dir / "configs.json"
    manifest_path = args.output_dir / "manifest.json"

    torch.save(loaded_model.model.autoencoder.state_dict(), state_path)

    clean_configs = {
        "data_config": dataclass_to_clean_dict(data_config),
        "model_config": dataclass_to_clean_dict(model_config),
        "training_config": dataclass_to_clean_dict(training_config),
        "inference_config": dataclass_to_clean_dict(inference_config),
        "datagen_config": dataclass_to_clean_dict(datagen_config),
    }
    config_path.write_text(json.dumps(clean_configs, indent=2, sort_keys=True))

    manifest = {
        "run_id": run_id,
        "model_name": model_name,
        "state_file": state_path.name,
        "config_file": config_path.name,
        "notes": (
            "Inference-only Cerebras bundle. Load configs, instantiate "
            "ptychopinn_torch.model.Autoencoder, load autoencoder_state.pt, "
            "and emit real/imag or amp/phase real tensors."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"Saved state:    {state_path}")
    print(f"Saved configs:  {config_path}")
    print(f"Saved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
