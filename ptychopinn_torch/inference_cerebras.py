"""Inference-only helpers for Cerebras porting experiments.

This module intentionally does not modify the normal GPU inference path.  It
keeps the wafer-facing part as real-valued tensor math by returning either
amplitude/phase or real/imag channels from the trained PtychoPINN autoencoder.
Complex reconstruction and ptychographic assembly can stay on the user node.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import torch
from torch import nn

from ptychopinn_torch.config_params import update_existing_config
from ptychopinn_torch.dataloader import Collate, PtychoDataset, TensorDictDataLoader
from ptychopinn_torch.inference import load_all_configs
from ptychopinn_torch.utils import load_all_configs_from_mlflow


OutputFormat = Literal["realimag", "amp_phase"]


class RealTensorInferenceWrapper(nn.Module):
    """Wrap a trained PtychoPINN Lightning model for real-valued inference.

    The existing ``forward_predict`` returns a complex tensor.  Cerebras
    compilation is much more likely to accept the autoencoder-only path if we
    expose real-valued outputs and leave complex reconstruction to host code.
    """

    def __init__(self, loaded_model: nn.Module, output_format: OutputFormat = "realimag"):
        super().__init__()
        if output_format not in ("realimag", "amp_phase"):
            raise ValueError(f"Unsupported output_format: {output_format}")
        self.loaded_model = loaded_model
        self.output_format = output_format

        if not hasattr(loaded_model, "model"):
            raise TypeError("Expected a loaded PtychoPINN_Lightning model with a .model attribute.")
        core_model = loaded_model.model
        if not hasattr(core_model, "autoencoder") or not hasattr(core_model, "scaler"):
            raise TypeError("Expected loaded_model.model to expose .autoencoder and .scaler.")
        self.core_model = core_model

    def forward(self, x: torch.Tensor, input_scale_factor: torch.Tensor) -> torch.Tensor:
        x = self.core_model.scaler.scale(x, input_scale_factor)
        amp, phase = self.core_model.autoencoder(x)

        if self.output_format == "amp_phase":
            return torch.stack((amp, phase), dim=-1)

        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        return torch.stack((real, imag), dim=-1)


def load_cerebras_inference_model(
    run_id: str,
    *,
    relative_mlflow_path: str = "mlruns",
    config_override_path: str | None = None,
    file_index: int = 0,
    device: str | torch.device = "cpu",
    batch_size: int | None = None,
    output_format: OutputFormat = "realimag",
):
    """Load a trained model and return a real-valued inference wrapper."""

    tracking_uri = f"file:{os.path.abspath(relative_mlflow_path)}"
    mlflow.set_tracking_uri(tracking_uri)

    if config_override_path is None:
        data_config, model_config, training_config, inference_config, datagen_config = (
            load_all_configs_from_mlflow(run_id, tracking_uri)
        )
    else:
        data_config, model_config, training_config, inference_config, datagen_config = load_all_configs(
            config_override_path,
            file_index,
        )

    update_existing_config(inference_config, {"experiment_number": file_index})
    if batch_size is not None:
        update_existing_config(inference_config, {"batch_size": batch_size})

    device = torch.device(device)
    model_uri = f"runs:/{run_id}/model"
    loaded_model = mlflow.pytorch.load_model(model_uri, map_location=device)
    loaded_model.to(device)
    loaded_model.eval()

    wrapper = RealTensorInferenceWrapper(loaded_model, output_format=output_format)
    wrapper.to(device)
    wrapper.eval()

    configs = (data_config, model_config, training_config, inference_config, datagen_config)
    return wrapper, configs


def make_inference_dataset(
    ptycho_files_dir: str | os.PathLike[str],
    model_config,
    data_config,
    *,
    data_dir: str | os.PathLike[str] | None = None,
    remake_map: bool = False,
) -> PtychoDataset:
    if data_dir is None:
        data_dir = "_memmap_cerebras"
    return PtychoDataset(
        str(ptycho_files_dir),
        model_config,
        data_config,
        data_dir=str(data_dir),
        remake_map=remake_map,
    )


def select_experiment_dataset(dataset: PtychoDataset, inference_config) -> PtychoDataset:
    if dataset.n_files > 1:
        return dataset.get_experiment_dataset(inference_config.experiment_number)
    return dataset


def export_patch_predictions(
    model: nn.Module,
    dataset: PtychoDataset,
    configs,
    *,
    output_npz: str | os.PathLike[str],
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> dict[str, object]:
    """Run real-valued patch inference and save outputs for host-side assembly."""

    data_config, _model_config, training_config, inference_config, _datagen_config = configs
    device = torch.device(device)
    ptycho_subset = select_experiment_dataset(dataset, inference_config)

    loader = TensorDictDataLoader(
        ptycho_subset,
        batch_size=inference_config.batch_size,
        num_workers=training_config.num_workers,
        collate_fn=Collate(device=device),
        pin_memory=device.type == "cuda",
        persistent_workers=training_config.num_workers > 0,
    )

    outputs = []
    coords_global = []
    coords_relative = []
    rms_scales = []

    start = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch_data = batch[0]
            x = batch_data["images"].to(device, non_blocking=True)
            in_scale = batch_data["rms_scaling_constant"].to(device, non_blocking=True)
            pred = model(x, in_scale)

            outputs.append(pred.detach().cpu().numpy())
            coords_global.append(batch_data["coords_global"].detach().cpu().numpy())
            coords_relative.append(batch_data["coords_relative"].detach().cpu().numpy())
            rms_scales.append(batch_data["rms_scaling_constant"].detach().cpu().numpy())

    elapsed_s = time.time() - start
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    predictions = np.concatenate(outputs, axis=0) if outputs else np.empty((0,))
    np.savez(
        output_npz,
        predictions=predictions,
        coords_global=np.concatenate(coords_global, axis=0) if coords_global else np.empty((0,)),
        coords_relative=np.concatenate(coords_relative, axis=0) if coords_relative else np.empty((0,)),
        rms_scaling_constant=np.concatenate(rms_scales, axis=0) if rms_scales else np.empty((0,)),
        output_format=getattr(model, "output_format", "unknown"),
        data_N=data_config.N,
        data_C=data_config.C,
        middle_trim=inference_config.middle_trim,
        batch_size=inference_config.batch_size,
    )

    return {
        "output_npz": str(output_npz),
        "num_predictions": int(predictions.shape[0]) if predictions.ndim else 0,
        "prediction_shape": tuple(predictions.shape),
        "elapsed_s": elapsed_s,
    }
