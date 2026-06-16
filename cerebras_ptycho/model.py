"""Model Zoo wrapper for PtychoPINN autoencoder wafer smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from cerebras_ptycho.modeling import RealImagAutoencoder, namespace_from_dict


def _config_get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class PtychoPINNWaferSmokeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        bundle_dir = Path(_config_get(config, "bundle_dir", "cerebras_bundle/TP1_PS_TP1"))
        with (bundle_dir / "configs.json").open("r") as f:
            bundle_config = json.load(f)

        self.data_config = namespace_from_dict(bundle_config["data_config"])
        self.model_config = namespace_from_dict(bundle_config["model_config"])
        self.model = RealImagAutoencoder(self.model_config, self.data_config)

        state_path = bundle_dir / "autoencoder_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            self.model.autoencoder.load_state_dict(state)
        elif _config_get(config, "require_checkpoint", False):
            raise FileNotFoundError(f"Missing exported autoencoder checkpoint: {state_path}")

    def forward(self, data):
        images = data["images"]
        scale = data["rms_scaling_constant"]
        pred = self.model(images, scale)
        target = data.get("target")
        if target is None:
            return pred.sum() * 0.0
        return torch.mean((pred - target) ** 2)
