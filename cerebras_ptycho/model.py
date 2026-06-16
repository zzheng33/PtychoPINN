"""Model Zoo wrapper for PtychoPINN autoencoder wafer smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from cerebras.modelzoo.config import ModelConfig
from cerebras_ptycho.modeling import RealImagAutoencoder, namespace_from_dict


class PtychoPINNWaferSmokeModelConfig(ModelConfig):
    name: Literal["ptychopinn"]
    bundle_dir: str = "cerebras_bundle/TP1_PS_TP1"
    require_checkpoint: bool = True

    @property
    def __model_cls__(self):
        return PtychoPINNWaferSmokeModel


class PtychoPINNWaferSmokeModel(nn.Module):
    def __init__(self, config: PtychoPINNWaferSmokeModelConfig):
        if isinstance(config, dict):
            if "model" in config:
                config = config["model"]
            config = PtychoPINNWaferSmokeModelConfig(**config)
        super().__init__()
        bundle_dir = Path(config.bundle_dir)
        with (bundle_dir / "configs.json").open("r") as f:
            bundle_config = json.load(f)

        self.data_config = namespace_from_dict(bundle_config["data_config"])
        self.model_config = namespace_from_dict(bundle_config["model_config"])
        self.model = RealImagAutoencoder(self.model_config, self.data_config)

        state_path = bundle_dir / "autoencoder_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            self.model.autoencoder.load_state_dict(state)
        elif config.require_checkpoint:
            raise FileNotFoundError(f"Missing exported autoencoder checkpoint: {state_path}")

    def forward(self, data):
        images = data["images"]
        scale = data["rms_scaling_constant"]
        pred = self.model(images, scale)
        target = data.get("target")
        if target is None:
            return pred.sum() * 0.0
        return torch.mean((pred - target) ** 2)
