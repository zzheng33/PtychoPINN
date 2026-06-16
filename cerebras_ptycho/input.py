"""Synthetic input for PtychoPINN Cerebras wafer smoke tests."""

from __future__ import annotations

from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset

from cerebras.modelzoo.config import DataConfig


class PtychoSyntheticDataset(Dataset):
    def __init__(self, num_samples=8, channels=4, image_size=64):
        self.num_samples = int(num_samples)
        self.channels = int(channels)
        self.image_size = int(image_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.ones(self.channels, self.image_size, self.image_size, dtype=torch.float32)
        image = image * (1.0 + float(idx % 7) * 0.01)
        scale = torch.ones(1, 1, 1, dtype=torch.float32)
        target = torch.zeros(self.channels, self.image_size, self.image_size, 2, dtype=torch.float32)
        return {
            "images": image,
            "rms_scaling_constant": scale,
            "target": target,
        }


class PtychoSyntheticDataProcessorConfig(DataConfig):
    data_processor: Literal["PtychoSyntheticDataProcessor"]
    batch_size: int = 1
    num_samples: int = 2
    channels: int = 4
    image_size: int = 64


class PtychoSyntheticDataProcessor:
    def __init__(self, config: PtychoSyntheticDataProcessorConfig):
        if isinstance(config, dict):
            if "params" in config:
                config = config["params"]
            config = PtychoSyntheticDataProcessorConfig(**config)
        self.config = config

    def create_dataloader(self):
        dataset = PtychoSyntheticDataset(
            num_samples=self.config.num_samples,
            channels=self.config.channels,
            image_size=self.config.image_size,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
        )
