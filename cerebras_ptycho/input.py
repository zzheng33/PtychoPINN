"""Synthetic input for PtychoPINN Cerebras wafer smoke tests."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


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


class PtychoSyntheticDataProcessor:
    def __init__(self, params):
        self.params = params or {}

    def create_dataloader(self):
        dataset = PtychoSyntheticDataset(
            num_samples=self.params.get("num_samples", 8),
            channels=self.params.get("channels", 4),
            image_size=self.params.get("image_size", 64),
        )
        return DataLoader(
            dataset,
            batch_size=self.params.get("batch_size", 1),
            shuffle=False,
            drop_last=True,
            num_workers=0,
        )
