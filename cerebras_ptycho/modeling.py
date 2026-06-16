"""Standalone real-valued PtychoPINN autoencoder for Cerebras smoke jobs."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


def namespace_from_dict(values: dict) -> SimpleNamespace:
    return SimpleNamespace(**values)


class TanhCustomAct(nn.Module):
    def forward(self, x):
        return math.pi * torch.tanh(x)


class AmplitudeActivation(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.activation_type = getattr(model_config, "amp_activation", "silu")

    def forward(self, x):
        if self.activation_type == "sigmoid":
            return torch.sigmoid(x)
        return F.silu(x)


class ConvBaseBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w1=3,
        w2=3,
        padding="same",
        activation="relu",
        batch_norm=False,
    ):
        super().__init__()
        padding_size = w1 // 2 if padding == "same" else 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, (w1, w2), padding=padding_size)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (w1, w2), padding=padding_size)
        self.activation = getattr(F, activation) if activation else None
        self.batch_norm = batch_norm
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else None

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        return self.activation(x) if self.activation else F.relu(x)


class ConvPoolBlock(ConvBaseBlock):
    def __init__(self, in_channels, out_channels, model_config, p1=2, p2=2, batch_norm=False):
        super().__init__(in_channels, out_channels, batch_norm=batch_norm)
        if any(
            getattr(model_config, name, False)
            for name in ("cbam_encoder", "eca_encoder")
        ):
            raise ValueError("Cerebras smoke adapter does not include attention blocks.")
        self.pool = nn.MaxPool2d(kernel_size=(p1, p2), padding=0)

    def forward(self, x):
        return self.pool(super().forward(x))


class ConvUpBlock(ConvBaseBlock):
    def __init__(self, in_channels, out_channels, p1=2, p2=2, batch_norm=False):
        super().__init__(in_channels, out_channels, batch_norm=batch_norm)
        self.up = nn.Upsample(scale_factor=(p1, p2), mode="nearest")

    def forward(self, x):
        return self.up(super().forward(x))


class Encoder(nn.Module):
    def __init__(self, model_config, data_config):
        super().__init__()
        self.N = data_config.N
        n_filters_scale = model_config.n_filters_scale
        self.filters = [model_config.C_model if model_config.object_big else 1]
        if self.N == 64:
            self.filters += [n_filters_scale * 32, n_filters_scale * 64, n_filters_scale * 128]
        elif self.N == 128:
            self.filters += [
                n_filters_scale * 16,
                n_filters_scale * 32,
                n_filters_scale * 64,
                n_filters_scale * 128,
            ]
        elif self.N == 256:
            self.filters += [
                n_filters_scale * 8,
                n_filters_scale * 16,
                n_filters_scale * 32,
                n_filters_scale * 64,
                n_filters_scale * 128,
            ]
        else:
            raise ValueError(f"Unsupported input size: {self.N}")

        self.blocks = nn.ModuleList(
            [
                ConvPoolBlock(
                    self.filters[i - 1],
                    self.filters[i],
                    model_config,
                    batch_norm=model_config.batch_norm,
                )
                for i in range(1, len(self.filters))
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class DecoderFilters(nn.Module):
    def __init__(self, model_config, data_config):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.n_filters_scale = model_config.n_filters_scale
        self.N = data_config.N
        self.filters = [self.n_filters_scale * 128]
        if self.N == 64:
            self.filters += [self.n_filters_scale * 64, self.n_filters_scale * 32]
        elif self.N == 128:
            self.filters += [
                self.n_filters_scale * 128,
                self.n_filters_scale * 64,
                self.n_filters_scale * 32,
            ]
        elif self.N == 256:
            self.filters += [
                self.n_filters_scale * 256,
                self.n_filters_scale * 128,
                self.n_filters_scale * 64,
                self.n_filters_scale * 32,
            ]
        else:
            raise ValueError(f"Unsupported input size: {self.N}")


class DecoderBase(DecoderFilters):
    def __init__(self, model_config, data_config, batch_norm=False):
        super().__init__(model_config, data_config)
        if any(
            getattr(model_config, name, False)
            for name in ("cbam_decoder", "eca_decoder", "spatial_decoder")
        ):
            raise ValueError("Cerebras smoke adapter does not include attention blocks.")
        self.blocks = nn.ModuleList(
            [
                ConvUpBlock(self.filters[i - 1], self.filters[i], batch_norm=batch_norm)
                for i in range(1, len(self.filters))
            ]
        )


class DecoderLast(nn.Module):
    def __init__(self, model_config, data_config, in_channels, out_channels, activation, batch_norm=False):
        super().__init__()
        self.model_config = model_config
        self.N = data_config.N
        n_filters_scale = model_config.n_filters_scale
        c_outer_fraction = max(0.0, min(0.5, getattr(model_config, "decoder_last_c_outer_fraction", 0.25)))
        self.c_outer = max(1, int(in_channels * c_outer_fraction))
        self.conv1 = nn.Conv2d(in_channels - self.c_outer, out_channels, (3, 3), padding=1)
        self.conv_up_block = ConvUpBlock(self.c_outer, n_filters_scale * 32, batch_norm=batch_norm)
        self.conv2 = nn.Conv2d(n_filters_scale * 32, out_channels, (3, 3), padding=1)
        self.batch_norm = batch_norm
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation
        self.padding = nn.ConstantPad2d((self.N // 4, self.N // 4, self.N // 4, self.N // 4), 0)

    def forward(self, x):
        x1 = self.conv1(x[:, :-self.c_outer])
        if self.batch_norm:
            x1 = self.bn1(x1)
        x1 = self.padding(self.activation(x1))
        if not self.model_config.probe_big:
            return x1
        x2 = self.conv_up_block(x[:, -self.c_outer :])
        x2 = self.conv2(x2)
        if self.batch_norm:
            x2 = self.bn2(x2)
        return x1 + F.silu(x2)


class DecoderPhase(DecoderBase):
    def __init__(self, model_config, data_config):
        super().__init__(model_config, data_config, batch_norm=model_config.batch_norm)
        num_channels = model_config.C_model if model_config.object_big else 1
        self.phase_activation = TanhCustomAct()
        self.phase = DecoderLast(
            model_config,
            data_config,
            self.n_filters_scale * 32,
            num_channels,
            self.phase_activation,
            batch_norm=model_config.batch_norm,
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.phase(x)


class DecoderAmp(DecoderBase):
    def __init__(self, model_config, data_config):
        super().__init__(model_config, data_config, batch_norm=False)
        num_channels = model_config.decoder_last_amp_channels
        self.amp_activation = AmplitudeActivation(model_config)
        self.amp = DecoderLast(
            model_config,
            data_config,
            self.n_filters_scale * 32,
            num_channels,
            self.amp_activation,
            batch_norm=False,
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.amp(x)


class Autoencoder(nn.Module):
    def __init__(self, model_config, data_config):
        super().__init__()
        if getattr(model_config, "cbam_bottleneck", False):
            raise ValueError("Cerebras smoke adapter does not include attention blocks.")
        self.encoder = Encoder(model_config, data_config)
        self.bottleneck_cbam = nn.Identity()
        self.decoder_amp = DecoderAmp(model_config, data_config)
        self.decoder_phase = DecoderPhase(model_config, data_config)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck_cbam(x)
        return self.decoder_amp(x), self.decoder_phase(x)


class RealImagAutoencoder(nn.Module):
    def __init__(self, model_config, data_config):
        super().__init__()
        self.autoencoder = Autoencoder(model_config, data_config)

    def forward(self, images, rms_scaling_constant):
        amp, phase = self.autoencoder(images * rms_scaling_constant)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        return torch.stack((real, imag), dim=-1)
