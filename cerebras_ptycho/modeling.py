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


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=("avg", "max")):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                pooled = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == "max":
                pooled = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            else:
                continue
            raw = self.mlp(pooled)
            channel_att_sum = raw if channel_att_sum is None else channel_att_sum + raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.compress = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

    def forward(self, x):
        compressed = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        return x * torch.sigmoid(self.compress(compressed))


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=("avg", "max"), spatial_kernel_size=7):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(kernel_size=spatial_kernel_size)

    def forward(self, x):
        return self.SpatialGate(self.ChannelGate(x))


class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)


class BasicSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.spatial_gate = SpatialGate(kernel_size=kernel_size)

    def forward(self, x):
        return self.spatial_gate(x)


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
        self.use_cbam = model_config.cbam_encoder
        self.use_eca = model_config.eca_encoder
        if self.use_cbam:
            self.attention = CBAM(gate_channels=out_channels)
        elif self.use_eca:
            self.attention = ECALayer(out_channels)
        else:
            self.attention = nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=(p1, p2), padding=0)

    def forward(self, x):
        x_new = super().forward(x)
        if self.use_cbam:
            x_new = self.attention(x_new)
        return self.pool(x_new)


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
        self.use_cbam = model_config.cbam_decoder
        self.use_eca = model_config.eca_decoder
        self.use_spatial = model_config.spatial_decoder
        self.spatial_kernel = model_config.decoder_spatial_kernel
        self.blocks = nn.ModuleList(
            [
                ConvUpBlock(self.filters[i - 1], self.filters[i], batch_norm=batch_norm)
                for i in range(1, len(self.filters))
            ]
        )
        self.attention_blocks = nn.ModuleList()
        for i in range(1, len(self.filters)):
            out_ch = self.filters[i]
            if self.use_eca:
                self.attention_blocks.append(ECALayer(channel=out_ch))
            elif self.use_spatial:
                self.attention_blocks.append(BasicSpatialAttention(kernel_size=self.spatial_kernel))
            elif self.use_cbam:
                self.attention_blocks.append(CBAM(gate_channels=out_ch))
            else:
                self.attention_blocks.append(nn.Identity())


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
        self.encoder = Encoder(model_config, data_config)
        if getattr(model_config, "cbam_bottleneck", False):
            self.bottleneck_cbam = CBAM(gate_channels=self.encoder.filters[-1])
        else:
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
