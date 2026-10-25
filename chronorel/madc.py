from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ops import resize_map, validity_resample
from .wavelets import WaveletLevel


class WindowedAmplitude(nn.Module):
    def __init__(self, channels: int, condition_channels: int, hidden_dim: int,
                 window_size: int = 16, stride: int = 8, padding: int = 8,
                 amplitude_bound: float = .1):
        super().__init__()
        if window_size < 2 or not 0 < stride < window_size or padding < stride:
            raise ValueError('require window_size > stride > 0 and padding >= stride')
        if amplitude_bound <= 0:
            raise ValueError('amplitude_bound must be positive')
        self.channels = channels
        self.window_size, self.stride, self.padding = window_size, stride, padding
        self.modulator = nn.Sequential(nn.Conv2d(channels + condition_channels + 2, hidden_dim, 3, padding=1),
                                       nn.SiLU(), nn.Conv2d(hidden_dim, channels, 1))
        self.raw_amplitude_bound = nn.Parameter(torch.tensor(math.log(math.expm1(amplitude_bound))))
        hann = torch.hann_window(window_size, periodic=False)
        self.register_buffer('window', torch.outer(hann, hann), persistent=False)
        self.register_buffer('negative_frequency', (-torch.arange(window_size)) % window_size, persistent=False)

    @property
    def amplitude_bound(self) -> Tensor:
        return F.softplus(self.raw_amplitude_bound)

    def forward(self, decoder: Tensor, condition: Tensor, mask: Tensor, support: Tensor) -> Tensor:
        batch, channels, height, width = decoder.shape
        if min(height, width) <= self.padding:
            raise ValueError('decoder spatial dimensions must exceed the reflection padding')
        size, stride, pad = self.window_size, self.stride, self.padding
        state = torch.cat((decoder, condition, mask, support), dim=1)
        padded = F.pad(state, (pad, pad, pad, pad), mode='reflect')
        extra_h = (-(padded.shape[-2] - size)) % stride
        extra_w = (-(padded.shape[-1] - size)) % stride
        if extra_h or extra_w:
            padded = F.pad(padded, (0, extra_w, 0, extra_h), mode='reflect')
        columns = F.unfold(padded, kernel_size=size, stride=stride)
        windows = columns.transpose(1, 2).reshape(-1, state.shape[1], size, size)
        modulation = self.modulator(windows)
        negative = self.negative_frequency
        modulation = .5 * (modulation + modulation.index_select(-2, negative).index_select(-1, negative))
        work_dtype = torch.float64 if decoder.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=decoder.device.type, enabled=False):
            window = self.window.to(device=decoder.device, dtype=work_dtype)[None, None]
            patches = windows[:, :channels].to(work_dtype)
            spectrum = torch.fft.fft2(patches * window, norm='ortho')
            scale = (self.amplitude_bound.to(work_dtype) * modulation.to(work_dtype).tanh()).exp()
            reconstructed = torch.fft.ifft2(spectrum * scale, norm='ortho').real * window
            count = columns.shape[-1]
            reconstructed = reconstructed.reshape(batch, count, channels * size * size).transpose(1, 2)
            merged = F.fold(reconstructed, output_size=padded.shape[-2:], kernel_size=size, stride=stride)
            weights = window.square().reshape(1, size * size, 1).expand(1, -1, count)
            denominator = F.fold(weights, output_size=padded.shape[-2:], kernel_size=size, stride=stride)
            normalized = merged / denominator.clamp_min(torch.finfo(work_dtype).tiny)
            normalized = normalized[:, :, pad:pad + height, pad:pad + width]
            return (normalized - decoder.to(work_dtype)).to(decoder.dtype)


class MADC(nn.Module):
    def __init__(self, channels: int, condition_channels: int, hidden_dim: int,
                 detail_channels: int, image_channels: int = 3, residual_init: float = 1e-4):
        super().__init__()
        self.amplitude = WindowedAmplitude(channels, condition_channels, hidden_dim)
        self.detail_projection = nn.Conv2d(3 * image_channels, detail_channels, 1, bias=False)
        self.spatial = nn.Sequential(
            nn.Conv2d(channels + condition_channels + detail_channels + 2, hidden_dim, 3, padding=1),
            nn.SiLU(), nn.Conv2d(hidden_dim, channels, 3, padding=1))
        self.gates = nn.Sequential(nn.Conv2d(channels + condition_channels + 3, hidden_dim, 3, padding=1),
                                   nn.SiLU(), nn.Conv2d(hidden_dim, 2 * channels, 1))
        self.amplitude_scale = nn.Parameter(torch.tensor(residual_init))
        self.spatial_scale = nn.Parameter(torch.tensor(residual_init))

    def forward(self, decoder: Tensor, condition: Tensor, support: Tensor, mask: Tensor,
                boundary_distance: Tensor, wavelet: WaveletLevel) -> Tensor:
        size = decoder.shape[-2:]
        condition, support = validity_resample(condition, support, size)
        mask = resize_map(mask, size, 'area')
        distance = resize_map(boundary_distance, size)
        detail = self.detail_projection(wavelet.detail.to(decoder))
        detail, detail_support = validity_resample(detail, wavelet.validity.to(detail), size)
        amplitude_residual = self.amplitude(decoder, condition, mask, support)
        spatial_residual = self.spatial(torch.cat((decoder, condition, detail, mask, detail_support), dim=1))
        amplitude_gate, spatial_gate = self.gates(torch.cat((decoder, condition, support, mask, distance), dim=1)
                                                ).sigmoid().chunk(2, dim=1)
        return (decoder + self.amplitude_scale * amplitude_gate * amplitude_residual +
                self.spatial_scale * spatial_gate * spatial_residual)
