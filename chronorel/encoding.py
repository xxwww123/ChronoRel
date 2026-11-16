from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import accumulate
from operator import mul

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ops import normalize_support


class ValidityConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError('kernel_size must be odd')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.area = kernel_size ** 2
        self.register_buffer('support_kernel', torch.ones(1, 1, kernel_size, kernel_size), persistent=False)

    def forward(self, x: Tensor, support: Tensor) -> tuple[Tensor, Tensor]:
        count = F.conv2d(support, self.support_kernel.to(support), stride=self.conv.stride,
                         padding=self.conv.padding)
        features = self.conv(x * support)
        features = normalize_support(features, count, scale=self.area) + self.bias[None, :, None, None]
        features = torch.where(count > 0, F.silu(features), 0)
        return features, count / self.area


class ObservationEncoder(nn.Module):
    def __init__(self, widths: Sequence[int], outputs: Mapping[int, int], image_channels: int = 3,
                 strides: Sequence[int] | None = None):
        super().__init__()
        if not widths or any(c <= 0 for c in widths):
            raise ValueError('widths must contain positive channel counts')
        strides = tuple(strides) if strides is not None else (2,) * len(widths)
        if len(strides) != len(widths) or any(s not in (1, 2) for s in strides):
            raise ValueError('provide one stride of 1 or 2 for each encoder stage')
        self.output_strides = tuple(accumulate(strides, mul))
        if len(set(self.output_strides)) != len(widths):
            raise ValueError('encoder stages must have distinct cumulative strides')
        available = dict(zip(self.output_strides, widths))
        if not outputs or not set(outputs).issubset(available):
            raise ValueError('output strides must correspond to encoder stages')
        channels = (image_channels, *widths)
        self.stages = nn.ModuleList(ValidityConv2d(a, b, stride=s)
                                    for a, b, s in zip(channels, channels[1:], strides))
        self.projections = nn.ModuleDict({str(s): nn.Conv2d(available[s], c, 1, bias=False)
                                         for s, c in outputs.items()})

    def forward(self, visible: Tensor, mask: Tensor) -> dict[int, tuple[Tensor, Tensor]]:
        support = 1 - mask
        x = visible
        result = {}
        for i, stage in enumerate(self.stages):
            x, support = stage(x, support)
            stride = self.output_strides[i]
            if str(stride) in self.projections:
                result[stride] = self.projections[str(stride)](x), support
        return result
