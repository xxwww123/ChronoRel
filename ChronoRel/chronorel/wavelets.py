from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class WaveletLevel:
    detail: Tensor
    validity: Tensor


class HaarDetailPyramid(nn.Module):
    def __init__(self, levels: int = 2):
        super().__init__()
        if levels < 1:
            raise ValueError('levels must be positive')
        self.levels = levels
        low = torch.tensor((1., 1.)) / 2 ** .5
        high = torch.tensor((1., -1.)) / 2 ** .5
        filters = torch.stack([torch.outer(a, b) for a, b in
                               ((low, low), (low, high), (high, low), (high, high))])
        self.register_buffer('filters', filters[:, None], persistent=False)

    def forward(self, image: Tensor, mask: Tensor) -> tuple[WaveletLevel, ...]:
        if any(s % (2 ** self.levels) for s in image.shape[-2:]):
            raise ValueError('image dimensions must be divisible by 2**levels')
        if mask.shape != (image.shape[0], 1, *image.shape[-2:]):
            raise ValueError('mask must have shape B x 1 x H x W')
        image = image.float() if image.dtype in (torch.float16, torch.bfloat16) else image
        valid = (mask == 0).to(image)
        result = []
        for _ in range(self.levels):
            batch, channels, _, _ = image.shape
            coefficients = F.conv2d(image, self.filters.to(image).repeat(channels, 1, 1, 1),
                                     stride=2, groups=channels)
            coefficients = coefficients.reshape(batch, channels, 4, *coefficients.shape[-2:])
            valid = 1 - F.max_pool2d(1 - valid, kernel_size=2, stride=2)
            detail = coefficients[:, :, 1:].reshape(batch, 3 * channels, *coefficients.shape[-2:])
            result.append(WaveletLevel(detail, valid))
            image = coefficients[:, :, 0]
        return tuple(result)
