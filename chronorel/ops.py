from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


def resize_map(x: Tensor, size: tuple[int, int], mode: str = 'bilinear') -> Tensor:
    if x.shape[-2:] == size:
        return x
    if mode in ('bilinear', 'bicubic'):
        return F.interpolate(x, size=size, mode=mode, align_corners=False)
    return F.interpolate(x, size=size, mode=mode)


def normalize_support(numerator: Tensor, support: Tensor, scale: float = 1.) -> Tensor:
    output_dtype = torch.promote_types(numerator.dtype, support.dtype)
    work_dtype = torch.float32 if output_dtype in (torch.float16, torch.bfloat16) else output_dtype
    support = support.to(work_dtype)
    valid = support > 0
    denominator = torch.where(valid, support.clamp_min(1e-8), 1.)
    normalized = (numerator.to(work_dtype) * scale) / denominator
    return torch.where(valid, normalized, 0.).to(output_dtype)


def validity_resample(features: Tensor, support: Tensor, size: tuple[int, int]) -> tuple[Tensor, Tensor]:
    weight = resize_map(support, size)
    numerator = resize_map(features * support, size)
    return normalize_support(numerator, weight), weight


def weighted_softmax(logits: Tensor, weights: Tensor, dim: int = -1) -> Tensor:
    scores = logits.float().masked_fill(weights <= 0, -torch.inf)
    peak = scores.amax(dim=dim, keepdim=True)
    peak = torch.where(torch.isfinite(peak), peak, 0)
    values = (scores - peak).exp() * weights.float()
    result = values / values.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return result.to(logits.dtype)


def time_map(embedding: Tensor, size: tuple[int, int]) -> Tensor:
    return embedding[:, :, None, None].expand(-1, -1, *size)


def composite(raw: Tensor, visible: Tensor, mask: Tensor) -> Tensor:
    return (1 - mask) * visible + mask * raw
