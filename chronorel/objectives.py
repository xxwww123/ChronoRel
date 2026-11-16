from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def regional_mse(prediction: Tensor, reference: Tensor, region: Tensor, delta: float = 1e-8) -> Tensor:
    dtype = torch.promote_types(torch.promote_types(prediction.dtype, reference.dtype), region.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    prediction, reference, region = prediction.to(dtype), reference.to(dtype), region.to(dtype)
    axes = tuple(range(1, prediction.ndim))
    numerator = ((prediction - reference).square() * region).sum(dim=axes)
    denominator = prediction.shape[1] * region.sum(dim=axes) + delta
    return numerator / denominator


@torch.no_grad()
def response_targets(shared_raw: Tensor, candidate_raw: Tensor, reference: Tensor,
                     missing_mask: Tensor) -> tuple[Tensor, Tensor]:
    gain = 100 * (regional_mse(shared_raw, reference, missing_mask) -
                  regional_mse(candidate_raw, reference, missing_mask))
    perturbation = regional_mse(candidate_raw, shared_raw, 1 - missing_mask)
    return gain, perturbation


def response_fitting_loss(gain: Tensor, curvature: Tensor, perturbation: Tensor, strength: Tensor,
                          target_gain: Tensor, target_perturbation: Tensor, lambda_d: float = 1.) -> Tensor:
    prediction_gain = gain * strength - .5 * curvature * strength.square()
    prediction_perturbation = perturbation * strength.square()
    return (F.mse_loss(prediction_gain, target_gain.detach()) +
            lambda_d * F.mse_loss(prediction_perturbation, target_perturbation.detach()))
