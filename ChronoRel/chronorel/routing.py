from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import torch
from torch import Tensor, nn


class CandidateBank(nn.Module):
    def __init__(self, num_experts: int = 4, max_cardinality: int = 2):
        super().__init__()
        if not 1 <= max_cardinality <= num_experts:
            raise ValueError('require 1 <= max_cardinality <= num_experts')
        self.combinations = ((),) + tuple(c for k in range(1, max_cardinality + 1)
                                         for c in combinations(range(num_experts), k))
        membership = torch.zeros(len(self.combinations), num_experts, dtype=torch.bool)
        for i, candidate in enumerate(self.combinations):
            membership[i, list(candidate)] = True
        self.register_buffer('membership', membership)

    def __len__(self) -> int:
        return len(self.combinations)


@dataclass
class RoutingDecision:
    candidate_index: Tensor
    strength: Tensor
    candidate_strengths: Tensor
    regularized_scores: Tensor
    feasible: Tensor


def closed_form_strength(gain: Tensor, curvature: Tensor, perturbation: Tensor,
                         epsilon: float = 1e-4, eta: float = .1) -> Tensor:
    if not epsilon >= 0 or not math.isfinite(eta) or eta <= 0:
        raise ValueError('epsilon must be nonnegative and eta must be positive')
    gain, curvature, perturbation = gain.float(), curvature.float(), perturbation.float()
    unconstrained = (gain.clamp_min(0) / (curvature + eta)).clamp_max(1)
    if math.isinf(epsilon):
        return unconstrained
    if epsilon == 0:
        cap = (perturbation == 0).to(unconstrained)
    else:
        positive_cap = (epsilon / perturbation.clamp_min(torch.finfo(perturbation.dtype).tiny)).sqrt()
        cap = torch.where(perturbation > 0, positive_cap, torch.ones_like(positive_cap))
    return torch.minimum(unconstrained, cap)


class JointDecision(nn.Module):
    def __init__(self, epsilon: float = 1e-4, eta: float = .1, fixed_strength: float | None = None):
        super().__init__()
        if not epsilon >= 0 or not math.isfinite(eta) or eta <= 0:
            raise ValueError('epsilon must be nonnegative and eta must be positive')
        if fixed_strength is not None and not 0 <= fixed_strength <= 1:
            raise ValueError('fixed_strength must lie in [0, 1]')
        self.epsilon, self.eta, self.fixed_strength = epsilon, eta, fixed_strength

    def forward(self, gain: Tensor, curvature: Tensor, perturbation: Tensor,
                ranking: Literal['regularized', 'gain', 'random'] = 'regularized',
                generator: torch.Generator | None = None) -> RoutingDecision:
        if gain.shape != curvature.shape or gain.shape != perturbation.shape or gain.shape[-1] < 2:
            raise ValueError('response tensors must have matching shapes with an empty candidate at index 0')
        gain, curvature, perturbation = gain.float(), curvature.float(), perturbation.float()
        if self.fixed_strength is None:
            strength = closed_form_strength(gain, curvature, perturbation, self.epsilon, self.eta)
        else:
            strength = torch.full_like(gain, self.fixed_strength)
        strength = torch.cat((torch.zeros_like(strength[..., :1]), strength[..., 1:]), dim=-1)
        measured_gain = gain * strength - .5 * curvature * strength.square()
        scores = measured_gain - .5 * self.eta * strength.square()
        if math.isinf(self.epsilon):
            feasible = torch.ones_like(strength, dtype=torch.bool)
        else:
            tolerance = torch.finfo(strength.dtype).eps * self.epsilon
            feasible = perturbation * strength.square() <= self.epsilon + tolerance
        feasible = torch.cat((torch.ones_like(feasible[..., :1]), feasible[..., 1:]), dim=-1)
        if ranking == 'random':
            samples = torch.multinomial(feasible.float().reshape(-1, feasible.shape[-1]), 1,
                                        generator=generator)
            index = samples.reshape(gain.shape[:-1])
        elif ranking in ('regularized', 'gain'):
            ranking_scores = scores if ranking == 'regularized' else measured_gain
            index = ranking_scores.masked_fill(~feasible, -torch.inf).argmax(dim=-1)
        else:
            raise ValueError(f'unknown ranking: {ranking}')
        chosen_strength = strength.gather(-1, index[..., None]).squeeze(-1)
        return RoutingDecision(index, chosen_strength, strength, scores, feasible)
