from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ops import LayerNorm2d, resize_map, weighted_softmax
from .routing import CandidateBank, JointDecision, RoutingDecision


class MultiScaleConditionAligner(nn.Module):
    def __init__(self, brush_channels: Sequence[int], channels: int, projection_dim: int):
        super().__init__()
        if not brush_channels:
            raise ValueError('brush_channels must not be empty')
        self.projections = nn.ModuleList(nn.Conv2d(c, projection_dim, 1) for c in brush_channels)
        self.fusion = nn.Conv2d(len(brush_channels) * projection_dim, channels, 1)

    def forward(self, features: Sequence[Tensor], size: tuple[int, int]) -> Tensor:
        if len(features) != len(self.projections):
            raise ValueError('expected one feature tensor for each Brush projection')
        aligned = [resize_map(projection(feature), size) for projection, feature in zip(self.projections, features)]
        return self.fusion(torch.cat(aligned, dim=1))


class LocalStateEncoder(nn.Module):
    def __init__(self, channels: int, time_dim: int, state_dim: int):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.projection = nn.Sequential(nn.Conv2d(channels + 2, state_dim, 1), nn.SiLU(),
                                        nn.Conv2d(state_dim, state_dim, 1))
        self.time_modulation = nn.Linear(time_dim, 2 * state_dim)
        nn.init.zeros_(self.time_modulation.weight)
        nn.init.zeros_(self.time_modulation.bias)

    def forward(self, hidden: Tensor, support: Tensor, distance: Tensor, time_embedding: Tensor) -> Tensor:
        state = self.projection(torch.cat((self.norm(hidden), support, distance), dim=1))
        scale, shift = self.time_modulation(time_embedding).chunk(2, dim=-1)
        return ((1 + scale[:, :, None, None]) * state + shift[:, :, None, None]).movedim(1, -1)


class ResponsePredictor(nn.Module):
    def __init__(self, state_dim: int, num_candidates: int):
        super().__init__()
        self.num_nonempty = num_candidates - 1
        self.head = nn.Linear(state_dim, 3 * self.num_nonempty)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        coefficients = self.head(state).reshape(*state.shape[:-1], self.num_nonempty, 3)
        gain, curvature, perturbation = coefficients.unbind(-1)
        return F.pad(gain, (1, 0)), F.pad(F.softplus(curvature), (1, 0)), F.pad(F.softplus(perturbation), (1, 0))


class CorrectionAdapter(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        hidden_dim = max(channels // 4, 64)
        self.norm = nn.LayerNorm(channels)
        self.time_modulation = nn.Linear(time_dim, 3 * channels)
        self.value = nn.Linear(channels, hidden_dim)
        self.activation = nn.Linear(channels, hidden_dim)
        self.output = nn.Linear(hidden_dim, channels)
        nn.init.zeros_(self.time_modulation.weight)
        nn.init.zeros_(self.time_modulation.bias)

    def forward(self, tokens: Tensor, time_embedding: Tensor) -> Tensor:
        scale, shift, gate = self.time_modulation(time_embedding).chunk(3, dim=-1)
        state = (1 + scale) * self.norm(tokens) + shift
        return gate * self.output(F.silu(self.activation(state)) * self.value(state))


def load_balancing_loss(decision: RoutingDecision, expert_logits: Tensor,
                         membership: Tensor, temperature: float = 1.) -> Tensor:
    candidate_probabilities = (decision.regularized_scores / temperature).masked_fill(
        ~decision.feasible, -torch.inf).softmax(dim=-1)
    mixture = weighted_softmax(expert_logits[..., None, :] / temperature, membership)
    soft_usage = (candidate_probabilities[..., :, None] * mixture).sum(dim=-2)
    num_experts = expert_logits.shape[-1]
    marginal = soft_usage.reshape(-1, num_experts).mean(dim=0)
    return num_experts * (marginal - 1 / num_experts).square().sum()


@dataclass
class DSEAOutput:
    hidden_state: Tensor
    correction: Tensor
    decision: RoutingDecision
    responses: tuple[Tensor, Tensor, Tensor]
    balance_loss: Tensor


class DSEA(nn.Module):
    def __init__(self, channels: int, brush_channels: Sequence[int], time_dim: int,
                 state_dim: int, projection_dim: int, num_experts: int = 4,
                 max_cardinality: int = 2, epsilon: float = 1e-4, eta: float = .1,
                 fixed_strength: float | None = None, residual_init: float = 1e-4):
        super().__init__()
        self.candidates = CandidateBank(num_experts, max_cardinality)
        self.aligner = MultiScaleConditionAligner(brush_channels, channels, projection_dim)
        self.state_encoder = LocalStateEncoder(channels, time_dim, state_dim)
        self.response_predictor = ResponsePredictor(state_dim, len(self.candidates))
        self.expert_logits = nn.Linear(state_dim, num_experts)
        self.shared_adapter = CorrectionAdapter(channels, time_dim)
        self.experts = nn.ModuleList(CorrectionAdapter(channels, time_dim) for _ in range(num_experts))
        self.decision_rule = JointDecision(epsilon, eta, fixed_strength)
        self.interface_scale = nn.Parameter(torch.tensor(residual_init))

    def forward(self, hidden: Tensor, brush_features: Sequence[Tensor], zero_conv: Tensor,
                support: Tensor, boundary_distance: Tensor, time_embedding: Tensor,
                ranking: Literal['regularized', 'gain', 'random'] = 'regularized',
                generator: torch.Generator | None = None) -> DSEAOutput:
        batch, channels, height, width = hidden.shape
        if zero_conv.shape != hidden.shape:
            raise ValueError('zero_conv must match the receiving U-Net feature')
        size = (height, width)
        support = resize_map(support, size, 'area')
        distance = resize_map(boundary_distance, size)
        aligned = self.aligner(brush_features, size)
        state = self.state_encoder(hidden, support, distance, time_embedding)
        responses = self.response_predictor(state)
        decision = self.decision_rule(*responses, ranking=ranking, generator=generator)
        logits = self.expert_logits(state)
        active = self.candidates.membership[decision.candidate_index]
        weights = weighted_softmax(logits, active).reshape(-1, len(self.experts))
        tokens = aligned.movedim(1, -1).reshape(-1, channels)
        token_times = time_embedding[:, None, None, :].expand(-1, height, width, -1).reshape(
            -1, time_embedding.shape[-1])
        strength = decision.strength.reshape(-1)
        active = active.reshape(-1, len(self.experts)) & (strength[:, None] > 0)
        routed = torch.zeros_like(tokens)
        for e, expert in enumerate(self.experts):
            indices = active[:, e].nonzero(as_tuple=True)[0]
            if indices.numel():
                delta = expert(tokens[indices], token_times[indices])
                routed = routed.index_add(0, indices, delta * weights[indices, e, None])
        correction = self.shared_adapter(tokens, token_times) + strength[:, None] * routed
        correction = correction.reshape(batch, height, width, channels).movedim(-1, 1)
        updated = hidden + zero_conv + self.interface_scale * correction
        balance = load_balancing_loss(decision, logits, self.candidates.membership)
        return DSEAOutput(updated, correction, decision, responses, balance)
