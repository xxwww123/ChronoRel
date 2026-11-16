from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ops import normalize_support, resize_map, time_map, weighted_softmax


class MEA(nn.Module):
    def __init__(self, channels: int, time_dim: int, attention_dim: int,
                 hidden_dim: int, scales: Sequence[float] = (1, 2, 4), residual_init: float = 1e-4):
        super().__init__()
        if not scales or any(s <= 0 for s in scales):
            raise ValueError('scales must be positive')
        self.points = 4 * len(scales)
        self.attention_dim = attention_dim
        state_channels = 2 * channels + 2 + time_dim
        self.condition_adapter = nn.Sequential(nn.Conv2d(state_channels, hidden_dim, 1), nn.SiLU(),
                                              nn.Conv2d(hidden_dim, channels, 1))
        self.offset_predictor = nn.Sequential(nn.Conv2d(channels + 2 + time_dim, hidden_dim, 3, padding=1),
                                             nn.SiLU(), nn.Conv2d(hidden_dim, 2 * self.points, 1))
        nn.init.zeros_(self.offset_predictor[-1].weight)
        nn.init.zeros_(self.offset_predictor[-1].bias)
        self.observation_query = nn.Linear(channels, attention_dim, bias=False)
        self.observation_key = nn.Linear(channels, attention_dim, bias=False)
        self.observation_value = nn.Linear(channels, attention_dim, bias=False)
        self.context_query = nn.Linear(channels, attention_dim, bias=False)
        self.context_key = nn.Linear(channels, attention_dim, bias=False)
        self.context_value = nn.Linear(channels, attention_dim, bias=False)
        self.observation_bias = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.context_bias = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.gates = nn.Conv2d(channels + 2 + time_dim, 2 * attention_dim, 1)
        self.output_projection = nn.Conv2d(attention_dim, channels, 1)
        self.adapter_scale = nn.Parameter(torch.tensor(residual_init))
        self.aggregation_scale = nn.Parameter(torch.tensor(residual_init))
        cross = torch.tensor(((1., 0.), (0., 1.), (-1., 0.), (0., -1.)))
        self.register_buffer('base_offsets', torch.cat([s * cross for s in scales]), persistent=False)

    @staticmethod
    def _sample(x: Tensor, grid: Tensor) -> Tensor:
        batch, height, width, points, _ = grid.shape
        values = F.grid_sample(x, grid.reshape(batch, height, width * points, 2),
                               mode='bilinear', padding_mode='zeros', align_corners=False)
        return values.reshape(batch, x.shape[1], height, width, points).permute(0, 2, 3, 4, 1)

    def forward(self, brush: Tensor, observation: Tensor, support: Tensor, mask: Tensor,
                time_embedding: Tensor, base_attention: Callable[[Tensor], Tensor]) -> Tensor:
        batch, _, height, width = brush.shape
        size = (height, width)
        if observation.shape != brush.shape or support.shape != (batch, 1, height, width):
            raise ValueError('observation and support must match the Brush grid')
        mask = resize_map(mask, size, 'area')
        embedded_time = time_map(time_embedding, size)
        state = torch.cat((brush, observation, support, mask, embedded_time), dim=1)
        conditioned = brush + self.adapter_scale * self.condition_adapter(state)
        spatial_state = torch.cat((conditioned, support, mask, embedded_time), dim=1)
        offsets = self.offset_predictor(spatial_state)
        offsets = offsets.reshape(batch, self.points, 2, height, width).permute(0, 3, 4, 1, 2)
        y, x = torch.meshgrid(torch.arange(height, device=brush.device, dtype=brush.dtype),
                              torch.arange(width, device=brush.device, dtype=brush.dtype), indexing='ij')
        centers = torch.stack((x, y), dim=-1)[None, :, :, None]
        displacement = self.base_offsets.to(brush)[None, None, None] + offsets
        locations = centers + displacement
        inside = ((locations[..., 0] >= 0) & (locations[..., 0] <= width - 1) &
                  (locations[..., 1] >= 0) & (locations[..., 1] <= height - 1))
        grid = torch.stack((2 * (locations[..., 0] + .5) / width - 1,
                            2 * (locations[..., 1] + .5) / height - 1), dim=-1)
        sampled_support = self._sample(support, grid).squeeze(-1) * inside
        sampled_observation = self._sample(observation * support, grid)
        sampled_observation = normalize_support(sampled_observation, sampled_support[..., None])
        sampled_context = self._sample(conditioned, grid)
        queries = conditioned.permute(0, 2, 3, 1)
        scale = math.sqrt(self.attention_dim)
        obs_logits = (self.observation_query(queries)[..., None, :] *
                      self.observation_key(sampled_observation)).sum(-1) / scale
        gen_logits = (self.context_query(queries)[..., None, :] * self.context_key(sampled_context)).sum(-1) / scale
        obs_logits = obs_logits + self.observation_bias(displacement).squeeze(-1)
        gen_logits = gen_logits + self.context_bias(displacement).squeeze(-1)
        obs_weights = weighted_softmax(obs_logits, sampled_support)
        gen_weights = weighted_softmax(gen_logits, inside)
        observed = (obs_weights[..., None] * self.observation_value(sampled_observation)).sum(-2)
        generated = (gen_weights[..., None] * self.context_value(sampled_context)).sum(-2)
        observed, generated = observed.movedim(-1, 1), generated.movedim(-1, 1)
        obs_gate, gen_gate = self.gates(spatial_state).sigmoid().chunk(2, dim=1)
        aggregate = self.output_projection(obs_gate * observed + gen_gate * generated)
        base = base_attention(conditioned)
        if base.shape != brush.shape:
            raise ValueError('base_attention must return an NCHW feature of the same shape')
        return base + self.aggregation_scale * aggregate

    def rotation_consistency_loss(self, brush: Tensor, observation: Tensor, support: Tensor,
                                  mask: Tensor, time_embedding: Tensor,
                                  base_attention: Callable[[Tensor], Tensor], quarter_turns: int) -> Tensor:
        if quarter_turns not in (1, 2, 3):
            raise ValueError('quarter_turns must be 1, 2, or 3')
        mask = resize_map(mask, brush.shape[-2:], 'area')
        original = self(brush, observation, support, mask, time_embedding, base_attention)
        rotated_inputs = [torch.rot90(x, quarter_turns, (-2, -1))
                          for x in (brush, observation, support, mask)]
        rotated = self(*rotated_inputs, time_embedding, base_attention)
        return F.l1_loss(torch.rot90(rotated, -quarter_turns, (-2, -1)), original)
