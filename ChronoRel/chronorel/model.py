from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from torch import Tensor, nn

from .dsea import DSEA, DSEAOutput
from .encoding import ObservationEncoder
from .madc import MADC
from .mea import MEA
from .wavelets import HaarDetailPyramid, WaveletLevel


@dataclass
class PreparedConditions:
    mea: dict[int, tuple[Tensor, Tensor]]
    madc: dict[int, tuple[Tensor, Tensor]]
    wavelets: tuple[WaveletLevel, ...]


class ChronoRelAdapters(nn.Module):
    def __init__(self, mea_encoder: ObservationEncoder, madc_encoder: ObservationEncoder,
                 mea: Mapping[int, MEA], dsea: Mapping[int, DSEA], madc: Mapping[int, MADC]):
        super().__init__()
        if mea_encoder is madc_encoder:
            raise ValueError('MEA and MADC require independent condition encoders')
        if set(mea) != set(dsea) or len(mea) != 2 or len(madc) != 2:
            raise ValueError('provide two matching MEA/DSEA interfaces and two MADC levels')
        if not set(map(str, mea)).issubset(mea_encoder.projections):
            raise ValueError('MEA encoder outputs must cover both interfaces')
        if not set(map(str, madc)).issubset(madc_encoder.projections):
            raise ValueError('MADC encoder outputs must cover both decoder levels')
        self.mea_encoder, self.madc_encoder = mea_encoder, madc_encoder
        self.mea = nn.ModuleDict({str(s): module for s, module in mea.items()})
        self.dsea = nn.ModuleDict({str(s): module for s, module in dsea.items()})
        self.madc = nn.ModuleDict({str(s): module for s, module in madc.items()})
        self.detail_level = {stride: i for i, stride in enumerate(sorted(madc))}
        self.haar = HaarDetailPyramid(2)

    def prepare_conditions(self, visible: Tensor, mask: Tensor) -> PreparedConditions:
        return PreparedConditions(self.mea_encoder(visible, mask), self.madc_encoder(visible, mask),
                                   self.haar(visible, mask))

    def aggregate(self, stride: int, brush: Tensor, mask: Tensor, time_embedding: Tensor,
                  conditions: PreparedConditions, base_attention: Callable[[Tensor], Tensor]) -> Tensor:
        observation, support = conditions.mea[stride]
        return self.mea[str(stride)](brush, observation, support, mask, time_embedding, base_attention)

    def adapt(self, stride: int, hidden: Tensor, brush_features: Sequence[Tensor], zero_conv: Tensor,
              boundary_distance: Tensor, time_embedding: Tensor, conditions: PreparedConditions) -> DSEAOutput:
        _, support = conditions.mea[stride]
        return self.dsea[str(stride)](hidden, brush_features, zero_conv, support,
                                     boundary_distance, time_embedding)

    def decode(self, stride: int, decoder: Tensor, mask: Tensor, boundary_distance: Tensor,
               conditions: PreparedConditions) -> Tensor:
        condition, support = conditions.madc[stride]
        wavelet = conditions.wavelets[self.detail_level[stride]]
        return self.madc[str(stride)](decoder, condition, support, mask, boundary_distance, wavelet)
