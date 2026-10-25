import io
import unittest

import torch
from torch import nn

from chronorel import ChronoRelAdapters, DSEA, MEA, MADC, ObservationEncoder


def small_model():
    mea_encoder = ObservationEncoder((4, 8), {2: 4, 4: 8})
    madc_encoder = ObservationEncoder((4, 6), {1: 3, 2: 3}, strides=(1, 2))
    mea = {2: MEA(4, 6, 4, 8), 4: MEA(8, 6, 4, 8)}
    dsea = {s: DSEA(c, (4, 8), 6, state_dim=8, projection_dim=4) for s, c in ((2, 4), (4, 8))}
    madc = {1: MADC(4, 3, 8, 4), 2: MADC(4, 3, 8, 4)}
    return ChronoRelAdapters(mea_encoder, madc_encoder, mea, dsea, madc)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_component_interfaces(self):
        model = small_model()
        mask = torch.zeros(1, 1, 24, 24)
        mask[:, :, 8:16, 8:16] = 1
        conditions = model.prepare_conditions(torch.rand(1, 3, 24, 24) * (1 - mask), mask)
        brush = (torch.randn(1, 4, 12, 12), torch.randn(1, 8, 6, 6))
        time = torch.randn(1, 6)
        enhanced = [model.aggregate(stride, feature, mask, time, conditions, nn.Identity())
                    for stride, feature in zip((2, 4), brush)]
        adapted = model.adapt(2, torch.randn_like(brush[0]), enhanced, torch.zeros_like(brush[0]),
                               torch.zeros_like(mask), time, conditions)
        fine = model.decode(1, torch.randn(1, 4, 24, 24), mask, torch.zeros_like(mask), conditions)
        coarse = model.decode(2, torch.randn(1, 4, 12, 12), mask, torch.zeros_like(mask), conditions)
        self.assertEqual(adapted.hidden_state.shape, (1, 4, 12, 12))
        self.assertEqual(fine.shape, (1, 4, 24, 24))
        self.assertEqual(coarse.shape, (1, 4, 12, 12))
        loss = fine.square().mean() + coarse.square().mean() + adapted.balance_loss
        loss.backward()
        self.assertTrue(torch.isfinite(model.madc['1'].spatial[-1].weight.grad).all())

    def test_state_dict_roundtrip(self):
        model = small_model()
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        other = small_model()
        other.load_state_dict(torch.load(stream, weights_only=True))
        for a, b in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_mea_bfloat16_autocast(self):
        model = MEA(4, 6, 4, 8)
        feature = torch.randn(1, 4, 8, 10)
        support = torch.ones(1, 1, 8, 10)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = model(feature, feature, support, 1 - support, torch.randn(1, 6), nn.Identity())
        self.assertTrue(torch.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
