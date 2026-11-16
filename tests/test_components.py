import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from chronorel import (DSEA, MEA, MADC, HaarDetailPyramid, ObservationEncoder,
                       ValidityConv2d, WindowedAmplitude, composite)
from chronorel.objectives import regional_mse, response_fitting_loss, response_targets
from chronorel.ops import validity_resample, weighted_softmax


class EncodingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_no_support_produces_zero(self):
        layer = ValidityConv2d(3, 4)
        x = torch.randn(2, 3, 12, 16, requires_grad=True)
        output, support = layer(x, torch.zeros(2, 1, 12, 16))
        torch.testing.assert_close(output, torch.zeros_like(output))
        torch.testing.assert_close(support, torch.zeros_like(support))
        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.zeros_like(x))
        torch.testing.assert_close(layer.conv.weight.grad, torch.zeros_like(layer.conv.weight))

    def test_support_normalization(self):
        layer = ValidityConv2d(1, 1, stride=1)
        nn.init.ones_(layer.conv.weight)
        output, _ = layer(torch.ones(1, 1, 5, 5), torch.ones(1, 1, 5, 5))
        torch.testing.assert_close(output, F.silu(torch.full_like(output, 9.)))

    def test_encoder_output_strides(self):
        encoder = ObservationEncoder((4, 8, 12), {2: 5, 8: 7})
        features = encoder(torch.randn(1, 3, 32, 40), torch.zeros(1, 1, 32, 40))
        self.assertEqual(features[2][0].shape, (1, 5, 16, 20))
        self.assertEqual(features[8][0].shape, (1, 7, 4, 5))

    def test_full_resolution_encoder(self):
        encoder = ObservationEncoder((4, 8), {1: 6, 2: 9}, strides=(1, 2))
        features = encoder(torch.randn(1, 3, 16, 20), torch.zeros(1, 1, 16, 20))
        self.assertEqual(features[1][0].shape, (1, 6, 16, 20))
        self.assertEqual(features[2][0].shape, (1, 9, 8, 10))

    def test_validity_resampling_excludes_invalid_values(self):
        features = torch.tensor([[[[2., 999.]]]])
        support = torch.tensor([[[[1., 0.]]]])
        result, weight = validity_resample(features, support, (1, 1))
        torch.testing.assert_close(result, torch.tensor([[[[2.]]]]))
        torch.testing.assert_close(weight, torch.tensor([[[[.5]]]]))

    def test_fp16_resampling_zero_support_gradients(self):
        features = torch.tensor([[[[2., 3.]]]], dtype=torch.float16, requires_grad=True)
        support = torch.tensor([[[[0., .5]]]], dtype=torch.float16, requires_grad=True)
        result, _ = validity_resample(features, support, (1, 2))
        torch.testing.assert_close(result, features.new_tensor([[[[0., 3.]]]]))
        result.sum().backward()
        torch.testing.assert_close(features.grad, features.new_tensor([[[[0., 1.]]]]))
        torch.testing.assert_close(support.grad, torch.zeros_like(support))

    def test_all_masked_softmax(self):
        result = weighted_softmax(torch.full((2, 4), 1000.), torch.zeros(2, 4))
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_compositing_preserves_observed_pixels(self):
        raw, visible = torch.randn(1, 3, 8, 8), torch.rand(1, 3, 8, 8)
        mask = torch.randint(0, 2, (1, 1, 8, 8)).float()
        out = composite(raw, visible, mask)
        torch.testing.assert_close(out * (1 - mask), visible * (1 - mask), rtol=0, atol=0)


class MEATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.layer = MEA(8, 6, attention_dim=4, hidden_dim=12)
        self.brush = torch.randn(2, 8, 7, 9, requires_grad=True)
        self.observation = torch.randn(2, 8, 7, 9)
        self.support = torch.rand(2, 1, 7, 9)
        self.mask = torch.randint(0, 2, (2, 1, 14, 18)).float()
        self.time = torch.randn(2, 6)

    def test_forward_and_backward(self):
        out = self.layer(self.brush, self.observation, self.support, self.mask, self.time, nn.Identity())
        self.assertEqual(out.shape, self.brush.shape)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(self.brush.grad).all())
        self.assertTrue(torch.isfinite(self.layer.offset_predictor[-1].weight.grad).all())

    def test_no_visible_support(self):
        out = self.layer(self.brush, self.observation, torch.zeros_like(self.support),
                         self.mask, self.time, nn.Identity())
        self.assertTrue(torch.isfinite(out).all())
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(self.layer.offset_predictor[-1].weight.grad).all())

    def test_rotation_consistency_on_rectangular_grid(self):
        loss = self.layer.rotation_consistency_loss(self.brush, self.observation, self.support,
                                                     self.mask, self.time, nn.Identity(), quarter_turns=1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(self.layer.condition_adapter[-1].weight.grad)


class DSEATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.layer = DSEA(8, (6, 10), 6, state_dim=12, projection_dim=5)
        self.hidden = torch.randn(2, 8, 4, 5, requires_grad=True)
        self.features = (torch.randn(2, 6, 8, 10), torch.randn(2, 10, 2, 3))
        self.zero = torch.randn_like(self.hidden)
        self.support = torch.rand(2, 1, 8, 10)
        self.distance = torch.randn(2, 1, 16, 20) * .1
        self.time = torch.randn(2, 6)

    def run_layer(self):
        return self.layer(self.hidden, self.features, self.zero, self.support, self.distance, self.time)

    def test_zero_gate_initialization(self):
        out = self.run_layer()
        torch.testing.assert_close(out.hidden_state, self.hidden + self.zero, rtol=0, atol=0)
        torch.testing.assert_close(out.correction, torch.zeros_like(out.correction))
        self.assertEqual(out.decision.strength.shape, (2, 4, 5))

    def test_sparse_dispatch(self):
        with torch.no_grad():
            self.layer.response_predictor.head.weight.zero_()
            bias = self.layer.response_predictor.head.bias.reshape(10, 3)
            bias[:, 0] = -1
            bias[:, 1] = math.log(math.expm1(1.))
            bias[:, 2] = math.log(math.expm1(1e-5))
            bias[5, 0] = 2
            for expert in self.layer.experts:
                expert.time_modulation.bias[16:] = 1
        calls = [0] * 4
        handles = []
        for e, expert in enumerate(self.layer.experts):
            def record(module, args, result, index=e):
                calls[index] += args[0].shape[0]
            handles.append(expert.register_forward_hook(record))
        out = self.run_layer()
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, [40, 0, 40, 0])
        self.assertTrue((out.decision.candidate_index == 6).all())

    def test_response_and_gate_gradients(self):
        out = self.run_layer()
        (out.hidden_state.square().mean() + .01 * out.balance_loss).backward()
        self.assertTrue(torch.isfinite(self.layer.response_predictor.head.weight.grad).all())
        self.assertGreater(self.layer.shared_adapter.time_modulation.bias.grad.abs().sum().item(), 0)

    def test_empty_response_is_zero(self):
        out = self.run_layer()
        for response in out.responses:
            torch.testing.assert_close(response[..., 0], torch.zeros_like(response[..., 0]))


class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_haar_energy(self):
        image = torch.randn(1, 3, 32, 40)
        haar = HaarDetailPyramid()
        levels = haar(image, torch.zeros(1, 1, 32, 40))
        low_filter = haar.filters[:1].repeat(3, 1, 1, 1)
        low = image
        for _ in range(2):
            low = F.conv2d(low, low_filter, stride=2, groups=3)
        energy = low.square().sum() + sum(level.detail.square().sum() for level in levels)
        torch.testing.assert_close(energy, image.square().sum(), rtol=2e-6, atol=1e-5)

    def test_haar_validity(self):
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 3, 5] = 1
        levels = HaarDetailPyramid()(torch.randn(1, 3, 16, 16), mask)
        self.assertEqual((levels[0].validity == 0).sum().item(), 1)
        self.assertEqual((levels[1].validity == 0).sum().item(), 1)

    def test_overlap_add_identity_with_odd_dimensions(self):
        layer = WindowedAmplitude(2, 3, hidden_dim=6)
        for parameter in layer.modulator.parameters():
            nn.init.zeros_(parameter)
        x = torch.randn(1, 2, 19, 23)
        residual = layer(x, torch.randn(1, 3, 19, 23), torch.zeros(1, 1, 19, 23),
                          torch.ones(1, 1, 19, 23))
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=1e-6)

    def test_custom_hann_window_at_coverage_limit(self):
        layer = WindowedAmplitude(2, 3, hidden_dim=6, window_size=7, stride=5, padding=5)
        for parameter in layer.modulator.parameters():
            nn.init.zeros_(parameter)
        x = torch.randn(1, 2, 19, 23)
        mask = torch.zeros(1, 1, 19, 23)
        residual = layer(x, torch.randn(1, 3, 19, 23), mask, 1 - mask)
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=2e-6)

    def test_rejects_hann_window_with_uncovered_seams(self):
        with self.assertRaisesRegex(ValueError, 'Hann windows require'):
            WindowedAmplitude(2, 3, hidden_dim=6, window_size=16, stride=15, padding=15)

    def test_rejects_grid_smaller_than_hann_window(self):
        layer = WindowedAmplitude(2, 3, hidden_dim=6, window_size=48, stride=8, padding=8)
        x = torch.randn(1, 2, 12, 64)
        mask = torch.zeros(1, 1, 12, 64)
        with self.assertRaisesRegex(ValueError, 'padded decoder grid is smaller'):
            layer(x, torch.randn(1, 3, 12, 64), mask, 1 - mask)

    def test_zero_amplitude_stays_zero(self):
        layer = WindowedAmplitude(2, 3, hidden_dim=6)
        x = torch.zeros(1, 2, 24, 24)
        residual = layer(x, torch.randn(1, 3, 24, 24), torch.zeros(1, 1, 24, 24),
                          torch.ones(1, 1, 24, 24))
        torch.testing.assert_close(residual, x, rtol=0, atol=0)

    def test_positive_amplitude_bound(self):
        layer = WindowedAmplitude(2, 3, hidden_dim=6)
        torch.testing.assert_close(layer.amplitude_bound, torch.tensor(.1))

    def test_madc_forward_backward_and_invalid_cues(self):
        layer = MADC(4, 3, hidden_dim=8, detail_channels=5)
        decoder = torch.randn(1, 4, 24, 24, requires_grad=True)
        condition = torch.randn(1, 3, 12, 12)
        support = torch.rand(1, 1, 12, 12)
        mask = torch.zeros(1, 1, 48, 48)
        mask[:, :, 16:30, 12:32] = 1
        image = torch.rand(1, 3, 48, 48) * (1 - mask)
        distance = torch.zeros_like(mask)
        haar = HaarDetailPyramid()
        wavelet = haar(image, mask)[0]
        out = layer(decoder, condition, support, mask, distance, wavelet)
        altered = haar(image + 999 * mask, mask)[0]
        other = layer(decoder, condition, support, mask, distance, altered)
        torch.testing.assert_close(out, other, rtol=0, atol=0)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(decoder.grad).all())
        self.assertTrue(torch.isfinite(layer.amplitude.raw_amplitude_bound.grad))

    def test_cpu_bfloat16_autocast(self):
        layer = MADC(4, 3, hidden_dim=8, detail_channels=5)
        mask = torch.zeros(1, 1, 24, 24)
        wavelet = HaarDetailPyramid()(torch.rand(1, 3, 24, 24), mask)[0]
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = layer(torch.randn(1, 4, 24, 24), torch.randn(1, 3, 24, 24),
                            torch.ones_like(mask), mask, mask, wavelet)
        self.assertTrue(torch.isfinite(result).all())


class ObjectiveTests(unittest.TestCase):
    def test_response_target_scales(self):
        shared = torch.full((2, 3, 8, 8), .7, requires_grad=True)
        candidate = torch.full_like(shared, .9, requires_grad=True)
        target = torch.ones_like(shared)
        mask = torch.zeros(2, 1, 8, 8)
        mask[:, :, :4] = 1
        gain, perturbation = response_targets(shared, candidate, target, mask)
        torch.testing.assert_close(gain, torch.full((2,), 8.))
        torch.testing.assert_close(perturbation, torch.full((2,), .04))
        self.assertFalse(gain.requires_grad)

    def test_empty_region(self):
        result = regional_mse(torch.ones(1, 3, 4, 4), torch.zeros(1, 3, 4, 4), torch.zeros(1, 1, 4, 4))
        self.assertEqual(result.item(), 0)

    def test_fp16_empty_region_gradients(self):
        prediction = torch.ones(1, 3, 4, 4, dtype=torch.float16, requires_grad=True)
        reference = torch.zeros_like(prediction)
        region = torch.zeros(1, 1, 4, 4, dtype=torch.float16)
        result = regional_mse(prediction, reference, region)
        torch.testing.assert_close(result, torch.zeros(1, dtype=torch.float32))
        result.sum().backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_fitting_loss_detaches_targets(self):
        g, h, d = [torch.tensor([v], requires_grad=True) for v in (.3, .2, .01)]
        target_gain, target_d = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
        loss = response_fitting_loss(g, h, d, torch.tensor([.5]), target_gain, target_d)
        loss.backward()
        self.assertIsNone(target_gain.grad)
        self.assertIsNone(target_d.grad)
        self.assertIsNotNone(g.grad)


if __name__ == '__main__':
    unittest.main()
