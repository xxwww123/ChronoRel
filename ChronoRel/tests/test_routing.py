import math
import unittest

import torch

from chronorel import CandidateBank, JointDecision, closed_form_strength


class RoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_candidate_bank(self):
        bank = CandidateBank()
        self.assertEqual(len(bank), 11)
        self.assertEqual(bank.combinations[0], ())
        self.assertEqual(bank.combinations[1:5], ((0,), (1,), (2,), (3,)))
        self.assertEqual(bank.membership.sum(dim=1).tolist(), [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])

    def test_invalid_bank(self):
        with self.assertRaises(ValueError):
            CandidateBank(2, 3)

    def test_closed_form_matches_grid_search(self):
        g = torch.rand(32) * 2 - .3
        h = torch.rand(32)
        d = torch.rand(32) * .01
        a = closed_form_strength(g, h, d)
        grid = torch.linspace(0, 1, 10001)[None]
        scores = g[:, None] * grid - .5 * (h[:, None] + .1) * grid.square()
        scores = scores.masked_fill(d[:, None] * grid.square() > 1e-4, -torch.inf)
        score = g * a - .5 * (h + .1) * a.square()
        self.assertTrue(torch.all(score >= scores.amax(dim=-1) - 1e-6))
        self.assertTrue(torch.all(d * a.square() <= 1e-4 + 1e-10))

    def test_nonpositive_gain(self):
        a = closed_form_strength(torch.tensor([-1., 0.]), torch.zeros(2), torch.ones(2))
        torch.testing.assert_close(a, torch.zeros(2))

    def test_zero_perturbation(self):
        a = closed_form_strength(torch.tensor([.2, .03]), torch.zeros(2), torch.zeros(2))
        torch.testing.assert_close(a, torch.tensor([1., .3]))

    def test_zero_budget(self):
        a = closed_form_strength(torch.ones(2), torch.ones(2), torch.tensor([1., 0.]), epsilon=0)
        torch.testing.assert_close(a, torch.tensor([0., 1 / 1.1]))

    def test_unconstrained_retains_eta(self):
        a = closed_form_strength(torch.tensor([.1]), torch.tensor([.2]), torch.tensor([100.]), epsilon=math.inf)
        torch.testing.assert_close(a, torch.tensor([1 / 3]))

    def test_empty_fallback(self):
        decision = JointDecision()(torch.tensor([[0., -1., -.5]]), torch.zeros(1, 3), torch.ones(1, 3))
        self.assertEqual(decision.candidate_index.item(), 0)
        self.assertEqual(decision.strength.item(), 0)

    def test_equal_scores_follow_candidate_order(self):
        g = torch.tensor([[0., .5, .5]])
        decision = JointDecision()(g, torch.zeros_like(g), torch.zeros_like(g))
        self.assertEqual(decision.candidate_index.item(), 1)

    def test_gain_ranking_shares_strengths(self):
        g, h, d = torch.tensor([[0., .105, .4]]), torch.tensor([[0., 0., 1.]]), torch.zeros(1, 3)
        joint = JointDecision()(g, h, d)
        gain = JointDecision()(g, h, d, ranking='gain')
        torch.testing.assert_close(joint.candidate_strengths, gain.candidate_strengths)
        torch.testing.assert_close(joint.feasible, gain.feasible)
        self.assertEqual(joint.candidate_index.item(), 2)
        self.assertEqual(gain.candidate_index.item(), 1)

    def test_fixed_strength_retains_budget(self):
        decision = JointDecision(fixed_strength=.5)(torch.tensor([[0., 20., .1]]),
                                                   torch.zeros(1, 3), torch.tensor([[0., 1e-3, 0.]]))
        self.assertEqual(decision.candidate_index.item(), 2)
        self.assertEqual(decision.strength.item(), .5)
        self.assertFalse(decision.feasible[0, 1])

    def test_fixed_strength_nonpositive_gain_falls_back(self):
        decision = JointDecision(fixed_strength=.5)(torch.tensor([[0., -.1, -1.]]),
                                                   torch.zeros(1, 3), torch.zeros(1, 3))
        self.assertEqual(decision.candidate_index.item(), 0)

    def test_random_choice_is_uniform(self):
        shape = (11000, 11)
        decision = JointDecision()(torch.zeros(shape), torch.zeros(shape), torch.zeros(shape),
                                    ranking='random', generator=torch.Generator().manual_seed(73))
        frequencies = torch.bincount(decision.candidate_index, minlength=11) / shape[0]
        self.assertTrue(torch.all((frequencies - 1 / 11).abs() < .02))

    def test_strength_gradients(self):
        g = torch.tensor([.05], requires_grad=True)
        h = torch.tensor([.1], requires_grad=True)
        d = torch.tensor([0.], requires_grad=True)
        closed_form_strength(g, h, d).sum().backward()
        self.assertTrue(torch.isfinite(g.grad).all())
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertTrue(torch.isfinite(d.grad).all())


if __name__ == '__main__':
    unittest.main()
