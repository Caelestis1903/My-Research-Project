"""S4 local algebra/installed-Adam gate; no formal data or training."""

import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage_S4_gradient_controls import (adam_deltas, assert_match, cosine,
                                        gradients_one_forward, scale_for, vector_norm,
                                        validate_adam_delta_against_clone,
                                        assert_vector_match)
from stage_S4_run import S4Runner
from stage_S3_run import _make_smoke_model
from stage_S3_gradient_access import partition_parameters


class StageS4ControlTests(unittest.TestCase):
    def test_installed_adam_reference(self):
        self.assertTrue(validate_adam_delta_against_clone())

    def test_vector_tolerance_uses_reference_move_norm(self):
        reference = [torch.tensor([0.1], dtype=torch.float64)]
        observed = [torch.tensor([0.1000012], dtype=torch.float64)]
        self.assertGreater(assert_vector_match(observed, reference, "relative"), 1e-6)
        with self.assertRaises(AssertionError):
            assert_vector_match([torch.tensor([0.10002], dtype=torch.float64)],
                                reference, "too-large")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_real_cuda_m3oe_adam_first_warmed_zero_and_none(self):
        device = torch.device("cuda:0")
        self.assertTrue(validate_adam_delta_against_clone(device))
        model = _make_smoke_model("kuairand", device)
        partition = partition_parameters(model)
        params = dict(model.named_parameters())
        for group_name, lr in (("C", .01), ("A_arch", .001)):
            names = partition["groups"][group_name]
            optimizer = torch.optim.Adam([params[name] for name in names], lr=lr,
                                         betas=(.9, .999), weight_decay=1e-5)
            self.assertIsNone(optimizer.param_groups[0]["foreach"])
            self.assertIsNone(optimizer.param_groups[0]["fused"])
            from torch.optim.optimizer import _default_to_fused_or_foreach
            self.assertEqual(_default_to_fused_or_foreach(
                [params[name] for name in names], False, use_fused=False), (False, True))
            for warmup in (0, 3):
                for _ in range(warmup if warmup else 0):
                    for index, name in enumerate(names):
                        params[name].grad = torch.full_like(params[name], .001 * (index + 1))
                    optimizer.step()
                gradients = [None if index == 0 else torch.zeros_like(params[name]) if index == 1
                             else torch.full_like(params[name], .002 * (index + 1))
                             for index, name in enumerate(names)]
                expected, _ = adam_deltas(params, names, gradients, optimizer)
                before = [params[name].detach().clone() for name in names]
                for name, grad in zip(names, gradients):
                    params[name].grad = grad
                optimizer.step()
                torch.cuda.synchronize(device)
                observed = [params[name].detach() - prior for name, prior in zip(names, before)]
                assert_vector_match(observed, expected, "real CUDA {} warmup{}".format(
                    group_name, warmup))

    def test_same_forward_gradient_partition_and_controls(self):
        shared = torch.nn.Parameter(torch.tensor([.3, -.1]))
        private = torch.nn.Parameter(torch.tensor([.2, .4]))
        params = {"shared": shared, "private": private}
        optimizer = torch.optim.Adam(list(params.values()), lr=.01, weight_decay=1e-5)
        # Both losses derive from one constructed forward tensor.
        prediction = shared.square().sum() + private.square().sum()
        target = shared[0].square() + private[0].square()
        gf, gt, gp = gradients_one_forward(prediction, target, params, ["shared"], ["private"])
        self.assertTrue(torch.equal(private.grad, gp[0]))
        ratio, zero = scale_for(vector_norm(gt), vector_norm(gf), "toy")
        self.assertEqual(zero, "regular")
        scaled = [gf[0] * ratio]
        assert_match(vector_norm(scaled), vector_norm(gt), "toy N")
        self.assertGreaterEqual(cosine(gf, scaled), .99999)
        df, _ = adam_deltas(params, ["shared"], gf, optimizer)
        dt, _ = adam_deltas(params, ["shared"], gt, optimizer)
        r_u, zero = scale_for(vector_norm(dt), vector_norm(df), "toy")
        self.assertEqual(zero, "regular")
        before = shared.detach().clone()
        p_before = private.detach().clone()
        optimizer.step()
        full = shared.detach() - before
        private_full = private.detach() - p_before
        assert_match(vector_norm([full - df[0]]), 0, "toy U full")
        with torch.no_grad():
            shared.copy_(before + r_u * full)
        assert_match(vector_norm([shared.detach() - before]), vector_norm(dt), "toy U target")
        self.assertTrue(torch.any(private_full != 0))
        self.assertEqual(int(optimizer.state[shared]["step"]), 1)
        self.assertEqual(int(optimizer.state[private]["step"]), 1)

    def test_undefined_direction_fails_closed(self):
        with self.assertRaises(ValueError):
            scale_for(1., 0., "toy")
        self.assertEqual(scale_for(0., 0., "toy"), (1., "zero_zero"))

    def test_runner_step_keeps_private_full_and_once_only_state(self):
        for arm in ("F", "T", "N", "U"):
            with self.subTest(arm=arm):
                class Toy(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.shared = torch.nn.Parameter(torch.tensor([.3, -.1]))
                        self.private = torch.nn.Parameter(torch.tensor([.2, .4]))
                model = Toy()
                r = object.__new__(S4Runner)
                r.model = model
                r.arm = arm
                r.formal_run_key = "toy"
                r.args = SimpleNamespace(seed=2022)
                r.parameter_by_name = dict(model.named_parameters())
                r.partition = {"groups": {"C": ["shared"], "P": ["private"], "A_arch": []}}
                r.optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
                r.gradient_rows, r.probe_rows, r.batch_rows = [], [], [{}]
                r.main_steps_seen = r.zero_target_main_steps = 0
                r.optimizer_steps = r.architecture_optimizer_steps = 0
                full = model.shared.square().sum() + model.private.square().sum()
                target = model.shared[0].square() + model.private[0].square()
                losses = {"full": full, "target": target,
                          "non_target": full - target,
                          "equality_residual": full - (target + (full - target))}
                r._controlled_step(losses, "toy-batch", torch.tensor([[0., 1.]]), 0, 0, "main")
                self.assertEqual(len(r.gradient_rows), 1)
                self.assertEqual(len(r.probe_rows), 1)
                self.assertEqual(int(r.optimizer.state[model.shared]["step"]), 1)
                self.assertEqual(int(r.optimizer.state[model.private]["step"]), 1)
                self.assertLess(float(r.gradient_rows[0]["private_full_residual"]), 1e-6)

    def test_architecture_group_is_controlled(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.arch = torch.nn.Parameter(torch.tensor([.2, -.4]))
        for arm in ("F", "T", "N", "U"):
            with self.subTest(arm=arm):
                model = Toy()
                r = object.__new__(S4Runner)
                r.model, r.arm, r.formal_run_key = model, arm, "toy"
                r.args = SimpleNamespace(seed=2022)
                r.parameter_by_name = dict(model.named_parameters())
                r.partition = {"groups": {"C": [], "P": [], "A_arch": ["arch"]}}
                r.arch_optimizer = torch.optim.Adam(model.parameters(), lr=.001,
                                                    weight_decay=1e-5)
                r.gradient_rows, r.probe_rows, r.batch_rows = [], [], [{}]
                r.optimizer_steps = r.architecture_optimizer_steps = 0
                full = model.arch.square().sum()
                target = model.arch[0].square()
                losses = {"full": full, "target": target, "non_target": full - target,
                          "equality_residual": full - (target + (full - target))}
                r._controlled_step(losses, "toy-architecture", torch.tensor([[0., 1.]]),
                                   0, -1, "architecture")
                self.assertEqual(len(r.probe_rows), 1)
                self.assertEqual(int(r.arch_optimizer.state[model.arch]["step"]), 1)


if __name__ == "__main__":
    unittest.main()
