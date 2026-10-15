from __future__ import annotations

import os
import sys

import unittest

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S2_objectives import compute_equal_coefficients
from stage_S3_gradient_access import (
    GROUP_ARCH, GROUP_PRIVATE, GROUP_SHARED, apply_architecture_gradients,
    apply_main_gradients, classify_parameter, loss_components, partition_parameters,
)
from stage_S3_run import run_s3_reachability_smoke, run_s3_update_smoke


class TinyPartition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Linear(2, 3, bias=False)
        self.slot_weight = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1, 1))
                                                   for _ in range(3)])
        self.tower = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(6)])
        self._weight_exp_d = torch.nn.Module()
        self._weight_exp_d.deep_weights = torch.nn.Parameter(torch.tensor([0.25]))


def _fixture():
    model = TinyPartition()
    partition = partition_parameters(model)
    params = dict(model.named_parameters())
    x = torch.tensor([[0.2, 0.7], [0.8, -0.1], [0.1, 0.4], [0.9, 0.3]], dtype=torch.float32)
    ids = torch.tensor([1, 0, 1, 2])
    prediction = torch.sigmoid(
        model.embedding(x)[:, 0:1] + model._weight_exp_d.deep_weights[0] * x[:, 0:1]).repeat(1, 2)
    # Ensure the private parameter contributes in a controlled, differentiable path.
    prediction = torch.sigmoid(prediction + torch.stack(list(model.slot_weight))[ids, 0, :])
    target = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
    table = compute_equal_coefficients({0: 1, 1: 2, 2: 1})
    losses = loss_components(prediction, target, ids, 1, "equal", table)
    return model, partition, params, losses


class GradientAccessTests(unittest.TestCase):
    def test_real_m3oe_sample_full_one_step_matches_direct_full_adam_updates(self):
        result = run_s3_update_smoke(
            torch.device("cpu"), objective="sample", access="FULL", representative=False)
        self.assertEqual(result["main_steps"], 1)
        self.assertEqual(result["architecture_steps"], 1)
        self.assertTrue(result["reachability_passed"])
        self.assertTrue(result["full_equivalence_passed"])

    def test_real_m3oe_reachability_passes_for_both_frozen_schemas(self):
        for dataset, expected_parameters in (("kuairand", 1614258), ("movielens", 773474)):
            with self.subTest(dataset=dataset):
                result = run_s3_reachability_smoke(dataset, device=torch.device("cpu"))
                self.assertTrue(result["reachability_passed"])
                self.assertEqual(result["parameter_count"], expected_parameters)
                self.assertEqual(result["architecture_count"], 4)

    def test_parameter_classification_is_exhaustive_and_architecture_is_explicit(self):
        self.assertEqual(classify_parameter("domain_expert.0.layers.0.weight"), GROUP_SHARED)
        self.assertEqual(classify_parameter("task_expert.1.layers.0.weight"), GROUP_SHARED)
        self.assertEqual(classify_parameter("slot_weight.2"), GROUP_PRIVATE)
        self.assertEqual(classify_parameter("gate.5.layers.0.weight"), GROUP_PRIVATE)
        self.assertEqual(classify_parameter("_weight_bal_t.deep_weights"), GROUP_ARCH)
        with self.assertRaisesRegex(ValueError, "unclassified"):
            classify_parameter("new_unreviewed_parameter")
        with self.assertRaisesRegex(ValueError, "unapproved architecture"):
            classify_parameter("_weight_new.deep_weights")


    def test_loss_decomposition_preserves_original_batch_denominator(self):
        _, _, _, losses = _fixture()
        self.assertTrue(torch.allclose(losses["full"], losses["target"] + losses["non_target"],
                                       atol=1e-7, rtol=1e-7))
        # Target mask is two out of four records, so the target component stays half-sized.
        self.assertLess(float(losses["target"]), float(losses["full"]))


    def test_target_only_routes_shared_and_private_gradients_to_distinct_losses(self):
        model, partition, params, losses = _fixture()
        model.zero_grad(set_to_none=True)
        target_expected = torch.autograd.grad(
            losses["target"], [params[name] for name in partition["groups"][GROUP_SHARED]],
            retain_graph=True, allow_unused=True)
        private_expected = torch.autograd.grad(
            losses["full"], [params[name] for name in partition["groups"][GROUP_PRIVATE]],
            retain_graph=True, allow_unused=True)
        apply_main_gradients(losses["full"], losses["target"], params, partition,
                             access="TARGET_ONLY_SHARED")
        for name, expected in zip(partition["groups"][GROUP_SHARED], target_expected):
            self.assertIsNotNone(expected)
            self.assertTrue(torch.allclose(params[name].grad, expected, atol=1e-7, rtol=1e-7))
        for name, expected in zip(partition["groups"][GROUP_PRIVATE], private_expected):
            observed = params[name].grad
            if expected is None:
                self.assertIsNone(observed)
            else:
                self.assertTrue(torch.allclose(observed, expected, atol=1e-7, rtol=1e-7))
        # The architecture optimizer receives only the target component.
        model.zero_grad(set_to_none=True)
        prediction = torch.sigmoid(
            model.embedding(torch.tensor([[0.2, 0.7], [0.8, -0.1], [0.1, 0.4], [0.9, 0.3]]))[:, 0:1]
            + model._weight_exp_d.deep_weights[0]
            * torch.tensor([[0.2], [0.8], [0.1], [0.9]])).repeat(1, 2)
        prediction = torch.sigmoid(
            prediction + torch.stack(list(model.slot_weight))[torch.tensor([1, 0, 1, 2]), 0, :])
        losses_arch = loss_components(
            prediction, torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]]),
            torch.tensor([1, 0, 1, 2]), 1, "equal", compute_equal_coefficients({0: 1, 1: 2, 2: 1}))
        arch = [params[name] for name in partition["groups"][GROUP_ARCH]]
        expected_arch = torch.autograd.grad(losses_arch["target"], arch, retain_graph=True)
        apply_architecture_gradients(losses_arch["full"], losses_arch["target"], params, partition,
                                     access="TARGET_ONLY_SHARED")
        for parameter, expected in zip(arch, expected_arch):
            self.assertTrue(torch.allclose(parameter.grad, expected, atol=1e-7, rtol=1e-7))


    def test_full_compatibility_matches_direct_backward_and_single_adam_step(self):
        model_a, partition, params_a, losses_a = _fixture()
        model_b = TinyPartition(); model_b.load_state_dict(model_a.state_dict())
        params_b = dict(model_b.named_parameters())
        table = compute_equal_coefficients({0: 1, 1: 2, 2: 1})
        x = torch.tensor([[0.2, 0.7], [0.8, -0.1], [0.1, 0.4], [0.9, 0.3]])
        ids = torch.tensor([1, 0, 1, 2])
        target = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])

        def compute(model):
            pred = torch.sigmoid(
                model.embedding(x)[:, 0:1] + model._weight_exp_d.deep_weights[0] * x[:, 0:1]).repeat(1, 2)
            pred = torch.sigmoid(pred + torch.stack(list(model.slot_weight))[ids, 0, :])
            return loss_components(pred, target, ids, 1, "equal", table)

        opt_a = torch.optim.Adam([params_a[name] for name in partition["groups"][GROUP_SHARED]
                                  + partition["groups"][GROUP_PRIVATE]], lr=0.01, weight_decay=1e-5)
        opt_b = torch.optim.Adam([params_b[name] for name in partition["groups"][GROUP_SHARED]
                                  + partition["groups"][GROUP_PRIVATE]], lr=0.01, weight_decay=1e-5)
        opt_a.zero_grad(set_to_none=True)
        apply_main_gradients(losses_a["full"], losses_a["target"], params_a, partition, access="FULL")
        opt_a.step()
        opt_b.zero_grad(set_to_none=True)
        losses_b = compute(model_b)
        losses_b["full"].backward()
        opt_b.step()
        for name, value in model_a.state_dict().items():
            self.assertTrue(torch.allclose(value, model_b.state_dict()[name], atol=1e-8, rtol=1e-8))
        for name, value in opt_a.state_dict()["state"].items():
            other = opt_b.state_dict()["state"][name]
            for field in value:
                if isinstance(value[field], torch.Tensor):
                    self.assertTrue(torch.equal(value[field], other[field]))
                else:
                    self.assertEqual(value[field], other[field])


if __name__ == "__main__":
    unittest.main()
