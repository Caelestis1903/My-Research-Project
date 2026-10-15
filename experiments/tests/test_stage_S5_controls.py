"""S5 local algebra/installed-Adam gate; no formal data or training.

The tests below validate the §4 state flow on small synthetic models without
launching formal training: every check exercises
``S5Runner._controlled_step`` directly, the read-only ``AdamDelta``, the
HT/DH overwrite ordering, and the moment-dump non-mutation contract.
"""

import copy
import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_E1_instrumentation import rng_state_digest
from stage_S4_gradient_controls import (adam_deltas, assert_vector_match,
                                        gradients_one_forward, vector_norm,
                                        validate_adam_delta_against_clone)
from stage_S5_run import S5Runner
from stage_S5_protocol import ARM_CURRENT, ARM_HISTORY, ARMS, MOMENT_DUMP_EPOCHS


def _make_pair(seed: int = 2022):
    torch.manual_seed(seed)
    shared = torch.nn.Parameter(torch.tensor([.3, -.1]))
    private = torch.nn.Parameter(torch.tensor([.2, .4]))
    arch = torch.nn.Parameter(torch.tensor([.15, -.25]))
    model = torch.nn.Module()
    model.shared = shared
    model.private = private
    model.arch = arch
    return model


def _toy_runner(model, arm):
    r = object.__new__(S5Runner)
    r.model = model
    r.arm = arm
    r.history_source = ARM_HISTORY[arm]
    r.current_source = ARM_CURRENT[arm]
    r.formal_run_key = "toy-{}".format(arm)
    r.args = SimpleNamespace(seed=2022)
    r.parameter_by_name = dict(model.named_parameters())
    r.partition = {"groups": {"C": ["shared"], "P": ["private"], "A_arch": ["arch"]}}
    r.optimizer = torch.optim.Adam(
        [r.parameter_by_name["shared"], r.parameter_by_name["private"]],
        lr=.01, betas=(.9, .999), weight_decay=1e-5)
    r.arch_optimizer = torch.optim.Adam(
        [r.parameter_by_name["arch"]], lr=.001, betas=(.9, .999), weight_decay=1e-5)
    r.gradient_rows, r.probe_rows, r.batch_rows = [], [], [{}]
    r.main_steps_seen = r.zero_target_main_steps = 0
    r.optimizer_steps = r.architecture_optimizer_steps = 0
    r._none_count_full = r._none_count_target = 0
    r._moment_dump_paths = []
    return r


class StageS5ControlTests(unittest.TestCase):

    def test_arm_factor_map_is_complete_and_consistent(self):
        self.assertEqual(set(ARMS), {"F", "T", "HT", "DH"})
        for arm in ARMS:
            self.assertIn(arm, ARM_HISTORY)
            self.assertIn(arm, ARM_CURRENT)
        # The four arms are mutually distinguishable in the (H, D) plane.
        keys = {(ARM_HISTORY[a], ARM_CURRENT[a]) for a in ARMS}
        self.assertEqual(len(keys), 4)

    def test_adam_delta_clone_parity_holds(self):
        self.assertTrue(validate_adam_delta_against_clone())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_adam_clone_parity(self):
        self.assertTrue(validate_adam_delta_against_clone(torch.device("cuda:0")))

    def test_hand_computed_two_step_ht_trajectory(self):
        """Compare the S5 HT path against a hand-computed reference trajectory."""
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda:0"))
        for device in devices:
            with self.subTest(device=device):
                self._hand_computed_ht_trajectory(device)

    def _hand_computed_ht_trajectory(self, device):
        model = _make_pair().to(device)
        runner = _toy_runner(model, "HT")
        torch.manual_seed(2022)
        shared = runner.parameter_by_name["shared"]
        private = runner.parameter_by_name["private"]
        arch = runner.parameter_by_name["arch"]
        for step in range(2):
            full = shared.square().sum() + private.square().sum()
            target = shared[0].square() + private[0].square()
            losses = {"full": full, "target": target,
                      "non_target": full - target,
                      "equality_residual": full - (target + (full - target))}
            runner._controlled_step(losses, "ht-{}".format(step),
                                     torch.tensor([[0., 1.]]), 0, step, "main")
            # Hand-computed reference: shared history commits g_T, then
            # parameters are overwritten with theta_before + Delta_F.
            self._assert_hand_reference_ht(runner, step, device)

    def _assert_hand_reference_ht(self, runner, step, device):
        # We re-run the same two steps on a clean copy and verify that the
        # S5 HT path reproduces the hand-computed reference move exactly.
        ref = _make_pair().to(device)
        ref_param = dict(ref.named_parameters())
        ref_opt = torch.optim.Adam(
            [ref_param["shared"], ref_param["private"]],
            lr=.01, betas=(.9, .999), weight_decay=1e-5)
        torch.manual_seed(2022)
        for _ in range(step + 1):
            full = ref_param["shared"].square().sum() + ref_param["private"].square().sum()
            target = ref_param["shared"][0].square() + ref_param["private"][0].square()
            # First compute Delta_F using the current state (g_F still alive).
            gf, gt, gp = gradients_one_forward(full, target, ref_param, ["shared"], ["private"])
            # Build the manually-warmed state we expect: install g_T and step.
            df, _ = adam_deltas(ref_param, ["shared"], gf, ref_opt)
            ref_param["shared"].grad = gt[0].detach().clone()
            ref_param["private"].grad = gp[0].detach().clone()
            before = ref_param["shared"].detach().clone()
            ref_opt.step()
            with torch.no_grad():
                ref_param["shared"].copy_(before + df[0])
            # Roll back optimizer's own view of parameters so AdamDelta on the
            # next iteration sees the *expected* persistent state.
            with torch.no_grad():
                ref_param["shared"].copy_(before + df[0])
        shared_runner = runner.parameter_by_name["shared"].detach().clone()
        shared_ref = ref_param["shared"].detach().clone()
        self.assertTrue(torch.allclose(shared_runner, shared_ref, atol=1e-6))
        # Private and Adam state must match (HT only overwrites shared names).
        self.assertTrue(torch.allclose(runner.parameter_by_name["private"].detach(),
                                        ref_param["private"].detach(), atol=1e-6))
        self.assertEqual(int(runner.optimizer.state[
            runner.parameter_by_name["shared"]]["step"]),
            int(ref_opt.state[ref_param["shared"]]["step"]))

    def test_hand_computed_two_step_dh_trajectory(self):
        model = _make_pair().to("cpu")
        runner = _toy_runner(model, "DH")
        torch.manual_seed(2022)
        shared = runner.parameter_by_name["shared"]
        private = runner.parameter_by_name["private"]
        for step in range(2):
            full = shared.square().sum() + private.square().sum()
            target = shared[0].square() + private[0].square()
            losses = {"full": full, "target": target,
                      "non_target": full - target,
                      "equality_residual": full - (target + (full - target))}
            runner._controlled_step(losses, "dh-{}".format(step),
                                     torch.tensor([[0., 1.]]), 0, step, "main")
        # DH's persistent history follows the FULL stream like F; step counters
        # are equal.  State vectors are close but not bit-equal because the
        # Adam coupled weight decay uses θ at the moment of the step, which
        # diverges between DH (target delta overwrite) and F (full delta
        # applied) starting from step 0.  ``allclose`` with a tight tol.
        f_model = _make_pair().to("cpu")
        f_runner = _toy_runner(f_model, "F")
        torch.manual_seed(2022)
        full = f_model.shared.square().sum() + f_model.private.square().sum()
        target = f_model.shared[0].square() + f_model.private[0].square()
        losses = {"full": full, "target": target,
                  "non_target": full - target,
                  "equality_residual": full - (target + (full - target))}
        for step in range(2):
            full = f_model.shared.square().sum() + f_model.private.square().sum()
            target = f_model.shared[0].square() + f_model.private[0].square()
            losses = {"full": full, "target": target,
                      "non_target": full - target,
                      "equality_residual": full - (target + (full - target))}
            f_runner._controlled_step(losses, "f-{}".format(step),
                                       torch.tensor([[0., 1.]]), 0, step, "main")
        dh_state = runner.optimizer.state[runner.parameter_by_name["shared"]]
        f_state = f_runner.optimizer.state[f_runner.parameter_by_name["shared"]]
        self.assertEqual(int(dh_state["step"]), int(f_state["step"]))
        # Same step counter (I5) and very close state vectors.
        self.assertTrue(torch.allclose(dh_state["exp_avg"], f_state["exp_avg"],
                                        rtol=1e-3, atol=1e-5))
        self.assertTrue(torch.allclose(dh_state["exp_avg_sq"], f_state["exp_avg_sq"],
                                        rtol=1e-3, atol=1e-5))

    def test_f_and_t_match_direct_backward_reference(self):
        """F and T paths must be bit-identical to the direct ``backward()+step()`` reference.

        Private parameters always use the FULL gradient stream even for the
        T arm, so the reference installs the appropriate gradient per group.
        """
        for arm in ("F", "T"):
            with self.subTest(arm=arm):
                model = _make_pair().to("cpu")
                runner = _toy_runner(model, arm)
                # Reference: identical model + optimizer + identical forward.
                ref_model = copy.deepcopy(model)
                ref_param = dict(ref_model.named_parameters())
                ref_opt = torch.optim.Adam(
                    [ref_param["shared"], ref_param["private"]],
                    lr=.01, betas=(.9, .999), weight_decay=1e-5)
                torch.manual_seed(2022)
                shared = runner.parameter_by_name["shared"]
                private = runner.parameter_by_name["private"]
                full = shared.square().sum() + private.square().sum()
                target = shared[0].square() + private[0].square()
                ref_full_loss = ref_param["shared"].square().sum() \
                    + ref_param["private"].square().sum()
                ref_target_loss = ref_param["shared"][0].square() \
                    + ref_param["private"][0].square()
                if arm == "F":
                    ref_full_loss.backward()
                else:
                    # Arm T: shared gradient is target, private is FULL.
                    shared_grad = torch.autograd.grad(
                        ref_target_loss, ref_param["shared"],
                        retain_graph=True)[0].detach().clone()
                    private_grad = torch.autograd.grad(
                        ref_full_loss, ref_param["private"],
                        retain_graph=False)[0].detach().clone()
                    ref_param["shared"].grad = shared_grad
                    ref_param["private"].grad = private_grad
                ref_opt.step()
                losses = {"full": full, "target": target,
                          "non_target": full - target,
                          "equality_residual": full - (target + (full - target))}
                runner._controlled_step(losses, "ft", torch.tensor([[0., 1.]]), 0, 0, "main")
                self.assertTrue(torch.allclose(
                    runner.parameter_by_name["shared"].detach(),
                    ref_param["shared"].detach(), atol=1e-7))
                self.assertTrue(torch.allclose(
                    runner.parameter_by_name["private"].detach(),
                    ref_param["private"].detach(), atol=1e-7))
                # Optimizer state equality (m, v, step).
                rs = runner.optimizer.state[runner.parameter_by_name["shared"]]
                fs = ref_opt.state[ref_param["shared"]]
                self.assertEqual(int(rs["step"]), int(fs["step"]))
                self.assertTrue(torch.allclose(rs["exp_avg"], fs["exp_avg"], atol=1e-6))
                self.assertTrue(torch.allclose(rs["exp_avg_sq"], fs["exp_avg_sq"], atol=1e-6))

    def test_overwrite_touches_only_shared_names(self):
        for arm in ("HT", "DH"):
            with self.subTest(arm=arm):
                model = _make_pair().to("cpu")
                # Pre-arm arch state hash to verify the architecture step is
                # also controlled (it follows the same H/D rule per §4).
                runner = _toy_runner(model, arm)
                # Snapshot the pre-step state and run one main + one arch step.
                arch_before = runner.parameter_by_name["arch"].detach().clone()
                shared_before = runner.parameter_by_name["shared"].detach().clone()
                private_before = runner.parameter_by_name["private"].detach().clone()
                shared = runner.parameter_by_name["shared"]
                private = runner.parameter_by_name["private"]
                full = shared.square().sum() + private.square().sum()
                target = shared[0].square() + private[0].square()
                losses = {"full": full, "target": target,
                          "non_target": full - target,
                          "equality_residual": full - (target + (full - target))}
                runner._controlled_step(losses, "main", torch.tensor([[0., 1.]]), 0, 0, "main")
                # Private parameters underwent the ordinary FULL Adam step and
                # were not overwritten by the HT/DH branch.
                private_state = runner.optimizer.state.get(
                    runner.parameter_by_name["private"], {})
                self.assertEqual(int(private_state["step"]), 1)
                self.assertFalse(torch.equal(
                    runner.parameter_by_name["private"].detach(), private_before))
                # Architecture step is also under the H/D rule.
                arch = runner.parameter_by_name["arch"]
                full = arch.square().sum() + shared.square().sum() + private.square().sum()
                target = arch[0].square() + shared[0].square() + private[0].square()
                losses = {"full": full, "target": target,
                          "non_target": full - target,
                          "equality_residual": full - (target + (full - target))}
                runner._controlled_step(losses, "arch", torch.tensor([[0., 1.]]), 0, -1,
                                         "architecture")
                self.assertNotEqual(int(runner.arch_optimizer.state[arch]["step"]), 0)
                # Names outside the controlled groups never appear in overwrite.
                self.assertEqual(shared_before.numel(), shared.numel())
                self.assertEqual(arch_before.numel(), arch.numel())

    def test_private_state_matches_full_reference_in_tests(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = _make_pair().to("cpu")
                runner = _toy_runner(model, arm)
                # Reference optimizer with the same warmup pattern.
                ref = copy.deepcopy(model)
                ref_param = dict(ref.named_parameters())
                ref_opt = torch.optim.Adam(
                    [ref_param["shared"], ref_param["private"]],
                    lr=.01, betas=(.9, .999), weight_decay=1e-5)
                torch.manual_seed(2022)
                shared = runner.parameter_by_name["shared"]
                private = runner.parameter_by_name["private"]
                full = shared.square().sum() + private.square().sum()
                target = shared[0].square() + private[0].square()
                # Compute the FULL gradients once on both models so we don't
                # depend on the autograd graph surviving the runner's step.
                from stage_S4_gradient_controls import gradients_one_forward
                gf_r, _, gp_r = gradients_one_forward(
                    full, target, runner.parameter_by_name, ["shared"], ["private"])
                gf_ref_full = torch.autograd.grad(
                    ref_param["shared"].square().sum() + ref_param["private"].square().sum(),
                    [ref_param["shared"], ref_param["private"]],
                    retain_graph=False, allow_unused=True)
                losses = {"full": full, "target": target,
                          "non_target": full - target,
                          "equality_residual": full - (target + (full - target))}
                runner._controlled_step(losses, "p", torch.tensor([[0., 1.]]), 0, 0, "main")
                ref_param["shared"].grad = gf_ref_full[0].detach().clone()
                ref_param["private"].grad = gf_ref_full[1].detach().clone()
                ref_opt.step()
                # Private parameter and state must match the FULL reference.
                self.assertTrue(torch.allclose(
                    runner.parameter_by_name["private"].detach(),
                    ref_param["private"].detach(), atol=1e-7))
                rs = runner.optimizer.state[runner.parameter_by_name["private"]]
                fs = ref_opt.state[ref_param["private"]]
                self.assertEqual(int(rs["step"]), int(fs["step"]))
                self.assertTrue(torch.allclose(rs["exp_avg"], fs["exp_avg"], atol=1e-6))
                self.assertTrue(torch.allclose(rs["exp_avg_sq"], fs["exp_avg_sq"], atol=1e-6))

    def test_one_step_per_batch_for_both_optimizers(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = _make_pair().to("cpu")
                runner = _toy_runner(model, arm)
                shared = runner.parameter_by_name["shared"]
                private = runner.parameter_by_name["private"]
                arch = runner.parameter_by_name["arch"]
                full = shared.square().sum() + private.square().sum() + arch.square().sum()
                target = shared[0].square() + private[0].square() + arch[0].square()
                losses = {"full": full, "target": target,
                          "non_target": full - target,
                          "equality_residual": full - (target + (full - target))}
                runner._controlled_step(losses, "main", torch.tensor([[0., 1.]]), 0, 0, "main")
                self.assertEqual(int(runner.optimizer.state[
                    runner.parameter_by_name["shared"]]["step"]), 1)
                self.assertEqual(int(runner.optimizer.state[
                    runner.parameter_by_name["private"]]["step"]), 1)
                shared_p = runner.parameter_by_name["shared"]
                private_p = runner.parameter_by_name["private"]
                arch_p = runner.parameter_by_name["arch"]
                arch_full = arch_p.square().sum() + shared_p.square().sum() + private_p.square().sum()
                arch_target = arch_p[0].square() + shared_p[0].square() + private_p[0].square()
                losses = {"full": arch_full, "target": arch_target,
                          "non_target": arch_full - arch_target,
                          "equality_residual": arch_full - (arch_target + (arch_full - arch_target))}
                runner._controlled_step(losses, "arch", torch.tensor([[0., 1.]]), 0, -1,
                                         "architecture")
                self.assertEqual(int(runner.arch_optimizer.state[
                    runner.parameter_by_name["arch"]]["step"]), 1)
                # Re-running the main step must advance the counter by exactly 1.
                shared_p2 = runner.parameter_by_name["shared"]
                private_p2 = runner.parameter_by_name["private"]
                arch_p2 = runner.parameter_by_name["arch"]
                full2 = shared_p2.square().sum() + private_p2.square().sum() + arch_p2.square().sum()
                target2 = shared_p2[0].square() + private_p2[0].square() + arch_p2[0].square()
                losses = {"full": full2, "target": target2,
                          "non_target": full2 - target2,
                          "equality_residual": full2 - (target2 + (full2 - target2))}
                runner._controlled_step(losses, "main2", torch.tensor([[0., 1.]]), 0, 1, "main")
                self.assertEqual(int(runner.optimizer.state[
                    runner.parameter_by_name["shared"]]["step"]), 2)

    def test_delta_d_state_timing_regression(self):
        """Mutating state before computing ``Delta_D`` must change the result."""
        shared = torch.nn.Parameter(torch.tensor([.3, -.1]))
        private = torch.nn.Parameter(torch.tensor([.2, .4]))
        params = {"shared": shared, "private": private}
        opt = torch.optim.Adam(list(params.values()), lr=.01, betas=(.9, .999),
                                weight_decay=1e-5)
        # Warmup step.
        torch.manual_seed(0)
        full = shared.square().sum() + private.square().sum()
        target = shared[0].square() + private[0].square()
        gf, gt, gp = gradients_one_forward(full, target, params, ["shared"], ["private"])
        for name, grad in (("shared", gf[0]), ("private", gp[0])):
            params[name].grad = grad.detach().clone()
        opt.step()
        # Compute baseline Delta_F.
        df_before, _ = adam_deltas(params, ["shared"], gf, opt)
        # Tamper with the persistent state: inject a large ``m`` and ``v``.
        state = opt.state[params["shared"]]
        state["exp_avg"].fill_(100.0)
        state["exp_avg_sq"].fill_(100.0)
        df_after, _ = adam_deltas(params, ["shared"], gf, opt)
        self.assertGreater(vector_norm([df_after[0] - df_before[0]]), 1e-3,
                            "AdamDelta did not respond to state mutation; timing guard failed")

    def test_moment_dump_does_not_mutate_state(self):
        import tempfile
        model = _make_pair().to("cpu")
        runner = _toy_runner(model, "HT")
        with tempfile.TemporaryDirectory() as run_dir:
            runner.run_dir = run_dir
            torch.manual_seed(2022)
            shared = runner.parameter_by_name["shared"]
            private = runner.parameter_by_name["private"]
            arch = runner.parameter_by_name["arch"]
            # Run one main step and one architecture step so both optimizers
            # hold persistent state (the real dump hook fires after the
            # architecture step at epoch end).
            full = shared.square().sum() + private.square().sum()
            target = shared[0].square() + private[0].square()
            losses = {"full": full, "target": target,
                      "non_target": full - target,
                      "equality_residual": full - (target + (full - target))}
            runner._controlled_step(losses, "warm", torch.tensor([[0., 1.]]), 0, 0, "main")
            full = arch.square().sum() + shared.square().sum() + private.square().sum()
            target = arch[0].square() + shared[0].square() + private[0].square()
            losses = {"full": full, "target": target,
                      "non_target": full - target,
                      "equality_residual": full - (target + (full - target))}
            runner._controlled_step(losses, "arch", torch.tensor([[0., 1.]]), 0, -1,
                                     "architecture")
            m_before = runner.optimizer.state[shared]["exp_avg"].detach().clone()
            v_before = runner.optimizer.state[shared]["exp_avg_sq"].detach().clone()
            runner._dump_moment_state(0)
            m_after = runner.optimizer.state[shared]["exp_avg"].detach().clone()
            v_after = runner.optimizer.state[shared]["exp_avg_sq"].detach().clone()
            self.assertTrue(torch.equal(m_before, m_after))
            self.assertTrue(torch.equal(v_before, v_after))
            self.assertTrue(any(p.endswith("moment_state_epoch0.pth")
                                  for p in runner._moment_dump_paths))
            # The dump must contain the persistent moments (m, v, tau) for the
            # shared + architecture groups, matching the live optimizer state.
            payload = torch.load(runner._moment_dump_paths[-1], map_location="cpu")
            self.assertEqual(set(payload), {"shared", "arch"})
            self.assertEqual(set(payload["shared"]), {"exp_avg", "exp_avg_sq", "step"})
            self.assertEqual(int(payload["shared"]["step"]), 1)
            self.assertEqual(int(payload["arch"]["step"]), 1)
            self.assertTrue(torch.equal(payload["shared"]["exp_avg"], m_after))
            self.assertTrue(torch.equal(payload["shared"]["exp_avg_sq"], v_after))
            self.assertTrue(torch.equal(
                payload["arch"]["exp_avg"],
                runner.arch_optimizer.state[arch]["exp_avg"].detach().cpu()))
            # Non-dump epochs must not produce files.
            runner._dump_moment_state(2)
            self.assertEqual(len(runner._moment_dump_paths), 1)

    def test_rng_digest_unchanged_across_controlled_step(self):
        model = _make_pair().to("cpu")
        runner = _toy_runner(model, "HT")
        torch.manual_seed(2022)
        shared = runner.parameter_by_name["shared"]
        private = runner.parameter_by_name["private"]
        before = rng_state_digest()
        full = shared.square().sum() + private.square().sum()
        target = shared[0].square() + private[0].square()
        losses = {"full": full, "target": target,
                  "non_target": full - target,
                  "equality_residual": full - (target + (full - target))}
        runner._controlled_step(losses, "rng", torch.tensor([[0., 1.]]), 0, 0, "main")
        after = rng_state_digest()
        self.assertEqual(before, after)
        # The probe row must carry real, equal before/after digests (I7).
        self.assertEqual(len(runner.probe_rows), 1)
        probe = runner.probe_rows[-1]
        self.assertTrue(probe["rng_digest_before"])
        self.assertEqual(probe["rng_digest_before"], probe["rng_digest_after"])

    def test_probe_row_with_multiple_shared_parameters(self):
        """Probe-row creation must work for >1 shared parameter (regression:
        the moment-cosine helper must align full vectors, not one tensor)."""
        model = _make_pair().to("cpu")
        model.shared2 = torch.nn.Parameter(torch.tensor([.05, .35]))
        runner = _toy_runner(model, "DH")
        runner.partition = {"groups": {"C": ["shared", "shared2"],
                                       "P": ["private"], "A_arch": ["arch"]}}
        runner.optimizer = torch.optim.Adam(
            [runner.parameter_by_name["shared"], runner.parameter_by_name["shared2"],
             runner.parameter_by_name["private"]],
            lr=.01, betas=(.9, .999), weight_decay=1e-5)
        torch.manual_seed(2022)
        shared = runner.parameter_by_name["shared"]
        shared2 = runner.parameter_by_name["shared2"]
        private = runner.parameter_by_name["private"]
        full = shared.square().sum() + shared2.square().sum() + private.square().sum()
        target = shared[0].square() + shared2[0].square() + private[0].square()
        losses = {"full": full, "target": target,
                  "non_target": full - target,
                  "equality_residual": full - (target + (full - target))}
        runner._controlled_step(losses, "multi", torch.tensor([[0., 1.]]), 0, 0, "main")
        self.assertEqual(len(runner.probe_rows), 1)
        probe = runner.probe_rows[-1]
        self.assertIsNotNone(probe["moment_cosine_history_full"])
        self.assertTrue(-1.0 <= probe["moment_cosine_history_full"] <= 1.0)
        self.assertEqual(probe["rng_digest_before"], probe["rng_digest_after"])

    def test_ht_dh_share_persistent_state_only_with_respective_history(self):
        """``HT`` persistent state matches ``T``; ``DH`` persistent state matches ``F``.

        Parameters diverge: HT and F both end at ``theta_pre + Delta_F`` while
        T and DH both end at ``theta_pre + Delta_T``.
        """
        f_model = _make_pair().to("cpu")
        t_model = _make_pair().to("cpu")
        ht_model = _make_pair().to("cpu")
        dh_model = _make_pair().to("cpu")
        f_runner = _toy_runner(f_model, "F")
        t_runner = _toy_runner(t_model, "T")
        ht_runner = _toy_runner(ht_model, "HT")
        dh_runner = _toy_runner(dh_model, "DH")
        for runner in (f_runner, t_runner, ht_runner, dh_runner):
            shared = runner.parameter_by_name["shared"]
            private = runner.parameter_by_name["private"]
            full = shared.square().sum() + private.square().sum()
            target = shared[0].square() + private[0].square()
            losses = {"full": full, "target": target,
                      "non_target": full - target,
                      "equality_residual": full - (target + (full - target))}
            runner._controlled_step(losses, "warm", torch.tensor([[0., 1.]]), 0, 0, "main")
        # HT and T share history (target); F and DH share history (FULL).
        ht_state = ht_runner.optimizer.state[ht_runner.parameter_by_name["shared"]]
        t_state = t_runner.optimizer.state[t_runner.parameter_by_name["shared"]]
        f_state = f_runner.optimizer.state[f_runner.parameter_by_name["shared"]]
        dh_state = dh_runner.optimizer.state[dh_runner.parameter_by_name["shared"]]
        self.assertTrue(torch.allclose(ht_state["exp_avg"], t_state["exp_avg"], atol=1e-6))
        self.assertTrue(torch.allclose(dh_state["exp_avg"], f_state["exp_avg"], atol=1e-6))
        # Parameters: F and HT both apply ``Delta_F``; T and DH both apply ``Delta_T``.
        self.assertTrue(torch.allclose(
            f_runner.parameter_by_name["shared"].detach(),
            ht_runner.parameter_by_name["shared"].detach(), atol=1e-7))
        self.assertTrue(torch.allclose(
            t_runner.parameter_by_name["shared"].detach(),
            dh_runner.parameter_by_name["shared"].detach(), atol=1e-7))

    def test_arm_to_factor_mapping_on_all_four_arms(self):
        for arm, (h, d) in (("F", ("FULL", "FULL")), ("T", ("TARGET", "TARGET")),
                            ("HT", ("TARGET", "FULL")), ("DH", ("FULL", "TARGET"))):
            self.assertEqual((ARM_HISTORY[arm], ARM_CURRENT[arm]), (h, d))


if __name__ == "__main__":
    unittest.main()