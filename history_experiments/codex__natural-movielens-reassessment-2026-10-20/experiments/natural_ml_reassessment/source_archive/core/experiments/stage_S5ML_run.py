"""One frozen Stage S5ML MovieLens Equal M3OE run.

This is the MovieLens cross-dataset replication of the Stage S5 four-arm
mechanism factorisation.  The arm semantics (persistent Adam history source
``H`` × current applied update source ``D``; HT/DH post-step overwrite) and
every per-step invariant (I1–I9) are bit-identical to ``stage_S5_run.py``;
only ``dataset`` (``movielens``), ``setting`` (``d2_large``), and the
frozen protocol numeric values sourced from the audited S2/S3 MovieLens
formal runs are different.

Formal execution is orchestrator-gated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from typing import Mapping

import torch

from stage_E1_instrumentation import rng_state_digest
from stage_S3_run import (GROUP_ARCH, GROUP_PRIVATE, GROUP_SHARED, S3Runner,
                          _sha256, write_json, write_rows_csv,
                          build_argparser as s3_parser)
from stage_S4_gradient_controls import (adam_deltas, assert_vector_match,
                                        cosine, gradients_one_forward,
                                        validate_adam_delta_against_clone, vector_norm)
from stage_S4_run import S4Runner
from stage_S5ML_protocol import (ARM_CURRENT, ARM_HISTORY, ARMS, BACKBONE,
                                 COMMON_EPOCHS, DATASET, EQUAL_TABLE_SHA256,
                                 INITIAL_PARAMETER_SHA256,
                                 MODEL_SOURCE_KEY as _MODEL_SOURCE_KEY,
                                 MODEL_SOURCE_SHA256 as _MODEL_SOURCE_SHA256,
                                 MOMENT_DUMP_EPOCHS, MOVIELENS_DATA_SHA256,
                                 OBJECTIVE_WEIGHTING, SEEDS, SETTING, STAGE,
                                 TARGET_LOCAL_DOMAIN, TARGET_PHYSICAL_DOMAIN,
                                 TASKS, inventory, inventory_hash, key)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Field contract bit-compatible with the S5 runner (see stage_S5_run.py);
# S5ML only changes identity strings, not the recorded numeric identities.
S5ML_FIELDS = (
    "stage formal_run_key arm seed epoch step kind batch_fingerprint batch_size domain_counts "
    "L_full L_target L_non decomposition_residual g_full_norm g_target_norm g_non_norm "
    "gradient_cosine history_source current_source "
    "m_norm_persistent v_norm_persistent "
    "delta_full_norm delta_target_norm delta_history_norm delta_applied_norm delta_observed_step_norm "
    "private_delta_norm "
    "applied_cosine_full applied_cosine_target installed_cosine_history "
    "history_formula_residual applied_formula_residual private_full_residual "
    "adam_state_step_min adam_state_step_max optimizer_steps architecture_optimizer_steps "
    "none_count_full none_count_target zero_target_gradient_step "
    "rng_digest_before rng_digest_after"
).split()
S5ML_PROBE_FIELDS = S5ML_FIELDS + [
    "delta_cosine", "m_norm_target_stream", "v_norm_target_stream",
    "moment_cosine_history_target", "moment_cosine_history_full",
]


class S5MLRunner(S4Runner):
    """HT/DH factorises Adam history source from current applied update source.

    The ``_controlled_step`` body is identical to S5's; only the arm-to-factor
    mapping (which is module-level) and the frozen protocol identity values
    are different.  F and T inherit the S4 path bit-for-bit; HT/DH install the
    alternate gradient for ``optimizer.step()`` (committing ``g_H`` to
    ``m/v/tau``) and then overwrite shared parameters with the read-only
    counterfactual ``Delta_D`` calculated from the pre-step persistent state.
    """

    def _output_formal_protocol(self):
        return {"stage": STAGE, "dataset": DATASET, "setting": SETTING,
                "backbone": BACKBONE, "objective": OBJECTIVE_WEIGHTING,
                "arm": self.arm,
                "history_source": ARM_HISTORY[self.arm],
                "current_source": ARM_CURRENT[self.arm],
                "seed": int(self.args.seed), "epochs": 50, "batch_size": 40960,
                "device": "cuda:0", "main_lr": .01, "architecture_lr": .001,
                "betas": [.9, .999], "weight_decay": 1e-5,
                "common_epochs": list(COMMON_EPOCHS),
                "moment_dump_epochs": list(MOMENT_DUMP_EPOCHS)}

    def _validate_args(self):
        a = self.args
        if (str(a.dataset).lower(), str(a.setting), str(a.backbone).lower(),
                str(a.objective_weighting).lower(), str(a.run_type)) != (
                DATASET, SETTING, BACKBONE, OBJECTIVE_WEIGHTING, "joint"):
            raise ValueError(
                "S5ML is {}/{}/{}/{} joint only".format(
                    DATASET, SETTING, BACKBONE, OBJECTIVE_WEIGHTING))
        if a.arm not in ARMS or int(a.seed) not in SEEDS or a.formal_run_key != key(a.seed, a.arm):
            raise ValueError("S5ML arm, seed, or key outside frozen matrix")
        expected = {"epoch": 50, "batch_size": 40960, "learning_rate": 0.01,
                    "learning_rate_darts": 0.001, "weight_decay": 1e-5,
                    "expert_num": 4, "num_workers": 0, "device": "cuda:0",
                    "fcn_dims": [512, 256, 256, 64], "shadow_patience": 10}
        for name, value in expected.items():
            if getattr(a, name) != value:
                raise ValueError("S5ML frozen protocol mismatch: " + name)
        if int(a.allow_protocol_deviation) or int(a.allow_nonformal_seed):
            raise ValueError("S5ML forbids protocol deviation")
        if bool(a.smoke_only) or bool(a.smoke_representative):
            raise ValueError("S5ML runner has no smoke-mode training shortcut")
        for name in ("co_run_pair_id", "planned_peer_run_key", "worker_id", "attempt_id"):
            if not getattr(a, name, None):
                raise ValueError("S5ML missing launch provenance: " + name)

    def __init__(self, args):
        args.objective_weighting = OBJECTIVE_WEIGHTING
        args.shared_access = "FULL"
        args.fresh_role = "fresh_s5ml"
        args.source_formal_run_key = ""
        self.arm = args.arm
        self.history_source = ARM_HISTORY[self.arm]
        self.current_source = ARM_CURRENT[self.arm]
        validate_adam_delta_against_clone(args.device)
        super().__init__(args)
        self.probe_rows = []
        self.zero_target_main_steps = 0
        self.main_steps_seen = 0
        self.initial_inventory = inventory(ROOT)
        if self.initial_inventory[_MODEL_SOURCE_KEY] != _MODEL_SOURCE_SHA256:
            raise RuntimeError("M3OE forward source differs from S3 audited source")
        self._moment_dump_paths: list[str] = []
        self._none_count_full = 0
        self._none_count_target = 0

    def _condition_name(self):
        return "s5ml_" + self.arm

    def _access_formula(self):
        return ("history_source={history_source}; current_source={current_source}; "
                "single optimizer.step() per batch; post-step overwrite only for HT/DH "
                "shared params with pre-step Delta_D".format(
                    history_source=self.history_source, current_source=self.current_source))

    def _dump_moment_state(self, epoch):
        """Snapshot the persistent Adam moments (shared + arch) at epoch end.

        Identical to S5: saves ``{name: {"exp_avg", "exp_avg_sq", "step"}}`` as
        detached CPU clones; strictly read-only with respect to
        ``optimizer.state``.  The file is written immediately so no moment
        history is held in memory.
        """
        if epoch not in MOMENT_DUMP_EPOCHS:
            return
        payload = {}
        for group, optimizer in ((GROUP_SHARED, self.optimizer),
                                 (GROUP_ARCH, self.arch_optimizer)):
            for name in self.partition["groups"][group]:
                state = optimizer.state.get(self.parameter_by_name[name])
                if not state:
                    raise RuntimeError("S5ML moment dump missing Adam state: " + name)
                payload[name] = {"exp_avg": state["exp_avg"].detach().cpu().clone(),
                                 "exp_avg_sq": state["exp_avg_sq"].detach().cpu().clone(),
                                 "step": int(state["step"])}
        path = os.path.join(self.run_dir, "moment_state_epoch{}.pth".format(epoch))
        torch.save(payload, path)
        self._moment_dump_paths.append(path)

    def _controlled_step(self, losses, fingerprint, x_batch, epoch, step, kind):
        groups = self.partition["groups"]
        names = list(groups[GROUP_ARCH] if kind == "architecture" else groups[GROUP_SHARED])
        private = [] if kind == "architecture" else list(groups[GROUP_PRIVATE])
        optimizer = self.arch_optimizer if kind == "architecture" else self.optimizer
        is_probe = step == 0 or kind == "architecture"
        rng_before = rng_state_digest()
        gf, gt, gp = gradients_one_forward(losses["full"], losses["target"],
                                             self.parameter_by_name, names, private)
        none_full = sum(1 for g in gf if g is None)
        none_target = sum(1 for g in gt if g is None)
        self._none_count_full += none_full
        self._none_count_target += none_target
        for name, f, t in zip(names, gf, gt):
            if f is None and t is not None:
                raise ValueError("target gradient without FULL gradient: " + name)
        gnon = [None if f is None and t is None else
                (torch.zeros_like(t) if f is None else f) -
                (torch.zeros_like(f) if t is None else t)
                for f, t in zip(gf, gt)]
        gf_norm, gt_norm, gn_norm = map(vector_norm, (gf, gt, gnon))

        before = [self.parameter_by_name[name].detach().clone() for name in names]
        private_before = [self.parameter_by_name[name].detach().clone() for name in private]
        m_norms, v_norms = [], []
        for name in names:
            state = optimizer.state.get(self.parameter_by_name[name], {})
            if state:
                m_norms.append(vector_norm([state["exp_avg"]]))
                v_norms.append(vector_norm([state["exp_avg_sq"]]))
            else:
                m_norms.append(0.0)
                v_norms.append(0.0)
        m_norm_persistent = float(sum(m_norms))
        v_norm_persistent = float(sum(v_norms))

        df, decay = adam_deltas(self.parameter_by_name, names, gf, optimizer)
        dt, _ = adam_deltas(self.parameter_by_name, names, gt, optimizer)
        dp, _ = adam_deltas(self.parameter_by_name, private, gp, optimizer)
        df_norm, dt_norm = vector_norm(df), vector_norm(dt)

        g_h = gf if self.history_source == "FULL" else gt
        g_d = gf if self.current_source == "FULL" else gt
        installed = [None if g is None else g.detach().clone() for g in g_h]
        for name, grad in zip(names, installed):
            self.parameter_by_name[name].grad = grad
        for name, p_grad in zip(private, gp):
            self.parameter_by_name[name].grad = (p_grad.detach().clone()
                                                  if p_grad is not None else None)

        optimizer.step()
        state_digest_post_step = (self._optimizer_state_digest(optimizer, names + private)
                                  if is_probe else None)

        observed_step = [self.parameter_by_name[name].detach() - old
                         for name, old in zip(names, before)]
        delta_history = df if self.history_source == "FULL" else dt
        history_residual = assert_vector_match(observed_step, delta_history,
                                                "S5ML history formula " + self.arm)

        delta_applied = df if self.current_source == "FULL" else dt
        if self.arm in ("HT", "DH"):
            with torch.no_grad():
                for name, old, full_delta, target_delta in zip(names, before, df, dt):
                    chosen = full_delta if self.current_source == "FULL" else target_delta
                    self.parameter_by_name[name].copy_(old + chosen)
        actual = [self.parameter_by_name[name].detach() - old for name, old in zip(names, before)]
        applied_residual = assert_vector_match(actual, delta_applied,
                                                "S5ML applied formula " + self.arm)

        if all(g is not None for g in installed) and all(g is not None for g in g_h):
            installed_cosine_history = cosine(installed, g_h)
            if installed_cosine_history is None or installed_cosine_history < .99999:
                raise AssertionError("S5ML installed-history cosine below threshold")

        private_delta = [self.parameter_by_name[name].detach() - old
                        for name, old in zip(private, private_before)]
        private_residual = assert_vector_match(private_delta, dp, "S5ML private FULL Adam")

        state_steps = [int(optimizer.state[self.parameter_by_name[name]]["step"])
                       for name in names if self.parameter_by_name[name] in optimizer.state]
        if not state_steps:
            raise AssertionError("shared Adam state absent")
        if min(state_steps) < 1 or max(state_steps) - min(state_steps) != 0:
            raise AssertionError("S5ML Adam step counter non-uniform")

        zero_target = float(gt_norm == 0)
        if kind == "main":
            self.main_steps_seen += 1
            if zero_target:
                self.zero_target_main_steps += 1

        if none_full or none_target:
            raise ValueError("S5ML shared/arch None gradient aborting run")

        loss_values = {label: float(losses[label].detach().item())
                       for label in ("full", "target", "non_target", "equality_residual")}
        if not all(torch.isfinite(torch.tensor(v)).item() for v in loss_values.values()):
            raise FloatingPointError("nonfinite S5ML loss")
        ids = x_batch[:, -1].detach().cpu().long().tolist()
        rng_after = rng_state_digest()
        if is_probe and rng_before != rng_after:
            raise AssertionError("S5ML controlled step consumed RNG")
        applied_cosine_full = cosine(actual, df)
        applied_cosine_target = cosine(actual, dt)
        observed_step_norm = vector_norm(observed_step)
        row = {"stage": STAGE, "formal_run_key": self.formal_run_key, "arm": self.arm,
               "seed": int(self.args.seed), "epoch": epoch, "step": step, "kind": kind,
               "batch_fingerprint": fingerprint, "batch_size": len(x_batch),
               "domain_counts": json.dumps([ids.count(d) for d in range(3)]),
               "L_full": loss_values["full"], "L_target": loss_values["target"],
               "L_non": loss_values["non_target"],
               "decomposition_residual": loss_values["equality_residual"],
               "g_full_norm": gf_norm, "g_target_norm": gt_norm, "g_non_norm": gn_norm,
               "gradient_cosine": cosine(gf, gt),
               "history_source": self.history_source, "current_source": self.current_source,
               "m_norm_persistent": m_norm_persistent,
               "v_norm_persistent": v_norm_persistent,
               "delta_full_norm": df_norm, "delta_target_norm": dt_norm,
               "delta_history_norm": vector_norm(delta_history),
               "delta_applied_norm": vector_norm(delta_applied),
               "delta_observed_step_norm": observed_step_norm,
               "private_delta_norm": vector_norm(private_delta),
               "applied_cosine_full": applied_cosine_full,
               "applied_cosine_target": applied_cosine_target,
               "installed_cosine_history": (cosine(installed, g_h)
                                            if all(g is not None for g in installed)
                                               and all(g is not None for g in g_h) else None),
               "history_formula_residual": history_residual,
               "applied_formula_residual": applied_residual,
               "private_full_residual": private_residual,
               "adam_state_step_min": min(state_steps),
               "adam_state_step_max": max(state_steps),
               "optimizer_steps": self.optimizer_steps + (kind == "main"),
               "architecture_optimizer_steps": self.architecture_optimizer_steps
                                                + (kind == "architecture"),
               "none_count_full": none_full, "none_count_target": none_target,
               "zero_target_gradient_step": int(zero_target),
               "rng_digest_before": rng_before, "rng_digest_after": rng_after}
        self.gradient_rows.append(row)
        if is_probe:
            probe = dict(row)
            probe.update({"delta_cosine": cosine(df, dt),
                          "m_norm_target_stream": sum(
                              vector_norm([optimizer.state[self.parameter_by_name[name]]["exp_avg"]])
                              for name in names
                              if self.parameter_by_name[name] in optimizer.state),
                          "v_norm_target_stream": sum(
                              vector_norm([optimizer.state[self.parameter_by_name[name]]["exp_avg_sq"]])
                              for name in names
                              if self.parameter_by_name[name] in optimizer.state),
                          "moment_cosine_history_target": self._moment_cosine(
                              optimizer, names, g_h),
                          "moment_cosine_history_full": self._moment_cosine(
                              optimizer, names, gf)})
            self.probe_rows.append(probe)
        self.batch_rows[-1].update({"L_full": loss_values["full"],
                                    "L_target": loss_values["target"],
                                    "L_non": loss_values["non_target"],
                                    "decomposition_residual": loss_values["equality_residual"],
                                    "applied_shared_gradient_norm": vector_norm(installed),
                                    "applied_private_gradient_norm": vector_norm(gp),
                                    "applied_architecture_gradient_norm": vector_norm(installed)
                                    if kind == "architecture" else "",
                                    "gradient_substitution_valid": 1})
        if not all(bool(torch.isfinite(p).all().item()) for p in self.model.parameters()):
            raise FloatingPointError("nonfinite parameter")
        if is_probe and state_digest_post_step != self._optimizer_state_digest(
                optimizer, names + private):
            raise AssertionError("S5ML optimizer state mutated outside optimizer.step()")

    def _optimizer_state_digest(self, optimizer, names):
        digest = hashlib.sha256()
        for name in names:
            state = optimizer.state.get(self.parameter_by_name[name])
            if not state:
                continue
            digest.update(state["exp_avg"].detach().cpu().numpy().tobytes())
            digest.update(state["exp_avg_sq"].detach().cpu().numpy().tobytes())
            digest.update(str(int(state["step"])).encode("utf-8"))
        return digest.hexdigest()

    def _moment_cosine(self, optimizer, names, gradients):
        ms = [optimizer.state[self.parameter_by_name[name]]["exp_avg"]
              for name in names if self.parameter_by_name[name] in optimizer.state]
        gs = [g for g in gradients if g is not None]
        if not ms or len(ms) != len(gs):
            return None
        return cosine(ms, gs)

    def _architecture_update(self, architecture_iterator, epoch):
        iterator = super()._architecture_update(architecture_iterator, epoch)
        self._dump_moment_state(epoch)
        return iterator

    def _check_source_drift(self):
        if inventory(ROOT) != self.initial_inventory:
            raise RuntimeError("S5ML producer source changed during training")

    def write_outputs(self, fit_info: Mapping[str, object], wall_seconds: float):
        self._check_source_drift()
        S3Runner.write_outputs(self, fit_info, wall_seconds)
        for filename in ("batch_trace.csv", "metrics_epoch.csv", "metrics_test.csv",
                         "metrics_summary.csv"):
            path = os.path.join(self.run_dir, filename)
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields, rows = list(reader.fieldnames or []), list(reader)
            if "stage" not in fields:
                raise RuntimeError("missing stage field in " + filename)
            write_rows_csv(path, [dict(row, stage=STAGE) for row in rows], fields)
        write_rows_csv(os.path.join(self.run_dir, "gradient_step.csv"),
                       self.gradient_rows, S5ML_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "probe.csv"),
                       self.probe_rows, S5ML_PROBE_FIELDS)
        sources = inventory(ROOT)
        config_path = os.path.join(self.run_dir, "config.json")
        manifest_path = os.path.join(self.run_dir, "run_manifest.json")
        protocol = self._output_formal_protocol()
        observed = {"stage": STAGE, "dataset": str(self.args.dataset).lower(),
                    "setting": str(self.args.setting), "backbone": str(self.args.backbone).lower(),
                    "objective": str(self.objective_weighting), "arm": self.arm,
                    "history_source": ARM_HISTORY[self.arm],
                    "current_source": ARM_CURRENT[self.arm],
                    "seed": int(self.args.seed), "epochs": int(self.args.epoch),
                    "batch_size": int(self.args.batch_size), "device": str(self.args.device),
                    "main_lr": float(self.args.learning_rate),
                    "architecture_lr": float(self.args.learning_rate_darts),
                    "betas": list(self.optimizer.param_groups[0]["betas"]),
                    "weight_decay": float(self.args.weight_decay),
                    "common_epochs": list(COMMON_EPOCHS),
                    "moment_dump_epochs": list(MOMENT_DUMP_EPOCHS)}
        mismatches = {name: {"expected": value, "observed": observed.get(name)}
                      for name, value in protocol.items() if observed.get(name) != value}
        if mismatches:
            raise RuntimeError("S5ML observed output protocol mismatch: " + str(mismatches))
        moment_hashes = {os.path.basename(path): _sha256(path)
                         for path in self._moment_dump_paths}
        for path in (config_path, manifest_path):
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
            value.update({"stage": STAGE, "condition": self.arm, "arm": self.arm,
                          "shared_access": self.arm, "fresh_role": "fresh_s5ml",
                          "source_stage": STAGE, "provenance_type": "fresh_s5ml",
                          "history_source": ARM_HISTORY[self.arm],
                          "current_source": ARM_CURRENT[self.arm],
                          "formal_protocol": protocol,
                          "formal_protocol_observed": observed,
                          "formal_protocol_mismatches": {}, "formal_valid": True,
                          "formal_run_key": self.formal_run_key,
                          "code_hashes": sources,
                          "code_inventory_hash": inventory_hash(sources),
                          "source_inventory_digest": inventory_hash(sources),
                          "source_code_hash": sources["experiments/stage_S5ML_run.py"],
                          "forward_model_source_sha256": sources[_MODEL_SOURCE_KEY],
                          "probe_count": len(self.probe_rows),
                          "gradient_batch_count": len(self.gradient_rows),
                          "zero_target_main_steps": self.zero_target_main_steps,
                          "main_steps_seen": self.main_steps_seen,
                          "moment_dump_epochs": list(MOMENT_DUMP_EPOCHS),
                          "moment_dumps": sorted(moment_hashes),
                          "moment_dump_hashes": moment_hashes,
                          "none_count_full": self._none_count_full,
                          "none_count_target": self._none_count_target,
                          "dataset_file_hash": MOVIELENS_DATA_SHA256,
                          "coefficient_table_hash": EQUAL_TABLE_SHA256,
                          "initial_parameter_state_hash": INITIAL_PARAMETER_SHA256[int(self.args.seed)],
                          "target_local_domain": TARGET_LOCAL_DOMAIN,
                          "target_physical_domain": TARGET_PHYSICAL_DOMAIN,
                          "target_task_names": dict(TASKS)})
            if path == manifest_path:
                value["artifact_hashes"].update({f: _sha256(os.path.join(self.run_dir, f))
                                                   for f in ("config.json", "batch_trace.csv",
                                                             "metrics_epoch.csv",
                                                             "metrics_test.csv",
                                                             "metrics_summary.csv",
                                                             "gradient_step.csv",
                                                             "probe.csv")})
                value["artifact_hashes"].update(moment_hashes)
                value["artifacts"] = sorted(set(value["artifacts"])
                                              | {"gradient_step.csv", "probe.csv"}
                                              | set(moment_hashes))
            write_json(path, value)


def build_argparser():
    parser = s3_parser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    for action in parser._actions:
        if action.dest in ("objective_weighting", "shared_access", "fresh_role",
                           "source_formal_run_key"):
            action.required = False
        if action.dest == "dataset":
            action.choices = ("movielens", "kuairand")
        if action.dest == "fresh_role":
            action.choices = ("fresh_s5ml", "fresh_target_only",
                              "fresh_sample_full_replacement")
    return parser


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.run_dir, exist_ok=False)
    started_ns = time.time_ns()
    started = started_ns / 1e9
    interval_path = os.path.join(args.run_dir, "process_interval.json")
    try:
        runner = S5MLRunner(args)
        info = runner.fit()
        runner.write_outputs(info, time.time() - started)
        write_json(interval_path, {"pid": os.getpid(), "started_ns": started_ns,
                                   "finished_ns": time.time_ns(), "status": "OK"})
        print("[stage_S5ML] completed {}".format(args.formal_run_key), flush=True)
        return 0
    except Exception as exc:
        write_json(interval_path, {"pid": os.getpid(), "started_ns": started_ns,
                                   "finished_ns": time.time_ns(), "status": "FAILED"})
        write_json(os.path.join(args.run_dir, "failure.json"),
                   {"stage": STAGE, "key": args.formal_run_key,
                    "exception_type": type(exc).__name__, "exception": str(exc),
                    "exception_errno": getattr(exc, "errno", None),
                    "wall_seconds": time.time() - started})
        raise


if __name__ == "__main__":
    raise SystemExit(main())