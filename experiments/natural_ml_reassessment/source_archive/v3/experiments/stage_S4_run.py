"""One frozen S4 Equal/KuaiRand/M3OE run. Formal execution is orchestrator-gated."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from typing import Mapping

import torch
from torch.utils.data import DataLoader

from stage_S3_run import (S3Runner, GROUP_ARCH, GROUP_PRIVATE, GROUP_SHARED,
                          TorchDataset, _cell_records, _optimizer_audit,
                          _require_finite, _runtime_metadata, _sha256,
                          write_json, write_rows_csv, build_argparser as s3_parser)
from stage_S4_gradient_controls import (adam_deltas, assert_match, assert_vector_match,
                                        cosine, dot,
                                        gradients_one_forward, scale_for, vector_norm,
                                        validate_adam_delta_against_clone)
from stage_S4_protocol import (ARMS, COMMON_EPOCHS, MODEL_SOURCE_KEY,
                               MODEL_SOURCE_SHA256, SEEDS,
                               STAGE, inventory, inventory_hash, key)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIELDS = ("stage formal_run_key arm seed epoch step kind batch_fingerprint batch_size domain_counts "
          "L_full L_target L_non decomposition_residual g_full_norm g_target_norm g_non_norm "
          "gradient_cosine r_N r_U actual_gradient_norm delta_full_norm delta_target_norm "
          "delta_N_predicted_norm delta_actual_norm matching_residual private_delta_norm decay_norm adam_state_step_min "
          "adam_state_step_max optimizer_steps architecture_optimizer_steps zero_case "
          "full_direction_cosine private_full_residual shared_formula_residual").split()
PROBE_FIELDS = FIELDS + ["delta_cosine", "target_gain_full", "target_gain_target",
                         "target_gain_equal_norm_full", "delta_N_norm"]


class S4Runner(S3Runner):
    def _output_formal_protocol(self):
        return {"stage": STAGE, "dataset": "kuairand", "setting": "natural",
                "backbone": "m3oe", "objective": "equal", "arm": self.arm,
                "seed": int(self.args.seed), "epochs": 50, "batch_size": 40960,
                "device": "cuda:0", "main_lr": .01, "architecture_lr": .001,
                "betas": [.9, .999], "weight_decay": 1e-5,
                "common_epochs": list(COMMON_EPOCHS)}

    def _validate_args(self):
        a = self.args
        if (str(a.dataset).lower(), str(a.setting), str(a.backbone).lower(),
                str(a.objective_weighting).lower(), str(a.run_type)) != (
                "kuairand", "natural", "m3oe", "equal", "joint"):
            raise ValueError("S4 is Equal/KuaiRand/natural/M3OE joint only")
        if a.arm not in ARMS or int(a.seed) not in SEEDS or a.formal_run_key != key(a.seed, a.arm):
            raise ValueError("S4 arm, seed, or key outside frozen matrix")
        expected = {"epoch": 50, "batch_size": 40960, "learning_rate": 0.01,
                    "learning_rate_darts": 0.001, "weight_decay": 1e-5,
                    "expert_num": 4, "num_workers": 0, "device": "cuda:0",
                    "fcn_dims": [512, 256, 256, 64], "shadow_patience": 10}
        for name, value in expected.items():
            if getattr(a, name) != value:
                raise ValueError("S4 frozen protocol mismatch: " + name)
        if int(a.allow_protocol_deviation) or int(a.allow_nonformal_seed):
            raise ValueError("S4 forbids protocol deviation")
        if bool(a.smoke_only) or bool(a.smoke_representative):
            raise ValueError("S4 runner has no smoke-mode training shortcut")
        for name in ("co_run_pair_id", "planned_peer_run_key", "worker_id", "attempt_id"):
            if not getattr(a, name, None):
                raise ValueError("S4 missing launch provenance: " + name)

    def __init__(self, args):
        args.objective_weighting = "equal"
        args.shared_access = "TARGET_ONLY_SHARED" if args.arm == "T" else "FULL"
        args.fresh_role = "target_only_shared"
        args.source_formal_run_key = ""
        self.arm = args.arm
        validate_adam_delta_against_clone(args.device)
        super().__init__(args)
        self.probe_rows = []
        self.zero_target_main_steps = 0
        self.main_steps_seen = 0
        self.initial_inventory = inventory(ROOT)
        if self.initial_inventory[MODEL_SOURCE_KEY] != MODEL_SOURCE_SHA256:
            raise RuntimeError("M3OE forward source differs from S3 audited source")

    def _condition_name(self):
        return "s4_" + self.arm

    def _access_formula(self):
        return {"F": "FULL shared Adam", "T": "target-only shared Adam",
                "N": "FULL shared gradient scaled to target gradient norm before Adam",
                "U": "FULL shared Adam moments; actual shared delta scaled to same-state target Adam delta norm"}[self.arm]

    def _controlled_step(self, losses, fingerprint, x_batch, epoch, step, kind):
        groups = self.partition["groups"]
        names = list(groups[GROUP_ARCH] if kind == "architecture" else groups[GROUP_SHARED])
        private = [] if kind == "architecture" else list(groups[GROUP_PRIVATE])
        optimizer = self.arch_optimizer if kind == "architecture" else self.optimizer
        gf, gt, gp = gradients_one_forward(losses["full"], losses["target"],
                                             self.parameter_by_name, names, private)
        for name, f, t in zip(names, gf, gt):
            if f is None and t is not None:
                raise ValueError("target gradient without FULL gradient: " + name)
        gnon = [None if f is None and t is None else
                (torch.zeros_like(t) if f is None else f) -
                (torch.zeros_like(f) if t is None else t)
                for f, t in zip(gf, gt)]
        gf_norm, gt_norm, gn_norm = map(vector_norm, (gf, gt, gnon))
        r_n, zero_n = scale_for(gt_norm, gf_norm, "N {} {} {}".format(epoch, step, kind))
        scaled_n = [None if g is None else g * r_n for g in gf]
        assert_match(vector_norm(scaled_n), gt_norm, "N")
        df, decay = adam_deltas(self.parameter_by_name, names, gf, optimizer)
        dt, _ = adam_deltas(self.parameter_by_name, names, gt, optimizer)
        dn, _ = adam_deltas(self.parameter_by_name, names, scaled_n, optimizer)
        dp, _ = adam_deltas(self.parameter_by_name, private, gp, optimizer)
        df_norm, dt_norm = vector_norm(df), vector_norm(dt)
        r_u, zero_u = scale_for(dt_norm, df_norm, "U {} {} {}".format(epoch, step, kind))
        if self.arm == "T":
            installed = gt
        elif self.arm == "N":
            installed = scaled_n
        else:
            installed = gf
        for name, grad in zip(names, installed):
            self.parameter_by_name[name].grad = None if grad is None else grad.detach().clone()
        before = [self.parameter_by_name[name].detach().clone() for name in names]
        private_before = [self.parameter_by_name[name].detach().clone() for name in private]
        optimizer.step()
        full_observed = [self.parameter_by_name[name].detach() - old
                         for name, old in zip(names, before)]
        if self.arm == "U":
            assert_vector_match(full_observed, df, "Adam FULL formula")
            with torch.no_grad():
                for name, old, full_delta in zip(names, before, full_observed):
                    if full_delta is not None:
                        self.parameter_by_name[name].copy_(old + r_u * full_delta)
        actual = [self.parameter_by_name[name].detach() - old for name, old in zip(names, before)]
        expected_shared = dt if self.arm == "T" else dn if self.arm == "N" else df
        expected_shared = [None if delta is None else delta * r_u if self.arm == "U" else delta
                           for delta in expected_shared]
        shared_residual = assert_vector_match(actual, expected_shared, "shared Adam formula")
        actual_norm = vector_norm(actual)
        reference = dt_norm if self.arm in ("T", "U") else gt_norm if self.arm == "N" else df_norm
        match = assert_match(actual_norm if self.arm != "N" else vector_norm(installed),
                             reference, self.arm)
        private_delta = [self.parameter_by_name[name].detach() - old
                         for name, old in zip(private, private_before)]
        private_residual = assert_vector_match(private_delta, dp, "private FULL Adam")
        direction = cosine(installed, gf)
        if self.arm in ("F", "N", "U") and direction is not None and direction < .99999:
            raise AssertionError("FULL gradient direction changed")
        state_steps = [int(optimizer.state[self.parameter_by_name[name]]["step"])
                       for name in names if self.parameter_by_name[name] in optimizer.state]
        if not state_steps:
            raise AssertionError("shared Adam state absent")
        if kind == "main":
            self.main_steps_seen += 1
            if zero_n == "target_zero" or zero_u == "target_zero":
                self.zero_target_main_steps += 1
        loss_values = {label: float(losses[label].detach().item())
                       for label in ("full", "target", "non_target", "equality_residual")}
        if not all(torch.isfinite(torch.tensor(v)).item() for v in loss_values.values()):
            raise FloatingPointError("nonfinite loss")
        ids = x_batch[:, -1].detach().cpu().long().tolist()
        row = {"stage": STAGE, "formal_run_key": self.formal_run_key, "arm": self.arm,
               "seed": int(self.args.seed), "epoch": epoch, "step": step, "kind": kind,
               "batch_fingerprint": fingerprint, "batch_size": len(x_batch),
               "domain_counts": json.dumps([ids.count(d) for d in range(3)]),
               "L_full": loss_values["full"], "L_target": loss_values["target"],
               "L_non": loss_values["non_target"],
               "decomposition_residual": loss_values["equality_residual"],
               "g_full_norm": gf_norm, "g_target_norm": gt_norm, "g_non_norm": gn_norm,
               "gradient_cosine": cosine(gf, gt), "r_N": r_n, "r_U": r_u,
               "actual_gradient_norm": vector_norm(installed),
               "delta_full_norm": df_norm, "delta_target_norm": dt_norm,
               "delta_N_predicted_norm": vector_norm(dn),
               "delta_actual_norm": actual_norm, "matching_residual": match,
               "private_delta_norm": vector_norm(private_delta), "decay_norm": vector_norm(decay),
               "adam_state_step_min": min(state_steps), "adam_state_step_max": max(state_steps),
               "optimizer_steps": self.optimizer_steps + (kind == "main"),
               "architecture_optimizer_steps": self.architecture_optimizer_steps + (kind == "architecture"),
               "zero_case": "N:{};U:{}".format(zero_n, zero_u),
               "full_direction_cosine": direction, "private_full_residual": private_residual,
               "shared_formula_residual": shared_residual}
        self.gradient_rows.append(row)
        if step == 0 or kind == "architecture":
            probe = dict(row)
            probe.update({"delta_cosine": cosine(df, dt),
                          "target_gain_full": -dot(gt, df),
                          "target_gain_target": -dot(gt, dt),
                          "target_gain_equal_norm_full": -r_u * dot(gt, df),
                          "delta_N_norm": vector_norm(dn)})
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

    def _train_epoch(self, epoch):
        self.model.train()
        total, count = 0., 0
        for step, (x_batch, y_batch) in enumerate(self.train_loader):
            fingerprint = self._record_batch(epoch, step, "train", x_batch, y_batch)
            self.batch_fingerprints.append(fingerprint)
            x = x_batch.transpose(0, 1).to(self.device, dtype=torch.float32)
            y = y_batch.to(self.device, dtype=torch.float32)
            pred = self.model(x)
            losses = self._objective_components(pred, y, x[-1, :].long())
            self.optimizer.zero_grad(set_to_none=True)
            self.arch_optimizer.zero_grad(set_to_none=True)
            self._controlled_step(losses, fingerprint, x_batch, epoch, step, "main")
            self.optimizer_steps += 1
            n = len(y_batch)
            self.examples_processed += n
            count += n
            total += float(losses["full"].detach().item()) * n
            self.batch_rows[-1]["train_objective"] = float(losses["full"].detach().item())
        loader = DataLoader(TorchDataset(self.bundle.train_x, self.bundle.train_y),
                            batch_size=self.args.batch_size, shuffle=False, num_workers=0, drop_last=False)
        return total / count, _cell_records(self.bundle, self.model, loader, self.device)

    def _architecture_update(self, architecture_iterator, epoch):
        from stage_E1_instrumentation import preserve_rng_state, rng_state_digest
        self.model.train()
        before = rng_state_digest()
        with preserve_rng_state():
            try:
                if architecture_iterator is None:
                    architecture_iterator = iter(self.train_loader)
                x_batch, y_batch = next(architecture_iterator)
            except StopIteration:
                architecture_iterator = iter(self.train_loader)
                x_batch, y_batch = next(architecture_iterator)
        after = rng_state_digest()
        self.rng_evidence.append({"epoch": epoch, "kind": "architecture_batch",
                                  "before": before, "after": after, "restored": before == after})
        fingerprint = self._record_batch(epoch, -1, "architecture", x_batch, y_batch)
        self.architecture_batch_fingerprints.append(fingerprint)
        self.architecture_examples_processed += len(y_batch)
        x = x_batch.transpose(0, 1).to(self.device, dtype=torch.float32)
        y = y_batch.to(self.device, dtype=torch.float32)
        losses = self._objective_components(self.model(x), y, x[-1, :].long())
        self.optimizer.zero_grad(set_to_none=True)
        self.arch_optimizer.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)
        self._controlled_step(losses, fingerprint, x_batch, epoch, -1, "architecture")
        self.architecture_optimizer_steps += 1
        self.batch_rows[-1]["train_objective"] = float(losses["full"].detach().item())
        _optimizer_audit(self.arch_optimizer)
        return architecture_iterator

    def write_outputs(self, fit_info: Mapping[str, object], wall_seconds: float):
        if inventory(ROOT) != self.initial_inventory:
            raise RuntimeError("S4 producer source changed during training")
        super().write_outputs(fit_info, wall_seconds)
        for filename in ("batch_trace.csv", "metrics_epoch.csv", "metrics_test.csv",
                         "metrics_summary.csv"):
            path = os.path.join(self.run_dir, filename)
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields, rows = list(reader.fieldnames or []), list(reader)
            if "stage" not in fields:
                raise RuntimeError("missing stage field in " + filename)
            write_rows_csv(path, [dict(row, stage=STAGE) for row in rows], fields)
        write_rows_csv(os.path.join(self.run_dir, "gradient_step.csv"), self.gradient_rows, FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "probe.csv"), self.probe_rows, PROBE_FIELDS)
        sources = inventory(ROOT)
        config_path = os.path.join(self.run_dir, "config.json")
        manifest_path = os.path.join(self.run_dir, "run_manifest.json")
        protocol = self._output_formal_protocol()
        observed = {"stage": STAGE, "dataset": str(self.args.dataset).lower(),
                    "setting": str(self.args.setting), "backbone": str(self.args.backbone).lower(),
                    "objective": str(self.objective_weighting), "arm": self.arm,
                    "seed": int(self.args.seed), "epochs": int(self.args.epoch),
                    "batch_size": int(self.args.batch_size), "device": str(self.args.device),
                    "main_lr": float(self.args.learning_rate),
                    "architecture_lr": float(self.args.learning_rate_darts),
                    "betas": list(self.optimizer.param_groups[0]["betas"]),
                    "weight_decay": float(self.args.weight_decay),
                    "common_epochs": list(COMMON_EPOCHS)}
        mismatches = {name: {"expected": value, "observed": observed.get(name)}
                      for name, value in protocol.items() if observed.get(name) != value}
        if mismatches:
            raise RuntimeError("S4 observed output protocol mismatch: " + str(mismatches))
        for path in (config_path, manifest_path):
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
            value.update({"stage": STAGE, "condition": self.arm, "arm": self.arm,
                          "shared_access": self.arm, "fresh_role": "fresh_s4",
                          "source_stage": STAGE, "provenance_type": "fresh_s4",
                          "formal_protocol": protocol,
                          "formal_protocol_observed": observed,
                          "formal_protocol_mismatches": {}, "formal_valid": True,
                          "formal_run_key": self.formal_run_key, "code_hashes": sources,
                          "code_inventory_hash": inventory_hash(sources),
                          "source_inventory_digest": inventory_hash(sources),
                          "source_code_hash": sources["experiments/stage_S4_run.py"],
                          "forward_model_source_sha256": sources[MODEL_SOURCE_KEY],
                          "probe_count": len(self.probe_rows),
                          "gradient_batch_count": len(self.gradient_rows),
                          "zero_target_main_steps": self.zero_target_main_steps,
                          "main_steps_seen": self.main_steps_seen})
            if path == manifest_path:
                value["artifact_hashes"].update({f: _sha256(os.path.join(self.run_dir, f))
                                                   for f in ("config.json", "batch_trace.csv",
                                                             "metrics_epoch.csv", "metrics_test.csv",
                                                             "metrics_summary.csv", "gradient_step.csv",
                                                             "probe.csv")})
                value["artifacts"] = sorted(set(value["artifacts"]) | {"gradient_step.csv", "probe.csv"})
            write_json(path, value)


def build_argparser():
    parser = s3_parser()
    # Keep inherited frozen defaults and accepted identity fields, with S4 arm added.
    parser.add_argument("--arm", choices=ARMS, required=True)
    for action in parser._actions:
        if action.dest in ("objective_weighting", "shared_access", "fresh_role",
                           "source_formal_run_key"):
            action.required = False
        if action.dest == "dataset":
            action.choices = ("kuairand",)
    return parser


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.run_dir, exist_ok=False)
    started_ns = time.time_ns()
    started = started_ns / 1e9
    interval_path = os.path.join(args.run_dir, "process_interval.json")
    try:
        runner = S4Runner(args)
        info = runner.fit()
        runner.write_outputs(info, time.time() - started)
        write_json(interval_path, {"pid": os.getpid(), "started_ns": started_ns,
                                   "finished_ns": time.time_ns(), "status": "OK"})
        print("[stage_S4] completed {}".format(args.formal_run_key), flush=True)
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
