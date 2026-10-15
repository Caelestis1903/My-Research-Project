"""Stage S8 Phase 1 base retrain runner.

Thin subclass of the S6 (KuaiRand) / S6ML (MovieLens) F-arm runners that
re-trains the same F-arm trajectory bit-for-bit (design §0.b) and emits,
in addition to the inherited S6 artifacts, ``full_state_epoch{e}.pth``
snapshots at every epoch in ``stage_S8_protocol.full_state_dump_epochs``.

Two subclasses:

- ``S8BaseRunnerKR`` -- ``stage_S6_run.S6Runner`` clone for KuaiRand.
- ``S8BaseRunnerML`` -- ``stage_S6ML_run.S6MLRunner`` clone for MovieLens.

The only differences versus the S6 parent are:

1. ``_validate_args`` enforces ``arm='F'``, the S8 frozen key via
   ``stage_S8_protocol.key_base``, and rejects non-S8 backbones.
2. ``__init__`` sets ``args.arm='F'`` / ``args.fresh_role='fresh_s8base'`` /
   ``args.shared_access='FULL'``, swaps the inventory to the S8 inventory
   (which still references the S6 source files as imported dependencies).
3. Adds ``_dump_full_state(epoch)`` -- strictly read-only snapshot of the
   model state_dict, both optimizer state_dicts, the running step counters,
   and an ``rng_state_digest()``.  Called immediately after the inherited
   ``_dump_moment_state(epoch)`` so the dump point matches S6 exactly
   (post-arch-update, pre-validation).
4. ``write_outputs`` records ``stage='S8BASE'``, ``condition='s8base_F'``,
   ``fresh_role='fresh_s8base'``, ``provenance_type='fresh_s8base'``, the
   full-state dump filenames + sha256s, and the source-code hash pinned
   to this file.

The ``_controlled_step`` body, ``_architecture_update``, the S1/S2/S3/S4/S6
fit loop, and the checkpoint selection logic are inherited verbatim -- only
the dump call is interleaved at the exact post-arch-update pre-validation
point.

Per Algorithm Specification ``stage_S8_design.md`` §4.2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from typing import Mapping, Sequence

import torch

from stage_E1_instrumentation import rng_state_digest
from stage_S3_run import (GROUP_ARCH, GROUP_PRIVATE, GROUP_SHARED, S3Runner,
                          _sha256, write_json, write_rows_csv,
                          build_argparser as s3_parser)
from stage_S4_gradient_controls import (adam_deltas, assert_vector_match,
                                        cosine, gradients_one_forward,
                                        validate_adam_delta_against_clone,
                                        vector_norm)
from stage_S4_run import S4Runner
from stage_S6_run import S6Runner
from stage_S6ML_run import S6MLRunner
from stage_S6ML_protocol import (SAMPLE_TABLE_SHA256 as S6ML_SAMPLE_TABLE_SHA256,
                                  INITIAL_PARAMETER_SHA256 as S6ML_INITIAL_PARAMETER_SHA256,
                                  MOVIELENS_DATA_SHA256, TARGET_LOCAL_DOMAIN as S6ML_TARGET_LOCAL_DOMAIN,
                                  TARGET_PHYSICAL_DOMAIN as S6ML_TARGET_PHYSICAL_DOMAIN,
                                  TASKS as S6ML_TASKS, OBJECTIVE_WEIGHTING as S6ML_OBJECTIVE,
                                  SETTING as S6ML_SETTING, DATASET as S6ML_DATASET,
                                  ARM_CURRENT as S6ML_ARM_CURRENT, ARM_HISTORY as S6ML_ARM_HISTORY,
                                  ARMS as S6ML_ARMS, BACKBONE as S6ML_BACKBONE,
                                  MOMENT_DUMP_EPOCHS as S6ML_MOMENT_DUMP_EPOCHS,
                                  COMMON_EPOCHS as S6ML_COMMON_EPOCHS, key as s6ml_key)
from stage_S6_protocol import (ARM_CURRENT, ARM_HISTORY, ARMS, COMMON_EPOCHS,
                               INITIAL_PARAMETER_SHA256 as S6_INITIAL_PARAMETER_SHA256,
                               KUAIRAND_DATA_SHA256, MOMENT_DUMP_EPOCHS,
                               SAMPLE_TABLE_SHA256 as S6_SAMPLE_TABLE_SHA256,
                               key as s6_key)
from stage_S8_protocol import (S8_SOURCE_FILES, BRANCHES, VBASE_MACRO_AUC_ABS,
                                full_state_dump_epochs, full_state_filename,
                                key_base, inventory as s8_inventory,
                                inventory_hash as s8_inventory_hash,
                                target_local_domain as _unused_tld,
                                data_sha256 as _data_sha256_unused,
                                sample_table_sha256 as _sample_sha_unused)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Field contract: bit-compatible with S6 (and S6ML) -- the S8 phase 1 base
# runner records the same ``gradient_step`` / ``probe`` rows so the
# continuation runner can replay batch sequences against them.
S8BASE_FIELDS = (
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
S8BASE_PROBE_FIELDS = S8BASE_FIELDS + [
    "delta_cosine", "m_norm_target_stream", "v_norm_target_stream",
    "moment_cosine_history_target", "moment_cosine_history_full",
]

# The shared S8 frozen hyper-parameter dict (epoch / batch / lr / device /
# fcn / patience / wd) -- matches S6/S6ML and the protocol.
_S8_EXPECTED = {
    "epoch": 50, "batch_size": 40960, "learning_rate": 0.01,
    "learning_rate_darts": 0.001, "weight_decay": 1e-5,
    "expert_num": 4, "num_workers": 0, "device": "cuda:0",
    "fcn_dims": [512, 256, 256, 64], "shadow_patience": 10,
}


def _make_s8_key(dataset: str, seed: int) -> str:
    return key_base(str(dataset).lower(), int(seed))


# ===========================================================================
# KuaiRand subclass
# ===========================================================================

class S8BaseRunnerKR(S6Runner):
    """KuaiRand / natural / m3oe / sample / F arm retrain with full-state
    dumps.  Inherits the F-arm ``_controlled_step`` and ``fit`` from
    ``S6Runner`` and only adds the full-state dump hook + S8 identity
    re-stamping.  """

    def _output_formal_protocol(self):
        return {"stage": "S8BASE", "dataset": "kuairand", "setting": "natural",
                "backbone": "m3oe", "objective": "sample", "arm": self.arm,
                "history_source": ARM_HISTORY[self.arm],
                "current_source": ARM_CURRENT[self.arm],
                "seed": int(self.args.seed), "epochs": 50, "batch_size": 40960,
                "device": "cuda:0", "main_lr": .01, "architecture_lr": .001,
                "betas": [.9, .999], "weight_decay": 1e-5,
                "common_epochs": list(COMMON_EPOCHS),
                "moment_dump_epochs": list(MOMENT_DUMP_EPOCHS),
                "full_state_dump_epochs": list(
                    full_state_dump_epochs("kuairand", int(self.args.seed)))}

    def _validate_args(self):
        a = self.args
        # 5-tuple must be the S8 KuaiRand configuration: kuairand/natural/
        # m3oe/sample/joint.  Arm must be ``F`` and the formal_run_key must
        # be the S8 base key.
        if (str(a.dataset).lower(), str(a.setting), str(a.backbone).lower(),
                str(a.objective_weighting).lower(), str(a.run_type)) != (
                "kuairand", "natural", "m3oe", "sample", "joint"):
            raise ValueError("S8BASE-KR is Sample/KuaiRand/natural/M3OE joint only")
        if str(a.arm) != "F":
            raise ValueError("S8BASE-KR must run arm='F' (got {!r})".format(a.arm))
        if int(a.seed) not in ARMS and int(a.seed) not in {int(s) for s in (2022, 2023, 2024, 2025)}:
            raise ValueError("S8BASE-KR seed outside frozen matrix")
        expected_key = _make_s8_key("kuairand", int(a.seed))
        if str(a.formal_run_key) != expected_key:
            raise ValueError("S8BASE-KR formal_run_key {!r} != expected {!r}".format(
                a.formal_run_key, expected_key))
        for name, value in _S8_EXPECTED.items():
            if getattr(a, name) != value:
                raise ValueError("S8BASE-KR frozen protocol mismatch: " + name)
        if int(a.allow_protocol_deviation) or int(a.allow_nonformal_seed):
            raise ValueError("S8BASE-KR forbids protocol deviation")
        if bool(a.smoke_only) or bool(a.smoke_representative):
            raise ValueError("S8BASE-KR has no smoke-mode training shortcut")
        for name in ("co_run_pair_id", "planned_peer_run_key", "worker_id", "attempt_id"):
            if not getattr(a, name, None):
                raise ValueError("S8BASE-KR missing launch provenance: " + name)

    def __init__(self, args):
        # Inject the S8 identity stamps BEFORE S6Runner.__init__ runs, then
        # delegate to S6Runner.__init__ which already handles the objective
        # clobber bypass (see stage_S6_run.py:117-146 for the design
        # comment that explains why S3Runner.__init__ is used to bypass
        # S4Runner's ``args.objective_weighting = "equal"`` clobber).
        args.objective_weighting = "sample"
        args.shared_access = "FULL"
        args.fresh_role = "fresh_s8base"
        args.source_formal_run_key = ""
        self.arm = "F"
        self.history_source = ARM_HISTORY["F"]
        self.current_source = ARM_CURRENT["F"]
        validate_adam_delta_against_clone(args.device)
        S3Runner.__init__(self, args)
        # Replicate S6's instance-attribute setup so that ``_check_source_drift``
        # uses the S8 inventory (which is a strict superset of the S4
        # inventory but still hash-equivalent to the S6 inventory on the
        # S6-owned files; the new entries are the S8 source files only).
        self.probe_rows = []
        self.zero_target_main_steps = 0
        self.main_steps_seen = 0
        self.initial_inventory = s8_inventory(ROOT)
        # The S6 inventory check fires here (it compares the M3OE source
        # hash against the S3 audited value); the S8 inventory contains the
        # same MODEL_SOURCE_KEY entry, so this still raises on real drift.
        from stage_S3_protocol import MODEL_SOURCE_KEY, MODEL_SOURCE_SHA256
        if self.initial_inventory.get(MODEL_SOURCE_KEY) != MODEL_SOURCE_SHA256:
            raise RuntimeError("M3OE forward source differs from S3 audited source")
        self._moment_dump_paths = []
        self._full_state_dump_paths = []
        self._none_count_full = 0
        self._none_count_target = 0

    def _condition_name(self):
        return "s8base_F"

    def _access_formula(self):
        return ("history_source={h}; current_source={c}; single optimizer.step() "
                "per batch; post-step overwrite only for HT/DH shared params with "
                "pre-step Delta_D (F arm has no overwrite)").format(
            h=self.history_source, c=self.current_source)

    def _dump_full_state(self, epoch):
        """Strictly read-only full-state snapshot.

        Called after the inherited ``_dump_moment_state`` (which itself runs
        inside ``_architecture_update`` -- the post-arch-update, pre-validation
        point).  The snapshot covers:

        - model parameters (CPU detached clones of every parameter tensor);
        - main optimizer state_dict (incl. private m/v/step tensors);
        - arch optimizer state_dict;
        - the running ``optimizer_steps`` / ``architecture_optimizer_steps``;
        - an RNG digest via ``rng_state_digest``.

        ``torch.save`` does not advance any RNG; the clones are
        ``detach().cpu().clone()`` so the in-flight CUDA tensors are not
        aliased.  The dump is appended to ``self._full_state_dump_paths`` for
        the manifest to hash.
        """
        target_epochs = full_state_dump_epochs(str(self.args.dataset).lower(),
                                                 int(self.args.seed))
        if int(epoch) not in target_epochs:
            return None
        payload = {
            "epoch": int(epoch),
            "model": {name: value.detach().cpu().clone()
                      for name, value in self.model.state_dict().items()},
            "main_optimizer": self.optimizer.state_dict(),
            "arch_optimizer": (self.arch_optimizer.state_dict()
                                 if self.arch_optimizer is not None else {}),
            "optimizer_steps": int(self.optimizer_steps),
            "architecture_optimizer_steps": int(self.architecture_optimizer_steps),
            "rng_digest": str(rng_state_digest()),
        }
        path = os.path.join(self.run_dir, full_state_filename(int(epoch)))
        torch.save(payload, path)
        self._full_state_dump_paths.append(path)
        return path

    def _architecture_update(self, architecture_iterator, epoch):
        # Inherited S6 path: runs ``_controlled_step`` for architecture
        # parameters and then dumps the moment state.  After it returns we
        # dump the full state at exactly the same point.
        iterator = S6Runner._architecture_update(self, architecture_iterator, epoch)
        self._dump_full_state(int(epoch))
        return iterator

    def _check_source_drift(self):
        # ``S6Runner._check_source_drift`` would compare the current S6
        # inventory against ``self.initial_inventory`` -- which we have
        # replaced with the S8 inventory.  Re-implement the check using the
        # S8 inventory so it stays fail-closed.
        if s8_inventory(ROOT) != self.initial_inventory:
            raise RuntimeError("S8BASE producer source changed during training")

    def write_outputs(self, fit_info: Mapping[str, object], wall_seconds: float):
        # Inherited source-drift gate + S3 writer (writes batch_trace /
        # metrics_epoch / metrics_test / metrics_summary) -- the same files
        # the S6 runner writes.
        self._check_source_drift()
        S3Runner.write_outputs(self, fit_info, wall_seconds)
        # Rewrite batch_trace / metrics_* with stage="S8BASE" stamped in
        # (mirroring S6's stage-stamping pass).
        for filename in ("batch_trace.csv", "metrics_epoch.csv", "metrics_test.csv",
                         "metrics_summary.csv"):
            path = os.path.join(self.run_dir, filename)
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields, rows = list(reader.fieldnames or []), list(reader)
            if "stage" not in fields:
                raise RuntimeError("missing stage field in " + filename)
            write_rows_csv(path, [dict(row, stage="S8BASE") for row in rows], fields)
        write_rows_csv(os.path.join(self.run_dir, "gradient_step.csv"),
                       self.gradient_rows, S8BASE_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "probe.csv"),
                       self.probe_rows, S8BASE_PROBE_FIELDS)
        sources = s8_inventory(ROOT)
        config_path = os.path.join(self.run_dir, "config.json")
        manifest_path = os.path.join(self.run_dir, "run_manifest.json")
        protocol = self._output_formal_protocol()
        observed = {"stage": "S8BASE", "dataset": str(self.args.dataset).lower(),
                    "setting": str(self.args.setting),
                    "backbone": str(self.args.backbone).lower(),
                    "objective": str(self.objective_weighting), "arm": self.arm,
                    "history_source": ARM_HISTORY[self.arm],
                    "current_source": ARM_CURRENT[self.arm],
                    "seed": int(self.args.seed), "epochs": int(self.args.epoch),
                    "batch_size": int(self.args.batch_size),
                    "device": str(self.args.device),
                    "main_lr": float(self.args.learning_rate),
                    "architecture_lr": float(self.args.learning_rate_darts),
                    "betas": list(self.optimizer.param_groups[0]["betas"]),
                    "weight_decay": float(self.args.weight_decay),
                    "common_epochs": list(COMMON_EPOCHS),
                    "moment_dump_epochs": list(MOMENT_DUMP_EPOCHS),
                    "full_state_dump_epochs": list(
                        full_state_dump_epochs("kuairand", int(self.args.seed)))}
        mismatches = {name: {"expected": value, "observed": observed.get(name)}
                      for name, value in protocol.items() if observed.get(name) != value}
        if mismatches:
            raise RuntimeError("S8BASE observed output protocol mismatch: " + str(mismatches))
        moment_hashes = {os.path.basename(path): _sha256(path)
                         for path in self._moment_dump_paths}
        full_state_hashes = {os.path.basename(path): _sha256(path)
                             for path in self._full_state_dump_paths}
        for path in (config_path, manifest_path):
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
            value.update({
                "stage": "S8BASE",
                "condition": "s8base_F",
                "arm": self.arm,
                "shared_access": "FULL",
                "fresh_role": "fresh_s8base",
                "source_stage": "S6",
                "provenance_type": "fresh_s8base",
                "history_source": ARM_HISTORY[self.arm],
                "current_source": ARM_CURRENT[self.arm],
                "formal_protocol": protocol,
                "formal_protocol_observed": observed,
                "formal_protocol_mismatches": {},
                "formal_valid": True,
                "formal_run_key": self.formal_run_key,
                "code_hashes": sources,
                "code_inventory_hash": s8_inventory_hash(sources),
                "source_inventory_digest": s8_inventory_hash(sources),
                "source_code_hash": sources["experiments/stage_S8_base_run.py"],
                "forward_model_source_sha256": sources["m3oe/MDMTRec/models/multi_domain/model.py"],
                "probe_count": len(self.probe_rows),
                "gradient_batch_count": len(self.gradient_rows),
                "zero_target_main_steps": self.zero_target_main_steps,
                "main_steps_seen": self.main_steps_seen,
                "moment_dump_epochs": list(MOMENT_DUMP_EPOCHS),
                "moment_dumps": sorted(moment_hashes),
                "moment_dump_hashes": moment_hashes,
                "full_state_dump_epochs": list(
                    full_state_dump_epochs("kuairand", int(self.args.seed))),
                "full_state_dumps": sorted(full_state_hashes),
                "full_state_dump_hashes": full_state_hashes,
                "none_count_full": self._none_count_full,
                "none_count_target": self._none_count_target,
                "coefficient_table_hash": S6_SAMPLE_TABLE_SHA256,
                "initial_parameter_state_hash": S6_INITIAL_PARAMETER_SHA256[int(self.args.seed)],
                "dataset_file_hash": KUAIRAND_DATA_SHA256,
                "vbase_macro_auc_abs": VBASE_MACRO_AUC_ABS,
            })
            if path == manifest_path:
                value["artifact_hashes"].update({f: _sha256(os.path.join(self.run_dir, f))
                                                   for f in ("config.json", "batch_trace.csv",
                                                             "metrics_epoch.csv",
                                                             "metrics_test.csv",
                                                             "metrics_summary.csv",
                                                             "gradient_step.csv",
                                                             "probe.csv")})
                value["artifact_hashes"].update(moment_hashes)
                value["artifact_hashes"].update(full_state_hashes)
                value["artifacts"] = sorted(set(value.get("artifacts", []))
                                              | {"gradient_step.csv", "probe.csv"}
                                              | set(moment_hashes)
                                              | set(full_state_hashes))
            write_json(path, value)


# ===========================================================================
# MovieLens subclass
# ===========================================================================

class S8BaseRunnerML(S6MLRunner):
    """MovieLens / d2_large / m3oe / sample / F arm retrain with full-state
    dumps.  Mirrors ``S8BaseRunnerKR`` for the ML side.  Inherits the F-arm
    ``_controlled_step`` and ``fit`` from ``S6MLRunner``.  """

    def _output_formal_protocol(self):
        return {"stage": "S8BASE", "dataset": S6ML_DATASET, "setting": S6ML_SETTING,
                "backbone": S6ML_BACKBONE, "objective": S6ML_OBJECTIVE,
                "arm": self.arm,
                "history_source": S6ML_ARM_HISTORY[self.arm],
                "current_source": S6ML_ARM_CURRENT[self.arm],
                "seed": int(self.args.seed), "epochs": 50, "batch_size": 40960,
                "device": "cuda:0", "main_lr": .01, "architecture_lr": .001,
                "betas": [.9, .999], "weight_decay": 1e-5,
                "common_epochs": list(S6ML_COMMON_EPOCHS),
                "moment_dump_epochs": list(S6ML_MOMENT_DUMP_EPOCHS),
                "full_state_dump_epochs": list(
                    full_state_dump_epochs("movielens", int(self.args.seed)))}

    def _validate_args(self):
        a = self.args
        if (str(a.dataset).lower(), str(a.setting), str(a.backbone).lower(),
                str(a.objective_weighting).lower(), str(a.run_type)) != (
                S6ML_DATASET, S6ML_SETTING, S6ML_BACKBONE, S6ML_OBJECTIVE, "joint"):
            raise ValueError(
                "S8BASE-ML is {}/{}/{}/{} joint only".format(
                    S6ML_DATASET, S6ML_SETTING, S6ML_BACKBONE, S6ML_OBJECTIVE))
        if str(a.arm) != "F":
            raise ValueError("S8BASE-ML must run arm='F' (got {!r})".format(a.arm))
        if int(a.seed) not in {2022, 2023, 2024, 2025}:
            raise ValueError("S8BASE-ML seed outside frozen matrix")
        expected_key = _make_s8_key("movielens", int(a.seed))
        if str(a.formal_run_key) != expected_key:
            raise ValueError("S8BASE-ML formal_run_key {!r} != expected {!r}".format(
                a.formal_run_key, expected_key))
        for name, value in _S8_EXPECTED.items():
            if getattr(a, name) != value:
                raise ValueError("S8BASE-ML frozen protocol mismatch: " + name)
        if int(a.allow_protocol_deviation) or int(a.allow_nonformal_seed):
            raise ValueError("S8BASE-ML forbids protocol deviation")
        if bool(a.smoke_only) or bool(a.smoke_representative):
            raise ValueError("S8BASE-ML has no smoke-mode training shortcut")
        for name in ("co_run_pair_id", "planned_peer_run_key", "worker_id", "attempt_id"):
            if not getattr(a, name, None):
                raise ValueError("S8BASE-ML missing launch provenance: " + name)

    def __init__(self, args):
        args.objective_weighting = S6ML_OBJECTIVE
        args.shared_access = "FULL"
        args.fresh_role = "fresh_s8base"
        args.source_formal_run_key = ""
        self.arm = "F"
        self.history_source = S6ML_ARM_HISTORY["F"]
        self.current_source = S6ML_ARM_CURRENT["F"]
        validate_adam_delta_against_clone(args.device)
        S3Runner.__init__(self, args)
        self.probe_rows = []
        self.zero_target_main_steps = 0
        self.main_steps_seen = 0
        self.initial_inventory = s8_inventory(ROOT)
        from stage_S3_protocol import MODEL_SOURCE_KEY, MODEL_SOURCE_SHA256
        if self.initial_inventory.get(MODEL_SOURCE_KEY) != MODEL_SOURCE_SHA256:
            raise RuntimeError("M3OE forward source differs from S3 audited source")
        self._moment_dump_paths = []
        self._full_state_dump_paths = []
        self._none_count_full = 0
        self._none_count_target = 0

    def _condition_name(self):
        return "s8base_F"

    def _access_formula(self):
        return ("history_source={h}; current_source={c}; single optimizer.step() "
                "per batch; post-step overwrite only for HT/DH shared params with "
                "pre-step Delta_D (F arm has no overwrite)").format(
            h=self.history_source, c=self.current_source)

    def _dump_full_state(self, epoch):
        target_epochs = full_state_dump_epochs(str(self.args.dataset).lower(),
                                                 int(self.args.seed))
        if int(epoch) not in target_epochs:
            return None
        payload = {
            "epoch": int(epoch),
            "model": {name: value.detach().cpu().clone()
                      for name, value in self.model.state_dict().items()},
            "main_optimizer": self.optimizer.state_dict(),
            "arch_optimizer": (self.arch_optimizer.state_dict()
                                 if self.arch_optimizer is not None else {}),
            "optimizer_steps": int(self.optimizer_steps),
            "architecture_optimizer_steps": int(self.architecture_optimizer_steps),
            "rng_digest": str(rng_state_digest()),
        }
        path = os.path.join(self.run_dir, full_state_filename(int(epoch)))
        torch.save(payload, path)
        self._full_state_dump_paths.append(path)
        return path

    def _architecture_update(self, architecture_iterator, epoch):
        iterator = S6MLRunner._architecture_update(self, architecture_iterator, epoch)
        self._dump_full_state(int(epoch))
        return iterator

    def _check_source_drift(self):
        if s8_inventory(ROOT) != self.initial_inventory:
            raise RuntimeError("S8BASE producer source changed during training")

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
            write_rows_csv(path, [dict(row, stage="S8BASE") for row in rows], fields)
        write_rows_csv(os.path.join(self.run_dir, "gradient_step.csv"),
                       self.gradient_rows, S8BASE_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "probe.csv"),
                       self.probe_rows, S8BASE_PROBE_FIELDS)
        sources = s8_inventory(ROOT)
        config_path = os.path.join(self.run_dir, "config.json")
        manifest_path = os.path.join(self.run_dir, "run_manifest.json")
        protocol = self._output_formal_protocol()
        observed = {"stage": "S8BASE", "dataset": str(self.args.dataset).lower(),
                    "setting": str(self.args.setting),
                    "backbone": str(self.args.backbone).lower(),
                    "objective": str(self.objective_weighting), "arm": self.arm,
                    "history_source": S6ML_ARM_HISTORY[self.arm],
                    "current_source": S6ML_ARM_CURRENT[self.arm],
                    "seed": int(self.args.seed), "epochs": int(self.args.epoch),
                    "batch_size": int(self.args.batch_size),
                    "device": str(self.args.device),
                    "main_lr": float(self.args.learning_rate),
                    "architecture_lr": float(self.args.learning_rate_darts),
                    "betas": list(self.optimizer.param_groups[0]["betas"]),
                    "weight_decay": float(self.args.weight_decay),
                    "common_epochs": list(S6ML_COMMON_EPOCHS),
                    "moment_dump_epochs": list(S6ML_MOMENT_DUMP_EPOCHS),
                    "full_state_dump_epochs": list(
                        full_state_dump_epochs("movielens", int(self.args.seed)))}
        mismatches = {name: {"expected": value, "observed": observed.get(name)}
                      for name, value in protocol.items() if observed.get(name) != value}
        if mismatches:
            raise RuntimeError("S8BASE-ML observed output protocol mismatch: " + str(mismatches))
        moment_hashes = {os.path.basename(path): _sha256(path)
                         for path in self._moment_dump_paths}
        full_state_hashes = {os.path.basename(path): _sha256(path)
                             for path in self._full_state_dump_paths}
        for path in (config_path, manifest_path):
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
            value.update({
                "stage": "S8BASE",
                "condition": "s8base_F",
                "arm": self.arm,
                "shared_access": "FULL",
                "fresh_role": "fresh_s8base",
                "source_stage": "S6ML",
                "provenance_type": "fresh_s8base",
                "history_source": S6ML_ARM_HISTORY[self.arm],
                "current_source": S6ML_ARM_CURRENT[self.arm],
                "formal_protocol": protocol,
                "formal_protocol_observed": observed,
                "formal_protocol_mismatches": {},
                "formal_valid": True,
                "formal_run_key": self.formal_run_key,
                "code_hashes": sources,
                "code_inventory_hash": s8_inventory_hash(sources),
                "source_inventory_digest": s8_inventory_hash(sources),
                "source_code_hash": sources["experiments/stage_S8_base_run.py"],
                "forward_model_source_sha256": sources["m3oe/MDMTRec/models/multi_domain/model.py"],
                "probe_count": len(self.probe_rows),
                "gradient_batch_count": len(self.gradient_rows),
                "zero_target_main_steps": self.zero_target_main_steps,
                "main_steps_seen": self.main_steps_seen,
                "moment_dump_epochs": list(S6ML_MOMENT_DUMP_EPOCHS),
                "moment_dumps": sorted(moment_hashes),
                "moment_dump_hashes": moment_hashes,
                "full_state_dump_epochs": list(
                    full_state_dump_epochs("movielens", int(self.args.seed))),
                "full_state_dumps": sorted(full_state_hashes),
                "full_state_dump_hashes": full_state_hashes,
                "none_count_full": self._none_count_full,
                "none_count_target": self._none_count_target,
                "coefficient_table_hash": S6ML_SAMPLE_TABLE_SHA256,
                "initial_parameter_state_hash": S6ML_INITIAL_PARAMETER_SHA256[int(self.args.seed)],
                "dataset_file_hash": MOVIELENS_DATA_SHA256,
                "target_local_domain": int(S6ML_TARGET_LOCAL_DOMAIN),
                "target_physical_domain": str(S6ML_TARGET_PHYSICAL_DOMAIN),
                "target_task_names": dict(S6ML_TASKS),
                "vbase_macro_auc_abs": VBASE_MACRO_AUC_ABS,
            })
            if path == manifest_path:
                value["artifact_hashes"].update({f: _sha256(os.path.join(self.run_dir, f))
                                                   for f in ("config.json", "batch_trace.csv",
                                                             "metrics_epoch.csv",
                                                             "metrics_test.csv",
                                                             "metrics_summary.csv",
                                                             "gradient_step.csv",
                                                             "probe.csv")})
                value["artifact_hashes"].update(moment_hashes)
                value["artifact_hashes"].update(full_state_hashes)
                value["artifacts"] = sorted(set(value.get("artifacts", []))
                                              | {"gradient_step.csv", "probe.csv"}
                                              | set(moment_hashes)
                                              | set(full_state_hashes))
            write_json(path, value)


# ===========================================================================
# Argparser + main entrypoint (single file with --dataset dispatch).
# ===========================================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = s3_parser()
    parser.add_argument("--arm", choices=("F",), default="F")
    for action in parser._actions:
        if action.dest in ("objective_weighting", "shared_access", "fresh_role",
                           "source_formal_run_key", "arm"):
            action.required = False
        if action.dest == "dataset":
            action.choices = ("kuairand", "movielens")
        if action.dest == "fresh_role":
            action.choices = ("fresh_s8base", "fresh_s6", "fresh_s6ml",
                              "fresh_target_only", "fresh_sample_full_replacement")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    # ``os.makedirs(run_dir, exist_ok=False)`` -- non-empty refusal per
    # design §4.2 and §9.7.  The orchestrator archives failed run dirs
    # separately; a fresh run cannot silently overwrite a previous one.
    os.makedirs(args.run_dir, exist_ok=False)
    started_ns = time.time_ns()
    started = started_ns / 1e9
    interval_path = os.path.join(args.run_dir, "process_interval.json")
    dataset = str(args.dataset).lower()
    runner_cls = S8BaseRunnerKR if dataset == "kuairand" else S8BaseRunnerML
    try:
        runner = runner_cls(args)
        info = runner.fit()
        runner.write_outputs(info, time.time() - started)
        write_json(interval_path, {"pid": os.getpid(), "started_ns": started_ns,
                                   "finished_ns": time.time_ns(), "status": "OK",
                                   "dataset": dataset, "seed": int(args.seed),
                                   "formal_run_key": str(args.formal_run_key)})
        print("[stage_S8BASE] completed {} seed={} dataset={} wall={:.2f}s".format(
            args.formal_run_key, int(args.seed), dataset,
            time.time() - started), flush=True)
        return 0
    except Exception as exc:
        write_json(interval_path, {"pid": os.getpid(), "started_ns": started_ns,
                                   "finished_ns": time.time_ns(), "status": "FAILED",
                                   "dataset": dataset, "seed": int(args.seed),
                                   "formal_run_key": str(args.formal_run_key)})
        write_json(os.path.join(args.run_dir, "failure.json"),
                   {"stage": "S8BASE", "key": args.formal_run_key,
                    "exception_type": type(exc).__name__, "exception": str(exc),
                    "wall_seconds": time.time() - started,
                    "dataset": dataset, "seed": int(args.seed)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
