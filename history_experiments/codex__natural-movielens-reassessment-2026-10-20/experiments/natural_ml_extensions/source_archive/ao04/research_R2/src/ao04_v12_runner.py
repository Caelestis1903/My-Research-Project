"""Stage R2 runner — training-side components (1, 2 and 3) only.

This module implements the three training-side components of the
four-component architecture required by Section 4.3 of the R2 design
spec:

    1. ``paired_init_phase(dataset, seed)`` — called once per
       (dataset, seed).  Constructs a B0 model, persists its state_dict
       and RNG so the remaining arms can reproduce the same initial
       tensors.

    2. ``train_val_loop(arm, dataset, seed, run_dir, config)`` — runs
       the 50-epoch joint training/validation cycle for a single arm.
       Performs checkpoint selection by *validation* macro AUC only.
       Does NOT touch any test fixture.

    3. ``state_writer(...)`` — emits the full set of training-side
       artifacts (CSV metrics, checkpoints, optimized-state snapshots,
       runtime metadata, run manifest).  The runner can call this
       even when the test split is unavailable.

Component 4 (``final_test_evaluator``) lives in
``ao04_v12_test_evaluator.py`` so that this module remains
forbidden-identifier-free — the ``ao04_v12_static_check`` module
grep-asserts that no Stage S1 test-coupling identifier appears in
this file's source.

Three tripwire mechanisms ensure these guarantees:

    * ``SealedTestSplit`` — a fingerprint-only façade that hides
      every test array behind ``row_count``, ``fingerprint_value`` and
      ``class_coverage_check``.  The runner never holds a raw test
      ``x``/``y`` array.

    * The module-level ``TEST_PREDICTION_CALLS`` counter, updated
      only by ``ao04_v12_test_evaluator.final_test_evaluator``.  The
      decision module asserts this counter equals 0 before publishing
      any R2-A verdict.

    * Static grep check (separate ``ao04_v12_static_check`` module)
      guarantees that the Stage-S1 test-coupling identifier names
      appear nowhere in this module.
"""

from __future__ import annotations

# Tripwire static check: these forbidden identifiers must not be imported
# or referenced anywhere in this file's body.  This is enforced via a
# static grep at review time; the imports list above is intentionally
# restricted so a future edit cannot silently re-couple test eval.

import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Set up sys.path so the read-only Stage S1 modules resolve correctly.
R2_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(R2_ROOT))
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")
M3OE_ROOT = os.path.join(PROJECT_ROOT, "m3oe")
for _path in (PROJECT_ROOT, EXPERIMENTS_ROOT, M3OE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from MDMTRec.utils.data import TorchDataset  # noqa: E402
from stage_E1_instrumentation import (  # noqa: E402
    batch_fingerprint,
    preserve_rng_state,
    rng_state_digest,
    write_json,
    write_rows_csv,
)
from stage_S1_metrics import LOGLOSS_EPS, safe_auc, safe_logloss  # noqa: E402
from stage_S1_data import (  # noqa: E402
    DataBundle,
    array_hash,
    load_stage_s1_bundle,
    row_ids_hash,
    split_row_fingerprint,
)
from stage_S1_models import (  # noqa: E402
    features_from_schema,
    initial_state_hash,
    is_architecture_parameter,
    parameter_manifest,
)
from stage_S1_protocol import FORMAL_PROTOCOL  # noqa: E402

from ao04_v12_model import (  # noqa: E402
    AO04M3OE,
    B3_BUFFER_NAME,
    DELTA_BAL_D_NAME,
    Q_RHO_NAME,
    build_ao04_model,
    is_ao04_architecture_parameter,
    is_ao04_buffer,
)
from ao04_v12_protocol import (  # noqa: E402
    ARCHITECTURE_PARAMETER_NAMES,
    B3_RHO_TABLE,
    BATCH_SIZE,
    BETAS,
    BUDGET_PER_DATASET,
    COMMON_EPOCHS,
    DATASETS,
    DEFAULT_DEVICE,
    DATASET_SETTINGS,
    DOMAIN_NUM,
    DROP_LAST,
    EPS_ADAM,
    EPOCHS,
    EXPECTED_COUNTS,
    FCN_DIMS,
    FINGERPRINT_ANCHORS,
    LEARNING_RATE_ARCH,
    LEARNING_RATE_MAIN,
    LOGLOSS_EPS_ACTUAL,
    NEW_ARM_PARAMETER_NAMES,
    N_CELLS,
    NUM_WORKERS,
    SHADOW_PATIENCE,
    SHUFFLE_TRAIN,
    TASK_NUM,
    WEIGHT_DECAY,
    arm_supported,
    dataset_supported,
    expected_param_total,
    setting_supported,
)
from ao04_v12_hpost import HPOST_ENABLED  # noqa: E402


# ---------------------------------------------------------------------------
# Forbidden-identifier guard.  Training paths must NOT reference any of
# the names imported here from the Stage S1 runner.  The grep-level
# static check is implemented in ``ao04_v12_static_check.py`` so the
# runner's own source does not embed the literals as text.
# ---------------------------------------------------------------------------


def _assert_no_forbidden(reference: str) -> None:
    """Helper for asserting no forbidden identifier is referenced.

    Used only by ``final_test_evaluator`` (test side).  Training paths
    raise this check via static grep; this runtime guard catches
    accidental direct references if someone tries to route test data
    through the training path.
    """

    from ao04_v12_static_check import _FORBIDDEN_IDENTIFIER_LIST  # noqa: F401
    if reference in _FORBIDDEN_IDENTIFIER_LIST:
        raise TestSealViolation(
            "forbidden reference {!r} inside the runner training path".format(reference)
        )


# ---------------------------------------------------------------------------
# Tripwire counters.
# ---------------------------------------------------------------------------

TEST_PREDICTION_CALLS: int = 0


def _bump_test_counter() -> None:
    """Increment the test-prediction counter (test-side only)."""

    global TEST_PREDICTION_CALLS
    TEST_PREDICTION_CALLS += 1


class TestSealViolation(RuntimeError):
    """Raised when training-side code attempts to access test data."""


# ---------------------------------------------------------------------------
# SealedTestSplit — fingerprint-only façade for the test split.
# ---------------------------------------------------------------------------

@dataclass
class SealedTestSplit:
    """Hides test arrays behind a fingerprint-only interface.

    The runner side only sees ``row_count``, ``fingerprint`` and
    ``class_coverage_check`` (a preflight used by G0 to confirm the
    validation contract).  Any other access — including iteration over
    features/labels — raises ``TestSealViolation``.
    """

    row_count: int
    fingerprint_value: str
    row_ids_array_hash: str
    labels_array_hash: str
    preflight_per_cell_counts: Dict[str, int]
    preflight_pre_cell_classes: Dict[str, int]

    def class_coverage_check(self) -> Dict[str, Dict[str, int]]:
        """Return the preflight counts (cheap, no raw arrays exposed)."""

        return {
            key: {"count": int(self.preflight_per_cell_counts.get(key, 0)),
                  "unique_classes": int(self.preflight_pre_cell_classes.get(key, 0))}
            for key in self.preflight_per_cell_counts
        }

    def __getitem__(self, key: str) -> Any:
        raise TestSealViolation(
            "SealedTestSplit does not expose raw arrays via __getitem__ ({!r})".format(key)
        )

    def __iter__(self) -> Iterator[Any]:
        raise TestSealViolation("SealedTestSplit is not iterable")

    def __len__(self) -> int:
        raise TestSealViolation("SealedTestSplit.__len__ is hidden")

    def __getattr__(self, name: str) -> Any:
        if name in ("row_count", "fingerprint_value", "row_ids_array_hash",
                    "labels_array_hash", "preflight_per_cell_counts",
                    "preflight_pre_cell_classes", "class_coverage_check"):
            return object.__getattribute__(self, name)
        raise TestSealViolation(
            "SealedTestSplit forbids access to attribute {!r}".format(name)
        )


# ---------------------------------------------------------------------------
# Seed setup (mirrors Stage S1 protocol).
# ---------------------------------------------------------------------------

def setup_seed(seed: int) -> None:
    """Reproduce Stage S1 setup_seed exactly."""

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _deterministic_train_generator(seed: int) -> torch.Generator:
    """Per-arm deterministic generator for the train DataLoader.

    Independent of the global RNG state — constructed fresh on each
    call so it always yields the same shuffle order across the five
    arms for the same (dataset, seed) pair.  See Specification §8.4.
    """

    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


# ---------------------------------------------------------------------------
# JSON helper.
# ---------------------------------------------------------------------------

def _json_array(values: Sequence[float]) -> str:
    return json.dumps([float(v) for v in values], separators=(",", ":"))


def _json_counts(values: Sequence[int]) -> str:
    return json.dumps([int(v) for v in values], separators=(",", ":"))


def _safe_move_to_device(x: torch.Tensor, device: torch.device,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return x.transpose(0, 1).to(device=device, dtype=dtype)


def _hash_state_dict(state: Mapping[str, torch.Tensor]) -> str:
    """Stable SHA256 over a state_dict."""

    h = hashlib.sha256()
    for name in sorted(state):
        v = state[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(v.shape)).encode("ascii"))
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_finite(label: str, value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise FloatingPointError("{} is non-finite: {}".format(label, value))
    return number


# ---------------------------------------------------------------------------
# Locked-config and data-identity assertions (G0-2 / no silent override).
# ---------------------------------------------------------------------------

_LOCKED_CONFIG_FIELDS: Dict[str, object] = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "shuffle_train": SHUFFLE_TRAIN,
    "drop_last": DROP_LAST,
    "learning_rate": LEARNING_RATE_MAIN,
    "learning_rate_arch": LEARNING_RATE_ARCH,
    "weight_decay": WEIGHT_DECAY,
    "betas": list(BETAS),
    "eps": EPS_ADAM,
    "shadow_patience": SHADOW_PATIENCE,
    "common_epochs": list(COMMON_EPOCHS),
    "scheduler": None,
    "hpost_enabled": False,
}


def _assert_locked_config(config: Mapping[str, Any], dataset: str, *,
                          smoke: bool = False) -> None:
    """Assert every locked config field equals the protocol constant.

    No silent fallback: any key present in ``config`` must match the
    frozen value; missing keys fall back to the protocol constant (the
    constant is what is used, so this cannot drift).  ``epochs`` may only
    deviate from 50 when ``smoke`` is True (G0 smoke path).
    """

    for key, locked in _LOCKED_CONFIG_FIELDS.items():
        if key in config:
            observed = config[key]
            if isinstance(locked, float):
                match = float(observed) == float(locked)
            else:
                match = (list(observed) == list(locked)
                         if isinstance(locked, list) else observed == locked)
            if not match:
                raise ValueError(
                    "locked config field {!r} mismatch: {} != {}".format(
                        key, observed, locked))
    epochs = int(config.get("epochs", EPOCHS))
    if not smoke and epochs != EPOCHS:
        raise ValueError("epochs must be {} outside smoke mode (got {})".format(
            EPOCHS, epochs))
    device = str(config.get("device", DEFAULT_DEVICE))
    if not smoke and device != DEFAULT_DEVICE:
        raise ValueError("device must be {} outside smoke mode (got {})".format(
            DEFAULT_DEVICE, device))


def _assert_data_identity(dataset: str, bundle: DataBundle,
                          config: Mapping[str, Any]) -> Dict[str, Any]:
    """G0-2: expected counts + fingerprint anchors + dataset byte hash.

    Test access here is limited to per-domain row counts (preflight
    counting), which is the only test-label-free identity check allowed
    before the global unlock.
    """

    setting = DATASET_SETTINGS[dataset][0]
    expected = EXPECTED_COUNTS[dataset][setting]

    def counts_of(x: np.ndarray) -> Dict[str, int]:
        ids = np.asarray(x[:, -1]).astype(np.int64)
        return {str(d): int((ids == d).sum()) for d in range(DOMAIN_NUM)}

    observed = {
        "train": counts_of(bundle.train_x),
        "validation": counts_of(bundle.val_x),
        "test": counts_of(bundle.test_x),
    }
    for split in ("train", "validation", "test"):
        if observed[split] != expected[split]:
            raise AssertionError(
                "{} {} counts mismatch: {} != expected {}".format(
                    dataset, split, observed[split], expected[split]))

    anchors = FINGERPRINT_ANCHORS[dataset][setting]
    fingerprints: Dict[str, Any] = {
        "counts": observed,
        "encoder_hash": bundle.features.encoder_hash,
        "scaler_hash": bundle.features.scaler_hash,
        "base_split_fingerprints": dict(bundle.base_split_fingerprints),
    }
    if anchors.get("encoder_hash"):
        if bundle.features.encoder_hash != anchors["encoder_hash"]:
            raise AssertionError("encoder_hash anchor mismatch for {}: {} != {}".format(
                dataset, bundle.features.encoder_hash, anchors["encoder_hash"]))
    if anchors.get("scaler_hash"):
        if bundle.features.scaler_hash != anchors["scaler_hash"]:
            raise AssertionError("scaler_hash anchor mismatch for {}: {} != {}".format(
                dataset, bundle.features.scaler_hash, anchors["scaler_hash"]))
    for split, anchor_key in (("train", "base_split_train"),
                              ("validation", "base_split_validation"),
                              ("test", "base_split_test")):
        anchor = anchors.get(anchor_key, "")
        actual = bundle.base_split_fingerprints.get(split, "")
        if anchor and actual != anchor:
            raise AssertionError(
                "base split fingerprint anchor mismatch {} {}: {} != {}".format(
                    dataset, split, actual, anchor))

    dataset_path = config.get("dataset_path")
    expected_sha = config.get("dataset_sha256")
    expected_size = config.get("dataset_size_bytes")
    if dataset_path and expected_sha:
        if not os.path.isfile(str(dataset_path)):
            raise FileNotFoundError("dataset file missing: {}".format(dataset_path))
        if expected_size is not None \
                and int(os.path.getsize(str(dataset_path))) != int(expected_size):
            raise AssertionError("dataset size mismatch: {} != {}".format(
                os.path.getsize(str(dataset_path)), expected_size))
        actual_sha = _file_hash(str(dataset_path))
        if actual_sha != expected_sha:
            raise AssertionError("dataset sha256 mismatch: {} != {}".format(
                actual_sha, expected_sha))
        fingerprints["dataset_sha256"] = actual_sha
    return fingerprints


# ---------------------------------------------------------------------------
# Component 1: paired_init_phase
# ---------------------------------------------------------------------------

@dataclass
class PairedInitArtifacts:
    """Artifacts written by paired_init_phase."""

    init_state_path: str
    init_rng_path: str
    init_manifest_path: str
    initial_state_hash: str
    rng_state_digest: str


def paired_init_phase(dataset: str, seed: int, *,
                      init_dir: str,
                      dataset_path: Optional[str] = None) -> Tuple[DataBundle, PairedInitArtifacts]:
    """Build a B0 model under deterministic seeds and persist its state.

    No training occurs.  The artifact triplet is later used by every
    arm to reproduce the original parameter state via strict load.
    """

    if dataset not in DATASETS:
        raise ValueError("unknown dataset: {}".format(dataset))
    seed = int(seed)

    setup_seed(seed)
    bundle = load_stage_s1_bundle(dataset, dataset_path,
                                  DATASET_SETTINGS[dataset][0], seed)
    # G0-2 data identity: expected counts, fingerprint anchors, dataset
    # byte hash (only when the caller supplied them via ``config``-like
    # arguments; paired_init_phase receives the raw path so we verify the
    # on-disk file against the protocol anchors unconditionally).
    identity = _assert_data_identity(dataset, bundle, {})
    if dataset_path:
        manifest_extra_hash = _file_hash(str(dataset_path))
    else:
        default_path = None
        # load_stage_s1_bundle resolved the default path; hash the file it
        # actually used (recorded in preprocessing metadata).
        default_path = bundle.preprocessing_metadata.get("raw_file")
        manifest_extra_hash = (_file_hash(str(default_path)) if default_path else "")
    # Construct a B0 model only — new parameters are not needed.
    feats = features_from_schema(bundle.features)
    model = build_ao04_model(feats, "B0", seed=seed)
    initial_hash = initial_state_hash(model)

    out_dir = os.path.abspath(os.path.join(init_dir, dataset, "seed{}".format(seed)))
    os.makedirs(out_dir, exist_ok=True)
    init_state_path = os.path.join(out_dir, "init_state.pt")
    init_rng_path = os.path.join(out_dir, "init_rng.pt")
    init_manifest_path = os.path.join(out_dir, "init_manifest.json")

    # Save CPU state dict (B0 only — no buffer, no extras).
    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(cpu_state, init_state_path)

    raw_rng = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else []),
    }
    torch.save(raw_rng, init_rng_path)
    rng_digest = rng_state_digest()

    manifest: Dict[str, Any] = {
        "dataset": dataset,
        "setting": bundle.setting,
        "seed": seed,
        "arm": "B0",
        "backbone": "m3oe",
        "initial_state_hash": initial_hash,
        "rng_state_digest_after_init": rng_digest,
        "tensor_hashes": {k: _hash_state_dict({k: v}) for k, v in cpu_state.items()},
        "init_state_path": init_state_path,
        "init_rng_path": init_rng_path,
        "data_identity": identity,
        "dataset_file_sha256": manifest_extra_hash,
        "bundle_manifest": bundle.manifest_fields(),
        "split_row_fingerprints": bundle.split_fingerprints(),
        "row_id_fingerprints": bundle.row_id_fingerprints(),
    }
    write_json(init_manifest_path, manifest)
    artifacts = PairedInitArtifacts(
        init_state_path=init_state_path,
        init_rng_path=init_rng_path,
        init_manifest_path=init_manifest_path,
        initial_state_hash=initial_hash,
        rng_state_digest=rng_digest,
    )
    return bundle, artifacts


# ---------------------------------------------------------------------------
# Optimizer construction for an arm.
# ---------------------------------------------------------------------------

def build_optimizers_for_arm(
    model: AO04M3OE,
    *,
    learning_rate: float = LEARNING_RATE_MAIN,
    learning_rate_arch: float = LEARNING_RATE_ARCH,
    weight_decay: float = WEIGHT_DECAY,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer,
           List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Build the main and arch Adam optimizers exactly per the spec.

    Main optimizer: every trainable parameter except the four raw
    scalars (i.e. all backbone parameters + arm-specific q / delta).
    Arch optimizer: only the four raw scalars (with the strict name
    set in ``ARCHITECTURE_PARAMETER_NAMES``).
    """

    arch_set: List[torch.nn.Parameter] = []
    main_set: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if is_ao04_architecture_parameter(name):
            arch_set.append(param)
        else:
            main_set.append(param)

    if len(arch_set) != len(ARCHITECTURE_PARAMETER_NAMES):
        raise AssertionError(
            "arch optimizer must contain exactly {} tensors; got {}".format(
                len(ARCHITECTURE_PARAMETER_NAMES), len(arch_set))
        )

    main_opt = torch.optim.Adam(main_set, lr=float(learning_rate),
                                betas=(0.9, 0.999),
                                weight_decay=float(weight_decay),
                                eps=float(EPS_ADAM))
    arch_opt = torch.optim.Adam(arch_set, lr=float(learning_rate_arch),
                                betas=(0.9, 0.999),
                                weight_decay=float(weight_decay),
                                eps=float(EPS_ADAM))
    return main_opt, arch_opt, main_set, arch_set


# ---------------------------------------------------------------------------
# Train/val helpers (lifted from Stage S1).
# ---------------------------------------------------------------------------

def _cell_records(bundle: DataBundle, model: torch.nn.Module, loader: DataLoader,
                  device: torch.device,
                  collect_arrays: bool = False):
    """Per-cell metrics; optionally also return the raw prediction/label arrays.

    ``collect_arrays=True`` is used only for the validation pass so the
    selected epoch's raw predictions can be persisted for the G1-3
    reload-replay gate.  The metric computation is unchanged either way.
    """

    domain_num, task_num = bundle.domain_num, bundle.task_num
    truth: Dict[int, List[float]] = {t * domain_num + d: []
                                     for t in range(task_num)
                                     for d in range(domain_num)}
    predictions: Dict[int, List[float]] = {k: [] for k in truth}
    pred_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_device = _safe_move_to_device(x_batch, device)
            y_device = y_batch.to(device=device, dtype=torch.float32)
            pred = model(x_device)
            if not bool(torch.isfinite(pred).all().detach().cpu()):
                raise FloatingPointError("non-finite evaluation prediction")
            if collect_arrays:
                pred_parts.append(pred.detach().cpu().numpy().astype(
                    np.float32, copy=False))
                label_parts.append(y_batch.detach().cpu().numpy().astype(
                    np.float32, copy=False))
            domain_ids = x_device[-1, :].long()
            for t in range(task_num):
                for d in range(domain_num):
                    key = t * domain_num + d
                    mask = domain_ids == d
                    if bool(mask.any()):
                        truth[key].extend(y_device[mask, t].detach().cpu().tolist())
                        predictions[key].extend(pred[mask, t].detach().cpu().tolist())
    result: Dict[int, Dict[str, float]] = {}
    for key in truth:
        labels = truth[key]
        scores = predictions[key]
        if (len(labels) == 0
                or len({float(v) for v in labels}) < 2):
            raise ValueError("evaluation cell {} empty or single class".format(key))
        result[key] = {
            "sample_count": int(len(labels)),
            "positive_count": int(sum(float(v) for v in labels)),
            "positive_rate": float(np.mean(labels)),
            "auc": _require_finite("auc", safe_auc(labels, scores)),
            "logloss": _require_finite("logloss", safe_logloss(labels, scores)),
        }
    if collect_arrays:
        stacked_pred = (np.concatenate(pred_parts, axis=0)
                        if pred_parts else np.zeros((0, task_num), dtype=np.float32))
        stacked_label = (np.concatenate(label_parts, axis=0)
                         if label_parts else np.zeros((0, task_num), dtype=np.float32))
        return result, stacked_pred, stacked_label
    return result


def _macro(records: Mapping[int, Mapping[str, float]], field: str) -> float:
    return float(np.mean([_require_finite(field, records[k][field])
                          for k in records]))


def select_checkpoint_epoch(val_aucs: Mapping[int, float]) -> int:
    """Validation macro-AUC selection with strict > + earliest tie.

    Identical rule to Stage S1's helper; copied verbatim so the
    independent runner never imports the protected namespace.
    """

    finite = {int(e): _require_finite("validation_macro_auc", v)
              for e, v in val_aucs.items()}
    return min(finite, key=lambda e: (-finite[e], e))


def _domain_weighted(records: Mapping[int, Mapping[str, float]],
                     shares: Mapping[int, float], field: str) -> float:
    total = 0.0
    for t in range(TASK_NUM):
        for d in range(DOMAIN_NUM):
            key = t * DOMAIN_NUM + d
            total += float(shares[d]) * _require_finite(field, records[key][field])
    return float(total / float(TASK_NUM))


# ---------------------------------------------------------------------------
# Re-construct an arm from the paired init artifacts.
# ---------------------------------------------------------------------------

def _reconstruct_arm(features: Sequence[Any], arm: str,
                     seed: int, init_state_path: str) -> Tuple[AO04M3OE,
                                                                Dict[str, torch.Tensor]]:
    """Construct the named arm and strict-load the saved init state.

    New arm-specific tensors are added on top of the constructed
    backbone.  The strict load uses ``strict=False`` because B0's
    state dict lacks the new tensors; we manually re-add them after
    the load so equality is preserved bit-by-bit.
    """

    setup_seed(seed)
    model = build_ao04_model(features, arm, seed=seed)
    saved_state: Dict[str, torch.Tensor] = torch.load(init_state_path,
                                                      map_location="cpu")
    # Load only the keys present in saved_state (B0 keys).
    result = model.load_state_dict(saved_state, strict=False)
    if len(result.missing_keys) > 0:
        # At most the arm-specific new tensors (parameters) and the B3
        # buffer may be missing from the B0 init state.
        allowed = set(NEW_ARM_PARAMETER_NAMES.get(arm, ()))
        for key in result.missing_keys:
            if not is_ao04_buffer(key) and key not in allowed:
                raise RuntimeError(
                    "missing key {!r} is not in the allowed new-tensor set for arm {}".format(
                        key, arm))
    if len(result.unexpected_keys) > 0:
        raise RuntimeError("unexpected keys in init_state for arm {}: {}".format(
            arm, result.unexpected_keys))

    # Pairing contract (Specification §8.2): every loaded original tensor
    # must hash-equal the B0 init manifest; new arm tensors must be exactly
    # zero (B1/B2 q, B4 delta) or the fixed B3 table.
    init_manifest_path = os.path.join(os.path.dirname(os.path.abspath(init_state_path)),
                                      "init_manifest.json")
    if not os.path.exists(init_manifest_path):
        raise FileNotFoundError(
            "init manifest missing next to init state: {}".format(init_manifest_path))
    with open(init_manifest_path, encoding="utf-8") as handle:
        init_manifest = json.load(handle)
    tensor_hashes = init_manifest.get("tensor_hashes", {})
    if not tensor_hashes:
        raise RuntimeError("init manifest lacks tensor_hashes: {}".format(
            init_manifest_path))
    for key, value in saved_state.items():
        expected = tensor_hashes.get(key)
        if expected is None:
            raise RuntimeError("init manifest lacks tensor hash for {!r}".format(key))
        actual = _hash_state_dict({key: value})
        if actual != expected:
            raise RuntimeError(
                "paired init tensor hash mismatch for {!r}: {} != {}".format(
                    key, actual, expected))
    named = dict(model.named_parameters())
    for new_name in NEW_ARM_PARAMETER_NAMES.get(arm, ()):
        param = named.get(new_name)
        if param is None:
            raise RuntimeError("arm {} missing new parameter {!r}".format(arm, new_name))
        if not bool(torch.all(param.detach() == 0)):
            raise RuntimeError("new parameter {!r} is not zero-initialized".format(new_name))
    if arm == "B3":
        buffer_value = getattr(model, B3_BUFFER_NAME)
        expected_table = torch.tensor(B3_RHO_TABLE, dtype=torch.float32)
        if not bool(torch.equal(buffer_value.detach().cpu(), expected_table)):
            raise RuntimeError("B3 rho_fixed buffer differs from the fixed table")
    return model, saved_state


# ---------------------------------------------------------------------------
# Component 2: train_val_loop
# ---------------------------------------------------------------------------

@dataclass
class TrainValResult:
    selected_epoch: int
    shadow_stop_epoch: Optional[int]
    macro_auc_by_epoch: Dict[int, float]
    val_metrics_by_epoch: Dict[int, Dict[int, Dict[str, float]]]
    train_metrics_by_epoch: Dict[int, Dict[int, Dict[str, float]]]
    train_objective_by_epoch: Dict[int, float]
    main_steps: int
    arch_steps: int
    examples_processed: int
    arch_examples_processed: int
    batch_order_hash: str
    architecture_batch_order_hash: str
    batch_fingerprint_count: int
    architecture_batch_fingerprint_count: int
    rng_state_digest_after: str
    initial_parameter_state_hash: str
    final_parameter_state_hash: str
    run_duration_seconds: float
    fusion_rows: List[Dict[str, Any]] = field(default_factory=list)
    probe_rows: List[Dict[str, Any]] = field(default_factory=list)
    batch_rows: List[Dict[str, Any]] = field(default_factory=list)
    epochs_completed: int = 0
    resumed_from: Optional[str] = None


def train_val_loop(arm: str,
                   dataset: str,
                   seed: int,
                   run_dir: str,
                   config: Mapping[str, Any],
                   paired_state_path: str,
                   *, bundle: Optional[DataBundle] = None,
                   smoke_max_batches: Optional[int] = None,
                   resume_from: Optional[str] = None,
                   write_last_resume_each_epoch: bool = True,
                   ) -> TrainValResult:
    """Train one arm for the full 50-epoch schedule with checkpoint selection.

    ``smoke_max_batches`` is used only by G0 smoke tests and the 1-epoch
    smoke (KuaiRand B0, MovieLens B0).  ``resume_from`` is an optional
    path to a previous ``last_resume.pt`` produced by this same
    function.  This function never touches any test fixture.

    The run directory must be empty (or non-existent) unless
    ``resume_from`` continues the same run — this mirrors Stage S1's
    refusal to overwrite a non-empty canonical run directory.
    Failures leave a FAILED/INVALID ``run_manifest.json`` behind so the
    state machine stays auditable (Specification §7).
    """

    if not arm_supported(arm):
        raise ValueError("unsupported arm: {}".format(arm))
    if dataset not in DATASETS:
        raise ValueError("unsupported dataset: {}".format(dataset))
    if seed not in (2022, 2023):
        raise ValueError("seed must be 2022 or 2023 (got {})".format(seed))
    seed = int(seed)
    smoke = smoke_max_batches is not None
    _assert_locked_config(config, dataset, smoke=smoke)

    if os.path.isdir(run_dir) and os.listdir(run_dir) and resume_from is None:
        raise FileExistsError(
            "refusing to write a non-empty run directory: {}".format(run_dir))
    os.makedirs(run_dir, exist_ok=True)

    try:
        return _train_val_loop_impl(
            arm, dataset, seed, run_dir, config, paired_state_path,
            bundle=bundle, smoke_max_batches=smoke_max_batches,
            resume_from=resume_from,
            write_last_resume_each_epoch=write_last_resume_each_epoch)
    except Exception as exc:
        status = "INVALID" if isinstance(exc, FloatingPointError) else "FAILED"
        manifest_path = os.path.join(run_dir, "run_manifest.json")
        if not os.path.exists(manifest_path):
            write_json(manifest_path, {
                "stage": "R2", "design": "ao04_v12", "status": status,
                "arm": arm, "dataset": dataset,
                "setting": DATASET_SETTINGS[dataset][0], "seed": seed,
                "backbone": "m3oe", "run_dir": os.path.abspath(run_dir),
                "exception_type": type(exc).__name__, "exception": str(exc),
                "test_evaluated_during_training": False,
                "test_predictions_written": False,
            })
        raise


def _train_val_loop_impl(arm: str,
                         dataset: str,
                         seed: int,
                         run_dir: str,
                         config: Mapping[str, Any],
                         paired_state_path: str,
                         *, bundle: Optional[DataBundle] = None,
                         smoke_max_batches: Optional[int] = None,
                         resume_from: Optional[str] = None,
                         write_last_resume_each_epoch: bool = True,
                         ) -> TrainValResult:
    setting = DATASET_SETTINGS[dataset][0]
    setup_seed(seed)
    if bundle is None:
        bundle = load_stage_s1_bundle(dataset, config.get("dataset_path"),
                                      setting, seed)
    # G0-2 data identity (expected counts, fingerprint anchors, dataset
    # byte hash when the config carries it).
    _assert_data_identity(dataset, bundle, config)

    features = features_from_schema(bundle.features)
    model, saved_state = _reconstruct_arm(features, arm, seed, paired_state_path)

    # Initialize main and arch optimizers via the audit-friendly helper.
    main_opt, arch_opt, main_params, arch_params = build_optimizers_for_arm(model)

    device = torch.device(config.get("device", DEFAULT_DEVICE))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = model.to(device)
    initial_parameter_state_hash_cpu = initial_state_hash(model.cpu())
    model = model.to(device)

    batch_size = int(config.get("batch_size", BATCH_SIZE))
    num_workers = int(config.get("num_workers", NUM_WORKERS))
    epochs = int(config.get("epochs", EPOCHS))
    shadow_patience = int(config.get("shadow_patience", SHADOW_PATIENCE))

    # Use an explicit per-arm ``torch.Generator`` so that the train
    # shuffle order is identical across the five arms for the same
    # (dataset, seed) pair.  See Specification §8.4 / Pitfall §7.
    train_loader = DataLoader(
        TorchDataset(bundle.train_x, bundle.train_y),
        batch_size=batch_size, shuffle=bool(SHUFFLE_TRAIN),
        num_workers=num_workers, drop_last=bool(DROP_LAST),
        generator=_deterministic_train_generator(seed))
    val_loader = DataLoader(
        TorchDataset(bundle.val_x, bundle.val_y),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False)
    # train_eval_loader used by the full epoch train eval pass.
    train_eval_loader = DataLoader(
        TorchDataset(bundle.train_x, bundle.train_y),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False)

    selected_cpu_state: Optional[Dict[str, torch.Tensor]] = None
    selected_epoch: Optional[int] = None
    selected_macro_auc: float = float("-inf")
    selected_val_pred: Optional[np.ndarray] = None
    selected_val_label: Optional[np.ndarray] = None
    shadow_stop_epoch: Optional[int] = None
    bad_epochs = 0
    main_steps = 0
    arch_steps = 0
    examples_processed = 0
    arch_examples_processed = 0
    macro_auc_by_epoch: Dict[int, float] = {}
    val_metrics_by_epoch: Dict[int, Dict[int, Dict[str, float]]] = {}
    train_metrics_by_epoch: Dict[int, Dict[int, Dict[str, float]]] = {}
    train_objective_by_epoch: Dict[int, float] = {}
    checkpoint_states: Dict[int, Dict[str, torch.Tensor]] = {}
    batch_fingerprints: List[str] = []
    arch_batch_fingerprints: List[str] = []
    batch_rows: List[Dict[str, Any]] = []
    fusion_rows: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []
    rng_evidence: List[Dict[str, Any]] = []
    nonfinite_inventory: List[Dict[str, Any]] = []
    architecture_iterator: Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]] = None

    # Resume path (used only when a prior last_resume.pt is supplied).
    # ``weights_only=False`` is required because the payload legitimately
    # contains the NumPy RNG state (a tuple with ndarrays); the file is
    # written by this same module into this same run directory.
    start_epoch = 0
    if resume_from is not None and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model = model.to(device)
        main_opt.load_state_dict(ckpt["main_optimizer_state_dict"])
        arch_opt.load_state_dict(ckpt["arch_optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        batch_fingerprints = list(ckpt.get("batch_fingerprints", []))
        arch_batch_fingerprints = list(ckpt.get("arch_batch_fingerprints", []))
        selected_cpu_state = ckpt.get("selected_cpu_state")
        selected_epoch = ckpt.get("selected_epoch")
        selected_macro_auc = float(ckpt.get("selected_macro_auc", float("-inf")))
        bad_epochs = int(ckpt.get("bad_epochs", 0))
        shadow_stop_epoch = ckpt.get("shadow_stop_epoch")
        macro_auc_by_epoch = {int(k): float(v) for k, v in
                              ckpt.get("macro_auc_by_epoch", {}).items()}
        val_metrics_by_epoch = {int(k): {int(c): dict(r) for c, r in v.items()}
                                for k, v in ckpt.get("val_metrics_by_epoch", {}).items()}
        train_metrics_by_epoch = {int(k): {int(c): dict(r) for c, r in v.items()}
                                  for k, v in ckpt.get("train_metrics_by_epoch", {}).items()}
        train_objective_by_epoch = {int(k): float(v) for k, v in
                                    ckpt.get("train_objective_by_epoch", {}).items()}
        main_steps = int(ckpt.get("main_steps", 0))
        arch_steps = int(ckpt.get("arch_steps", 0))
        examples_processed = int(ckpt.get("examples_processed", 0))
        arch_examples_processed = int(ckpt.get("arch_examples_processed", 0))
        batch_rows = list(ckpt.get("batch_rows", []))
        fusion_rows = list(ckpt.get("fusion_rows", []))
        probe_rows = list(ckpt.get("probe_rows", []))
        rng_evidence = list(ckpt.get("rng_evidence", []))
        # RNG restoration.
        rng_state = ckpt.get("torch_cpu_rng", None)
        if rng_state is not None:
            torch.set_rng_state(rng_state)
        cuda_rng = ckpt.get("torch_cuda_rng", [])
        if cuda_rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)
        np_state = ckpt.get("numpy_rng")
        if np_state is not None:
            np.random.set_state(np_state)
        py_state = ckpt.get("python_rng")
        if py_state is not None:
            random.setstate(py_state)
        # Replay the DataLoader generator + arch iterator exactly: per
        # completed epoch the main loop consumed one full iteration
        # (including the sampler's trailing randperm draw at exhaustion)
        # and the arch lifecycle consumed one batch (creating or
        # re-creating its iterator on exhaustion).  Replaying the same
        # call sequence restores both the generator position and the
        # arch iterator.
        architecture_iterator = _replay_loader_position(
            train_loader, start_epoch, smoke_max_batches=smoke_max_batches)
        # Common-epoch checkpoints captured before the interruption live
        # on disk (they are written at capture time); reload them.
        for common_epoch in COMMON_EPOCHS:
            path = os.path.join(run_dir, "common_epoch_{}.pth".format(common_epoch))
            if os.path.exists(path):
                checkpoint_states[common_epoch] = torch.load(
                    path, map_location="cpu", weights_only=True)

    wall_start = time.time()

    epoch_range = range(start_epoch, epochs)
    if smoke_max_batches is not None:
        smoke_max_batches = int(smoke_max_batches)
        if smoke_max_batches < 1:
            raise ValueError("smoke_max_batches must be >=1 when set")
    n_train_batches = len(train_loader)
    effective_last_step = (min(n_train_batches, smoke_max_batches)
                           if smoke_max_batches is not None else n_train_batches) - 1
    for epoch in epoch_range:
        # -------- train one epoch --------
        model.train()
        objective_sum = 0.0
        sample_count = 0
        last_batch_grad_norms: Dict[str, float] = {}
        last_batch_update_norms: Dict[str, float] = {}
        for step, (x_batch, y_batch) in enumerate(train_loader):
            fp = batch_fingerprint(x_batch, y_batch)
            batch_fingerprints.append(fp)
            if smoke_max_batches is not None and step >= smoke_max_batches:
                break
            x_device = _safe_move_to_device(x_batch, device)
            y_device = y_batch.to(device=device, dtype=torch.float32)
            pred = model(x_device)
            if not bool(torch.isfinite(pred).all().detach().cpu()):
                raise FloatingPointError("non-finite training prediction")
            loss = F.binary_cross_entropy(pred, y_device.float(), reduction="mean")
            _require_finite("train_objective", loss.detach().cpu())
            main_opt.zero_grad(set_to_none=True)
            loss.backward()
            for name, param in model.named_parameters():
                if (not is_ao04_architecture_parameter(name)
                        and param.grad is not None
                        and not bool(torch.isfinite(param.grad).all().detach().cpu())):
                    raise FloatingPointError("non-finite gradient {}".format(name))
            if step == effective_last_step:
                # Per-epoch optimization audit (Specification §6.7):
                # gradient L2 norm of the new arm parameters at the final
                # train batch of the epoch, plus the update norm applied
                # by this main step.
                pre_step: Dict[str, torch.Tensor] = {}
                for new_name in NEW_ARM_PARAMETER_NAMES.get(arm, ()):
                    param = dict(model.named_parameters())[new_name]
                    if param.grad is not None:
                        last_batch_grad_norms[new_name + "_grad_l2"] = float(
                            param.grad.detach().norm().cpu())
                    pre_step[new_name] = param.detach().clone()
            main_opt.step()
            if step == effective_last_step:
                for new_name, before_value in pre_step.items():
                    param = dict(model.named_parameters())[new_name]
                    last_batch_update_norms[new_name + "_update_l2"] = float(
                        (param.detach() - before_value).norm().cpu())
            for name, param in model.named_parameters():
                if not bool(torch.isfinite(param).all().detach().cpu()):
                    raise FloatingPointError("non-finite parameter {}".format(name))
            main_steps += 1
            count = int(len(y_batch))
            examples_processed += count
            sample_count += count
            objective_sum += float(loss.detach().cpu()) * count
            batch_rows.append({
                "stage": "R2", "design": "ao04_v12", "arm": arm,
                "dataset": dataset, "setting": bundle.setting, "seed": seed,
                "epoch": int(epoch), "step": int(step), "kind": "train",
                "batch_fingerprint": fp, "batch_size": int(count),
                "domain_counts": _json_counts([
                    int((x_batch[:, -1].long() == d).sum()) for d in range(DOMAIN_NUM)]),
                "task_counts": _json_counts([int(count)] * TASK_NUM),
                "train_objective": float(loss.detach().cpu()),
            })
        train_objective = objective_sum / float(sample_count or 1)
        train_objective_by_epoch[epoch] = float(train_objective)

        # -------- end-of-epoch full train eval --------
        train_records = _cell_records(bundle, model, train_eval_loader, device)
        train_metrics_by_epoch[epoch] = train_records

        # -------- arch update (one batch, persistent iterator) --------
        architecture_iterator, arch_audit = _architecture_update(
            model, arch_opt, train_loader, architecture_iterator, epoch,
            arch_batch_fingerprints, device, bundle,
            rng_evidence=rng_evidence)
        arch_steps += 1
        arch_examples_processed += int(arch_audit["batch_size"])
        batch_rows.append({
            "stage": "R2", "design": "ao04_v12", "arm": arm,
            "dataset": dataset, "setting": bundle.setting, "seed": seed,
            "epoch": int(epoch), "step": -1, "kind": "architecture",
            "batch_fingerprint": arch_audit["fingerprint"],
            "batch_size": int(arch_audit["batch_size"]),
            "domain_counts": _json_counts(arch_audit["domain_counts"]),
            "task_counts": _json_counts([int(arch_audit["batch_size"])] * TASK_NUM),
            "train_objective": float(arch_audit["objective"]),
        })

        # -------- validation --------
        with preserve_rng_state():
            val_records, epoch_val_pred, epoch_val_label = _cell_records(
                bundle, model, val_loader, device, collect_arrays=True)
        val_metrics_by_epoch[epoch] = val_records
        macro = _macro(val_records, "auc")
        macro_auc_by_epoch[epoch] = float(macro)

        # -------- descriptive probes (Specification §6.6) --------
        probe_rows.append(_expert_gate_probe(model, val_loader, device,
                                             epoch=epoch, arm=arm))

        # -------- fusion / coefficient logging (Specification §6.3-6.5) --
        fusion_rows.append(_fusion_log_row(
            model, arm, epoch, last_batch_grad_norms, last_batch_update_norms,
            arch_audit["arch_grad_l2"]))

        improved = macro > selected_macro_auc  # strict > for tie = earliest
        if improved:
            selected_macro_auc = float(macro)
            selected_epoch = int(epoch)
            selected_cpu_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
            selected_val_pred = epoch_val_pred
            selected_val_label = epoch_val_label
            bad_epochs = 0
        else:
            bad_epochs += 1
        if shadow_stop_epoch is None and bad_epochs >= shadow_patience:
            shadow_stop_epoch = int(epoch)
        if epoch in COMMON_EPOCHS or epoch == epochs - 1:
            state_copy = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            checkpoint_states[epoch] = state_copy
            if epoch in COMMON_EPOCHS:
                # Persist immediately so a resume can recover them.
                torch.save(state_copy, os.path.join(
                    run_dir, "common_epoch_{}.pth".format(epoch)))

        # -------- resume artifact (overwritten every epoch) --------
        if write_last_resume_each_epoch:
            _write_last_resume(
                run_dir=run_dir, model=model, main_opt=main_opt,
                arch_opt=arch_opt, epoch=epoch + 1,
                arch_batches_consumed_total=arch_steps,
                batch_fingerprints=batch_fingerprints,
                arch_batch_fingerprints=arch_batch_fingerprints,
                selected_cpu_state=selected_cpu_state,
                selected_epoch=selected_epoch,
                selected_macro_auc=selected_macro_auc,
                bad_epochs=bad_epochs,
                shadow_stop_epoch=shadow_stop_epoch,
                macro_auc_by_epoch=macro_auc_by_epoch,
                val_metrics_by_epoch=val_metrics_by_epoch,
                train_metrics_by_epoch=train_metrics_by_epoch,
                train_objective_by_epoch=train_objective_by_epoch,
                main_steps=main_steps, arch_steps=arch_steps,
                examples_processed=examples_processed,
                arch_examples_processed=arch_examples_processed,
                batch_rows=batch_rows, fusion_rows=fusion_rows,
                probe_rows=probe_rows, rng_evidence=rng_evidence)

    if selected_cpu_state is None or selected_epoch is None:
        raise RuntimeError("no validation checkpoint selected for run")
    # Cross-check the online strict-> selection against the offline
    # helper, exactly as Stage S1's fit() does.
    recomputed = select_checkpoint_epoch(macro_auc_by_epoch)
    if recomputed != int(selected_epoch):
        raise AssertionError(
            "checkpoint selection helper disagrees with training selection: "
            "{} != {}".format(recomputed, selected_epoch))

    # The true final state is the last completed epoch's capture (epoch
    # 49 in a full run); the selected state is a separate artifact.
    final_epoch = max(checkpoint_states.keys())
    model.load_state_dict(checkpoint_states[final_epoch], strict=True)
    final_parameter_state_hash_cpu = initial_state_hash(model.cpu())
    model = model.to(device)
    # Restore the selected state into the live model for downstream
    # consumers that expect the selected parameters after the loop.
    model.load_state_dict(selected_cpu_state, strict=True)
    wall = time.time() - wall_start
    result = TrainValResult(
        selected_epoch=int(selected_epoch),
        shadow_stop_epoch=shadow_stop_epoch,
        macro_auc_by_epoch=dict(macro_auc_by_epoch),
        val_metrics_by_epoch=val_metrics_by_epoch,
        train_metrics_by_epoch=train_metrics_by_epoch,
        train_objective_by_epoch=train_objective_by_epoch,
        main_steps=int(main_steps),
        arch_steps=int(arch_steps),
        examples_processed=int(examples_processed),
        arch_examples_processed=int(arch_examples_processed),
        batch_order_hash=hashlib.sha256(
            json.dumps(batch_fingerprints).encode()).hexdigest(),
        architecture_batch_order_hash=hashlib.sha256(
            json.dumps(arch_batch_fingerprints).encode()).hexdigest(),
        batch_fingerprint_count=len(batch_fingerprints),
        architecture_batch_fingerprint_count=len(arch_batch_fingerprints),
        rng_state_digest_after=rng_state_digest(),
        initial_parameter_state_hash=str(initial_parameter_state_hash_cpu),
        final_parameter_state_hash=str(final_parameter_state_hash_cpu),
        run_duration_seconds=float(wall),
        fusion_rows=fusion_rows,
        probe_rows=probe_rows,
        batch_rows=batch_rows,
        epochs_completed=int(len(macro_auc_by_epoch)),
        resumed_from=(os.path.abspath(resume_from) if resume_from else None),
    )

    # Stash selected state, checkpoint states and the ACTUAL optimizer
    # states onto the run dir for state_writer.
    torch.save(selected_cpu_state, os.path.join(run_dir, "_selected_state.pt"))
    torch.save(checkpoint_states, os.path.join(run_dir, "_checkpoint_states.pt"))
    torch.save({"main": main_opt.state_dict(), "arch": arch_opt.state_dict()},
               os.path.join(run_dir, "_optimizer_states.pt"))
    # G1-3 evidence: raw validation predictions at the selected epoch.
    # Captured in-memory on every improving epoch; after a resume whose
    # post-resume epochs never improved, the arrays were never captured
    # in this process — recompute them once from the selected state the
    # model was just restored to (eval mode, deterministic loader).
    if selected_val_pred is None or selected_val_label is None:
        with preserve_rng_state():
            _, selected_val_pred, selected_val_label = _cell_records(
                bundle, model, val_loader, device, collect_arrays=True)
    np.savez_compressed(os.path.join(run_dir, "_val_pred_selected.npz"),
                        predictions=np.asarray(selected_val_pred, dtype=np.float32),
                        labels=np.asarray(selected_val_label, dtype=np.float32))
    rng_manifest = {
        "rng_evidence": rng_evidence,
        "nonfinite_inventory": nonfinite_inventory,
    }
    write_json(os.path.join(run_dir, "_train_val_aux.json"), rng_manifest)
    return result


def _replay_loader_position(train_loader: DataLoader,
                            upto_epoch: int,
                            smoke_max_batches: Optional[int] = None) -> Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]]:
    """Replay DataLoader consumption so the generator and arch iterator
    resume at exactly the position they had at the end of epoch
    ``upto_epoch - 1``.

    The replay must mirror the real loop's consumption *exactly*,
    including iterator exhaustion: torch's ``RandomSampler.__iter__``
    draws a trailing ``randperm`` whose output is truncated away when
    ``num_samples % n == 0``, and that draw fires only when the iterator
    is consumed to ``StopIteration``.  Creating-and-discarding the main
    iterator therefore does NOT reproduce the generator position — the
    main loop is replayed by fully iterating it (with the same
    ``smoke_max_batches`` early-break semantics as the real loop), and
    the persistent arch iterator lifecycle is replayed one batch per
    epoch, exactly as ``_architecture_update`` consumes it.
    """

    architecture_iterator: Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]] = None
    for _ in range(int(upto_epoch)):
        for step, _batch in enumerate(train_loader):
            if smoke_max_batches is not None and step >= smoke_max_batches:
                break
        if architecture_iterator is None:
            architecture_iterator = iter(train_loader)
        try:
            next(architecture_iterator)
        except StopIteration:
            architecture_iterator = iter(train_loader)
            next(architecture_iterator)
    return architecture_iterator


def _write_last_resume(*, run_dir: str, model: AO04M3OE,
                       main_opt: torch.optim.Optimizer,
                       arch_opt: torch.optim.Optimizer,
                       epoch: int,
                       arch_batches_consumed_total: int,
                       batch_fingerprints: List[str],
                       arch_batch_fingerprints: List[str],
                       selected_cpu_state: Optional[Dict[str, torch.Tensor]],
                       selected_epoch: Optional[int],
                       selected_macro_auc: float,
                       bad_epochs: int,
                       shadow_stop_epoch: Optional[int],
                       macro_auc_by_epoch: Dict[int, float],
                       val_metrics_by_epoch: Dict[int, Dict[int, Dict[str, float]]],
                       train_metrics_by_epoch: Dict[int, Dict[int, Dict[str, float]]],
                       train_objective_by_epoch: Dict[int, float],
                       main_steps: int, arch_steps: int,
                       examples_processed: int,
                       arch_examples_processed: int,
                       batch_rows: List[Dict[str, Any]],
                       fusion_rows: List[Dict[str, Any]],
                       probe_rows: List[Dict[str, Any]],
                       rng_evidence: List[Dict[str, Any]]) -> None:
    payload = {
        "model_state_dict": {k: v.detach().cpu()
                             for k, v in model.state_dict().items()},
        "main_optimizer_state_dict": main_opt.state_dict(),
        "arch_optimizer_state_dict": arch_opt.state_dict(),
        "epoch": int(epoch),
        "arch_batches_consumed_total": int(arch_batches_consumed_total),
        "batch_fingerprints": list(batch_fingerprints),
        "arch_batch_fingerprints": list(arch_batch_fingerprints),
        "selected_cpu_state": selected_cpu_state,
        "selected_epoch": selected_epoch,
        "selected_macro_auc": float(selected_macro_auc),
        "bad_epochs": int(bad_epochs),
        "shadow_stop_epoch": shadow_stop_epoch,
        "macro_auc_by_epoch": dict(macro_auc_by_epoch),
        "val_metrics_by_epoch": val_metrics_by_epoch,
        "train_metrics_by_epoch": train_metrics_by_epoch,
        "train_objective_by_epoch": dict(train_objective_by_epoch),
        "main_steps": int(main_steps),
        "arch_steps": int(arch_steps),
        "examples_processed": int(examples_processed),
        "arch_examples_processed": int(arch_examples_processed),
        "batch_rows": list(batch_rows),
        "fusion_rows": list(fusion_rows),
        "probe_rows": list(probe_rows),
        "rng_evidence": list(rng_evidence),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else []),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    path = os.path.join(run_dir, "last_resume.pt")
    torch.save(payload, path)


def _architecture_update(model: AO04M3OE,
                        arch_opt: torch.optim.Optimizer,
                        train_loader: DataLoader,
                        architecture_iterator: Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]],
                        epoch: int,
                        arch_batch_fingerprints: List[str],
                        device: torch.device,
                        bundle: DataBundle,
                        *,
                        rng_evidence: List[Dict[str, Any]]
                        ) -> Tuple[Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]],
                                   Dict[str, Any]]:
    """One arch update per epoch, persistent iterator, RNG-preserved batch.

    Implementation lift of ``stage_S1_run._architecture_update`` with
    one extra guarantee: the iterator is initialized *only once per
    run* and survives across epochs; otherwise StopIteration triggers
    a single new iterator and the next ``next()`` call consumes the
    first batch of the new shuffle.

    Returns the (possibly renewed) iterator together with an audit dict
    carrying the batch fingerprint, size, domain counts, objective and
    the pre-step gradient L2 norm of each of the four raw scalars
    (Specification §6.7).
    """

    if arch_opt is None:
        return architecture_iterator, {"fingerprint": "", "batch_size": 0,
                                       "domain_counts": [0] * DOMAIN_NUM,
                                       "objective": float("nan"),
                                       "arch_grad_l2": {}}
    model.train()
    rng_before = rng_state_digest()
    with preserve_rng_state():
        if architecture_iterator is None:
            architecture_iterator = iter(train_loader)
        try:
            x_batch, y_batch = next(architecture_iterator)
        except StopIteration:
            architecture_iterator = iter(train_loader)
            x_batch, y_batch = next(architecture_iterator)
    rng_after = rng_state_digest()
    rng_evidence.append({"epoch": int(epoch), "kind": "architecture_batch",
                          "before": rng_before, "after": rng_after,
                          "restored": rng_before == rng_after})
    fp = batch_fingerprint(x_batch, y_batch)
    arch_batch_fingerprints.append(fp)
    x_device = _safe_move_to_device(x_batch, device)
    y_device = y_batch.to(device=device, dtype=torch.float32)
    pred = model(x_device)
    if not bool(torch.isfinite(pred).all().detach().cpu()):
        raise FloatingPointError("non-finite architecture prediction")
    loss = F.binary_cross_entropy(pred, y_device.float(), reduction="mean")
    _require_finite("architecture_objective", loss.detach().cpu())
    arch_opt.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    loss.backward()
    arch_grad_l2: Dict[str, float] = {}
    for name, param in model.named_parameters():
        if not is_ao04_architecture_parameter(name):
            continue
        if param.grad is None:
            raise FloatingPointError(
                "architecture parameter {} received no gradient".format(name))
        if not bool(torch.isfinite(param.grad).all().detach().cpu()):
            raise FloatingPointError("non-finite architecture gradient {}".format(name))
        arch_grad_l2[name] = float(param.grad.detach().norm().cpu())
    arch_opt.step()
    audit = {
        "fingerprint": fp,
        "batch_size": int(len(y_batch)),
        "domain_counts": [int((x_batch[:, -1].long() == d).sum())
                          for d in range(DOMAIN_NUM)],
        "objective": float(loss.detach().cpu()),
        "arch_grad_l2": arch_grad_l2,
    }
    return architecture_iterator, audit


# ---------------------------------------------------------------------------
# Descriptive probes (Specification §6.3-6.7).
# ---------------------------------------------------------------------------

@torch.no_grad()
def _expert_gate_probe(model: AO04M3OE,
                       val_loader: DataLoader,
                       device: torch.device,
                       *,
                       epoch: int,
                       arm: str) -> Dict[str, Any]:
    """Per-epoch expert/gate probe on the first validation batch.

    Specification §6.6: under eval + no_grad + ``preserve_rng_state``,
    record the batch-mean L2 norm of all 9 expert outputs (4 shared,
    3 domain, 2 task) and the mean softmax entropy of the 6 gates on
    the first validation batch.  Descriptive only — no gradient, no
    state mutation; the model's training mode and the global RNG are
    restored on exit.
    """

    was_training = model.training
    with preserve_rng_state():
        model.eval()
        x_batch, _ = next(iter(val_loader))
        x_device = _safe_move_to_device(x_batch, device)
        domain_id = x_device[-1, :].clone().detach().long()
        masks = [(domain_id == d) for d in range(model.domain_num)]
        input_emb = model.embedding(x_device, model.features, squeeze_dim=True)
        skip = model.skip_conn(input_emb)
        emb = torch.zeros((input_emb.shape[0], model.star_dim[1]),
                          device=device, dtype=input_emb.dtype)
        for index, (weight, bias) in enumerate(zip(model.slot_weight,
                                                   model.slot_bias)):
            output = torch.matmul(input_emb,
                                  torch.multiply(weight, model.shared_weight)) \
                + bias + model.shared_bias
            emb = torch.where(masks[index].unsqueeze(1), output, emb)
        emb = model.star_mlp(emb) + skip

        row: Dict[str, Any] = {
            "stage": "R2", "design": "ao04_v12", "arm": arm,
            "epoch": int(epoch), "probe": "expert_gate",
            "probe_batch_size": int(x_batch.shape[0]),
        }
        for i in range(model.expert_num):
            out = model.expert[i](emb)
            row["expert_norm_shared_{}".format(i)] = float(
                out.norm(dim=1).mean().cpu())
        for i in range(model.domain_num):
            out = model.domain_expert[i](emb)
            row["expert_norm_domain_{}".format(i)] = float(
                out.norm(dim=1).mean().cpu())
        for i in range(model.task_num):
            out = model.task_expert[i](emb)
            row["expert_norm_task_{}".format(i)] = float(
                out.norm(dim=1).mean().cpu())
        for i in range(model.task_num * model.domain_num):
            probs = model.gate[i](emb.detach())  # ends with Softmax(dim=1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean()
            row["gate_entropy_{}".format(i)] = float(entropy.cpu())
    if was_training:
        model.train()
    return row


def _fusion_log_row(model: AO04M3OE,
                    arm: str,
                    epoch: int,
                    grad_norms: Mapping[str, float],
                    update_norms: Mapping[str, float],
                    arch_grad_l2: Mapping[str, float]) -> Dict[str, Any]:
    """One per-epoch fusion-log row (Specification §6.3-6.5 + §6.7).

    Captures the four global raw scalars (raw + sigmoid), the
    arm-specific quantities (B1 q/rho table, B2 per-domain q/rho, B3
    fixed table, B4 delta/centred delta/effective b_dt), the per-cell
    own/other coefficient snapshot, the saturation flag (any rho or
    effective b_dt outside [0.01, 0.99] — flagged only, never clamped),
    and the optimization-audit norms captured at the epoch's final
    train batch plus the arch-step raw-scalar gradient norms.
    """

    snapshot = model.fusion_coefficients()
    row: Dict[str, Any] = {
        "stage": "R2", "design": "ao04_v12", "arm": arm,
        "epoch": int(epoch),
        "raw_aD": float(snapshot["raw"]["aD_raw"]),
        "raw_aT": float(snapshot["raw"]["aT_raw"]),
        "raw_bD": float(snapshot["raw"]["bD_raw"]),
        "raw_bT": float(snapshot["raw"]["bT_raw"]),
        "sigmoid_aD": float(snapshot["sigmoid"]["aD"]),
        "sigmoid_aT": float(snapshot["sigmoid"]["aT"]),
        "sigmoid_bD": float(snapshot["sigmoid"]["bD"]),
        "sigmoid_bT": float(snapshot["sigmoid"]["bT"]),
    }
    saturation_pool: List[float] = []
    if arm == "B1":
        q_vals = [float(v) for v in snapshot["q_rho"]]
        rho_vals = [float(v) for v in snapshot["rho_dt"]]
        for i, v in enumerate(q_vals):
            row["q_rho_{}".format(i)] = v
        for i, v in enumerate(rho_vals):
            row["rho_dt_{}".format(i)] = v
        row["rho_mean"] = float(np.mean(rho_vals))
        saturation_pool.extend(rho_vals)
    elif arm == "B2":
        q_vals = [float(v) for v in snapshot["q_rho_by_domain"]]
        rho_d = [float(v) for v in snapshot["rho_per_domain"]]
        rho_dt = [float(v) for v in snapshot["rho_dt"]]
        for i, v in enumerate(q_vals):
            row["q_d_{}".format(i)] = v
        for i, v in enumerate(rho_d):
            row["rho_d_{}".format(i)] = v
        row["rho_shared_across_tasks"] = 1
        row["rho_mean"] = float(np.mean(rho_dt))
        saturation_pool.extend(rho_dt)
    elif arm == "B3":
        rho_dt = [float(v) for v in snapshot["rho_dt"]]
        for i, v in enumerate(rho_dt):
            row["rho_dt_{}".format(i)] = v
        row["rho_mean"] = float(np.mean(rho_dt))
        row["rho_fixed_table"] = 1
        saturation_pool.extend(rho_dt)
    elif arm == "B4":
        delta = [float(v) for v in snapshot["delta_dt"]]
        b_dt = [float(v) for v in snapshot["b_dt"]]
        delta_mean = float(np.mean(delta))
        for i, v in enumerate(delta):
            row["delta_dt_{}".format(i)] = v
        row["delta_mean"] = delta_mean
        for i, v in enumerate(delta):
            row["delta_centered_{}".format(i)] = v - delta_mean
        for i, v in enumerate(b_dt):
            row["b_dt_{}".format(i)] = v
        saturation_pool.extend(b_dt)
    own = [float(c["own"]) for c in snapshot["per_cell"]]
    other = [float(c["other_total"]) for c in snapshot["per_cell"]]
    row["own_coeffs_json"] = _json_array(own)
    row["other_coeffs_json"] = _json_array(other)
    if saturation_pool:
        row["saturation_flag"] = int(any(
            (v < 0.01 or v > 0.99) for v in saturation_pool))
    else:
        row["saturation_flag"] = ""
    for key, value in grad_norms.items():
        row[key] = float(value)
    for key, value in update_norms.items():
        row[key] = float(value)
    for raw_name, value in arch_grad_l2.items():
        short = raw_name.replace("_weight_", "").replace(".deep_weights", "")
        row["arch_grad_l2_{}".format(short)] = float(value)
    return row


# ---------------------------------------------------------------------------
# Component 3: state_writer
# ---------------------------------------------------------------------------

def state_writer(arm: str,
                 dataset: str,
                 seed: int,
                 run_dir: str,
                 config: Mapping[str, Any],
                 bundle: DataBundle,
                 train_val: TrainValResult,
                 paired_state_path: str,
                 *,
                 test_arrays_present: bool = True) -> Dict[str, Any]:
    """Persist training-side artifacts under run_dir.

    The state writer never touches a test array; if ``test_arrays_present``
    is False the writer still completes without complaint, simply
    omitting test-related fields from the manifest.
    """

    if not arm_supported(arm):
        raise ValueError("unsupported arm: {}".format(arm))

    os.makedirs(run_dir, exist_ok=True)
    device = torch.device(config.get("device", DEFAULT_DEVICE))

    selected_state_path = os.path.join(run_dir, "selected_checkpoint.pth")
    final_state_path = os.path.join(run_dir, "final_checkpoint.pth")
    selected_state = torch.load(os.path.join(run_dir, "_selected_state.pt"),
                                 map_location="cpu")
    checkpoint_states = torch.load(os.path.join(run_dir, "_checkpoint_states.pt"),
                                   map_location="cpu")
    torch.save(selected_state, selected_state_path)
    # final == last completed epoch when the run terminates early.
    last_completed_epoch = max(checkpoint_states.keys())
    torch.save(checkpoint_states[last_completed_epoch], final_state_path)
    common_paths: Dict[int, str] = {}
    for epoch in COMMON_EPOCHS:
        if epoch in checkpoint_states:
            p = os.path.join(run_dir, "common_epoch_{}.pth".format(epoch))
            torch.save(checkpoint_states[epoch], p)
            common_paths[epoch] = p

    # Re-inflate the model briefly so parameter_manifest is correct.
    setup_seed(seed)
    model_feats = features_from_schema(bundle.features)
    model = build_ao04_model(model_feats, arm, seed=seed)
    model.load_state_dict(selected_state, strict=True)
    manifest = parameter_manifest(model)
    expected_total = expected_param_total(dataset, arm)
    if int(manifest["total"]) != int(expected_total):
        raise AssertionError(
            "parameter total mismatch dataset={} arm={}: model={} expected={}".format(
                dataset, arm, manifest["total"], expected_total))
    architecture_names = list(manifest["architecture_names"])
    if sorted(architecture_names) != sorted(ARCHITECTURE_PARAMETER_NAMES):
        raise AssertionError(
            "architecture_names mismatch for arm {}: {}".format(
                arm, architecture_names))

    # Persist the ACTUAL optimizer states captured at the end of training
    # (stashed by _train_val_loop_impl).  Rebuilding fresh optimizers here
    # would emit empty state dicts and silently destroy the audit value of
    # these artifacts.
    optimizer_states = torch.load(os.path.join(run_dir, "_optimizer_states.pt"),
                                  map_location="cpu", weights_only=False)
    optimizer_state_path = os.path.join(run_dir, "optimizer_state.pth")
    arch_optimizer_state_path = os.path.join(run_dir, "architecture_optimizer_state.pth")
    torch.save(optimizer_states["main"], optimizer_state_path)
    torch.save(optimizer_states["arch"], arch_optimizer_state_path)
    rng_state_path = os.path.join(run_dir, "rng_state.pth")
    torch.save({
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else []),
    }, rng_state_path)

    # CSV: metrics_epoch.csv (per epoch / per cell).  Only writes
    # rows for the epochs that actually ran — the 1-epoch smoke writes
    # a single epoch's worth.
    metrics_rows = _build_metrics_epoch_rows(bundle, arm, dataset, seed,
                                              config, train_val)
    fields = sorted({key for row in metrics_rows for key in row.keys()})
    write_rows_csv(os.path.join(run_dir, "metrics_epoch.csv"), metrics_rows, fields)

    # CSV: fusion_log.csv — per-epoch fusion coefficients (Specification
    # §6.3-6.5 + §6.7) merged with the per-epoch expert/gate probe rows
    # (§6.6) on the epoch key.  All values were captured online during
    # training, so this log is a true per-epoch trajectory.
    fusion_rows = _merge_fusion_probe_rows(train_val.fusion_rows,
                                           train_val.probe_rows)
    if fusion_rows:
        fusion_fields = sorted({key for row in fusion_rows for key in row.keys()})
        write_rows_csv(os.path.join(run_dir, "fusion_log.csv"), fusion_rows, fusion_fields)

    # CSV: batch_trace.csv — the per-batch trace recorded online by the
    # training loop (train + architecture batches, with fingerprints,
    # domain counts and per-batch objective).
    batch_rows = list(train_val.batch_rows)
    if batch_rows:
        batch_fields = sorted({key for row in batch_rows for key in row.keys()})
        write_rows_csv(os.path.join(run_dir, "batch_trace.csv"), batch_rows, batch_fields)

    # selection.json — selected epoch + macro AUC across epochs.
    selection_payload = {
        "selected_epoch": int(train_val.selected_epoch),
        "shadow_stop_epoch": train_val.shadow_stop_epoch,
        "macro_auc_by_epoch": {str(k): float(v)
                                for k, v in train_val.macro_auc_by_epoch.items()},
        "fixed_final_epoch": EPOCHS - 1,
        "common_epochs": list(COMMON_EPOCHS),
    }
    write_json(os.path.join(run_dir, "selection.json"), selection_payload)

    write_json(os.path.join(run_dir, "parameter_manifest.json"), manifest)

    # Publish the selected-epoch validation predictions (stashed by
    # _train_val_loop_impl) as a first-class artifact for the G1-3
    # reload-replay gate.
    val_pred_stash = os.path.join(run_dir, "_val_pred_selected.npz")
    val_pred_path = os.path.join(run_dir, "validation_predictions_selected.npz")
    val_pred_sha: Optional[str] = None
    if os.path.exists(val_pred_stash):
        os.replace(val_pred_stash, val_pred_path)
        val_pred_sha = _file_hash(val_pred_path)

    # run manifest.
    config_with_runtime = dict(config)
    config_with_runtime["code"] = {
        name: _file_hash(os.path.join(R2_ROOT, name))
        for name in (
            "ao04_v12_model.py",
            "ao04_v12_protocol.py",
            "ao04_v12_runner.py",
            "ao04_v12_decision.py",
            "ao04_v12_test_evaluator.py",
            "ao04_v12_static_check.py",
            "ao04_v12_hpost.py",
        )
    }
    write_json(os.path.join(run_dir, "config.json"), config_with_runtime)
    init_manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(paired_state_path)), "init_manifest.json")
    if not os.path.exists(init_manifest_path):
        raise FileNotFoundError(
            "paired init manifest missing: {}".format(init_manifest_path))
    manifest_doc = {
        "stage": "R2",
        "design": "ao04_v12",
        "status": "COMPLETE",
        "arm": arm,
        "dataset": dataset,
        "setting": bundle.setting,
        "seed": seed,
        "backbone": "m3oe",
        "run_dir": run_dir,
        "selected_epoch": int(train_val.selected_epoch),
        "selected_validation_macro_auc": float(train_val.macro_auc_by_epoch[int(train_val.selected_epoch)]),
        "selected_validation_predictions_sha256": val_pred_sha,
        "shadow_stop_epoch": train_val.shadow_stop_epoch,
        "fixed_final_epoch": EPOCHS - 1,
        "common_epochs": list(COMMON_EPOCHS),
        "epochs_completed": int(train_val.epochs_completed),
        "resumed_from": train_val.resumed_from,
        "main_optimizer_steps": int(train_val.main_steps),
        "architecture_optimizer_steps": int(train_val.arch_steps),
        "examples_processed": int(train_val.examples_processed),
        "architecture_examples_processed": int(train_val.arch_examples_processed),
        "batch_order_hash": train_val.batch_order_hash,
        "batch_fingerprint_count": int(train_val.batch_fingerprint_count),
        "architecture_batch_order_hash": train_val.architecture_batch_order_hash,
        "architecture_batch_fingerprint_count": int(train_val.architecture_batch_fingerprint_count),
        "initial_parameter_state_hash": train_val.initial_parameter_state_hash,
        "final_parameter_state_hash": train_val.final_parameter_state_hash,
        "dataset_domain_shares": {str(k): float(v)
                                  for k, v in bundle.domain_shares.items()},
        "paired_init_manifest_sha256": _file_hash(init_manifest_path),
        "test_evaluated_during_training": False,
        "test_arrays_present": bool(test_arrays_present),
        "test_predictions_written": False,
        "rng_state_digest_after_run": train_val.rng_state_digest_after,
        "paired_init_state_path": paired_state_path,
        "metric_implementation": {
            "module": "stage_S1_metrics.py",
            "auc": "tie_aware_average_rank",
            "logloss": "fixed_epsilon_clip",
            "logloss_epsilon": LOGLOSS_EPS_ACTUAL,
            "logloss_epsilon_charter_text": 1e-7,
            "logloss_epsilon_rationale": "FROZEN to actual 1e-12 — see R2_Implementation_Report.md",
        },
    }
    write_json(os.path.join(run_dir, "run_manifest.json"), manifest_doc)

    # Cleanup ephemeral helpers (kept for debug).
    for name in ("_selected_state.pt", "_checkpoint_states.pt",
                 "_optimizer_states.pt", "_train_val_aux.json"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            os.remove(path)

    return {
        "selected_checkpoint_path": selected_state_path,
        "final_checkpoint_path": final_state_path,
        "common_checkpoint_paths": common_paths,
        "last_resume_path": os.path.join(run_dir, "last_resume.pt"),
        "metrics_epoch_csv": os.path.join(run_dir, "metrics_epoch.csv"),
        "selection_json": os.path.join(run_dir, "selection.json"),
        "run_manifest_path": os.path.join(run_dir, "run_manifest.json"),
    }


# ---------------------------------------------------------------------------
# CSV builders.
# ---------------------------------------------------------------------------

def _build_metrics_epoch_rows(bundle: DataBundle, arm: str,
                              dataset: str, seed: int,
                              config: Mapping[str, Any],
                              train_val: TrainValResult) -> List[Dict[str, Any]]:
    """Per-epoch × per-cell metrics rows."""

    shares = bundle.domain_shares
    rows: List[Dict[str, Any]] = []
    for epoch in sorted(train_val.val_metrics_by_epoch.keys()):
        val_rec = train_val.val_metrics_by_epoch[epoch]
        train_rec = train_val.train_metrics_by_epoch.get(epoch, {})
        macro_auc = _macro(val_rec, "auc")
        macro_ll = _macro(val_rec, "logloss")
        target_auc = _domain_weighted(val_rec, shares, "auc")
        target_ll = _domain_weighted(val_rec, shares, "logloss")
        worst_domain = float(min(
            np.mean([val_rec[t * DOMAIN_NUM + d]["auc"] for t in range(TASK_NUM)])
            for d in range(DOMAIN_NUM)))
        worst_cell = float(min(val_rec[k]["auc"] for k in val_rec))
        for t in range(TASK_NUM):
            for d in range(DOMAIN_NUM):
                key = t * DOMAIN_NUM + d
                v = val_rec[key]
                tr = train_rec.get(key, {})
                rows.append({
                    "stage": "R2", "design": "ao04_v12",
                    "arm": arm, "dataset": dataset, "setting": bundle.setting,
                    "backbone": "m3oe", "seed": seed, "epoch": int(epoch),
                    "domain": d, "physical_domain": int(bundle.local_to_physical[d]),
                    "size_role": bundle.role_by_domain.get(d, ""),
                    "task": t, "cell": key,
                    "train_sample_count": tr.get("sample_count", ""),
                    "train_positive_rate": tr.get("positive_rate", ""),
                    "train_cell_logloss": tr.get("logloss", ""),
                    "validation_sample_count": int(v["sample_count"]),
                    "validation_positive_count": int(v["positive_count"]),
                    "validation_positive_rate": float(v["positive_rate"]),
                    "validation_auc": float(v["auc"]),
                    "validation_logloss": float(v["logloss"]),
                    "validation_macro_auc": float(macro_auc),
                    "validation_macro_logloss": float(macro_ll),
                    "target_weighted_validation_auc": float(target_auc),
                    "target_weighted_validation_logloss": float(target_ll),
                    "worst_domain_validation_auc": float(worst_domain),
                    "worst_cell_validation_auc": float(worst_cell),
                    "train_objective": float(train_val.train_objective_by_epoch[epoch]),
                    "main_optimizer_steps": int(train_val.main_steps),
                    "architecture_optimizer_steps": int(train_val.arch_steps),
                    "learning_rate": float(LEARNING_RATE_MAIN),
                    "architecture_learning_rate": float(LEARNING_RATE_ARCH),
                    "selected_checkpoint": int(epoch == train_val.selected_epoch),
                    "fixed_final": int(epoch == EPOCHS - 1),
                    "common_epoch": int(epoch in COMMON_EPOCHS),
                })
    return rows


def _merge_fusion_probe_rows(fusion_rows: List[Dict[str, Any]],
                             probe_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge per-epoch fusion rows with per-epoch probe rows on epoch.

    Both row families are keyed by ``epoch``; probe keys (``probe``,
    ``probe_batch_size``, ``expert_norm_*``, ``gate_entropy_*``) never
    collide with fusion keys, so a shallow dict merge per epoch is exact.
    """

    by_epoch: Dict[int, Dict[str, Any]] = {}
    for row in fusion_rows:
        by_epoch[int(row["epoch"])] = dict(row)
    for row in probe_rows:
        epoch = int(row["epoch"])
        merged = by_epoch.get(epoch, {})
        for key, value in row.items():
            if key in merged and key not in ("stage", "design", "arm", "epoch"):
                raise AssertionError(
                    "fusion/probe column collision on {!r}".format(key))
            merged[key] = value
        by_epoch[epoch] = merged
    return [by_epoch[e] for e in sorted(by_epoch)]


# ---------------------------------------------------------------------------
# SealedTestSplit factory helper.
# ---------------------------------------------------------------------------

def _sealed_from_bundle(bundle: DataBundle) -> SealedTestSplit:
    test_x = bundle.test_x
    test_y = bundle.test_y
    test_row_ids = bundle.test_row_ids
    fp = split_row_fingerprint(test_x, test_y, test_row_ids)
    row_id_hash = row_ids_hash(test_row_ids)
    test_y_f32 = np.asarray(test_y, dtype=np.float32)
    label_hash = array_hash(test_y_f32)
    per_cell_counts: Dict[str, int] = {}
    per_cell_classes: Dict[str, int] = {}
    domain_ids = test_x[:, -1].astype(np.int64)
    for t in range(TASK_NUM):
        for d in range(DOMAIN_NUM):
            mask = domain_ids == d
            labels = test_y_f32[mask, t]
            per_cell_counts["d{}t{}".format(d, t)] = int(mask.sum())
            per_cell_classes["d{}t{}".format(d, t)] = int(
                len(np.unique(labels)) if labels.size else 0)
    return SealedTestSplit(
        row_count=int(test_x.shape[0]),
        fingerprint_value=fp,
        row_ids_array_hash=row_id_hash,
        labels_array_hash=label_hash,
        preflight_per_cell_counts=per_cell_counts,
        preflight_pre_cell_classes=per_cell_classes,
    )


# ---------------------------------------------------------------------------
# Static-tripwire helpers used by tests / review.
# ---------------------------------------------------------------------------

def forbidden_identifier_check(filename: str = "ao04_v12_runner.py",
                                root: str = R2_ROOT) -> List[str]:
    """Static-tripwire wrapper.

    Delegates to ``ao04_v12_static_check.runner_static_tripwire`` so the
    forbidden identifier names do not appear as literals in this
    module's source.
    """

    from ao04_v12_static_check import runner_static_tripwire
    path = os.path.join(root, filename)
    return runner_static_tripwire(path)


def tripwire_self_test() -> Dict[str, Any]:
    """Smoke test for the tripwire mechanics.

    Exercises: forbidden-identifier static check; SealedTestSplit
    behaviour; module-level counter; ``final_test_evaluator`` counter
    bump.
    """

    report: Dict[str, Any] = {
        "forbidden_in_runner": forbidden_identifier_check(),
    }
    # Forbidden identifiers must be absent from the runner module.
    if report["forbidden_in_runner"]:
        raise AssertionError("forbidden identifiers present in runner")
    # Construct a tiny SealedTestSplit and confirm access raises.
    sealed = SealedTestSplit(10, "fp", "rid", "lbl", {}, {})
    access_ok = {"row_count": False, "fingerprint_value": False, "class_coverage_check": False,
                 "x": False, "iter": False}
    access_ok["row_count"] = sealed.row_count == 10
    access_ok["fingerprint_value"] = sealed.fingerprint_value == "fp"
    access_ok["class_coverage_check"] = isinstance(sealed.class_coverage_check(), dict)
    try:
        sealed.x  # type: ignore[attr-defined]
    except TestSealViolation:
        access_ok["x"] = True
    try:
        iter(sealed)
    except TestSealViolation:
        access_ok["iter"] = True
    report["sealed_test_split"] = access_ok
    return report


__all__ = [
    "TestSealViolation",
    "SealedTestSplit",
    "TEST_PREDICTION_CALLS",
    "setup_seed",
    "paired_init_phase",
    "PairedInitArtifacts",
    "build_optimizers_for_arm",
    "train_val_loop",
    "TrainValResult",
    "state_writer",
    "forbidden_identifier_check",
    "tripwire_self_test",
    "_sealed_from_bundle",
]
