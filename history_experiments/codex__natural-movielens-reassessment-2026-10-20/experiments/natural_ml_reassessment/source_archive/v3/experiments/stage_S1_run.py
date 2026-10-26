"""One fixed-budget raw Stage S1 run.

The runner emits observations only.  It never compares a joint result to a
reference and never makes a GO/STOP decision.  Test evaluation is deferred
until all 50 validation epochs have completed and checkpoint selection is
frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
M3OE_ROOT = os.path.join(ROOT, "m3oe")
EXAMPLES_ROOT = os.path.join(M3OE_ROOT, "examples")
for _path in (HERE, M3OE_ROOT, EXAMPLES_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from MDMTRec.utils.data import TorchDataset  # noqa: E402
from stage_E1_instrumentation import (  # noqa: E402
    batch_fingerprint,
    parameter_state_hash,
    preserve_rng_state,
    rng_state_digest,
    write_json,
    write_rows_csv,
)
from stage_S1_metrics import LOGLOSS_EPS, safe_auc, safe_logloss  # noqa: E402
from stage_S1_data import DataBundle, array_hash, load_stage_s1_bundle  # noqa: E402
from stage_S1_models import (  # noqa: E402
    apply_semantic_initialization,
    build_model,
    estimate_forward_flops,
    initial_state_hash,
    is_architecture_parameter,
    parameter_manifest,
)
from stage_S1_protocol import (  # noqa: E402
    BACKBONES,
    COMMON_EPOCHS,
    EPOCHS,
    FORMAL_PROTOCOL,
    ROLE_SIZES,
    REFERENCE_SCOPE,
    SEEDS,
    THRESHOLDS,
    assert_formal_values,
    canonical_dataset,
    formal_key,
    protocol_mismatches,
)


METRICS_EPOCH_FIELDS = [
    "stage", "run_type", "dataset", "setting", "backbone", "seed", "epoch",
    "domain", "physical_domain", "size_role", "task", "cell",
    "train_sample_count", "validation_sample_count", "train_positive_rate",
    "validation_positive_rate", "train_cell_logloss", "validation_auc",
    "validation_logloss", "validation_macro_auc", "validation_macro_logloss",
    "target_weighted_validation_auc", "target_weighted_validation_logloss",
    "worst_domain_validation_auc", "worst_cell_validation_auc",
    "train_objective", "main_optimizer_steps", "architecture_optimizer_steps",
    "learning_rate", "architecture_learning_rate", "selected_checkpoint",
    "fixed_final", "common_epoch",
]
METRICS_TEST_FIELDS = [
    "stage", "run_type", "dataset", "setting", "backbone", "seed", "checkpoint",
    "stage_epoch", "training_stage", "domain", "physical_domain", "size_role",
    "task", "cell", "train_sample_count", "validation_sample_count",
    "test_sample_count", "test_positive_rate", "auc", "logloss",
    "validation_auc_at_stage", "validation_logloss_at_stage",
    "validation_macro_auc_at_stage", "validation_macro_logloss_at_stage",
    "selected_epoch", "selected_validation_macro_auc", "parameter_count",
    "training_stage_identifier", "domain_share", "physical_domain_identity",
    "task_identity", "train_positive_count", "validation_positive_count",
    "test_positive_count", "train_positive_rate", "validation_positive_rate",
    "train_label_entropy", "validation_label_entropy",
    "test_label_entropy", "reference_scope",
]
BATCH_FIELDS = [
    "stage", "run_type", "dataset", "setting", "backbone", "seed", "epoch", "step",
    "kind", "batch_fingerprint", "batch_size", "domain_counts", "task_counts",
    "train_objective",
]


def setup_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def direct_per_example_bce(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The sole Stage S1 objective: mean BCE over examples and tasks."""

    if prediction.shape != target.shape:
        raise ValueError("prediction/target shape mismatch: {} vs {}".format(
            tuple(prediction.shape), tuple(target.shape)))
    return F.binary_cross_entropy(prediction, target.float(), reduction="mean")


def normalized_cell_bce(prediction: torch.Tensor, target: torch.Tensor,
                       domain_ids: torch.Tensor, domain_num: int,
                       task_num: int) -> torch.Tensor:
    """Normalized p_d cell expression used only as an equivalence oracle."""

    total = prediction.new_zeros(())
    n = float(target.shape[0])
    if n == 0:
        raise ValueError("empty objective batch")
    shares = []
    for domain in range(int(domain_num)):
        shares.append((domain_ids == domain).float().mean())
    for task in range(int(task_num)):
        for domain in range(int(domain_num)):
            mask = domain_ids == domain
            if not bool(mask.any()):
                continue
            cell_mean = F.binary_cross_entropy(
                prediction[mask, task], target[mask, task].float(), reduction="mean")
            total = total + shares[domain] * cell_mean / float(task_num)
    return total


def objective_equivalence(prediction: torch.Tensor, target: torch.Tensor,
                          domain_ids: torch.Tensor, domain_num: int,
                          task_num: int, atol: float = 1e-7) -> Dict[str, float | bool]:
    direct = direct_per_example_bce(prediction, target)
    normalized = normalized_cell_bce(prediction, target, domain_ids, domain_num, task_num)
    difference = float(torch.abs(direct - normalized).detach().cpu())
    return {"direct": float(direct.detach().cpu()),
            "normalized": float(normalized.detach().cpu()),
            "absolute_difference": difference, "equivalent": difference <= float(atol)}


def select_checkpoint_epoch(validation_macro_auc: Mapping[int, float]) -> int:
    """Select max validation macro-AUC with a deterministic earliest tie break."""

    if not validation_macro_auc:
        raise ValueError("cannot select from an empty validation history")
    finite = {int(epoch): _require_finite("validation_macro_auc", score)
              for epoch, score in validation_macro_auc.items()}
    return min(finite, key=lambda epoch: (-finite[epoch], epoch))


# Explicit names used in design-review tests.
sample_objective = direct_per_example_bce
direct_sample_bce = direct_per_example_bce
normalized_sample_objective = normalized_cell_bce


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _require_finite(label: str, value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise FloatingPointError("{} is non-finite: {}".format(label, value))
    return number


def _state_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_metadata(device: torch.device) -> Dict[str, object]:
    """Persist the concrete child-runtime identity used for a raw run."""

    cuda_available = bool(torch.cuda.is_available())
    device_index = int(device.index if device.index is not None else 0) if device.type == "cuda" else None
    device_name = ""
    if cuda_available and device.type == "cuda":
        device_name = str(torch.cuda.get_device_name(device_index))
    return {
        "python_executable": os.path.abspath(sys.executable),
        "python_version": sys.version,
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_version": str(torch.version.cuda or ""),
        "device_identity": str(device),
        "device_index": device_index,
        "device_name": device_name,
        "torch_cudnn_version": (int(torch.backends.cudnn.version())
                                 if torch.backends.cudnn.is_available() else None),
        "num_threads": int(torch.get_num_threads()),
    }


def _code_hash_inventory() -> Dict[str, str]:
    """Hash every Stage S1 implementation file used by the formal runner."""

    names = ("stage_S1_run.py", "stage_S1_models.py", "stage_S1_data.py",
             "stage_S1_protocol.py", "stage_S1_report.py", "stage_S1_orchestrator.py",
             "stage_S1_metrics.py")
    return {name: _file_hash(os.path.join(HERE, name)) for name in names}


def _raw_rng_state() -> Dict[str, object]:
    """Return serializable CPU/CUDA/NumPy/Python RNG state for audit."""

    state: Dict[str, object] = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = []
    return state


def _json_counts(values: Sequence[int]) -> str:
    return json.dumps([int(value) for value in values], separators=(",", ":"))


def _optimizer_steps(optimizer: torch.optim.Optimizer) -> int:
    steps: List[int] = []
    for state in optimizer.state.values():
        value = state.get("step")
        if value is None:
            continue
        steps.append(int(value.detach().cpu()) if isinstance(value, torch.Tensor) else int(value))
    return max(steps, default=0)


def _optimizer_audit(optimizer: Optional[torch.optim.Optimizer]) -> Dict[str, object]:
    if optimizer is None:
        return {"group_count": 0, "groups": [], "state_parameter_count": 0,
                "step_min": 0, "step_max": 0}
    steps: List[int] = []
    groups = []
    for group in optimizer.param_groups:
        groups.append({"parameter_count": len(group.get("params", [])),
                       "betas": list(group.get("betas", ())),
                       "lr": float(group.get("lr", 0.0)),
                       "weight_decay": float(group.get("weight_decay", 0.0))})
    for state in optimizer.state.values():
        for field in ("exp_avg", "exp_avg_sq"):
            if field in state and not bool(torch.isfinite(state[field]).all().detach().cpu()):
                raise FloatingPointError("non-finite optimizer state {}".format(field))
        value = state.get("step")
        if value is not None:
            steps.append(int(value.detach().cpu()) if isinstance(value, torch.Tensor) else int(value))
    return {"group_count": len(groups), "groups": groups,
            "state_parameter_count": len(optimizer.state),
            "step_min": min(steps) if steps else 0, "step_max": max(steps) if steps else 0}


def _cell_records(bundle: DataBundle, model: torch.nn.Module, loader: DataLoader,
                  device: torch.device) -> Dict[int, Dict[str, float]]:
    domain_num, task_num = bundle.domain_num, bundle.task_num
    truth: Dict[int, List[float]] = {task * domain_num + d: []
                                     for task in range(task_num) for d in range(domain_num)}
    predictions: Dict[int, List[float]] = {key: [] for key in truth}
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_device = x_batch.transpose(0, 1).to(device=device, dtype=torch.float32)
            y_device = y_batch.to(device=device, dtype=torch.float32)
            pred = model(x_device)
            if not bool(torch.isfinite(pred).all().detach().cpu()):
                raise FloatingPointError("non-finite evaluation prediction")
            domain_ids = x_device[-1, :].long()
            for task in range(task_num):
                for domain in range(domain_num):
                    key = task * domain_num + domain
                    mask = domain_ids == domain
                    if bool(mask.any()):
                        truth[key].extend(y_device[mask, task].detach().cpu().tolist())
                        predictions[key].extend(pred[mask, task].detach().cpu().tolist())
    result: Dict[int, Dict[str, float]] = {}
    for key in truth:
        labels = truth[key]
        scores = predictions[key]
        if len(labels) == 0 or len(set(float(item) for item in labels)) < 2:
            raise ValueError("evaluation cell {} is empty or missing one class".format(key))
        auc = safe_auc(labels, scores)  # unclipped ranking scores
        logloss = safe_logloss(labels, scores)  # clipping occurs inside LogLoss only
        result[key] = {
            "sample_count": int(len(labels)),
            "positive_count": int(sum(float(item) for item in labels)),
            "positive_rate": float(np.mean(labels)),
            "auc": _require_finite("auc", auc),
            "logloss": _require_finite("logloss", logloss),
        }
    return result


def _prediction_records(bundle: DataBundle, model: torch.nn.Module,
                        loader: DataLoader, device: torch.device,
                        row_ids: np.ndarray) -> Tuple[Dict[int, Dict[str, float]],
                                                       np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate once and retain row-aligned labels/predictions for audit/reporting."""

    labels_parts: List[np.ndarray] = []
    prediction_parts: List[np.ndarray] = []
    domain_parts: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_device = x_batch.transpose(0, 1).to(device=device, dtype=torch.float32)
            y_device = y_batch.to(device=device, dtype=torch.float32)
            prediction = model(x_device)
            if not bool(torch.isfinite(prediction).all().detach().cpu()):
                raise FloatingPointError("non-finite evaluation prediction")
            labels_parts.append(y_batch.detach().cpu().numpy().astype(np.float32, copy=False))
            prediction_parts.append(prediction.detach().cpu().numpy().astype(np.float32, copy=False))
            domain_parts.append(x_batch[:, -1].detach().cpu().numpy().astype(np.int64, copy=False))
    labels = np.concatenate(labels_parts, axis=0) if labels_parts else np.empty((0, bundle.task_num), dtype=np.float32)
    predictions = np.concatenate(prediction_parts, axis=0) if prediction_parts else np.empty_like(labels)
    domains = np.concatenate(domain_parts, axis=0) if domain_parts else np.empty((0,), dtype=np.int64)
    ordered_ids = np.asarray(row_ids, dtype=np.int64)
    if len(labels) != len(ordered_ids):
        raise ValueError("prediction row count does not match declared row IDs")
    truth: Dict[int, List[float]] = {task * bundle.domain_num + d: []
                                     for task in range(bundle.task_num)
                                     for d in range(bundle.domain_num)}
    scores: Dict[int, List[float]] = {key: [] for key in truth}
    for task in range(bundle.task_num):
        for domain in range(bundle.domain_num):
            key = task * bundle.domain_num + domain
            mask = domains == domain
            truth[key].extend(labels[mask, task].tolist())
            scores[key].extend(predictions[mask, task].tolist())
    result: Dict[int, Dict[str, float]] = {}
    for key in truth:
        cell_labels, cell_scores = truth[key], scores[key]
        if len(cell_labels) == 0 or len(set(float(item) for item in cell_labels)) < 2:
            raise ValueError("evaluation cell {} is empty or missing one class".format(key))
        result[key] = {
            "sample_count": int(len(cell_labels)),
            "positive_count": int(sum(float(item) for item in cell_labels)),
            "positive_rate": float(np.mean(cell_labels)),
            "auc": _require_finite("auc", safe_auc(cell_labels, cell_scores)),
            "logloss": _require_finite("logloss", safe_logloss(cell_labels, cell_scores)),
        }
    return result, ordered_ids, domains, labels, predictions


def _macro(records: Mapping[int, Mapping[str, float]], field: str) -> float:
    values = [_require_finite(field, row[field]) for row in records.values()]
    if not values:
        raise ValueError("empty metric records")
    return float(np.mean(values))


evaluate_cells = _cell_records


def _target_weighted(records: Mapping[int, Mapping[str, float]], field: str,
                     shares: Mapping[int, float], domain_num: int,
                     task_num: int) -> float:
    total = 0.0
    for task in range(task_num):
        for domain in range(domain_num):
            key = task * domain_num + domain
            total += float(shares[domain]) * _require_finite(field, records[key][field])
    return float(total / float(task_num))


def _worst_domain_auc(records: Mapping[int, Mapping[str, float]], domain_num: int,
                      task_num: int) -> float:
    return float(min(np.mean([records[task * domain_num + domain]["auc"]
                              for task in range(task_num)])
                     for domain in range(domain_num)))


def _cell_loss(records: Mapping[int, Mapping[str, float]], key: int) -> float:
    return _require_finite("cell_logloss", records[key]["logloss"])


def _rows_for_bundle(bundle: DataBundle, records: Mapping[int, Mapping[str, float]],
                     stage: str, epoch: int, train_records: Optional[Mapping[int, Mapping[str, float]]] = None,
                     selected: bool = False, fixed_final: bool = False,
                     train_objective: float = float("nan"), main_steps: int = 0,
                     architecture_steps: int = 0, learning_rate: float = 0.01,
                     architecture_learning_rate: float = 0.001) -> List[Dict[str, object]]:
    shares = bundle.domain_shares
    task_num, domain_num = bundle.task_num, bundle.domain_num
    macro_auc = _macro(records, "auc")
    macro_ll = _macro(records, "logloss")
    target_auc = _target_weighted(records, "auc", shares, domain_num, task_num)
    target_ll = _target_weighted(records, "logloss", shares, domain_num, task_num)
    worst_domain = _worst_domain_auc(records, domain_num, task_num)
    worst_cell = min(row["auc"] for row in records.values())
    rows: List[Dict[str, object]] = []
    train_stats = train_records or {}
    for task in range(task_num):
        for domain in range(domain_num):
            key = task * domain_num + domain
            stats = records[key]
            train = train_stats.get(key, {})
            physical = int(bundle.local_to_physical[domain])
            rows.append({
                "stage": "S1", "run_type": "joint" if bundle.is_joint else "single_domain",
                "dataset": bundle.dataset, "setting": bundle.setting,
                "backbone": "", "seed": bundle.seed, "epoch": int(epoch),
                "domain": domain, "physical_domain": physical,
                "size_role": bundle.role_by_domain.get(domain, ""), "task": task, "cell": key,
                "train_sample_count": train.get("sample_count", ""),
                "validation_sample_count": stats["sample_count"],
                "train_positive_rate": train.get("positive_rate", ""),
                "validation_positive_rate": stats["positive_rate"],
                "train_cell_logloss": train.get("logloss", ""),
                "validation_auc": stats["auc"], "validation_logloss": stats["logloss"],
                "validation_macro_auc": macro_auc,
                "validation_macro_logloss": macro_ll,
                "target_weighted_validation_auc": target_auc,
                "target_weighted_validation_logloss": target_ll,
                "worst_domain_validation_auc": worst_domain,
                "worst_cell_validation_auc": worst_cell,
                "train_objective": train_objective,
                "main_optimizer_steps": int(main_steps),
                "architecture_optimizer_steps": int(architecture_steps),
                "learning_rate": float(learning_rate),
                "architecture_learning_rate": float(architecture_learning_rate),
                "selected_checkpoint": int(selected), "fixed_final": int(fixed_final),
                "common_epoch": int(int(epoch) in COMMON_EPOCHS),
            })
    return rows


class S1Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dataset = canonical_dataset(args.dataset)
        self.setting = str(args.setting)
        self.backbone = str(args.backbone).lower()
        self.seed = int(args.seed)
        self.run_type = str(args.run_type)
        self.physical_domain = None if self.run_type == "joint" else int(args.physical_domain)
        if self.backbone not in BACKBONES:
            raise ValueError("unregistered Stage S1 backbone")
        if self.seed not in SEEDS and not bool(int(getattr(args, "allow_nonformal_seed", 0))):
            raise ValueError("formal Stage S1 seeds are {}".format(SEEDS))
        if self.run_type not in ("joint", "single_domain"):
            raise ValueError("run_type must be joint or single_domain")
        if self.run_type == "single_domain" and self.physical_domain not in (0, 1, 2):
            raise ValueError("single-domain run requires physical_domain 0/1/2")
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.run_dir = os.path.abspath(args.run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        prior_manifest = os.path.join(self.run_dir, "run_manifest.json")
        if os.path.exists(prior_manifest):
            try:
                prior = json.loads(open(prior_manifest, encoding="utf-8").read())
            except Exception as exc:
                raise RuntimeError(
                    "refusing to overwrite a corrupt canonical manifest; archive it first"
                ) from exc
            raise FileExistsError(
                "refusing to overwrite an existing canonical Stage S1 run (status={})".format(
                    prior.get("status")))
        existing = [name for name in os.listdir(self.run_dir)
                    if name not in ("failed_attempts", "stdout.log")]
        if existing:
            raise FileExistsError(
                "refusing to write a non-empty canonical run directory; archive it first")
        self._validate_args()
        setup_seed(self.seed)
        self.bundle_joint = load_stage_s1_bundle(
            self.dataset, args.dataset_path or None, self.setting, self.seed)
        self.bundle = (self.bundle_joint if self.run_type == "joint"
                       else self.bundle_joint.reference_view(self.physical_domain))
        self.train_loader = DataLoader(
            TorchDataset(self.bundle.train_x, self.bundle.train_y),
            batch_size=int(args.batch_size), shuffle=True, num_workers=0,
            drop_last=False)
        self.val_loader = DataLoader(
            TorchDataset(self.bundle.val_x, self.bundle.val_y),
            batch_size=int(args.batch_size), shuffle=False, num_workers=0,
            drop_last=False)
        self.test_loader = DataLoader(
            TorchDataset(self.bundle.test_x, self.bundle.test_y),
            batch_size=int(args.batch_size), shuffle=False, num_workers=0,
            drop_last=False)
        self._preflight_class_coverage()
        self._build_models_and_optimizers()
        self.metric_rows: List[Dict[str, object]] = []
        self.test_rows: List[Dict[str, object]] = []
        self.batch_rows: List[Dict[str, object]] = []
        self.nonfinite_inventory: List[Dict[str, object]] = []
        self.batch_fingerprints: List[str] = []
        self.architecture_batch_fingerprints: List[str] = []
        # Batch trajectory capture remains part of the deterministic training
        # contract in both branches; ``instrumentation_enabled`` controls
        # emission of the human-readable batch trace only.  The acceptance
        # fixture toggles this flag on the exact _train_epoch path.
        self.instrumentation_enabled = True
        self.rng_evidence: List[Dict[str, object]] = []
        self.examples_processed = 0
        self.architecture_examples_processed = 0
        self.optimizer_steps = 0
        self.architecture_optimizer_steps = 0
        self.scheduler_steps = 0
        self.selected_epoch: Optional[int] = None
        self.selected_val_auc = float("-inf")
        self.selected_state: Optional[Dict[str, torch.Tensor]] = None
        # Keep only the selected state and the fixed audit checkpoints.  The
        # prior implementation retained all 50 full snapshots unnecessarily.
        self.checkpoint_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.validation_records: Dict[int, Dict[int, Dict[str, float]]] = {}
        self.validation_summaries: Dict[int, Dict[str, float]] = {}
        self.shadow_stop_epoch: Optional[int] = None
        self.test_prediction_arrays: Dict[str, np.ndarray] = {}
        self.test_labels_array: Optional[np.ndarray] = None
        self.test_row_ids_array: Optional[np.ndarray] = None
        self.test_physical_domains_array: Optional[np.ndarray] = None

    def _validate_args(self) -> None:
        if bool(int(getattr(self.args, "allow_protocol_deviation", 0))):
            raise ValueError("formal Stage S1 does not allow protocol deviation")
        if bool(int(getattr(self.args, "allow_nonformal_seed", 0))):
            raise ValueError("formal Stage S1 does not allow non-formal seeds")
        if int(self.args.epoch) != EPOCHS:
            raise ValueError("Stage S1 requires exactly 50 epochs")
        if int(self.args.batch_size) != int(FORMAL_PROTOCOL["batch_size"]):
            raise ValueError("Stage S1 requires batch_size=40960")
        if float(self.args.learning_rate) != float(FORMAL_PROTOCOL["learning_rate"]):
            raise ValueError("Stage S1 requires learning_rate=0.01")
        if float(self.args.weight_decay) != float(FORMAL_PROTOCOL["weight_decay"]):
            raise ValueError("Stage S1 requires weight_decay=1e-5")
        if float(self.args.learning_rate_darts) != float(FORMAL_PROTOCOL["learning_rate_darts"]):
            raise ValueError("Stage S1 requires architecture learning_rate=0.001")
        if int(self.args.expert_num) != int(FORMAL_PROTOCOL["expert_num"]):
            raise ValueError("Stage S1 requires expert_num=4")
        if tuple(int(x) for x in self.args.fcn_dims) != tuple(FORMAL_PROTOCOL["fcn_dims"]):
            raise ValueError("Stage S1 requires fcn_dims=[512,256,256,64]")
        if int(self.args.shadow_patience) != int(FORMAL_PROTOCOL["shadow_patience"]):
            raise ValueError("Stage S1 requires shadow_patience=10")
        if int(self.args.num_workers) != 0:
            raise ValueError("Stage S1 requires num_workers=0")
        if str(self.args.device) != str(FORMAL_PROTOCOL["device"]):
            raise ValueError("Stage S1 requires device=cuda:0")

    def _preflight_class_coverage(self) -> None:
        for split_name, x_values, y_values in (
                ("validation", self.bundle.val_x, self.bundle.val_y),
                ("test", self.bundle.test_x, self.bundle.test_y)):
            ids = x_values[:, -1].astype(np.int64)
            for task in range(self.bundle.task_num):
                for domain in range(self.bundle.domain_num):
                    labels = y_values[ids == domain, task]
                    if len(labels) == 0 or len(np.unique(labels)) < 2:
                        raise ValueError("{} cell d{}t{} lacks both classes".format(
                            split_name, domain, task))

    def _build_models_and_optimizers(self) -> None:
        model_seed = self.seed
        if self.run_type == "joint":
            self.model = build_model(self.bundle.features, 3, 2, self.backbone,
                                     expert_num=int(self.args.expert_num),
                                     fcn_dims=tuple(self.args.fcn_dims), seed=model_seed)
            self.initialization_audit = {
                "source_scope": "joint", "target_scope": "joint",
                "backbone": self.backbone, "mapping_table": [],
                "common_tensor_hashes": {}, "unmapped_tensors": [],
                "unmapped_tensor_subseeds": {},
            }
        else:
            source_model = build_model(self.bundle_joint.features, 3, 2, self.backbone,
                                       expert_num=int(self.args.expert_num),
                                       fcn_dims=tuple(self.args.fcn_dims), seed=model_seed)
            self.model = build_model(self.bundle.features, 1, 2, self.backbone,
                                     expert_num=int(self.args.expert_num),
                                     fcn_dims=tuple(self.args.fcn_dims), seed=model_seed + 1)
            audit = apply_semantic_initialization(
                source_model, self.model, backbone=self.backbone,
                source_domain_num=3, target_domain_num=1,
                physical_domain=int(self.physical_domain), base_seed=self.seed)
            self.initialization_audit = audit.as_dict()
        self.model = self.model.to(self.device)
        self.architecture_parameters = [p for name, p in self.model.named_parameters()
                                        if is_architecture_parameter(name)]
        self.main_parameters = [p for name, p in self.model.named_parameters()
                                if not is_architecture_parameter(name)]
        if not self.main_parameters:
            raise ValueError("Stage S1 model has no main parameters")
        self.optimizer = torch.optim.Adam(
            self.main_parameters, lr=float(self.args.learning_rate),
            betas=(0.9, 0.999), weight_decay=float(self.args.weight_decay))
        self.arch_optimizer = None
        if self.backbone == "m3oe":
            if not self.architecture_parameters:
                raise ValueError("M3OE must expose architecture parameters")
            self.arch_optimizer = torch.optim.Adam(
                self.architecture_parameters, lr=float(self.args.learning_rate_darts),
                betas=(0.9, 0.999), weight_decay=float(self.args.weight_decay))
        elif self.architecture_parameters:
            raise ValueError("MMoE/PLE may not expose architecture parameters")
        self.initial_parameter_state_hash = initial_state_hash(self.model)

    def _record_batch(self, epoch: int, step: int, kind: str, x_batch: torch.Tensor,
                      y_batch: torch.Tensor, objective: object = "") -> str:
        fingerprint = batch_fingerprint(x_batch, y_batch)
        ids = x_batch[:, -1].detach().cpu().numpy().astype(np.int64)
        domain_counts = [int((ids == d).sum()) for d in range(self.bundle.domain_num)]
        if self.instrumentation_enabled:
            self.batch_rows.append({
                "stage": "S1", "run_type": self.run_type, "dataset": self.dataset,
                "setting": self.setting, "backbone": self.backbone, "seed": self.seed,
                "epoch": int(epoch), "step": int(step), "kind": kind,
                "batch_fingerprint": fingerprint, "batch_size": int(len(y_batch)),
                "domain_counts": _json_counts(domain_counts),
                "task_counts": _json_counts([int(len(y_batch))] * self.bundle.task_num),
                "train_objective": (float(objective.detach().cpu())
                                    if isinstance(objective, torch.Tensor) else objective),
            })
        return fingerprint

    def _train_epoch(self, epoch: int) -> Tuple[float, Dict[int, Dict[str, float]]]:
        self.model.train()
        objective_sum = 0.0
        sample_count = 0
        for step, (x_batch, y_batch) in enumerate(self.train_loader):
            fingerprint = self._record_batch(epoch, step, "train", x_batch, y_batch)
            self.batch_fingerprints.append(fingerprint)
            x_device = x_batch.transpose(0, 1).to(device=self.device, dtype=torch.float32)
            y_device = y_batch.to(device=self.device, dtype=torch.float32)
            pred = self.model(x_device)
            if not bool(torch.isfinite(pred).all().detach().cpu()):
                raise FloatingPointError("non-finite training prediction")
            loss = direct_per_example_bce(pred, y_device)
            _require_finite("train_objective", loss.detach().cpu())
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, param in self.model.named_parameters():
                if name not in {n for n, _ in self.model.named_parameters()
                                if is_architecture_parameter(n)} and param.grad is not None \
                        and not bool(torch.isfinite(param.grad).all().detach().cpu()):
                    raise FloatingPointError("non-finite gradient {}".format(name))
            self.optimizer.step()
            for name, param in self.model.named_parameters():
                if not bool(torch.isfinite(param).all().detach().cpu()):
                    raise FloatingPointError("non-finite parameter {}".format(name))
            self.optimizer_steps += 1
            count = int(len(y_batch))
            self.examples_processed += count
            sample_count += count
            objective_sum += float(loss.detach().cpu()) * count
            if self.instrumentation_enabled:
                self.batch_rows[-1]["train_objective"] = float(loss.detach().cpu())
        train_objective = objective_sum / float(sample_count or 1)
        train_loader = DataLoader(
            TorchDataset(self.bundle.train_x, self.bundle.train_y),
            batch_size=int(self.args.batch_size), shuffle=False, num_workers=0, drop_last=False)
        train_records = _cell_records(self.bundle, self.model, train_loader, self.device)
        return train_objective, train_records

    def _architecture_update(self, architecture_iterator: Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]],
                             epoch: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.arch_optimizer is None:
            return architecture_iterator
        self.model.train()
        rng_before = rng_state_digest()
        with preserve_rng_state():
            try:
                if architecture_iterator is None:
                    architecture_iterator = iter(self.train_loader)
                x_batch, y_batch = next(architecture_iterator)
            except StopIteration:
                architecture_iterator = iter(self.train_loader)
                x_batch, y_batch = next(architecture_iterator)
        rng_after = rng_state_digest()
        self.rng_evidence.append({"epoch": int(epoch), "kind": "architecture_batch",
                                  "before": rng_before, "after": rng_after,
                                  "restored": rng_before == rng_after})
        fingerprint = self._record_batch(epoch, -1, "architecture", x_batch, y_batch)
        self.architecture_batch_fingerprints.append(fingerprint)
        self.architecture_examples_processed += int(len(y_batch))
        x_device = x_batch.transpose(0, 1).to(device=self.device, dtype=torch.float32)
        y_device = y_batch.to(device=self.device, dtype=torch.float32)
        pred = self.model(x_device)
        loss = direct_per_example_bce(pred, y_device)
        self.arch_optimizer.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        for name, param in self.model.named_parameters():
            if is_architecture_parameter(name) and param.grad is not None \
                    and not bool(torch.isfinite(param.grad).all().detach().cpu()):
                raise FloatingPointError("non-finite architecture gradient {}".format(name))
        self.arch_optimizer.step()
        self.architecture_optimizer_steps += 1
        if self.instrumentation_enabled:
            self.batch_rows[-1]["train_objective"] = float(loss.detach().cpu())
        _optimizer_audit(self.arch_optimizer)
        return architecture_iterator

    def fit(self) -> Dict[str, object]:
        best_epoch: Optional[int] = None
        best_score = float("-inf")
        bad_epochs = 0
        architecture_iterator: Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]] = None
        for epoch in range(int(self.args.epoch)):
            train_objective, train_records = self._train_epoch(epoch)
            architecture_iterator = self._architecture_update(architecture_iterator, epoch)
            rng_before = rng_state_digest()
            with preserve_rng_state():
                val_records = _cell_records(self.bundle, self.model, self.val_loader, self.device)
            rng_after = rng_state_digest()
            self.rng_evidence.append({"epoch": int(epoch), "kind": "validation_eval",
                                      "before": rng_before, "after": rng_after,
                                      "restored": rng_before == rng_after})
            val_score = _macro(val_records, "auc")
            if not math.isfinite(val_score):
                raise FloatingPointError("non-finite validation macro AUC")
            improved = val_score > best_score  # strict > implements earliest tie break
            if improved:
                best_score = val_score
                best_epoch = epoch
                self.selected_state = _state_cpu(self.model)
                bad_epochs = 0
            else:
                bad_epochs += 1
            if self.shadow_stop_epoch is None and bad_epochs >= int(self.args.shadow_patience):
                self.shadow_stop_epoch = epoch
            if epoch in COMMON_EPOCHS or epoch == EPOCHS - 1:
                self.checkpoint_states[epoch] = _state_cpu(self.model)
            self.validation_records[epoch] = val_records
            self.validation_summaries[epoch] = {
                "macro_auc": _macro(val_records, "auc"),
                "macro_logloss": _macro(val_records, "logloss"),
            }
            self.metric_rows.extend(_rows_for_bundle(
                self.bundle, val_records, "validation", epoch,
                train_records=train_records, selected=False, fixed_final=epoch == EPOCHS - 1,
                train_objective=train_objective, main_steps=self.optimizer_steps,
                architecture_steps=self.architecture_optimizer_steps,
                learning_rate=float(self.args.learning_rate),
                architecture_learning_rate=float(self.args.learning_rate_darts)))
        if best_epoch is None or self.selected_state is None:
            raise RuntimeError("no validation checkpoint was selected")
        self.selected_epoch = select_checkpoint_epoch(self.validation_summaries and {
            epoch: summary["macro_auc"] for epoch, summary in self.validation_summaries.items()
        } or {best_epoch: best_score})
        # ``best_epoch`` is retained as an explicit audit cross-check; both
        # paths must agree under the strict earliest-tie rule.
        if self.selected_epoch != best_epoch:
            raise AssertionError("checkpoint selection helper disagrees with training selection")
        self.selected_val_auc = best_score
        for row in self.metric_rows:
            row["backbone"] = self.backbone
            if int(row["epoch"]) == best_epoch:
                row["selected_checkpoint"] = 1
        # Test starts only now: selection is frozen and all validation history
        # is already persisted in memory.  No test metric enters a retry or
        # selection path.
        self._evaluate_deferred_checkpoints()
        return {"selected_epoch": best_epoch, "selected_validation_macro_auc": best_score,
                "shadow_stop_epoch": self.shadow_stop_epoch, "fixed_final_epoch": EPOCHS - 1}

    def _evaluate_deferred_checkpoints(self) -> None:
        checkpoints: Dict[str, Tuple[int, Mapping[str, torch.Tensor]]] = {
            "selected": (int(self.selected_epoch), self.selected_state),
            "final": (EPOCHS - 1, self.checkpoint_states[EPOCHS - 1]),
        }
        for epoch in COMMON_EPOCHS:
            checkpoints["common_epoch_{}".format(epoch)] = (epoch, self.checkpoint_states[epoch])
        parameter_count = int(parameter_manifest(self.model)["total"])
        for checkpoint, (stage_epoch, state) in checkpoints.items():
            self.model.load_state_dict(state, strict=True)
            records, row_ids, local_domains, labels, predictions = _prediction_records(
                self.bundle, self.model, self.test_loader, self.device,
                self.bundle.test_row_ids)
            if self.test_labels_array is None:
                self.test_labels_array = labels
                self.test_row_ids_array = row_ids
                self.test_physical_domains_array = np.asarray(
                    [self.bundle.local_to_physical[int(value)] for value in local_domains],
                    dtype=np.int64)
            elif not np.array_equal(self.test_labels_array, labels):
                raise AssertionError("deferred test labels changed between checkpoints")
            self.test_prediction_arrays[checkpoint] = predictions
            validation_records = self.validation_records[int(stage_epoch)]
            validation_summary = self.validation_summaries[int(stage_epoch)]
            for task in range(self.bundle.task_num):
                for domain in range(self.bundle.domain_num):
                    key = task * self.bundle.domain_num + domain
                    row = records[key]
                    physical = int(self.bundle.local_to_physical[domain])
                    self.test_rows.append({
                        "stage": "S1", "run_type": self.run_type, "dataset": self.dataset,
                        "setting": self.setting, "backbone": self.backbone, "seed": self.seed,
                        "checkpoint": checkpoint, "stage_epoch": int(stage_epoch),
                        "training_stage": checkpoint, "domain": domain,
                        "physical_domain": physical,
                        "size_role": self.bundle.role_by_domain.get(domain, ""),
                        "task": task, "cell": key,
                        "train_sample_count": self.bundle.cell_stats()["train"]["d{}t{}".format(domain, task)]["count"],
                        "validation_sample_count": self.bundle.cell_stats()["validation"]["d{}t{}".format(domain, task)]["count"],
                        "test_sample_count": row["sample_count"],
                        "test_positive_rate": row["positive_rate"],
                        "auc": row["auc"], "logloss": row["logloss"],
                        "validation_auc_at_stage": validation_records[key]["auc"],
                        "validation_logloss_at_stage": validation_records[key]["logloss"],
                        "validation_macro_auc_at_stage": validation_summary["macro_auc"],
                        "validation_macro_logloss_at_stage": validation_summary["macro_logloss"],
                        "selected_epoch": int(self.selected_epoch),
                        "selected_validation_macro_auc": float(self.selected_val_auc),
                        "parameter_count": parameter_count,
                        "training_stage_identifier": "selected" if checkpoint == "selected"
                        else ("final_epoch_49" if checkpoint == "final" else checkpoint),
                        "domain_share": self.bundle.domain_shares[domain],
                        "physical_domain_identity": self.bundle.manifest_fields()["physical_domain_identity"][domain],
                        "task_identity": self.bundle.preprocessing_metadata.get("label_definitions", {}).get(
                            "task{}".format(task), "task{}".format(task)),
                        "train_positive_count": self.bundle.cell_stats()["train"]["d{}t{}".format(domain, task)]["positive_count"],
                        "validation_positive_count": self.bundle.cell_stats()["validation"]["d{}t{}".format(domain, task)]["positive_count"],
                        "test_positive_count": row["positive_count"],
                        "train_positive_rate": self.bundle.cell_stats()["train"]["d{}t{}".format(domain, task)]["positive_rate"],
                        "validation_positive_rate": self.bundle.cell_stats()["validation"]["d{}t{}".format(domain, task)]["positive_rate"],
                        "train_label_entropy": self.bundle.cell_stats()["train"]["d{}t{}".format(domain, task)]["entropy"],
                        "validation_label_entropy": self.bundle.cell_stats()["validation"]["d{}t{}".format(domain, task)]["entropy"],
                        "test_label_entropy": self.bundle.cell_stats()["test"]["d{}t{}".format(domain, task)]["entropy"],
                        "reference_scope": REFERENCE_SCOPE if self.run_type == "single_domain" else "joint",
                    })
        # Restore the final in-memory state for the manifest hash.
        self.model.load_state_dict(self.checkpoint_states[EPOCHS - 1], strict=True)

    def _checkpoint_payloads(self) -> Dict[str, Mapping[str, torch.Tensor]]:
        return {
            "selected_checkpoint.pth": self.selected_state,
            "final_checkpoint.pth": self.checkpoint_states[EPOCHS - 1],
            **{"common_epoch_{}.pth".format(epoch): self.checkpoint_states[epoch]
               for epoch in COMMON_EPOCHS},
        }

    def write_outputs(self, fit_info: Mapping[str, object], wall_seconds: float) -> None:
        write_rows_csv(os.path.join(self.run_dir, "metrics_epoch.csv"), self.metric_rows,
                       METRICS_EPOCH_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "metrics_test.csv"), self.test_rows,
                       METRICS_TEST_FIELDS)
        selected_test = [row for row in self.test_rows if row["checkpoint"] == "selected"]
        write_rows_csv(os.path.join(self.run_dir, "metrics_summary.csv"), selected_test,
                       METRICS_TEST_FIELDS)
        write_rows_csv(os.path.join(self.run_dir, "batch_trace.csv"), self.batch_rows,
                       BATCH_FIELDS)
        if self.test_labels_array is None or self.test_row_ids_array is None \
                or self.test_physical_domains_array is None:
            raise RuntimeError("deferred prediction arrays were not collected")
        prediction_payload = {
            "row_ids": self.test_row_ids_array,
            "physical_domain": self.test_physical_domains_array,
            "labels": self.test_labels_array,
        }
        for checkpoint, values in self.test_prediction_arrays.items():
            prediction_payload["pred_{}".format(checkpoint)] = values
        np.savez_compressed(os.path.join(self.run_dir, "predictions_test.npz"),
                            **prediction_payload)
        checkpoints = self._checkpoint_payloads()
        checkpoint_hashes: Dict[str, str] = {}
        for filename, state in checkpoints.items():
            path = os.path.join(self.run_dir, filename)
            torch.save(dict(state), path)
            checkpoint_hashes[filename] = _file_hash(path)
        write_json(os.path.join(self.run_dir, "initialization_audit.json"), self.initialization_audit)
        write_json(os.path.join(self.run_dir, "parameter_manifest.json"), parameter_manifest(self.model))
        write_json(os.path.join(self.run_dir, "nonfinite_inventory.json"), self.nonfinite_inventory)
        torch.save(self.optimizer.state_dict(), os.path.join(self.run_dir, "optimizer_state.pth"))
        if self.arch_optimizer is not None:
            torch.save(self.arch_optimizer.state_dict(),
                       os.path.join(self.run_dir, "architecture_optimizer_state.pth"))
        else:
            write_json(os.path.join(self.run_dir, "architecture_optimizer_absent.json"), {
                "present": False, "reason": "MMoE/PLE have no architecture optimizer by protocol"})
        torch.save(_raw_rng_state(), os.path.join(self.run_dir, "rng_state.pth"))
        with open(os.path.join(self.run_dir, "stdout.log"), "a", encoding="utf-8") as stdout:
            stdout.write("[stage_S1] run outputs written; status=OK\n")
        protocol_values = dict(FORMAL_PROTOCOL)
        protocol_values.update({
            "stage": "S1", "device": str(self.device), "batch_size": int(self.args.batch_size),
            "epoch": int(self.args.epoch), "learning_rate": float(self.args.learning_rate),
            "learning_rate_darts": float(self.args.learning_rate_darts),
            "weight_decay": float(self.args.weight_decay), "expert_num": int(self.args.expert_num),
            "fcn_dims": [int(x) for x in self.args.fcn_dims],
        })
        formal_mismatches = protocol_mismatches(protocol_values)
        runtime = _runtime_metadata(self.device)
        runtime_fingerprint = hashlib.sha256(
            json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        code_hashes = _code_hash_inventory()
        config = vars(self.args).copy()
        config.update({"stage": "S1", "formal_protocol": dict(FORMAL_PROTOCOL),
                       "formal_protocol_observed": protocol_values,
                       "formal_protocol_mismatches": formal_mismatches,
                       "formal_valid": not formal_mismatches,
                       "objective_formula": "mean_i,t BCE(p_it,y_it)",
                       "objective_equivalence_contract": "direct per-example BCE == normalized p_d cell expression",
                       "metric_implementation": {"module": "stage_S1_metrics.py",
                                                   "auc": "tie_aware_average_rank",
                                                   "logloss": "fixed_epsilon_clip",
                                                   "logloss_epsilon": LOGLOSS_EPS},
                       "runtime": runtime,
                       "runtime_fingerprint": runtime_fingerprint,
                       "code_hashes": code_hashes,
                       "source_code_hash": code_hashes["stage_S1_run.py"],
                       "test_evaluated_during_training": False,
                       "test_evaluated_after_selection": True,
                       "thresholds": dict(THRESHOLDS)})
        config["reference_scope"] = REFERENCE_SCOPE
        write_json(os.path.join(self.run_dir, "config.json"), config)
        architecture_artifact = ("architecture_optimizer_state.pth"
                                 if self.arch_optimizer is not None
                                 else "architecture_optimizer_absent.json")
        artifact_names = [
            "config.json", "metrics_epoch.csv", "metrics_test.csv", "metrics_summary.csv",
            "batch_trace.csv", "predictions_test.npz", "initialization_audit.json",
            "parameter_manifest.json", "nonfinite_inventory.json", "optimizer_state.pth",
            architecture_artifact, "rng_state.pth", "stdout.log",
        ] + list(checkpoints)
        artifact_hashes = {name: _file_hash(os.path.join(self.run_dir, name))
                           for name in artifact_names}
        manifest = {
            "stage": "S1", "status": "OK", "run_type": self.run_type,
            "formal_run_key": formal_key(self.run_type, self.dataset, self.setting,
                                          self.backbone, self.seed, self.physical_domain),
            "dataset": self.dataset, "dataset_label": self.bundle.manifest_fields()["dataset"],
            "reference_scope": (REFERENCE_SCOPE if self.run_type == "single_domain" else "joint"),
            "setting": self.setting, "backbone": self.backbone, "seed": self.seed,
            "physical_domain": self.physical_domain,
            "run_dir": self.run_dir, "device_identity": str(self.device),
            "runtime": runtime,
            "runtime_fingerprint": runtime_fingerprint,
            "dataset_identity": self.bundle.preprocessing_metadata.get("raw_file", ""),
            "dataset_file_hash": (_file_hash(self.bundle.preprocessing_metadata.get("raw_file", ""))
                                  if self.bundle.preprocessing_metadata.get("raw_file") else ""),
            "source_code_hash": code_hashes["stage_S1_run.py"],
            "code_hashes": code_hashes,
            **self.bundle.manifest_fields(),
            "formal_protocol": dict(FORMAL_PROTOCOL),
            "formal_protocol_observed": protocol_values,
            "formal_protocol_mismatches": formal_mismatches,
            "formal_valid": not formal_mismatches,
            "objective": "direct_per_example_task_mean_bce",
            "objective_coefficients": "none; normalized p_d equivalence is a test oracle",
            "metric_implementation": {"module": "stage_S1_metrics.py",
                                       "auc": "tie_aware_average_rank",
                                       "logloss": "fixed_epsilon_clip",
                                       "logloss_epsilon": LOGLOSS_EPS},
            "optimizer": {"name": "Adam", "lr": float(self.args.learning_rate),
                          "betas": [0.9, 0.999], "weight_decay": float(self.args.weight_decay),
                          "scheduler": None, "audit": _optimizer_audit(self.optimizer)},
            "architecture_optimizer": (None if self.arch_optimizer is None else {
                "name": "Adam", "lr": float(self.args.learning_rate_darts),
                "betas": [0.9, 0.999], "weight_decay": float(self.args.weight_decay),
                "steps_per_epoch": 1, "audit": _optimizer_audit(self.arch_optimizer)}),
            "model_parameter_manifest": parameter_manifest(self.model),
            "forward_flops_estimate": estimate_forward_flops(self.model),
            "initial_parameter_state_hash": self.initial_parameter_state_hash,
            "final_parameter_state_hash": initial_state_hash(self.model),
            "semantic_initialization_audit": "initialization_audit.json",
            "examples_processed": int(self.examples_processed),
            "architecture_examples_processed": int(self.architecture_examples_processed),
            "optimizer_steps": int(self.optimizer_steps),
            "architecture_optimizer_steps": int(self.architecture_optimizer_steps),
            "observed_epochs": list(range(EPOCHS)),
            "batches_per_epoch": int(len(self.train_loader)),
            "train_batches_total": int(len(self.batch_fingerprints)),
            "scheduler_steps": int(self.scheduler_steps),
            "batch_order_hash": hashlib.sha256(json.dumps(self.batch_fingerprints).encode()).hexdigest(),
            "batch_fingerprint_count": len(self.batch_fingerprints),
            "architecture_batch_order_hash": hashlib.sha256(
                json.dumps(self.architecture_batch_fingerprints).encode()).hexdigest(),
            "architecture_batch_fingerprint_count": len(self.architecture_batch_fingerprints),
            "rng_state_digest_after_run": rng_state_digest(),
            "rng_restore_evidence": self.rng_evidence,
            "selected_epoch": int(self.selected_epoch),
            "selected_validation_macro_auc": float(self.selected_val_auc),
            "shadow_stop_epoch": self.shadow_stop_epoch,
            "fixed_final_epoch": EPOCHS - 1,
            "common_epochs": list(COMMON_EPOCHS),
            "checkpoints": checkpoint_hashes,
            "artifact_hashes": artifact_hashes,
            "optimizer_artifact": "optimizer_state.pth",
            "architecture_optimizer_artifact": architecture_artifact,
            "prediction_artifact": "predictions_test.npz",
            "prediction_row_ids_hash": hashlib.sha256(
                np.asarray(self.test_row_ids_array, dtype=np.int64).tobytes()).hexdigest(),
            "prediction_labels_hash": hashlib.sha256(
                np.asarray(self.test_labels_array, dtype=np.float32).tobytes()).hexdigest(),
            "prediction_row_ids_array_hash": array_hash(
                np.asarray(self.test_row_ids_array, dtype=np.int64)),
            "prediction_labels_array_hash": array_hash(
                np.asarray(self.test_labels_array, dtype=np.float32)),
            "prediction_checkpoint_keys": sorted(self.test_prediction_arrays),
            "checkpoint_selection_frozen_before_test": True,
            "test_evaluated_during_training": False,
            "test_evaluation_order": ["selection_frozen", "selected", "final"]
            + ["common_epoch_{}".format(e) for e in COMMON_EPOCHS],
            "test_metric_in_epoch_history": False,
            "nonfinite_inventory": "nonfinite_inventory.json",
            "wall_seconds": float(wall_seconds),
            "peak_memory_bytes": (int(torch.cuda.max_memory_allocated(self.device))
                                  if self.device.type == "cuda" else 0),
            "artifacts": ["config.json", "run_manifest.json"] + artifact_names,
            "scope_note": "raw Stage S1 observation; no scientific conclusion",
        }
        # ``DataBundle.manifest_fields`` intentionally exposes the human label
        # under ``dataset`` for report tables.  The run manifest identity is a
        # machine key and must remain the same canonical value used by the
        # formal matrix and validator.  Keep the display label separately.
        manifest["dataset"] = self.dataset
        manifest["dataset_label"] = self.bundle.manifest_fields().get(
            "dataset", manifest.get("dataset_label", self.dataset))
        manifest["dataset_key"] = self.dataset
        write_json(os.path.join(self.run_dir, "run_manifest.json"), manifest)


def run_instrumentation_equivalence_fixture(device: Optional[torch.device] = None) -> Dict[str, object]:
    """Run one identical real ``S1Runner._train_epoch`` path with logging on/off.

    This is a non-research acceptance fixture.  It uses a tiny in-memory
    three-domain/two-task batch, but invokes the same batch recorder,
    objective, optimizer step, cell metrics, and batch-fingerprint path as a
    formal runner.  The OFF branch suppresses only emitted ``batch_rows``;
    trajectory capture remains available for equality auditing.
    """

    from types import SimpleNamespace

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    sample_count = 12
    features = np.zeros((sample_count, 3), dtype=np.float32)
    features[:, 0] = np.arange(sample_count, dtype=np.float32) / float(sample_count)
    features[:, 1] = np.arange(sample_count, dtype=np.float32) % 3
    features[:, 2] = np.repeat(np.arange(3, dtype=np.float32), 4)
    labels = np.zeros((sample_count, 2), dtype=np.float32)
    labels[:, 0] = np.arange(sample_count) % 2
    labels[:, 1] = (np.arange(sample_count) + 1) % 2
    bundle = SimpleNamespace(domain_num=3, task_num=2,
                             train_x=features, train_y=labels)

    class _FixtureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, values):
            # Stage S1 models consume [feature, batch] tensors.
            return torch.sigmoid(self.linear(values.transpose(0, 1)))

    setup_seed(731)
    base_model = _FixtureModel()
    base_state = {name: value.detach().clone() for name, value in base_model.state_dict().items()}

    def make_runner(enabled: bool) -> S1Runner:
        setup_seed(913)
        runner = S1Runner.__new__(S1Runner)
        runner.args = SimpleNamespace(batch_size=4)
        runner.dataset = "synthetic"
        runner.setting = "instrumentation_fixture"
        runner.backbone = "mmoe"
        runner.seed = 2022
        runner.run_type = "joint"
        runner.bundle = bundle
        runner.device = device
        runner.model = _FixtureModel().to(device)
        runner.model.load_state_dict(base_state, strict=True)
        runner.optimizer = torch.optim.Adam(runner.model.parameters(), lr=0.01)
        runner.train_loader = DataLoader(
            TorchDataset(bundle.train_x, bundle.train_y), batch_size=4,
            shuffle=False, num_workers=0, drop_last=False)
        runner.instrumentation_enabled = bool(enabled)
        runner.batch_rows = []
        runner.batch_fingerprints = []
        runner.architecture_batch_fingerprints = []
        runner.optimizer_steps = 0
        runner.examples_processed = 0
        runner.architecture_optimizer_steps = 0
        runner.architecture_examples_processed = 0
        runner.scheduler_steps = 0
        runner.nonfinite_inventory = []
        return runner

    def run_branch(enabled: bool) -> Dict[str, object]:
        runner = make_runner(enabled)
        train_objective, records = runner._train_epoch(0)
        torch_rng = torch.get_rng_state().clone()
        cuda_rng = ([value.clone() for value in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_available() else [])
        numpy_rng = np.random.get_state()
        python_rng = random.getstate()
        return {"runner": runner, "objective": float(train_objective),
                "records": records, "torch_rng": torch_rng, "cuda_rng": cuda_rng,
                "numpy_rng": numpy_rng, "python_rng": python_rng,
                "parameter_state": _state_cpu(runner.model),
                "optimizer_state": runner.optimizer.state_dict()}

    on = run_branch(True)
    off = run_branch(False)
    on_runner, off_runner = on["runner"], off["runner"]
    if len(on_runner.batch_rows) != len(on_runner.batch_fingerprints) \
            or not on_runner.batch_rows or off_runner.batch_rows:
        raise AssertionError("instrumentation ON/OFF batch-row emission contract failed")
    if on_runner.batch_fingerprints != off_runner.batch_fingerprints:
        raise AssertionError("instrumentation changed batch trajectory")
    if on_runner.optimizer_steps != off_runner.optimizer_steps \
            or on_runner.examples_processed != off_runner.examples_processed:
        raise AssertionError("instrumentation changed optimizer/exposure counters")
    if abs(float(on["objective"]) - float(off["objective"])) > 1e-12:
        raise AssertionError("instrumentation changed train objective")
    if _state_hash(on["parameter_state"]) != _state_hash(off["parameter_state"]):
        raise AssertionError("instrumentation changed parameters")

    def equal_nested(left: object, right: object) -> bool:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) \
                and torch.equal(left, right)
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            return isinstance(left, Mapping) and isinstance(right, Mapping) \
                and set(left) == set(right) \
                and all(equal_nested(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            return isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) \
                and len(left) == len(right) \
                and all(equal_nested(a, b) for a, b in zip(left, right))
        return left == right

    if not equal_nested(on["optimizer_state"], off["optimizer_state"]):
        raise AssertionError("instrumentation changed optimizer state")
    for key in on["records"]:
        for field in on["records"][key]:
            if abs(float(on["records"][key][field]) - float(off["records"][key][field])) > 1e-12:
                raise AssertionError("instrumentation changed cell metric {}".format(field))
    if not torch.equal(on["torch_rng"], off["torch_rng"]) \
            or len(on["cuda_rng"]) != len(off["cuda_rng"]) \
            or any(not torch.equal(a, b) for a, b in zip(on["cuda_rng"], off["cuda_rng"])):
        raise AssertionError("instrumentation changed torch RNG")
    if on["numpy_rng"][0] != off["numpy_rng"][0] \
            or not np.array_equal(on["numpy_rng"][1], off["numpy_rng"][1]) \
            or on["numpy_rng"][2:] != off["numpy_rng"][2:]:
        raise AssertionError("instrumentation changed NumPy RNG")
    if on["python_rng"] != off["python_rng"]:
        raise AssertionError("instrumentation changed Python RNG")
    return {"parameter_equal": True, "optimizer_equal": True, "metrics_equal": True,
            "rng_equal": True, "batch_trajectory_equal": True,
            "on_log_rows": len(on_runner.batch_rows), "off_log_rows": len(off_runner.batch_rows)}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage S1 one-run raw trainer")
    parser.add_argument("--run_type", choices=("joint", "single_domain"), required=True)
    parser.add_argument("--dataset", choices=("movielens", "kuairand"), required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--physical_domain", type=int, default=None)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--batch_size", type=int, default=int(FORMAL_PROTOCOL["batch_size"]))
    parser.add_argument("--epoch", type=int, default=EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=float(FORMAL_PROTOCOL["learning_rate"]))
    parser.add_argument("--learning_rate_darts", type=float,
                        default=float(FORMAL_PROTOCOL["learning_rate_darts"]))
    parser.add_argument("--weight_decay", type=float, default=float(FORMAL_PROTOCOL["weight_decay"]))
    parser.add_argument("--expert_num", type=int, default=int(FORMAL_PROTOCOL["expert_num"]))
    parser.add_argument("--fcn_dims", nargs=4, type=int, default=[512, 256, 256, 64])
    parser.add_argument("--shadow_patience", type=int, default=int(FORMAL_PROTOCOL["shadow_patience"]))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=str(FORMAL_PROTOCOL["device"]))
    parser.add_argument("--allow_protocol_deviation", type=int, choices=(0, 1), default=0)
    parser.add_argument("--allow_nonformal_seed", type=int, choices=(0, 1), default=0)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    started = time.time()
    os.makedirs(args.run_dir, exist_ok=True)
    try:
        runner = S1Runner(args)
        fit_info = runner.fit()
        runner.write_outputs(fit_info, time.time() - started)
        print("[stage_S1] completed", args.run_type, args.dataset, args.setting,
              args.backbone, args.seed, flush=True)
        return 0
    except Exception as exc:
        # A failed attempt is preserved; the orchestrator decides whether and
        # where a retry may be launched.
        prior_manifest = os.path.join(args.run_dir, "run_manifest.json")
        if not os.path.exists(prior_manifest):
            write_json(prior_manifest, {
                "stage": "S1", "status": "FAILED", "run_type": args.run_type,
                "dataset": args.dataset, "setting": args.setting, "backbone": args.backbone,
                "seed": int(args.seed), "physical_domain": args.physical_domain,
                "exception_type": type(exc).__name__, "exception": str(exc),
                "wall_seconds": time.time() - started,
            })
        raise


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "direct_per_example_bce", "normalized_cell_bce", "objective_equivalence",
    "run_instrumentation_equivalence_fixture",
    "sample_objective", "direct_sample_bce", "normalized_sample_objective", "evaluate_cells",
    "select_checkpoint_epoch", "S1Runner", "build_argparser", "main", "METRICS_EPOCH_FIELDS",
    "METRICS_TEST_FIELDS", "BATCH_FIELDS",
]
