"""Exact Stage S3 parameter partition and gradient-access intervention."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from stage_S2_objectives import compute_equal_coefficients, validate_coefficient_invariants

GROUP_SHARED = "C"
GROUP_PRIVATE = "P"
GROUP_ARCH = "A_arch"
_SHARED_PREFIXES = (
    "embedding.", "skip_conn.", "star_mlp.", "expert.", "domain_expert.", "task_expert.",
)
_PRIVATE_PREFIXES = ("slot_weight.", "slot_bias.", "gate.", "tower.")
_ARCH_NAMES = {
    "_weight_exp_d.deep_weights", "_weight_exp_t.deep_weights",
    "_weight_bal_d.deep_weights", "_weight_bal_t.deep_weights",
}
_EXPECTED_COUNTS = {
    "kuairand": {
        GROUP_SHARED: (80, 744_848), GROUP_PRIVATE: (54, 869_406), GROUP_ARCH: (4, 4),
    },
    "movielens": {
        GROUP_SHARED: (53, 566_080), GROUP_PRIVATE: (54, 207_390), GROUP_ARCH: (4, 4),
    },
}


def is_architecture_parameter(name: str) -> bool:
    return str(name) in _ARCH_NAMES


def classify_parameter(name: str) -> str:
    name = str(name)
    if name in _ARCH_NAMES:
        return GROUP_ARCH
    if name.startswith("_weight_") or name.startswith("architecture."):
        raise ValueError("unapproved architecture parameter: {}".format(name))
    if name in ("shared_weight", "shared_bias") or any(name.startswith(prefix) for prefix in _SHARED_PREFIXES):
        return GROUP_SHARED
    if any(name.startswith(prefix) for prefix in _PRIVATE_PREFIXES):
        return GROUP_PRIVATE
    raise ValueError("unclassified Stage S3 parameter: {}".format(name))


def partition_parameters(model: torch.nn.Module) -> Dict[str, object]:
    groups = {GROUP_SHARED: [], GROUP_PRIVATE: [], GROUP_ARCH: []}
    counts = {key: {"tensor_count": 0, "parameter_count": 0} for key in groups}
    seen = set()
    for name, parameter in model.named_parameters():
        if name in seen:
            raise ValueError("duplicate parameter name in Stage S3 partition: {}".format(name))
        seen.add(name)
        group = classify_parameter(name)
        groups[group].append(name)
        counts[group]["tensor_count"] += 1
        counts[group]["parameter_count"] += int(parameter.numel())
    if set(groups[GROUP_SHARED]) & set(groups[GROUP_PRIVATE]) \
            or set(groups[GROUP_SHARED]) & set(groups[GROUP_ARCH]) \
            or set(groups[GROUP_PRIVATE]) & set(groups[GROUP_ARCH]):
        raise AssertionError("Stage S3 partition groups overlap")
    digest_payload = {group: list(names) for group, names in groups.items()}
    digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "groups": groups,
        "counts": counts,
        "partition_hash": digest,
        "parameter_count": sum(value["parameter_count"] for value in counts.values()),
        "tensor_count": sum(value["tensor_count"] for value in counts.values()),
        "name_order": [name for names in groups.values() for name in names],
    }


def assert_expected_parameter_counts(partition: Mapping[str, object], dataset: str) -> None:
    key = str(dataset).lower()
    if key not in _EXPECTED_COUNTS:
        raise ValueError("no audited S3 parameter counts for dataset {}".format(dataset))
    counts = partition.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("partition count table is missing")
    for group, expected in _EXPECTED_COUNTS[key].items():
        observed = counts.get(group)
        if not isinstance(observed, Mapping):
            raise ValueError("partition group is missing: {}".format(group))
        actual = (int(observed.get("tensor_count", -1)),
                  int(observed.get("parameter_count", -1)))
        if actual != expected:
            raise ValueError("S3 parameter count mismatch for {} {}: {} != {}".format(
                key, group, actual, expected))
    if int(partition.get("parameter_count", -1)) != sum(
            value[1] for value in _EXPECTED_COUNTS[key].values()):
        raise ValueError("S3 total parameter count mismatch")


def loss_components(prediction: torch.Tensor, target: torch.Tensor,
                    domain_ids: torch.Tensor, target_domain: int,
                    objective: str, coefficient_table: Mapping[str, object]) -> Dict[str, torch.Tensor]:
    if prediction.ndim != 2 or target.shape != prediction.shape or prediction.shape[1] != 2:
        raise ValueError("Stage S3 loss requires matching [B,2] prediction and target tensors")
    if domain_ids.ndim != 1 or int(domain_ids.numel()) != int(prediction.shape[0]):
        raise ValueError("Stage S3 domain IDs must have shape [B]")
    if prediction.shape[0] == 0 or not bool(torch.isfinite(prediction).all().detach().cpu()) \
            or not bool(torch.isfinite(target).all().detach().cpu()):
        raise FloatingPointError("empty or non-finite Stage S3 objective input")
    if not bool(((domain_ids >= 0) & (domain_ids < 3)).all().detach().cpu()):
        raise ValueError("Stage S3 domain ID outside D=3")
    target_domain = int(target_domain)
    if target_domain not in (0, 1, 2):
        raise ValueError("Stage S3 target domain outside D=3")
    if not bool((domain_ids == target_domain).any().detach().cpu()):
        raise ValueError("target domain is absent from Stage S3 batch")
    objective = str(objective).lower()
    if objective not in ("equal", "sample"):
        raise ValueError("Stage S3 objective must be Equal or Sample")
    if objective == "equal":
        validate_coefficient_invariants(coefficient_table)
        weight_map = coefficient_table.get("loss_coefficient_w")
        if not isinstance(weight_map, Mapping):
            raise ValueError("Equal coefficient table has no loss weights")
        weights = torch.as_tensor([float(weight_map[str(d)]) for d in range(3)],
                                  dtype=prediction.dtype, device=prediction.device)
    else:
        # Sample's accepted S1/S2 definition is the unweighted per-example/task mean.
        weights = torch.ones(3, dtype=prediction.dtype, device=prediction.device)
    per_example = F.binary_cross_entropy(prediction, target.to(dtype=prediction.dtype),
                                         reduction="none").mean(dim=1)
    sample_weight = weights.index_select(0, domain_ids.to(device=prediction.device, dtype=torch.long))
    full = (sample_weight * per_example).mean()
    target_mask = (domain_ids.to(device=prediction.device) == target_domain).to(prediction.dtype)
    target_loss = (sample_weight * per_example * target_mask).mean()
    non_target = full - target_loss
    residual = full - (target_loss + non_target)
    if not all(bool(torch.isfinite(value).all().detach().cpu())
               for value in (full, target_loss, non_target, residual)):
        raise FloatingPointError("non-finite Stage S3 loss component")
    return {"full": full, "target": target_loss, "non_target": non_target,
            "equality_residual": residual}


def _norm(values: Iterable[torch.Tensor | None]) -> float:
    squares = []
    for value in values:
        if value is not None:
            if not bool(torch.isfinite(value).all().detach().cpu()):
                raise FloatingPointError("non-finite Stage S3 gradient")
            squares.append(float(torch.sum(value.detach().double() ** 2).cpu()))
    return math.sqrt(sum(squares)) if squares else 0.0


def _gradient_list(loss: torch.Tensor, params: Sequence[torch.nn.Parameter], *, retain_graph: bool):
    if not params:
        return ()
    return torch.autograd.grad(loss, list(params), retain_graph=retain_graph,
                               create_graph=False, allow_unused=True)


def _install(parameters: Mapping[str, torch.nn.Parameter], names: Sequence[str],
             gradients: Sequence[torch.Tensor | None]) -> None:
    if len(names) != len(gradients):
        raise AssertionError("gradient/name cardinality mismatch")
    for name, grad in zip(names, gradients):
        parameters[name].grad = None if grad is None else grad.detach().clone()


def apply_main_gradients(full_loss: torch.Tensor, target_loss: torch.Tensor,
                         parameters: Mapping[str, torch.nn.Parameter],
                         partition: Mapping[str, object], *, access: str) -> Dict[str, float]:
    groups = partition.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError("Stage S3 partition groups are missing")
    shared = list(groups[GROUP_SHARED])
    private = list(groups[GROUP_PRIVATE])
    if set(shared + private) & set(groups[GROUP_ARCH]):
        raise ValueError("architecture tensors entered the main optimizer partition")
    if any(name not in parameters for name in shared + private):
        raise ValueError("partition refers to absent model parameters")
    access = str(access)
    if access not in ("FULL", "TARGET_ONLY_SHARED"):
        raise ValueError("unknown Stage S3 shared-gradient access: {}".format(access))
    all_main = [parameters[name] for name in shared + private]
    if access == "FULL":
        grads = _gradient_list(full_loss, all_main, retain_graph=True)
        shared_grads = grads[:len(shared)]
        private_grads = grads[len(shared):]
        removed_norm = 0.0
    else:
        shared_grads = _gradient_list(target_loss, [parameters[name] for name in shared],
                                      retain_graph=True)
        private_grads = _gradient_list(full_loss, [parameters[name] for name in private],
                                       retain_graph=True)
        non_target = full_loss - target_loss
        removed = _gradient_list(non_target, [parameters[name] for name in shared],
                                 retain_graph=True)
        removed_norm = _norm(removed)
    _install(parameters, shared, shared_grads)
    _install(parameters, private, private_grads)
    return {
        "applied_shared_gradient_norm": _norm(shared_grads),
        "applied_private_gradient_norm": _norm(private_grads),
        "removed_non_target_shared_gradient_norm": removed_norm,
        "gradient_substitution_valid": 1.0,
    }


def apply_architecture_gradients(full_loss: torch.Tensor, target_loss: torch.Tensor,
                                 parameters: Mapping[str, torch.nn.Parameter],
                                 partition: Mapping[str, object], *, access: str) -> Dict[str, float]:
    groups = partition.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError("Stage S3 partition groups are missing")
    names = list(groups[GROUP_ARCH])
    if any(name not in parameters for name in names):
        raise ValueError("partition refers to absent architecture parameters")
    access = str(access)
    if access == "FULL":
        gradients = _gradient_list(full_loss, [parameters[name] for name in names],
                                   retain_graph=True)
        removed_norm = 0.0
    elif access == "TARGET_ONLY_SHARED":
        gradients = _gradient_list(target_loss, [parameters[name] for name in names],
                                   retain_graph=True)
        removed = _gradient_list(full_loss - target_loss,
                                 [parameters[name] for name in names], retain_graph=True)
        removed_norm = _norm(removed)
    else:
        raise ValueError("unknown Stage S3 architecture-gradient access")
    _install(parameters, names, gradients)
    return {"applied_architecture_gradient_norm": _norm(gradients),
            "removed_non_target_architecture_gradient_norm": removed_norm,
            "gradient_substitution_valid": 1.0}


def _private_owner_domain(name: str, domain_num: int) -> int:
    pieces = str(name).split(".")
    if len(pieces) < 2:
        raise ValueError("private parameter has no owner index: {}".format(name))
    try:
        index = int(pieces[1])
    except ValueError as exc:
        raise ValueError("private parameter has invalid owner index: {}".format(name)) from exc
    if pieces[0] in ("slot_weight", "slot_bias"):
        if not 0 <= index < domain_num:
            raise ValueError("slot parameter owner outside D=3")
        return index
    if pieces[0] in ("gate", "tower"):
        if index < 0 or index >= 2 * domain_num:
            raise ValueError("cell parameter owner outside T*D=6")
        return index % domain_num
    raise ValueError("unrecognized private parameter owner: {}".format(name))


def reachability_audit(model: torch.nn.Module, *, device: torch.device,
                       domain_num: int = 3) -> Dict[str, object]:
    domain_num = int(domain_num)
    if domain_num != 3:
        raise ValueError("Stage S3 reachability audit is fixed at D=3")
    partition = partition_parameters(model)
    parameters = dict(model.named_parameters())
    features = getattr(model, "features", None)
    if features is None:
        raise ValueError("real M3OE model does not expose feature descriptors")
    batch = torch.zeros((1, len(features) + 1), dtype=torch.float32, device=device)
    names = partition["groups"]
    reached = {group: {name: [] for name in names[group]} for group in names}
    for domain in range(domain_num):
        batch.fill_(0)
        batch[0, -1] = float(domain)
        output = model(batch.transpose(0, 1))
        loss = F.binary_cross_entropy(output, torch.ones_like(output), reduction="mean")
        grad_names = list(names[GROUP_SHARED]) + list(names[GROUP_PRIVATE]) + list(names[GROUP_ARCH])
        grads = _gradient_list(loss, [parameters[name] for name in grad_names], retain_graph=False)
        for name, grad in zip(grad_names, grads):
            if grad is not None and bool(torch.count_nonzero(grad).detach().cpu() > 0):
                group = classify_parameter(name)
                reached[group][name].append(domain)
    missing_cross = [name for group in (GROUP_SHARED, GROUP_ARCH)
                     for name, domains in reached[group].items()
                     if len(set(domains)) < 2]
    private_mismatch = []
    for name, domains in reached[GROUP_PRIVATE].items():
        owner = _private_owner_domain(name, domain_num)
        if set(domains) - {owner}:
            private_mismatch.append({"name": name, "owner": owner, "observed": domains})
        if owner not in domains:
            private_mismatch.append({"name": name, "owner": owner, "observed": domains,
                                     "reason": "owner_not_reached"})
    if missing_cross or private_mismatch:
        raise RuntimeError("Stage S3 reachability audit failed: shared={}, private={}".format(
            missing_cross[:8], private_mismatch[:8]))
    return {
        "passed": True, "domain_num": domain_num,
        "reached_domains_by_group": {
            group: {name: sorted(set(domains)) for name, domains in table.items()}
            for group, table in reached.items()},
        "partition_hash": partition["partition_hash"],
        "parameter_count": partition["parameter_count"],
        "architecture_count": len(names[GROUP_ARCH]),
        "cross_domain_shared_count": len(names[GROUP_SHARED]),
        "private_count": len(names[GROUP_PRIVATE]),
    }


def target_parameter_sets(model: torch.nn.Module) -> Dict[str, object]:
    partition = partition_parameters(model)
    parameters = dict(model.named_parameters())
    return {"partition": partition, "parameters": parameters,
            "shared": {name: parameters[name] for name in partition["groups"][GROUP_SHARED]},
            "private": {name: parameters[name] for name in partition["groups"][GROUP_PRIVATE]},
            "architecture": {name: parameters[name] for name in partition["groups"][GROUP_ARCH]}}
