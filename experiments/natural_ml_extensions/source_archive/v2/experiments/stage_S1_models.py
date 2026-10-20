"""Audited Stage S1 model constructors and semantic initialization pairing.

MMoE and PLE use the repository's FuxiCTR-compatible layer math from P7,
with the prohibited trainable dummy architecture parameter removed.  M3OE is
the existing MDMTRec implementation.  This module does not contain any
optimizer, curriculum, momentum, or Safe-MTL logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
M3OE_ROOT = os.path.join(ROOT, "m3oe")
P7_ROOT = os.path.join(HERE, "history_exp", "first")
for _path in (M3OE_ROOT, P7_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from MDMTRec.basic.features import DenseFeature, SparseFeature  # noqa: E402
from MDMTRec.basic.layers import EmbeddingLayer  # noqa: E402
from MDMTRec.models.multi_domain import MDMTRec  # noqa: E402
from p7_models import CGC_Layer, MLPBlock, MMoE_Layer  # noqa: E402

from stage_S1_data import FeatureSchema  # noqa: E402


def features_from_schema(schema: FeatureSchema) -> List[object]:
    """Build repository feature descriptors with stable integer column names."""

    descriptors: List[object] = []
    for column_index, column in enumerate(schema.columns):
        if column in schema.dense_columns:
            descriptors.append(DenseFeature(column_index))
        elif column in schema.sparse_columns:
            descriptors.append(SparseFeature(
                column_index, vocab_size=int(schema.vocab_sizes[column]),
                embed_dim=int(schema.embedding_dim)))
        else:
            raise ValueError("column {} is neither dense nor sparse".format(column))
    return descriptors


class StageS1Backbone(nn.Module):
    """MMoE/PLE wrapper with the exact D*T cell ordering used by MDMTRec."""

    def __init__(self, features: Sequence[object], domain_num: int, task_num: int,
                 backbone: str, expert_num: int = 4,
                 fcn_dims: Sequence[int] = (512, 256, 256, 64)):
        super().__init__()
        backbone = str(backbone).lower()
        if backbone not in ("mmoe", "ple"):
            raise ValueError("StageS1Backbone supports only mmoe/ple")
        self.features = list(features)
        self.input_dim = sum(int(fea.embed_dim) for fea in features)
        self.domain_num = int(domain_num)
        self.task_num = int(task_num)
        self.backbone_name = backbone
        self.expert_num = int(expert_num)
        self.fcn_dims = tuple(int(x) for x in fcn_dims)
        self.embedding = EmbeddingLayer(self.features)
        n_task = self.domain_num * self.task_num
        # Stage S1 fixes the FuxiCTR core at expert [256,64], gate [64],
        # followed by one [64 -> LayerNorm/ReLU -> 1] tower per cell.  The
        # M3OE fcn_dims list contains an additional 256-wide internal layer;
        # it is intentionally not copied into this P7 core.
        expert_hidden = [self.fcn_dims[1], self.fcn_dims[3]]
        gate_hidden = [self.fcn_dims[3]]
        if backbone == "mmoe":
            self.core = MMoE_Layer(
                num_experts=self.expert_num, num_tasks=n_task,
                input_dim=self.input_dim, expert_hidden_units=expert_hidden,
                gate_hidden_units=gate_hidden)
        else:
            self.core = CGC_Layer(
                num_shared_experts=self.expert_num, num_specific_experts=1,
                num_tasks=n_task, input_dim=self.input_dim,
                expert_hidden_units=expert_hidden, gate_hidden_units=gate_hidden)
        tower_in = self.fcn_dims[3]
        self.tower = nn.ModuleList([
            nn.Sequential(nn.Linear(tower_in, tower_in), nn.LayerNorm(tower_in),
                          nn.ReLU(), nn.Linear(tower_in, 1))
            for _ in range(n_task)
        ])

    def forward(self, x: torch.Tensor, test_flag: bool = False) -> torch.Tensor:
        domain_id = x[-1, :].long().detach()
        input_emb = self.embedding(x, self.features, squeeze_dim=True)
        n_task = self.domain_num * self.task_num
        if self.backbone_name == "mmoe":
            outputs = self.core(input_emb)
        else:
            outputs = self.core([input_emb] * n_task + [input_emb])
        cell_predictions = [torch.sigmoid(self.tower[i](outputs[i]).squeeze(1))
                            for i in range(n_task)]
        stacked = torch.stack(cell_predictions, dim=1)
        result = torch.zeros((stacked.shape[0], self.task_num), device=stacked.device,
                             dtype=stacked.dtype)
        for task in range(self.task_num):
            for domain in range(self.domain_num):
                cell = task * self.domain_num + domain
                result[:, task] = torch.where(
                    domain_id == domain, stacked[:, cell], result[:, task])
        return result

    def shared_param_names(self) -> List[str]:
        """Return names that have a semantic counterpart across D=3/D=1."""

        names: List[str] = []
        for name, _ in self.named_parameters():
            if name.startswith("tower."):
                continue
            if self.backbone_name == "ple" and (
                    name.startswith("core.specific_experts.") or ".gate." in name):
                continue
            if name.startswith("core.gate."):
                continue
            names.append(name)
        return names


class StageS1M3OEAdapter(MDMTRec):
    """Device-safe Stage S1 adapter for the legacy M3OE layer math.

    The historical MDMTRec forward assembles its final outputs with CPU
    ``torch.zeros``.  Stage S1 inherits its parameters and operations but
    reproduces only that forward assembly with allocations on ``x.device``.
    """

    def forward(self, x: torch.Tensor, test_flag: bool = False) -> torch.Tensor:
        device = x.device
        domain_id = x[-1, :].clone().detach().long()
        masks = [(domain_id == d) for d in range(self.domain_num)]
        input_emb = self.embedding(x, self.features, squeeze_dim=True)
        skip = self.skip_conn(input_emb)
        emb = torch.zeros((input_emb.shape[0], self.star_dim[1]),
                          device=device, dtype=input_emb.dtype)
        for index, (weight, bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            output = torch.matmul(input_emb, torch.multiply(weight, self.shared_weight)) \
                + bias + self.shared_bias
            emb = torch.where(masks[index].unsqueeze(1), output, emb)
        emb = self.star_mlp(emb) + skip
        gate_value = [self.gate[i](emb.detach()).unsqueeze(1)
                      for i in range(self.task_num * self.domain_num)]
        shared_outputs = [self.expert[i](emb) for i in range(self.expert_num)]
        domain_outputs = [self.domain_expert[i](emb) for i in range(self.domain_num)]
        task_outputs = [self.task_expert[i](emb) for i in range(self.task_num)]
        expert_features = torch.cat([value.unsqueeze(1) for value in shared_outputs], dim=1)
        domain_features = torch.cat([value.unsqueeze(1) for value in domain_outputs], dim=1)
        task_features = torch.cat([value.unsqueeze(1) for value in task_outputs], dim=1)
        weighted_domain = []
        for index in range(self.domain_num):
            value = self._weight_bal_d() * domain_features[:, index, :]
            for other in range(self.domain_num):
                if other != index:
                    value = value + (1 - self._weight_bal_d()) / (self.domain_num - 1) * domain_features[:, other, :]
            weighted_domain.append(value)
        weighted_task = []
        for index in range(self.task_num):
            value = self._weight_bal_t() * task_features[:, index, :]
            for other in range(self.task_num):
                if other != index:
                    value = value + (1 - self._weight_bal_t()) / (self.task_num - 1) * task_features[:, other, :]
            weighted_task.append(value)
        fused = [
            torch.bmm(gate_value[index], expert_features).squeeze(1)
            + self._weight_exp_d() * weighted_domain[index % self.domain_num]
            + self._weight_exp_t() * weighted_task[index // self.domain_num]
            for index in range(self.task_num * self.domain_num)
        ]
        results = [torch.sigmoid(self.tower[index](fused[index]).squeeze(1))
                   for index in range(self.task_num * self.domain_num)]
        cell_output = torch.stack(results, dim=1)
        result = torch.zeros((cell_output.shape[0], self.task_num),
                             device=device, dtype=cell_output.dtype)
        for task in range(self.task_num):
            for domain in range(self.domain_num):
                cell = task * self.domain_num + domain
                result[:, task] = torch.where(
                    masks[domain], cell_output[:, cell], result[:, task])
        return result


def build_model(schema: FeatureSchema, domain_num: int, task_num: int,
                backbone: str, *, expert_num: int = 4,
                fcn_dims: Sequence[int] = (512, 256, 256, 64),
                seed: Optional[int] = None) -> nn.Module:
    """Construct one of the three fixed Stage S1 backbones."""

    if seed is not None:
        torch.manual_seed(int(seed))
    features = features_from_schema(schema)
    key = str(backbone).lower()
    if key == "m3oe":
        return StageS1M3OEAdapter(
            features, domain_num=int(domain_num), task_num=int(task_num),
            fcn_dims=list(fcn_dims), expert_num=int(expert_num),
            exp_d=1, exp_t=1, bal_d=1, bal_t=1).to(dtype=torch.float32)
    if key in ("mmoe", "ple"):
        return StageS1Backbone(
            features, domain_num=int(domain_num), task_num=int(task_num),
            backbone=key, expert_num=int(expert_num), fcn_dims=fcn_dims).to(dtype=torch.float32)
    raise ValueError("unknown Stage S1 backbone: {}".format(backbone))


def is_architecture_parameter(name: str) -> bool:
    return str(name).startswith("_weight_") or str(name).startswith("architecture.")


def parameter_manifest(model: nn.Module) -> Dict[str, object]:
    named = list(model.named_parameters())
    names = [name for name, _ in named]
    shapes = {name: list(param.shape) for name, param in named}
    counts = {name: int(param.numel()) for name, param in named}
    architecture = [name for name in names if is_architecture_parameter(name)]
    main = [name for name in names if name not in architecture]
    shared: List[str] = []
    private: List[str] = []
    if hasattr(model, "shared_param_names"):
        shared = list(getattr(model, "shared_param_names")())
        shared_set = set(shared)
        private = [name for name in main if name not in shared_set]
    else:
        # M3OE's existing E-stage partition is kept explicit here.  This is
        # reporting only; it does not alter optimization behavior.
        for name in main:
            if name.startswith(("embedding.", "shared_weight", "shared_bias",
                                "skip_conn.", "star_mlp.", "expert.")):
                shared.append(name)
            else:
                private.append(name)
    return {
        "names": names,
        "shapes": shapes,
        "numel": counts,
        "total": int(sum(counts.values())),
        "architecture_names": architecture,
        "main_names": main,
        "shared_names": sorted(shared),
        "private_names": sorted(private),
        "shared_count": int(sum(counts.get(name, 0) for name in shared)),
        "private_count": int(sum(counts.get(name, 0) for name in private)),
        "architecture_count": int(sum(counts.get(name, 0) for name in architecture)),
    }


def estimate_forward_flops(model: nn.Module) -> int:
    """A transparent linear-layer multiply-add estimate for logging."""

    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += 2 * int(module.in_features) * int(module.out_features)
    return int(total)


def _tensor_hash(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(tuple(arr.shape)).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _stable_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256((str(int(base_seed)) + "|" + name).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31 - 1)


def _cell_index_from_name(name: str, source_domain_num: int,
                          target_domain_num: int, physical_domain: int) -> Optional[str]:
    """Map a target cell-indexed parameter to its joint source name."""

    # M3OE and StageS1Backbone both use task-major cell indices.
    match = re.search(r"\.(\d+)(?:\.|$)", name)
    if not match:
        return None
    index = int(match.group(1))
    if target_domain_num != 1:
        return None
    task = index // target_domain_num
    local_domain = index % target_domain_num
    if local_domain != 0:
        return None
    return str(task * source_domain_num + int(physical_domain))


def _semantic_source_name(target_name: str, source_state: Mapping[str, torch.Tensor],
                          target_domain_num: int, source_domain_num: int,
                          physical_domain: int, backbone: str,
                          target_n_task: int, source_n_task: int) -> Optional[str]:
    """Resolve one target tensor to its semantically paired joint tensor."""

    # Local domain-specific M3OE parameters.
    for prefix in ("slot_weight.", "slot_bias.", "domain_expert."):
        if target_name.startswith(prefix + "0"):
            candidate = prefix + str(int(physical_domain)) + target_name[len(prefix) + 1:]
            if candidate in source_state:
                return candidate
    # PLE's trailing gate is shared across cells and has a shifted index.
    if backbone == "ple" and target_name.startswith("core.gate."):
        match = re.match(r"core\.gate\.(\d+)(\..+)$", target_name)
        if match and int(match.group(1)) == target_n_task:
            candidate = "core.gate.{}{}".format(source_n_task, match.group(2))
            if candidate in source_state:
                return candidate
    # Cell-specific gates/towers and PLE specific experts.
    if target_name.startswith("tower.") or target_name.startswith("gate.") \
            or target_name.startswith("core.gate.") \
            or target_name.startswith("core.specific_experts."):
        mapped = _cell_index_from_name(target_name, source_domain_num, target_domain_num,
                                       physical_domain)
        if mapped is not None:
            candidate = re.sub(r"\.(\d+)(?=\.|$)", ".{}".format(mapped), target_name, count=1)
            if candidate in source_state:
                return candidate
    # Embedding table names are integer feature indices in the repository.
    if target_name.startswith("embedding.embed_dict.") and target_name.endswith(".weight"):
        return target_name if target_name in source_state else None
    if target_name in source_state:
        return target_name
    return None


@dataclass
class SemanticInitializationAudit:
    source_scope: str
    target_scope: str
    backbone: str
    physical_domain: Optional[int]
    mapping_table: List[Dict[str, object]]
    common_tensor_hashes: Dict[str, object]
    unmapped_tensors: List[str]
    unmapped_tensor_subseeds: Dict[str, int]
    source_parameter_manifest: Dict[str, object]
    target_parameter_manifest: Dict[str, object]

    def as_dict(self) -> Dict[str, object]:
        return {
            "source_scope": self.source_scope,
            "target_scope": self.target_scope,
            "backbone": self.backbone,
            "physical_domain": self.physical_domain,
            "mapping_table": self.mapping_table,
            "common_tensor_hashes": self.common_tensor_hashes,
            "unmapped_tensors": self.unmapped_tensors,
            "unmapped_tensor_subseeds": self.unmapped_tensor_subseeds,
            "source_parameter_manifest": self.source_parameter_manifest,
            "target_parameter_manifest": self.target_parameter_manifest,
        }


def _fill_unmapped_parameter(param: torch.Tensor, seed: int) -> None:
    # Reinitialize one tensor inside a forked RNG so the name-derived seed is
    # deterministic and does not perturb construction/training RNG streams.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        with torch.no_grad():
            if param.ndim >= 2:
                nn.init.xavier_uniform_(param)
            elif param.ndim == 1:
                nn.init.uniform_(param, -0.02, 0.02)
            else:
                param.zero_()


def apply_semantic_initialization(
    source_model: nn.Module,
    target_model: nn.Module,
    *, backbone: str,
    source_domain_num: int = 3,
    target_domain_num: int = 1,
    physical_domain: int = 0,
    base_seed: int = 2022,
    source_scope: str = "joint",
    target_scope: str = "single_domain_mtl_reference",
    domain_feature_index: Optional[int] = None,
) -> SemanticInitializationAudit:
    """Copy semantic joint tensors into one D=1 reference and audit every map.

    Exact-shape tensors are copied by semantic name.  Embedding tables with a
    different row count are copied row-wise; the local routing row is mapped
    from the requested physical-domain row.  Any tensor with no counterpart is
    explicitly reinitialized from a name-derived sub-seed and never silently
    treated as paired.
    """

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    source_named = dict(source_model.named_parameters())
    target_named = dict(target_model.named_parameters())
    source_n_task = int(source_domain_num) * 2
    target_n_task = int(target_domain_num) * 2
    if domain_feature_index is None:
        # The repository schema always puts the routing feature last.
        domain_feature_index = max(
            [int(key.split(".")[2]) for key in source_state
             if key.startswith("embedding.embed_dict.") and key.endswith(".weight")] or [0])

    mapping: List[Dict[str, object]] = []
    common_hashes: Dict[str, object] = {}
    unmapped: List[str] = []
    subseeds: Dict[str, int] = {}
    patched: Dict[str, torch.Tensor] = {}

    for target_name, target_value in target_state.items():
        source_name = _semantic_source_name(
            target_name, source_state, target_domain_num, source_domain_num,
            int(physical_domain), str(backbone).lower(), target_n_task, source_n_task)
        source_value = source_state.get(source_name) if source_name else None
        target_copy = target_value.detach().clone()
        mapping_kind = "none"
        if source_value is not None and source_value.shape == target_value.shape:
            target_copy.copy_(source_value)
            mapping_kind = "exact"
        elif source_value is not None and target_name.startswith("embedding.embed_dict.") \
                and target_name.endswith(".weight") and source_value.ndim == target_value.ndim == 2 \
                and source_value.shape[1] == target_value.shape[1]:
            if target_name == "embedding.embed_dict.{}.weight".format(domain_feature_index):
                source_row = int(physical_domain)
                if source_row >= source_value.shape[0]:
                    source_row = 0
                target_copy[0] = source_value[source_row]
                # Other rows are deterministic but unused by a D=1 view.
                if target_copy.shape[0] > 1:
                    for row in range(1, target_copy.shape[0]):
                        _fill_unmapped_parameter(target_copy[row],
                                                 _stable_seed(base_seed, target_name + str(row)))
            else:
                rows = min(int(source_value.shape[0]), int(target_value.shape[0]))
                target_copy[:rows] = source_value[:rows]
            mapping_kind = "row_subset"
        else:
            # A target tensor may have an exact-name source with an incompatible
            # shape (e.g. output gates).  Make the absence explicit.
            source_name = None

        source_slice: Optional[torch.Tensor] = None
        if source_name is not None and source_value is not None:
            if mapping_kind == "row_subset" and target_name.startswith("embedding.embed_dict."):
                if target_name == "embedding.embed_dict.{}.weight".format(domain_feature_index):
                    source_row = min(int(physical_domain), int(source_value.shape[0]) - 1)
                    source_slice = source_value[source_row:source_row + 1]
                else:
                    source_slice = source_value[:int(target_value.shape[0])]
            elif source_value.shape == target_value.shape:
                source_slice = source_value
        if source_name is None:
            subseed = _stable_seed(base_seed, target_name)
            _fill_unmapped_parameter(target_copy, subseed)
            subseeds[target_name] = subseed
            unmapped.append(target_name)
            mapping_kind = "name_derived_subseed"
        source_hash = _tensor_hash(source_value) if source_value is not None else ""
        source_slice_hash = _tensor_hash(source_slice) if source_slice is not None else ""
        target_hash = _tensor_hash(target_copy)
        target_slice = target_copy
        if mapping_kind == "row_subset" and target_name.startswith("embedding.embed_dict."):
            target_slice = target_copy[:1] if target_name == \
                "embedding.embed_dict.{}.weight".format(domain_feature_index) \
                else target_copy[:int(source_value.shape[0])]
        target_slice_hash = _tensor_hash(target_slice)
        if source_name is not None:
            common_hashes[target_name] = {
                "source": source_name,
                "source_hash": source_hash,
                "source_slice_hash": source_slice_hash,
                "target_hash": target_hash,
                "target_slice_hash": target_slice_hash,
                "hash_equal": bool(source_slice_hash and source_slice_hash == target_slice_hash),
            }
        patched[target_name] = target_copy
        mapping.append({
            "target": target_name,
            "source": source_name or "",
            "kind": mapping_kind,
            "target_shape": list(target_value.shape),
            "source_shape": list(source_value.shape) if source_value is not None else [],
            "source_hash": source_hash,
            "source_slice_hash": source_slice_hash,
            "target_hash": target_hash,
            "target_slice_hash": target_slice_hash,
            "hash_equal": bool(source_name is not None and source_slice_hash == target_slice_hash),
        })

    target_model.load_state_dict(patched, strict=True)
    return SemanticInitializationAudit(
        source_scope=source_scope, target_scope=target_scope,
        backbone=str(backbone).lower(), physical_domain=int(physical_domain),
        mapping_table=mapping, common_tensor_hashes=common_hashes,
        unmapped_tensors=unmapped, unmapped_tensor_subseeds=subseeds,
        source_parameter_manifest=parameter_manifest(source_model),
        target_parameter_manifest=parameter_manifest(target_model),
    )


def initial_state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(_tensor_hash(param).encode("ascii"))
    return h.hexdigest()


def semantic_initialize(*args, **kwargs) -> SemanticInitializationAudit:
    """Compatibility alias for the explicit semantic pairing routine."""
    return apply_semantic_initialization(*args, **kwargs)


def build_m3oe(schema: FeatureSchema, domain_num: int = 3, task_num: int = 2, **kwargs) -> nn.Module:
    return build_model(schema, domain_num, task_num, "m3oe", **kwargs)


def build_mmoe(schema: FeatureSchema, domain_num: int = 3, task_num: int = 2, **kwargs) -> nn.Module:
    return build_model(schema, domain_num, task_num, "mmoe", **kwargs)


def build_ple(schema: FeatureSchema, domain_num: int = 3, task_num: int = 2, **kwargs) -> nn.Module:
    return build_model(schema, domain_num, task_num, "ple", **kwargs)


__all__ = [
    "StageS1Backbone", "StageS1M3OEAdapter", "features_from_schema", "build_model", "parameter_manifest",
    "estimate_forward_flops", "is_architecture_parameter", "SemanticInitializationAudit",
    "apply_semantic_initialization", "initial_state_hash",
    "semantic_initialize", "build_m3oe", "build_mmoe", "build_ple",
]
