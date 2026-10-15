"""Stage F0 scalable M3OE, MMoE and terminal CGC-1 implementations."""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _path in (HERE, os.path.join(ROOT, "m3oe"),
              os.path.join(HERE, "history_exp", "first")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from MDMTRec.basic.layers import EmbeddingLayer  # noqa: E402
from p7_models import MLPBlock, MMoE_Layer  # noqa: E402
from stage_S1_data import FeatureSchema  # noqa: E402
from stage_S1_models import features_from_schema  # noqa: E402
from stage_F0_protocol import (  # noqa: E402
    BACKBONES,
    P_ML,
    Q,
    FROZEN_WIDTHS,
    scope_instance,
    width_for,
)


class MLP_N(nn.Module):
    """Legacy M3OE MLP_N module layout and Linear/LayerNorm/ReLU math."""

    def __init__(self, fcn_dim: Sequence[int]):
        super().__init__()
        self.fcn_dim = tuple(int(v) for v in fcn_dim)
        self.n = len(self.fcn_dim)
        self.domain_specific = nn.ModuleList()
        for left, right in zip(self.fcn_dim[:-1], self.fcn_dim[1:]):
            self.domain_specific.extend((nn.Linear(left, right), nn.LayerNorm(right), nn.ReLU()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x
        for layer in self.domain_specific:
            output = layer(output)
        return output


class PureSigmoidWeight(nn.Module):
    """State-compatible trainable scalar with side-effect-free sigmoid forward."""

    def __init__(self):
        super().__init__()
        self.deep_weights = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.deep_weights)


class F0M3OE(nn.Module):
    """F0 M3OE with normalized D/T mixing and no mutable tau state."""

    def __init__(self, features: Sequence[object], domain_num: int,
                 task_num: int, c: int, expert_num: int = 4):
        super().__init__()
        self.features = list(features)
        self.input_dim = sum(int(fea.embed_dim) for fea in self.features)
        self.domain_num = int(domain_num)
        self.task_num = int(task_num)
        self.expert_num = int(expert_num)
        if self.domain_num < 1 or self.task_num < 1:
            raise ValueError("D and T must be positive")
        if self.expert_num != 4:
            raise ValueError("F0 fixes M3OE shared expert count to four")
        self.c = int(c)
        self.a = 8 * self.c
        self.b = 4 * self.c
        self.star_dim = (self.input_dim, self.a, self.b)
        self.fcn_dim = (self.b, self.c)
        self.embedding = EmbeddingLayer(self.features)

        self._weight_exp_d = PureSigmoidWeight()
        self._weight_exp_t = PureSigmoidWeight()
        if self.domain_num > 1:
            self._weight_bal_d = PureSigmoidWeight()
        if self.task_num > 1:
            self._weight_bal_t = PureSigmoidWeight()

        self.skip_conn = MLP_N((self.input_dim, self.b))
        self.shared_weight = nn.Parameter(torch.empty(self.input_dim, self.a))
        self.shared_bias = nn.Parameter(torch.zeros(self.a))
        self.slot_weight = nn.ParameterList(
            [nn.Parameter(torch.empty(self.input_dim, self.a)) for _ in range(self.domain_num)])
        self.slot_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.a)) for _ in range(self.domain_num)])
        self.star_mlp = MLP_N((self.a, self.b))
        nn.init.xavier_uniform_(self.shared_weight)
        for weight in self.slot_weight:
            nn.init.xavier_uniform_(weight)

        self.expert = nn.ModuleList([MLP_N((self.b, self.c)) for _ in range(self.expert_num)])
        self.domain_expert = nn.ModuleList([MLP_N((self.b, self.c))
                                             for _ in range(self.domain_num)])
        self.task_expert = nn.ModuleList([MLP_N((self.b, self.c))
                                          for _ in range(self.task_num)])
        # Cell order is task-major: k = t*D+d, matching Stage S1.
        self.gate = nn.ModuleList([
            nn.Sequential(nn.Linear(self.b, self.expert_num), nn.Softmax(dim=1))
            for _ in range(self.domain_num * self.task_num)
        ])
        self.tower = nn.ModuleList([
            nn.Sequential(nn.Linear(self.c, self.c), nn.LayerNorm(self.c), nn.ReLU(),
                          nn.Linear(self.c, 1))
            for _ in range(self.domain_num * self.task_num)
        ])

    def architecture_named_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith(("_weight_exp_", "_weight_bal_")):
                yield name, parameter

    def forward(self, x: torch.Tensor, test_flag: bool = False) -> torch.Tensor:
        device = x.device
        # The repository EmbeddingLayer consumes feature-major [F, batch].
        domain_id = x[-1, :].detach().long()
        if torch.any((domain_id < 0) | (domain_id >= self.domain_num)):
            raise ValueError("F0 routing id is outside the local scope")
        input_emb = self.embedding(x, self.features, squeeze_dim=True)
        skip = self.skip_conn(input_emb)
        embedded = torch.zeros((input_emb.shape[0], self.a), dtype=input_emb.dtype,
                               device=device)
        for domain, (slot_weight, slot_bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            value = torch.matmul(input_emb, slot_weight * self.shared_weight) \
                + slot_bias + self.shared_bias
            embedded = torch.where((domain_id == domain).unsqueeze(1), value, embedded)
        embedded = self.star_mlp(embedded) + skip

        # Keep the legacy detach on gate input while making all scalar reads pure.
        gates = [self.gate[k](embedded.detach()) for k in range(self.domain_num * self.task_num)]
        shared = torch.stack([expert(embedded) for expert in self.expert], dim=1)
        by_domain = torch.stack([expert(embedded) for expert in self.domain_expert], dim=1)
        by_task = torch.stack([expert(embedded) for expert in self.task_expert], dim=1)

        weighted_domain: List[torch.Tensor] = []
        if self.domain_num == 1:
            weighted_domain = [by_domain[:, 0, :]]
        else:
            beta_d = self._weight_bal_d()
            for d in range(self.domain_num):
                weighted_domain.append(beta_d * by_domain[:, d, :] +
                                       (1.0 - beta_d) / float(self.domain_num - 1) *
                                       (by_domain.sum(dim=1) - by_domain[:, d, :]))
        weighted_task: List[torch.Tensor] = []
        if self.task_num == 1:
            weighted_task = [by_task[:, 0, :]]
        else:
            beta_t = self._weight_bal_t()
            for t in range(self.task_num):
                weighted_task.append(beta_t * by_task[:, t, :] +
                                     (1.0 - beta_t) / float(self.task_num - 1) *
                                     (by_task.sum(dim=1) - by_task[:, t, :]))

        alpha_d = self._weight_exp_d()
        alpha_t = self._weight_exp_t()
        cell_hidden = []
        for task in range(self.task_num):
            for domain in range(self.domain_num):
                k = task * self.domain_num + domain
                shared_cell = torch.sum(gates[k].unsqueeze(-1) * shared, dim=1)
                cell_hidden.append(shared_cell + alpha_d * weighted_domain[domain]
                                   + alpha_t * weighted_task[task])
        cell_prob = torch.stack([
            torch.sigmoid(self.tower[k](cell_hidden[k]).squeeze(1))
            for k in range(self.domain_num * self.task_num)
        ], dim=1)
        result = torch.zeros((x.shape[1], self.task_num), device=device, dtype=cell_prob.dtype)
        for task in range(self.task_num):
            for domain in range(self.domain_num):
                k = task * self.domain_num + domain
                result[:, task] = torch.where(domain_id == domain, cell_prob[:, k], result[:, task])
        return result


class F0MMoECore(nn.Module):
    """MMoE layer with the repository's P7/FuxiCTR MLP math."""

    def __init__(self, num_cells: int, input_dim: int, b: int, c: int):
        super().__init__()
        self.core = MMoE_Layer(num_experts=4, num_tasks=num_cells, input_dim=input_dim,
                               expert_hidden_units=[b, c], gate_hidden_units=[c])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.core(x)


class F0CGC1Layer(nn.Module):
    """One CGC layer exposing only the K cell gates consumed by towers."""

    def __init__(self, num_cells: int, input_dim: int, b: int, c: int):
        super().__init__()
        self.num_shared_experts = 4
        self.num_specific_experts = 1
        self.num_tasks = int(num_cells)
        self.shared_experts = nn.ModuleList([MLPBlock(input_dim, [b, c]) for _ in range(4)])
        self.specific_experts = nn.ModuleList([
            nn.ModuleList([MLPBlock(input_dim, [b, c])]) for _ in range(num_cells)
        ])
        self.gate = nn.ModuleList([
            MLPBlock(input_dim, [c], output_dim=5) for _ in range(num_cells)
        ])
        self.gate_activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        shared = [expert(x) for expert in self.shared_experts]
        outputs: List[torch.Tensor] = []
        for cell in range(self.num_tasks):
            specific = self.specific_experts[cell][0](x)
            candidates = torch.stack([specific] + shared, dim=1)
            gate = self.gate_activation(self.gate[cell](x))
            outputs.append(torch.sum(gate.unsqueeze(-1) * candidates, dim=1))
        return outputs


class F0MultiExpert(nn.Module):
    """Shared wrapper for MMoE and the terminal cell-output CGC-1."""

    def __init__(self, features: Sequence[object], domain_num: int,
                 task_num: int, c: int, backbone: str):
        super().__init__()
        self.features = list(features)
        self.input_dim = sum(int(fea.embed_dim) for fea in self.features)
        self.domain_num = int(domain_num)
        self.task_num = int(task_num)
        self.c = int(c)
        self.b = 4 * self.c
        self.backbone_name = str(backbone)
        self.embedding = EmbeddingLayer(self.features)
        num_cells = self.domain_num * self.task_num
        if self.backbone_name == "mmoe":
            self.core = F0MMoECore(num_cells, self.input_dim, self.b, self.c).core
        elif self.backbone_name == "cgc1":
            self.core = F0CGC1Layer(num_cells, self.input_dim, self.b, self.c)
        else:
            raise ValueError("F0MultiExpert supports mmoe/cgc1")
        self.tower = nn.ModuleList([
            nn.Sequential(nn.Linear(self.c, self.c), nn.LayerNorm(self.c), nn.ReLU(),
                          nn.Linear(self.c, 1)) for _ in range(num_cells)
        ])

    def forward(self, x: torch.Tensor, test_flag: bool = False) -> torch.Tensor:
        domain_id = x[-1, :].detach().long()
        if torch.any((domain_id < 0) | (domain_id >= self.domain_num)):
            raise ValueError("F0 routing id is outside the local scope")
        input_emb = self.embedding(x, self.features, squeeze_dim=True)
        if self.backbone_name == "mmoe":
            cell_hidden = self.core(input_emb)
        else:
            cell_hidden = self.core(input_emb)
        cell_prob = torch.stack([
            torch.sigmoid(self.tower[k](cell_hidden[k]).squeeze(1))
            for k in range(self.domain_num * self.task_num)
        ], dim=1)
        result = torch.zeros((x.shape[1], self.task_num), device=x.device, dtype=cell_prob.dtype)
        for task in range(self.task_num):
            for domain in range(self.domain_num):
                k = task * self.domain_num + domain
                result[:, task] = torch.where(domain_id == domain, cell_prob[:, k], result[:, task])
        return result


def build_f0_model(schema: FeatureSchema, domain_num: int, task_num: int,
                   backbone: str, c: int, seed: Optional[int] = None) -> nn.Module:
    model_name = str(backbone).lower()
    if model_name not in BACKBONES:
        raise ValueError("unknown F0 model: {}".format(backbone))
    if seed is None:
        if model_name == "m3oe":
            return F0M3OE(features_from_schema(schema), domain_num, task_num, c)
        return F0MultiExpert(features_from_schema(schema), domain_num, task_num, c, model_name)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if model_name == "m3oe":
            return F0M3OE(features_from_schema(schema), domain_num, task_num, c)
        return F0MultiExpert(features_from_schema(schema), domain_num, task_num, c, model_name)


def m3oe_parameter_formula(domain_num: int, task_num: int, c: int, p: int = P_ML) -> int:
    d, t, c = int(domain_num), int(task_num), int(c)
    b, a, k = 4 * c, 8 * c, d * t
    return ((d + 1) * (8 * p * c + 8 * c) + (4 * p * c + 12 * c) +
            (32 * c * c + 12 * c) + (4 + d + t) * (4 * c * c + 3 * c) +
            k * (16 * c + 4) + k * (c * c + 4 * c + 1) +
            2 + int(d > 1) + int(t > 1))


def mmoe_parameter_formula(domain_num: int, task_num: int, c: int, p: int = P_ML) -> int:
    d, t, c = int(domain_num), int(task_num), int(c)
    b, k = 4 * c, d * t
    h = p * b + 3 * b + b * c + 3 * c
    v = c * c + 4 * c + 1
    return 4 * h + k * (p * c + 3 * c + 4 * c + 4) + k * v


def cgc1_parameter_formula(domain_num: int, task_num: int, c: int, p: int = P_ML) -> int:
    d, t, c = int(domain_num), int(task_num), int(c)
    b, k = 4 * c, d * t
    h = p * b + 3 * b + b * c + 3 * c
    v = c * c + 4 * c + 1
    return (4 + k) * h + k * (p * c + 3 * c + 5 * c + 5) + k * v


def non_embedding_parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for name, parameter in model.named_parameters()
               if not name.startswith("embedding."))


def parameter_count_manifest(model: nn.Module) -> Dict[str, object]:
    parameters = [(name, parameter) for name, parameter in model.named_parameters()]
    embedding = {name: int(parameter.numel()) for name, parameter in parameters
                 if name.startswith("embedding.")}
    non_embedding = {name: int(parameter.numel()) for name, parameter in parameters
                     if not name.startswith("embedding.")}
    return {"all_parameter_count": sum(v for _, v in [(n, int(p.numel())) for n, p in parameters]),
            "embedding_parameter_count": sum(embedding.values()),
            "non_embedding_parameter_count": sum(non_embedding.values()),
            "embedding": embedding, "non_embedding": non_embedding}


def expected_parameter_count(backbone: str, domain_num: int, task_num: int,
                             c: int, p: int = P_ML) -> int:
    key = str(backbone).lower()
    if key == "m3oe":
        return m3oe_parameter_formula(domain_num, task_num, c, p)
    if key == "mmoe":
        return mmoe_parameter_formula(domain_num, task_num, c, p)
    if key == "cgc1":
        return cgc1_parameter_formula(domain_num, task_num, c, p)
    raise ValueError("unknown F0 model: {}".format(backbone))


def validate_frozen_width(backbone: str, scope: str, model: nn.Module,
                          q: int = Q, tolerance: float = 0.05) -> Dict[str, object]:
    d = 3 if str(scope).lower() == "mdmt" else 1
    t = 2
    width = width_for(backbone, scope)
    observed = non_embedding_parameter_count(model)
    expected = expected_parameter_count(backbone, d, t, width["c"], P_ML)
    if observed != expected:
        raise AssertionError("{} {} non-embedding count {} != formula {}".format(
            backbone, scope, observed, expected))
    if observed != int(width["non_embedding"]):
        raise AssertionError("{} {} count differs from frozen width table".format(backbone, scope))
    delta = observed / float(q) - 1.0
    if abs(delta) > float(tolerance):
        raise ValueError("frozen width is outside the 5% Q budget: {:.4%}".format(delta))
    return {"backbone": backbone, "scope": scope, "c": width["c"],
            "b": width["b"], "a": width["a"], "expected": expected,
            "observed": observed, "q": int(q), "relative_delta": delta,
            "within_tolerance": abs(delta) <= float(tolerance)}


def _stable_subseed(seed: int, model: str, semantic_name: str,
                    shape: Sequence[int]) -> int:
    payload = "{}|{}|{}|{}|init-v1".format(
        int(seed), str(model), str(semantic_name), tuple(int(v) for v in shape))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31 - 1)


def _target_semantic_source(name: str, backbone: str, physical_domain: Optional[int],
                            source_d: int, target_d: int) -> Tuple[Optional[str], str]:
    """Map one target parameter to an MDMT semantic name plus physical label."""

    model = str(backbone).lower()
    if physical_domain is None:
        return name, name
    domain = int(physical_domain)
    for prefix in ("slot_weight.", "slot_bias.", "domain_expert."):
        marker = prefix + "0"
        if name.startswith(marker):
            suffix = name[len(marker):]
            return prefix + str(domain) + suffix, prefix + "d" + str(domain) + suffix
    if name.startswith("embedding.embed_dict.") and name.endswith(".weight"):
        return name, name

    # Map cell-indexed tensors using task-major k=t*D+d.  CGC's
    # specific-expert list has the cell index before the inner expert index.
    if name.startswith("core.specific_experts."):
        match = re.match(r"^core\.specific_experts\.(\d+)(\..+)$", name)
        if match:
            local_index = int(match.group(1))
            task = local_index // int(target_d)
            local_domain = local_index % int(target_d)
            if local_domain == 0:
                source_index = task * int(source_d) + domain
                source_name = "core.specific_experts." + str(source_index) + match.group(2)
                semantic_name = "core.specific_experts.d{}t{}".format(domain, task) + match.group(2)
                return source_name, semantic_name
    match = re.match(r"^(gate|tower|core\.gate)\.(\d+)(\..*)?$", name)
    if match:
        local_index = int(match.group(2))
        task = local_index // int(target_d)
        local_domain = local_index % int(target_d)
        if local_domain == 0:
            source_index = task * int(source_d) + domain
            target_root = match.group(1) + "."
            suffix = match.group(3) or ""
            source_name = target_root + str(source_index) + suffix
            semantic_name = match.group(1) + ".d{}t{}".format(domain, task) + suffix
            return source_name, semantic_name
    return name, name


def _init_unmatched_tensor(model: nn.Module, name: str, tensor: torch.Tensor,
                           seed: int) -> None:
    """Use the module's default init, LayerNorm constants, or STAR Xavier."""

    modules = dict(model.named_modules())
    parent_name, _, leaf = name.rpartition(".")
    parent = modules.get(parent_name)
    star_tensor = name in ("shared_weight",) or name.startswith("slot_weight.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        with torch.no_grad():
            if star_tensor:
                nn.init.xavier_uniform_(tensor)
            elif isinstance(parent, nn.Linear):
                if leaf == "weight":
                    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
                elif leaf == "bias":
                    fan_in = int(parent.in_features)
                    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(tensor, -bound, bound)
                else:
                    raise ValueError("unexpected Linear parameter {}".format(name))
            elif isinstance(parent, nn.LayerNorm):
                if leaf == "weight":
                    nn.init.ones_(tensor)
                elif leaf == "bias":
                    nn.init.zeros_(tensor)
                else:
                    raise ValueError("unexpected LayerNorm parameter {}".format(name))
            elif isinstance(parent, nn.Embedding):
                nn.init.normal_(tensor, mean=0.0, std=1e-4)
            elif name.startswith("_weight_") and name.endswith("deep_weights"):
                nn.init.ones_(tensor)
            # ParameterList entries are named slot_bias.N, so they do not
            # end with the literal string "bias". Keep this semantic bias at
            # its frozen zero initialization when widths differ.
            elif name.startswith("slot_bias."):
                nn.init.zeros_(tensor)
            elif name.endswith("bias"):
                nn.init.zeros_(tensor)
            else:
                raise ValueError("no frozen initializer for unmatched parameter {}".format(name))


@dataclass
class F0InitializationAudit:
    backbone: str
    source_scope: str
    target_scope: str
    physical_domain: Optional[int]
    matched: List[Dict[str, object]]
    unmatched: List[Dict[str, object]]
    source_parameter_count: int
    target_parameter_count: int

    def as_dict(self) -> Dict[str, object]:
        return {"backbone": self.backbone, "source_scope": self.source_scope,
                "target_scope": self.target_scope, "physical_domain": self.physical_domain,
                "matched": list(self.matched), "unmatched": list(self.unmatched),
                "source_parameter_count": self.source_parameter_count,
                "target_parameter_count": self.target_parameter_count}


def semantic_initialize_f0(source_model: nn.Module, target_model: nn.Module, *,
                           backbone: str, seed: int, physical_domain: Optional[int],
                           source_scope: str = "mdmt", target_scope: str = "mdmt") -> F0InitializationAudit:
    """Copy exact-shape semantic MDMT tensors, seed all other target tensors."""

    source_state = source_model.state_dict()
    source_named = dict(source_model.named_parameters())
    target_named = dict(target_model.named_parameters())
    target_state = target_model.state_dict()
    if not getattr(target_model, "features", None):
        raise ValueError("F0 model is missing its feature schema")
    domain_feature_index = int(target_model.features[-1].name)
    patched = {name: tensor.detach().clone() for name, tensor in target_state.items()}
    matched: List[Dict[str, object]] = []
    unmatched: List[Dict[str, object]] = []

    for name, target_parameter in target_named.items():
        source_name, semantic_name = _target_semantic_source(
            name, backbone, physical_domain, 3 if physical_domain is not None else target_model.domain_num,
            target_model.domain_num)
        source_parameter = source_named.get(source_name or "")
        # For the domain route table, preserve its global rows but make local
        # row 0 semantically equal to the selected physical-domain row.
        if name.startswith("embedding.embed_dict.") and name.endswith(".weight") \
                and source_parameter is not None and source_parameter.shape == target_parameter.shape:
            value = source_parameter.detach().clone()
            if physical_domain is not None and name == "embedding.embed_dict.{}.weight".format(domain_feature_index):
                value[0].copy_(source_parameter[int(physical_domain)])
                semantic_name = "embedding.domain_indicator.physical_d{}".format(physical_domain)
            patched[name].copy_(value)
            matched.append({"target": name, "source": source_name,
                            "semantic_name": semantic_name, "kind": "exact_global_embedding",
                            "target_shape": list(target_parameter.shape)})
            continue
        if source_parameter is not None and tuple(source_parameter.shape) == tuple(target_parameter.shape):
            patched[name].copy_(source_parameter.detach())
            matched.append({"target": name, "source": source_name,
                            "semantic_name": semantic_name, "kind": "exact_shape",
                            "target_shape": list(target_parameter.shape)})
        else:
            subseed = _stable_subseed(seed, backbone, semantic_name, target_parameter.shape)
            _init_unmatched_tensor(target_model, name, patched[name], subseed)
            unmatched.append({"target": name, "source_candidate": source_name or "",
                              "semantic_name": semantic_name,
                              "target_shape": list(target_parameter.shape),
                              "source_shape": (list(source_parameter.shape)
                                               if source_parameter is not None else []),
                              "subseed": subseed})

    # Buffers are copied only when exact name/shape and semantic meaning agree.
    for name, target_value in target_state.items():
        if name in target_named:
            continue
        source_value = source_state.get(name)
        if source_value is not None and source_value.shape == target_value.shape:
            patched[name].copy_(source_value.detach())
    target_model.load_state_dict(patched, strict=True)
    return F0InitializationAudit(
        backbone=str(backbone), source_scope=str(source_scope), target_scope=str(target_scope),
        physical_domain=None if physical_domain is None else int(physical_domain),
        matched=matched, unmatched=unmatched,
        source_parameter_count=sum(int(p.numel()) for p in source_named.values()),
        target_parameter_count=sum(int(p.numel()) for p in target_named.values()))


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def build_semantically_paired_model(schema: FeatureSchema, *, backbone: str,
                                    domain_num: int, task_num: int, c: int,
                                    seed: int, physical_domain: Optional[int] = None,
                                    model_seed: Optional[int] = None) -> Tuple[nn.Module, Dict[str, object]]:
    """Create an MDMT source and return the requested scope model with audit."""

    init_seed = int(seed if model_seed is None else model_seed)
    source_c = int(width_for(backbone, "mdmt")["c"])
    source_root_seed = _stable_subseed(init_seed, backbone, "mdmt-root", (source_c,))
    source = build_f0_model(schema, 3, 2, backbone, source_c, seed=source_root_seed)
    source_hash = module_state_sha256(source)
    if physical_domain is None:
        return source, {"source_scope": "mdmt", "target_scope": "mdmt",
                        "matched": [], "unmatched": [], "initial_source_only": True,
                        "source_root_seed": source_root_seed,
                        "source_state_sha256": source_hash,
                        "target_state_sha256": source_hash,
                        "local_embedding_remap": None}
    target = build_f0_model(schema, domain_num, task_num, backbone, c,
                            seed=_stable_subseed(init_seed, backbone,
                                                 "sdmt-d{}-root".format(physical_domain), (c,)))
    audit = semantic_initialize_f0(source, target, backbone=backbone, seed=seed,
                                   physical_domain=physical_domain,
                                   source_scope="mdmt", target_scope="sdmt-d{}".format(physical_domain))
    result = audit.as_dict()
    domain_feature_index = int(target.features[-1].name)
    embedding_name = "embedding.embed_dict.{}.weight".format(domain_feature_index)
    source_embedding = source.state_dict()[embedding_name]
    target_embedding = target.state_dict()[embedding_name]
    result.update({
        "source_root_seed": source_root_seed,
        "source_state_sha256": source_hash,
        "target_state_sha256": module_state_sha256(target),
        "local_embedding_remap": {
            "feature_index": domain_feature_index,
            "physical_domain": int(physical_domain),
            "source_embedding_shape": list(source_embedding.shape),
            "target_embedding_shape": list(target_embedding.shape),
            "source_physical_row_sha256": tensor_sha256(source_embedding[int(physical_domain)]),
            "target_local_row0_sha256": tensor_sha256(target_embedding[0]),
            "source_table_sha256": tensor_sha256(source_embedding),
            "target_table_sha256": tensor_sha256(target_embedding),
            "local_row_matches_physical_source": tensor_sha256(
                source_embedding[int(physical_domain)]) == tensor_sha256(target_embedding[0]),
            "global_table_shape_preserved": tuple(source_embedding.shape) == tuple(target_embedding.shape),
        },
    })
    return target, result


def legacy_m3oe_tensor_compatibility_fixture(schema: FeatureSchema, *, c: int = 64,
                                              seed: int = 1701, batch_size: int = 19,
                                              atol: float = 1e-6) -> Dict[str, object]:
    """Compare F0 M3OE against the historical S1 forward at D=3/T=2."""

    from stage_S1_models import StageS1M3OEAdapter  # local import avoids changing S1
    from stage_S1_models import features_from_schema as s1_features_from_schema

    legacy_dims = [8 * int(c), 4 * int(c), 4 * int(c), int(c)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        legacy = StageS1M3OEAdapter(s1_features_from_schema(schema), domain_num=3,
                                    task_num=2, fcn_dims=legacy_dims, expert_num=4,
                                    exp_d=1, exp_t=1, bal_d=1, bal_t=1)
        f0 = build_f0_model(schema, 3, 2, "m3oe", c, seed=seed + 1)
    f0.load_state_dict(legacy.state_dict(), strict=True)
    torch.manual_seed(seed + 2)
    x = torch.zeros((len(schema.columns), batch_size), dtype=torch.float32)
    sparse_indices = {schema.columns.index(v) for v in schema.sparse_columns}
    for column in range(len(schema.columns) - 1):
        if column in sparse_indices:
            x[column, :] = torch.randint(0, int(schema.vocab_sizes[schema.columns[column]]),
                                         (batch_size,), dtype=torch.int64).float()
        else:
            x[column, :] = torch.rand(batch_size)
    x[-1, :] = torch.arange(batch_size).remainder(3).float()
    legacy.eval()
    f0.eval()
    with torch.no_grad():
        expected = legacy(x)
        observed = f0(x)
    error = float(torch.max(torch.abs(expected - observed)).item())
    return {"max_absolute_error": error, "atol": float(atol),
            "equivalent": error <= float(atol), "legacy_shape": list(expected.shape),
            "f0_shape": list(observed.shape)}


__all__ = [
    "F0M3OE", "F0MultiExpert", "F0CGC1Layer", "PureSigmoidWeight", "build_f0_model",
    "build_semantically_paired_model", "semantic_initialize_f0", "F0InitializationAudit",
    "m3oe_parameter_formula", "mmoe_parameter_formula", "cgc1_parameter_formula",
    "expected_parameter_count", "non_embedding_parameter_count", "parameter_count_manifest",
    "validate_frozen_width", "legacy_m3oe_tensor_compatibility_fixture",
    "tensor_sha256", "module_state_sha256",
]
