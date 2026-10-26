"""Logit/component adapter for the local, unchanged fixed1 implementation.

The numerical operations below mirror ``fixed.MDMTRec.forward``.  The only
training graph change in ``risk=True`` is removal of the gate-input detach.
This is a predictive-function derivative, not the model's training rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .vendor.pinned_fixed.basic.features import DenseFeature, SparseFeature
from .vendor.pinned_fixed.models.multi_domain.fixed import MDMTRec


@dataclass
class FeatureOutput:
    native_logits: Tensor
    shared_context: Tensor
    domain: Tensor
    expert_outputs: Tensor
    embedding: Tensor


def make_features(schema: Any) -> list:
    """Use the actual historical feature classes, including scalar dense age."""
    if isinstance(schema, (list, tuple)) and schema and hasattr(schema[0], "embed_dim"):
        return list(schema)
    def field(name: str, default=None):
        return schema.get(name, default) if isinstance(schema, dict) else getattr(schema, name, default)
    columns = field("columns")
    dense = set(field("dense_columns", ()))
    sizes = field("vocab_sizes", {})
    dim = int(field("embedding_dim", 16))
    if not columns:
        raise ValueError("A feature schema with ordered columns is required")
    return [DenseFeature(i) if name in dense else SparseFeature(i, int(sizes[name]), embed_dim=dim)
            for i, name in enumerate(columns)]


class _MechanicalAdapter(nn.Module):
    """Accept batch-major input and return two independent Bernoulli logits."""
    def __init__(self, features_or_schema=None, seed: int = 2022, native: nn.Module | None = None):
        super().__init__()
        if native is None:
            # Native initialization has its own stream and is unaffected by bank draws.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                native = MDMTRec(make_features(features_or_schema), domain_num=3, task_num=2,
                                 fcn_dims=[512, 256, 256, 64], expert_num=4,
                                 exp_d=1, exp_t=1, bal_d=1, bal_t=1)
        if (native.domain_num, native.task_num, native.expert_num) != (3, 2, 4):
            raise ValueError("MDMT-NEXT fixed1 requires 3 domains, 2 tasks and 4 shared experts")
        if any(float(getattr(native, k)) != 1.0 for k in ("exp_d", "exp_t", "bal_d", "bal_t")):
            raise ValueError("The native fixed1 coefficients must all equal one")
        if any(name.startswith("_weight_") for name, _ in native.named_parameters()):
            raise ValueError("Learned fusion/AutoML parameters are not part of fixed1")
        self.native = native
        self.context_dim = 64
        self.domain_num = 3
        self.task_num = 2

    def encode(self, x_batch_major: Tensor, risk: bool = False) -> FeatureOutput:
        m = self.native
        if x_batch_major.ndim != 2 or x_batch_major.shape[1] != len(m.features):
            raise ValueError("Expected batch-major X with one column per feature")
        domain_values = x_batch_major[:, -1]
        domain = domain_values.detach().long()
        if not torch.all((domain_values == domain) & (domain >= 0) & (domain < 3)):
            raise ValueError("Domain indicators must be integer 0, 1 or 2")
        x = x_batch_major.transpose(0, 1)
        masks = [domain == d for d in range(3)]
        input_emb = m.embedding(x, m.features, squeeze_dim=True)
        skip = m.skip_conn(input_emb)
        # Keep the native operation order, including the zero-valued fusion terms.
        emb = torch.zeros((input_emb.shape[0], m.star_dim[1]), device=x.device, dtype=input_emb.dtype)
        for d, (weight, bias) in enumerate(zip(m.slot_weight, m.slot_bias)):
            transformed = torch.matmul(input_emb, torch.multiply(weight, m.shared_weight)) + bias + m.shared_bias
            emb = torch.where(masks[d].unsqueeze(1), transformed, emb)
        emb = m.star_mlp(emb) + skip
        gate_input = emb if risk else emb.detach()
        gates = [m.gate[i](gate_input).unsqueeze(1) for i in range(6)]
        experts = torch.cat([e(emb).unsqueeze(1) for e in m.expert], dim=1)
        domain_experts = torch.cat([e(emb).unsqueeze(1) for e in m.domain_expert], dim=1)
        task_experts = torch.cat([e(emb).unsqueeze(1) for e in m.task_expert], dim=1)
        weighted_domain = []
        for d in range(3):
            value = m.bal_d * domain_experts[:, d, :]
            for other in range(3):
                if other != d:
                    value = value + (1 - m.bal_d) / 2 * domain_experts[:, other, :]
            weighted_domain.append(value)
        weighted_task = []
        for q in range(2):
            value = m.bal_t * task_experts[:, q, :]
            for other in range(2):
                if other != q:
                    value = value + (1 - m.bal_t) * task_experts[:, other, :]
            weighted_task.append(value)
        logits = []
        for i in range(6):
            fused = (torch.bmm(gates[i], experts).squeeze(1)
                     + m.exp_d * weighted_domain[i % 3] + m.exp_t * weighted_task[i // 3])
            logits.append(m.tower[i](fused).squeeze(1))
        selected = torch.zeros((len(x_batch_major), 2), device=x.device, dtype=input_emb.dtype)
        for q in range(2):
            for d in range(3):
                selected[:, q] = torch.where(masks[d], logits[q * 3 + d], selected[:, q])
        return FeatureOutput(selected, experts.mean(1), domain, experts, input_emb)

    def forward(self, x_batch_major: Tensor, *, risk: bool = False) -> Tensor:
        return self.encode(x_batch_major, risk=risk).native_logits

    def native_predict(self, x_batch_major: Tensor) -> Tensor:
        return self.forward(x_batch_major).sigmoid()

    def risk_forward(self, x_batch_major: Tensor) -> Tensor:
        return self.forward(x_batch_major, risk=True)


@dataclass
class NativePacket:
    logits: Tensor
    shared: Tensor
    h: Tensor


class NativeAdapter(_MechanicalAdapter):
    def encode(self, x_batch_major: Tensor, risk: bool=False) -> NativePacket:
        if risk:
            raise ValueError("risk=True changes the native gradient graph")
        out = super().encode(x_batch_major, risk=False)
        h = torch.nn.functional.layer_norm(out.expert_outputs, (64,), eps=1e-5).reshape(-1,256)
        return NativePacket(out.native_logits, out.expert_outputs, h)

    def forward(self, x_batch_major: Tensor, *, risk: bool=False) -> Tensor:
        return self.encode(x_batch_major, risk=risk).logits


def make_native(schema, seed, dataset, device="cpu"):
    from .vendor.pinned_fixed.models.multi_domain.original import MDMTRec as Original
    from .plans import torch_seed
    from .state import state_hash
    canonical_seed = torch_seed(seed,dataset,10)
    kwargs=dict(domain_num=3,task_num=2,fcn_dims=[512,256,256,64],expert_num=4,exp_d=1,exp_t=1,bal_d=1,bal_t=1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(canonical_seed)
        original=Original(make_features(schema),**kwargs)
        backbone={k:v for k,v in original.state_dict().items() if not k.startswith("_weight_")}
        native=MDMTRec(make_features(schema),**kwargs)
        native.load_state_dict(backbone,strict=True)
    receipt=dict(seed=canonical_seed,canonical_backbone_sha256=state_hash(backbone),copied_backbone_sha256=state_hash(native.state_dict()),native_parameter_count=sum(p.numel() for p in native.parameters()))
    assert receipt["canonical_backbone_sha256"]==receipt["copied_backbone_sha256"]
    return NativeAdapter(native=native).to(device),receipt
