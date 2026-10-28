"""Unified row-batch adapter around the pinned RecStudio HyperGate core."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .sources import load_hypergate_class


TASK_FIELDS = ("__mdmt_task_0", "__mdmt_task_1")


class _OrderedFieldSet(set):
    """Set-compatible RecStudio field collection with schema-stable iteration."""
    def __init__(self, values):
        self._ordered = tuple(dict.fromkeys(values))
        super().__init__(self._ordered)

    def __iter__(self):
        return iter(self._ordered)

    def __sub__(self, other):
        removed = set(other)
        return type(self)(name for name in self._ordered if name not in removed)


class _FeatureData:
    """Small RecStudio data contract used to construct its unchanged model."""
    def __init__(self, schema: dict):
        self.columns = tuple(schema["columns"])
        self.dense = set(schema.get("dense_columns", ()))
        self.sparse = set(schema.get("sparse_columns", ()))
        self.vocab_sizes = schema["vocab_sizes"]
        self.domain_key = "domain_indicator"
        self.frating = list(TASK_FIELDS)
        self.use_field = _OrderedFieldSet((*self.columns, *self.frating))
        self.field2type = {name: ("float" if name in self.dense else "token")
                           for name in self.columns}
        self.field2type.update({name: "float" for name in self.frating})
        self.field2token2idx = {self.domain_key: {"<PAD>": 0, 0: 1, 1: 2, 2: 3}}

    def num_values(self, name: str) -> int:
        return int(self.vocab_sizes[name]) + (name == self.domain_key)


class HyperGateAdapter(nn.Module):
    def __init__(self, schema: dict, config: dict, source_root):
        super().__init__()
        _validate_hypergate_config(config)
        self.schema = schema
        self.config = config
        self.columns = tuple(schema["columns"])
        if "domain_indicator" not in self.columns:
            raise ValueError("HyperGate input schema must include domain_indicator")
        self.domain_index = self.columns.index("domain_indicator")
        self.dense = frozenset(schema.get("dense_columns", ()))
        self.sparse = frozenset(schema.get("sparse_columns", ()))
        self._validate_schema()

        model_cfg = _recstudio_config(config)
        core_cfg = {
            "model": model_cfg,
            "eval": {"multi_domain_key": "domain_indicator"},
            "train": {"negative_count": None, "init_method": "xavier_normal"},
        }
        core_type = load_hypergate_class(source_root)
        self.network = core_type(config=core_cfg)
        data = _FeatureData(schema)
        self.network._init_model(data, drop_unused_field=False)
        self.network._init_parameter()

    def _validate_schema(self):
        if set(self.columns) != self.dense | self.sparse:
            raise ValueError("Every HyperGate input field must be dense or sparse")
        if self.dense & self.sparse:
            raise ValueError("A feature cannot be both dense and sparse")
        if "domain_indicator" not in self.sparse:
            raise ValueError("HyperGate domain_indicator must be sparse")
        if set(self.columns) - set(self.schema.get("vocab_sizes", {})) - self.dense:
            raise ValueError("Sparse feature vocab_sizes are incomplete")

    def _batch(self, x: torch.Tensor, domain=None, labels=None):
        if x.ndim != 2 or x.shape[1] != len(self.columns):
            raise ValueError(f"Expected [B,{len(self.columns)}] HyperGate input")
        if not torch.isfinite(x).all():
            raise ValueError("HyperGate input contains non-finite features")
        observed = x[:, self.domain_index]
        if not torch.equal(observed, observed.round()) or not torch.all((observed >= 0) & (observed < 3)):
            raise ValueError("Domain indicators must be integer 0, 1 or 2")
        d = observed.long()
        if domain is not None:
            given = torch.as_tensor(domain, device=x.device).reshape(-1)
            if (given.numel() != x.shape[0] or not torch.isfinite(given).all()
                    or not torch.equal(given, d.to(given.dtype))):
                raise ValueError("Explicit domain IDs do not match the input column")
        batch = {}
        for j, name in enumerate(self.columns):
            value = x[:, j]
            if name == "domain_indicator":
                # RecStudio's domain token reserves 0 for padding.
                batch[name] = d + 1
            elif name in self.dense:
                batch[name] = value.float()
            else:
                vocab = int(self.schema["vocab_sizes"][name])
                if not torch.equal(value, value.round()) or not torch.all((value >= 0) & (value < vocab)):
                    raise ValueError(f"Sparse feature {name} is outside its integer vocabulary")
                batch[name] = value.long()
        # The official score() computes auxiliary losses inside the same
        # method. Predictor evaluation uses neutral placeholders; objective()
        # supplies labels and returns the real auxiliary terms separately.
        if labels is None:
            task = torch.zeros((len(x), 2), dtype=torch.float32, device=x.device)
        else:
            task = labels
        batch[TASK_FIELDS[0]] = task[:, 0].float()
        batch[TASK_FIELDS[1]] = task[:, 1].float()
        return batch, d

    def _forward_terms(self, x: torch.Tensor, domain=None, labels=None, mask=None):
        batch, d = self._batch(x, domain, labels)
        core = self.network
        model_cfg = core.config["model"]
        emb = core.embedding(batch).flatten(1)
        anchor_ids = d + 1  # RecStudio reserves zero for padding.

        domain_aug = core.domain_aug_module(emb.detach())
        domain_anchor = core.domain_anchor_embedding(anchor_ids)
        domain_control = torch.cat((domain_aug, domain_anchor.detach()), dim=-1)
        task_aug = core.task_aug_module(emb.detach())
        task_anchor = core.task_anchor_embedding.weight

        aux_domain = F.cross_entropy(
            domain_aug @ core.domain_anchor_embedding.weight[1:].T, d
        )

        if core.enable_domain_gate:
            domain_emb = core.domain_embedding(anchor_ids)
            gated_emb, _ = core.domain_gate(domain_control, torch.cat((emb, domain_emb), dim=-1))
        else:  # The registered configuration requires this gate.
            gated_emb = emb

        if core.enable_biasing_gate:
            biases = [gate(domain_control) for gate in core.biasing_gate]
            experts = torch.stack([expert(gated_emb, biases) for expert in core.experts], dim=1)
        else:
            experts = torch.stack([expert(gated_emb) for expert in core.experts], dim=1)

        logits = []
        task_aux_values = []
        for task_i, task_name in enumerate(core.frating):
            task_anchor_i = task_anchor[task_i].unsqueeze(0).expand(len(x), -1)
            task_control = torch.cat((task_aug, task_anchor_i.detach(), domain_control), dim=-1)
            weights = core.fusion_gates[task_name](task_control)
            fused = (weights.unsqueeze(-1) * experts).sum(1)
            logits.append(core.towers[task_name](fused).squeeze(-1))

            if labels is not None:
                targets, valid = _task_targets(labels[:, task_i], None if mask is None else mask[:, task_i])
                raw = F.binary_cross_entropy_with_logits(
                    task_aug @ task_anchor[task_i], targets, reduction="none"
                )
                task_aux_values.append(_masked_mean(raw, valid))

        output = torch.stack(logits, dim=1)
        ld, lt = float(model_cfg["lambda_d"]), float(model_cfg["lambda_t"])
        task_aux = sum(task_aux_values, output.new_zeros(()))
        aux_domain = aux_domain * ld
        task_aux = task_aux * lt
        return output, {"domain_aux": aux_domain, "task_aux": task_aux,
                        "domain_id": d, "anchors": anchor_ids}

    def forward(self, x: torch.Tensor, domain=None) -> torch.Tensor:
        return self._forward_terms(x, domain)[0]

    def objective(self, x: torch.Tensor, labels: torch.Tensor, mask=None) -> dict[str, torch.Tensor]:
        labels = torch.as_tensor(labels, dtype=torch.float32, device=x.device)
        if labels.shape != (x.shape[0], 2):
            raise ValueError("HyperGate labels must have shape [B,2]")
        valid = torch.ones_like(labels, dtype=torch.bool) if mask is None else torch.as_tensor(
            mask, dtype=torch.bool, device=x.device)
        if valid.shape != labels.shape:
            raise ValueError("HyperGate label mask must have shape [B,2]")
        logits, aux = self._forward_terms(x, labels=labels, mask=valid)
        predictive = logits.new_zeros(())
        for t in range(2):
            targets, task_mask = _task_targets(labels[:, t], valid[:, t])
            raw = F.binary_cross_entropy_with_logits(logits[:, t], targets, reduction="none")
            predictive = predictive + _masked_mean(raw, task_mask)
        loss = predictive + aux["domain_aux"] + aux["task_aux"]
        return {"loss": loss, "predictive": predictive,
                "domain_aux": aux["domain_aux"], "task_aux": aux["task_aux"]}


def _validate_hypergate_config(config):
    required = {
        "embed_dim", "enable_embedding_gate", "enable_biasing_gate", "enable_fusion_gate",
        "gate_input_type", "aug_embed_dim", "aug_layers", "aug_dropout", "lambda_d", "lambda_t",
        "num_experts", "expert_mlp_layer", "expert_activation", "expert_dropout", "expert_batch_norm",
        "gate_mlp_layer", "gate_activation", "gate_dropout", "gate_batch_norm",
        "tower_mlp_layer", "tower_activation", "tower_dropout", "tower_batch_norm",
    }
    missing = required - set(config)
    extra = set(config) - required
    if missing or extra:
        raise ValueError(f"Invalid HyperGate config; missing={sorted(missing)}, unknown={sorted(extra)}")
    if not all(config[key] is True for key in
               ("enable_embedding_gate", "enable_biasing_gate", "enable_fusion_gate")):
        raise ValueError("All three official HyperGate gates must be enabled")
    if config["gate_input_type"] != "augmented_embedding":
        raise ValueError("This adapter implements the registered augmented_embedding path")
    if config["expert_batch_norm"] or config["gate_batch_norm"] or config["tower_batch_norm"]:
        raise ValueError("BatchNorm is unsupported in the registered no-BN HyperGate configuration")
    if config["embed_dim"] <= 0 or config["num_experts"] <= 0:
        raise ValueError("HyperGate embedding and expert counts must be positive")


def _recstudio_config(cfg):
    return {
        "embed_dim": int(cfg["embed_dim"]),
        "enable_embedding_gate": True,
        "enable_biasing_gate": True,
        "enable_fusion_gate": True,
        "gate_input_type": "augmented_embedding",
        "aug_embed_dim": int(cfg["aug_embed_dim"]),
        "aug_layers": list(cfg["aug_layers"]),
        "aug_dropout": float(cfg["aug_dropout"]),
        "lambda_d": float(cfg["lambda_d"]),
        "lambda_t": float(cfg["lambda_t"]),
        "num_experts": int(cfg["num_experts"]),
        "expert_mlp_layer": list(cfg["expert_mlp_layer"]),
        "expert_activation": cfg["expert_activation"],
        "expert_dropout": float(cfg["expert_dropout"]),
        "expert_batch_norm": bool(cfg["expert_batch_norm"]),
        "gate_mlp_layer": list(cfg["gate_mlp_layer"]),
        "gate_activation": cfg["gate_activation"],
        "gate_dropout": float(cfg["gate_dropout"]),
        "gate_batch_norm": bool(cfg["gate_batch_norm"]),
        "tower_mlp_layer": list(cfg["tower_mlp_layer"]),
        "tower_activation": cfg["tower_activation"],
        "tower_dropout": float(cfg["tower_dropout"]),
        "tower_batch_norm": bool(cfg["tower_batch_norm"]),
    }


def _task_targets(labels, mask):
    if mask is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    else:
        valid = mask
    valid = valid & torch.isfinite(labels)
    targets = torch.where(valid, labels, torch.zeros_like(labels)).float()
    if torch.any(valid & ((targets != 0) & (targets != 1))):
        raise ValueError("Observed task labels must be binary")
    return targets, valid


def _masked_mean(values, valid):
    selected = torch.where(valid, values, torch.zeros_like(values))
    return selected.sum() / valid.sum().clamp_min(1).to(values.dtype)


__all__ = ["HyperGateAdapter"]
