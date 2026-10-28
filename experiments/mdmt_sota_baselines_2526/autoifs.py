"""Unified row-batch adapter around the pinned external AutoIFS core."""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .sources import load_autoifs_class


class MixedFieldEmbedding(nn.Module):
    """Global-offset categorical rows plus a learned dense-field vector."""
    def __init__(self, schema: dict, embedding_dim: int):
        super().__init__()
        self.columns = tuple(schema["columns"])
        self.dense = frozenset(schema.get("dense_columns", ()))
        self.vocab_sizes = schema["vocab_sizes"]
        self.embedding_dim = int(embedding_dim)
        offsets = {}
        cursor = 0
        for name in self.columns:
            if name in self.dense:
                continue
            if name not in self.vocab_sizes:
                raise ValueError(f"Missing AutoIFS vocabulary size for {name}")
            offsets[name] = cursor
            cursor += int(self.vocab_sizes[name])
        self.offsets = offsets
        self.embedding = nn.Parameter(torch.empty(cursor, self.embedding_dim))
        nn.init.xavier_uniform_(self.embedding)
        dense_names = [name for name in self.columns if name in self.dense]
        self.dense_names = tuple(dense_names)
        if len(dense_names) > 1:
            raise ValueError("The registered schema supports at most one scalar dense field")
        self.dense_vector = nn.Parameter(torch.empty(len(dense_names), self.embedding_dim))
        if dense_names:
            nn.init.xavier_uniform_(self.dense_vector)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fields = []
        dense_i = 0
        for i, name in enumerate(self.columns):
            value = x[:, i]
            if name in self.dense:
                if not torch.isfinite(value).all():
                    raise ValueError(f"Dense feature {name} contains non-finite values")
                fields.append(value.float().unsqueeze(1) * self.dense_vector[dense_i].unsqueeze(0))
                dense_i += 1
            else:
                vocab = int(self.vocab_sizes[name])
                if not torch.equal(value, value.round()) or not torch.all((value >= 0) & (value < vocab)):
                    raise ValueError(f"Sparse feature {name} is outside its integer vocabulary")
                ids = value.long() + self.offsets[name]
                fields.append(F.embedding(ids, self.embedding))
        if not fields:
            return x.new_empty((x.shape[0], 0, self.embedding_dim))
        return torch.stack(fields, dim=1)


class AutoIFSAdapter(nn.Module):
    def __init__(self, schema: dict, config: dict, source_root):
        super().__init__()
        _validate_autoifs_config(config)
        self.schema = schema
        self.config = config
        self.columns = tuple(schema["columns"])
        if "domain_indicator" not in self.columns:
            raise ValueError("AutoIFS input schema must include domain_indicator")
        dense = set(schema.get("dense_columns", ()))
        sparse = set(schema.get("sparse_columns", ()))
        if dense & sparse or set(self.columns) != dense | sparse or "domain_indicator" not in sparse:
            raise ValueError("AutoIFS schema must classify each field and keep domain_indicator sparse")
        self.domain_index = self.columns.index("domain_indicator")
        feature_embedding = MixedFieldEmbedding(schema, int(config["latent_dim"]))
        embedding_rows = feature_embedding.embedding.shape[0]

        network_type = load_autoifs_class(source_root)
        opt = {
            "latent_dim": int(config["latent_dim"]),
            "feat_num": int(embedding_rows),
            "field_num": len(self.columns),
            "task_num": 2,
            "domain_list": [0, 1, 2],
            "lora_r": int(config["lora_r"]),
            "mlp_dims": list(config["mlp_dims"]),
            "mlp_dropout": float(config["mlp_dropout"]),
            "use_bn": bool(config["use_bn"]),
        }
        self.network = network_type(opt)
        # The official embedding is replaced only at its input boundary; all
        # four information paths and the selector remain the upstream class.
        self.network.embedding = feature_embedding
        self.network.gate_embedding = MixedFieldEmbedding(schema, int(config["latent_dim"]))
        self.network.gate_embedding.load_state_dict(feature_embedding.state_dict())
        self.network.gate_embedding.requires_grad_(False)
        self._patch_output_bias_shapes()
        self.lambdas = tuple(float(v) for v in config["gate_penalties"])

    def _patch_output_bias_shapes(self):
        network = self.network
        network.D_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(kernel.shape[1])) for kernel in network.D_kernels
        ])
        network.T_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(kernel.shape[1])) for kernel in network.T_kernels
        ])
        network.D_lora_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(network.domain_num, kernel.shape[1]))
            for kernel in network.D_kernels
        ])
        network.T_lora_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(network.task_num, kernel.shape[1]))
            for kernel in network.T_kernels
        ])

    def _input_domain(self, x: torch.Tensor, domain=None):
        if x.ndim != 2 or x.shape[1] != len(self.columns):
            raise ValueError(f"Expected [B,{len(self.columns)}] AutoIFS input")
        if not torch.isfinite(x).all():
            raise ValueError("AutoIFS input contains non-finite features")
        observed = x[:, self.domain_index]
        if not torch.equal(observed, observed.round()) or not torch.all((observed >= 0) & (observed < 3)):
            raise ValueError("Domain indicators must be integer 0, 1 or 2")
        d = observed.long()
        if domain is not None:
            given = torch.as_tensor(domain, device=x.device).reshape(-1)
            if (given.numel() != x.shape[0] or not torch.isfinite(given).all()
                    or not torch.equal(given, d.to(given.dtype))):
                raise ValueError("Explicit domain IDs do not match the input column")
        return d

    def forward(self, x: torch.Tensor, domain=None) -> torch.Tensor:
        d = self._input_domain(x, domain).reshape(-1, 1)
        result = self.network(x, d)
        return torch.cat((result[0], result[1]), dim=1)

    def objective(self, x: torch.Tensor, labels: torch.Tensor, mask=None) -> dict[str, torch.Tensor]:
        labels = torch.as_tensor(labels, dtype=torch.float32, device=x.device)
        if labels.shape != (x.shape[0], 2):
            raise ValueError("AutoIFS labels must have shape [B,2]")
        valid = torch.ones_like(labels, dtype=torch.bool) if mask is None else torch.as_tensor(
            mask, dtype=torch.bool, device=x.device)
        if valid.shape != labels.shape:
            raise ValueError("AutoIFS label mask must have shape [B,2]")
        logits = self.forward(x)
        predictive = logits.new_zeros(())
        for t in range(2):
            targets, task_mask = _task_targets(labels[:, t], valid[:, t])
            raw = F.binary_cross_entropy_with_logits(logits[:, t], targets, reduction="none")
            predictive = predictive + _masked_mean(raw, task_mask)
        gate_penalty = self.gate_penalty() if not self.network.ticket else predictive.new_zeros(())
        return {"loss": predictive + gate_penalty, "predictive": predictive,
                "gate_penalty": gate_penalty}

    def gate_penalty(self) -> torch.Tensor:
        """Official search penalty on mean sigmoid selector probabilities."""
        value = None
        for task, logits in enumerate(self.network.weights):
            term = self.lambdas[task] * torch.sigmoid(self.network.temp * logits).mean()
            value = term if value is None else value + term
        if value is None:
            return next(self.parameters()).new_zeros(())
        return value

    def optimizer_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {"base": [], "gate": []}
        for name, parameter in self.network.named_parameters():
            if not parameter.requires_grad:
                continue
            groups["gate" if "gate" in name else "base"].append(parameter)
        if not groups["base"]:
            raise ValueError("AutoIFS needs trainable non-selector parameters")
        if not groups["gate"] and not self.network.ticket:
            raise ValueError("AutoIFS search needs trainable selector parameters")
        if {id(p) for p in groups["base"]} & {id(p) for p in groups["gate"]}:
            raise AssertionError("AutoIFS optimizer parameter groups overlap")
        return groups

    def export_search_state(self) -> dict:
        def state_dict(module):
            return OrderedDict((name, value.detach().cpu().clone())
                               for name, value in module.state_dict().items())
        return {
            "gate_hypernet": state_dict(self.network.gate_hypernet),
            "gate_head0": state_dict(self.network.gate_head0),
            "gate_head1": state_dict(self.network.gate_head1),
            # The final search backbone becomes the fixed selector embedding
            # at reuse; the initial frozen gate embedding is not the ticket.
            "gate_embedding.embedding": self.network.embedding.embedding.detach().cpu().clone(),
            "gate_embedding.dense_vector": self.network.embedding.dense_vector.detach().cpu().clone(),
            "temp": float(self.network.temp),
            "ticket": bool(self.network.ticket),
        }

    def import_selector_state(self, snapshot: dict):
        for key in ("gate_hypernet", "gate_head0", "gate_head1"):
            getattr(self.network, key).load_state_dict(snapshot[key])
        with torch.no_grad():
            self.network.gate_embedding.embedding.copy_(snapshot["gate_embedding.embedding"])
            if self.network.gate_embedding.dense_names:
                self.network.gate_embedding.dense_vector.copy_(snapshot["gate_embedding.dense_vector"])
        self.network.temp = float(snapshot["temp"])
        self.network.ticket = bool(snapshot["ticket"])

    def freeze_gate(self):
        for name, parameter in self.network.named_parameters():
            if "gate" in name:
                parameter.requires_grad_(False)


def _validate_autoifs_config(config):
    required = {"latent_dim", "lora_r", "mlp_dims", "mlp_dropout", "use_bn", "gate_penalties"}
    missing, extra = required - set(config), set(config) - required
    if missing or extra:
        raise ValueError(f"Invalid AutoIFS config; missing={sorted(missing)}, unknown={sorted(extra)}")
    if config["use_bn"]:
        raise ValueError("BatchNorm is unsupported in the registered no-BN AutoIFS configuration")
    if list(config["mlp_dims"]) != [1024, 512, 256]:
        raise ValueError("AutoIFS adapter supports the pinned 1024→512→256 topology")
    if int(config["latent_dim"]) <= 0 or int(config["lora_r"]) <= 0:
        raise ValueError("AutoIFS latent dimension and LoRA rank must be positive")
    if len(config["gate_penalties"]) != 2:
        raise ValueError("AutoIFS gate_penalties must contain one value per task")


def _task_targets(labels, mask):
    valid = torch.ones_like(labels, dtype=torch.bool) if mask is None else mask
    valid = valid & torch.isfinite(labels)
    targets = torch.where(valid, labels, torch.zeros_like(labels)).float()
    if torch.any(valid & ((targets != 0) & (targets != 1))):
        raise ValueError("Observed task labels must be binary")
    return targets, valid


def _masked_mean(values, valid):
    return torch.where(valid, values, torch.zeros_like(values)).sum() / valid.sum().clamp_min(1).to(values.dtype)


__all__ = ["AutoIFSAdapter", "MixedFieldEmbedding"]
