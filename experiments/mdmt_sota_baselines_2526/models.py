"""Method factory for the two pinned upstream models and fixed1 comparator."""
from __future__ import annotations

import torch
from torch import nn

from experiments.mdmt_basp_v1.prefix import make_native

from .autoifs import AutoIFSAdapter
from .hypergate import HyperGateAdapter
from .sources import validate_source_roots


class M3OEFixed1Adapter(nn.Module):
    def __init__(self, schema: dict, config: dict):
        super().__init__()
        required = {
            "seed": config.get("seed", 2022),
            "dataset": config.get("dataset", schema.get("dataset")),
        }
        if required["dataset"] not in ("movielens", "kuairand"):
            raise ValueError("M3OE_FIXED1 requires dataset=movielens or kuairand in its model config")
        if int(config.get("domain_num", 3)) != 3 or int(config.get("task_num", 2)) != 2:
            raise ValueError("M3OE_FIXED1 is fixed to three domains and two tasks")
        if list(config.get("fcn_dims", [512, 256, 256, 64])) != [512, 256, 256, 64]:
            raise ValueError("M3OE_FIXED1 uses the existing frozen [512,256,256,64] topology")
        # The BASP helper initializes Original then copies fixed1 tensors and
        # checks direct tensor equality. It does not execute its legacy SHA path.
        self.network, self.initialization = make_native(
            schema, int(required["seed"]), required["dataset"], device="cpu"
        )
        self.columns = tuple(schema["columns"])
        self.domain_index = self.columns.index("domain_indicator")
        self.dense = frozenset(schema.get("dense_columns", ()))
        self.sparse = frozenset(schema.get("sparse_columns", ()))
        self.vocab_sizes = schema.get("vocab_sizes", {})
        if (self.dense & self.sparse or set(self.columns) != self.dense | self.sparse
                or "domain_indicator" not in self.sparse
                or self.sparse - set(self.vocab_sizes)):
            raise ValueError("M3OE schema must classify each field and keep domain_indicator sparse")

    def forward(self, x: torch.Tensor, domain=None) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != len(self.columns):
            raise ValueError(f"Expected [B,{len(self.columns)}] M3OE input")
        observed = x[:, self.domain_index]
        if not torch.isfinite(x).all() or not torch.equal(observed, observed.round()):
            raise ValueError("M3OE input must be finite with integral domains")
        d = observed.long()
        if not torch.all((d >= 0) & (d < 3)):
            raise ValueError("Domain indicators must be 0, 1 or 2")
        for index, name in enumerate(self.columns):
            if name not in self.sparse or name == "domain_indicator":
                continue
            value = x[:, index]
            vocab = int(self.vocab_sizes[name])
            if not torch.equal(value, value.round()) or not torch.all((value >= 0) & (value < vocab)):
                raise ValueError(f"Sparse feature {name} is outside its integer vocabulary")
        if domain is not None:
            given = torch.as_tensor(domain, device=x.device).reshape(-1)
            if (given.numel() != len(x) or not torch.isfinite(given).all()
                    or not torch.equal(given, d.to(given.dtype))):
                raise ValueError("Explicit domain IDs do not match the input column")
        return self.network(x)

    def objective(self, x: torch.Tensor, labels: torch.Tensor, mask=None) -> dict[str, torch.Tensor]:
        labels = torch.as_tensor(labels, dtype=torch.float32, device=x.device)
        if labels.shape != (x.shape[0], 2):
            raise ValueError("M3OE labels must have shape [B,2]")
        valid = torch.ones_like(labels, dtype=torch.bool) if mask is None else torch.as_tensor(
            mask, dtype=torch.bool, device=x.device)
        if valid.shape != labels.shape:
            raise ValueError("M3OE label mask must have shape [B,2]")
        logits = self.forward(x)
        observed = valid & torch.isfinite(labels)
        targets = torch.where(observed, labels, torch.zeros_like(labels))
        if torch.any(observed & ((targets != 0) & (targets != 1))):
            raise ValueError("Observed task labels must be binary")
        raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        predictive = torch.where(observed, raw, torch.zeros_like(raw)).sum() / observed.sum().clamp_min(1)
        return {"loss": predictive, "predictive": predictive,
                "domain_aux": predictive.new_zeros(()), "task_aux": predictive.new_zeros(())}


def build_model(method: str, schema: dict, model_config: dict, source_roots: dict):
    """Build one adapter without opening data or starting a training action."""
    method = str(method).upper()
    if method == "M3OE_FIXED1":
        return M3OEFixed1Adapter(schema, model_config)
    roots = validate_source_roots(source_roots)
    if method == "HYPERGATE":
        cfg = {
            "embed_dim": model_config["embedding_dim"],
            "enable_embedding_gate": model_config["gates"]["embedding"],
            "enable_biasing_gate": model_config["gates"]["expert_bias"],
            "enable_fusion_gate": model_config["gates"]["fusion"],
            "gate_input_type": model_config["gate_input_type"],
            "aug_embed_dim": model_config["aug_dim"],
            "aug_layers": model_config["aug_dims"],
            "aug_dropout": model_config["aug_dropout"],
            "lambda_d": model_config["lambda_domain"],
            "lambda_t": model_config["lambda_task"],
            "num_experts": model_config["expert_num"],
            "expert_mlp_layer": model_config["expert_dims"],
            "expert_activation": model_config["activation"],
            "expert_dropout": model_config["dropout"],
            "expert_batch_norm": model_config["batch_norm"],
            "gate_mlp_layer": model_config["gate_dims"],
            "gate_activation": model_config["activation"],
            "gate_dropout": model_config["dropout"],
            "gate_batch_norm": model_config["batch_norm"],
            "tower_mlp_layer": model_config["tower_dims"],
            "tower_activation": model_config["activation"],
            "tower_dropout": model_config["dropout"],
            "tower_batch_norm": model_config["batch_norm"],
        }
        return HyperGateAdapter(schema, cfg, roots["recstudio"])
    if method == "AUTOIFS":
        cfg = {
            "latent_dim": model_config["embedding_dim"],
            "lora_r": model_config["rank"][model_config["dataset"]],
            "mlp_dims": model_config["mlp_dims"],
            "mlp_dropout": model_config["dropout"],
            "use_bn": model_config["batch_norm"],
            "gate_penalties": model_config["lambda"],
        }
        return AutoIFSAdapter(schema, cfg, roots["autoifs"])
    raise ValueError(f"Unknown baseline method: {method}")


__all__ = ["build_model", "M3OEFixed1Adapter", "HyperGateAdapter", "AutoIFSAdapter"]
