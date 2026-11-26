"""Factories for the frozen published adapters without legacy training imports."""
from __future__ import annotations

from .vendor.sota2526.autoifs import AutoIFSAdapter
from .vendor.sota2526.hypergate import HyperGateAdapter


def build_published(method, bundle, seed, cfg):
    """Build the fixed adapter; session creation owns the initialization RNG."""
    method = method.upper()
    schema = bundle.features.as_dict()
    if method == "HYPERGATE":
        mapped = {
            "embed_dim": cfg["embedding_dim"],
            "enable_embedding_gate": cfg["gates"]["embedding"],
            "enable_biasing_gate": cfg["gates"]["expert_bias"],
            "enable_fusion_gate": cfg["gates"]["fusion"],
            "gate_input_type": cfg["gate_input_type"],
            "aug_embed_dim": cfg["aug_dim"], "aug_layers": cfg["aug_dims"],
            "aug_dropout": cfg["aug_dropout"], "lambda_d": cfg["lambda_domain"],
            "lambda_t": cfg["lambda_task"], "num_experts": cfg["expert_num"],
        }
        for name in ("expert", "gate", "tower"):
            mapped.update({f"{name}_mlp_layer": cfg[f"{name}_dims"],
                           f"{name}_activation": cfg["activation"],
                           f"{name}_dropout": cfg["dropout"],
                           f"{name}_batch_norm": cfg["batch_norm"]})
        return HyperGateAdapter(schema, mapped, cfg["source_root"])
    if method == "AUTOIFS":
        mapped = dict(latent_dim=cfg["embedding_dim"], lora_r=cfg["rank"],
                      mlp_dims=cfg["mlp_dims"], mlp_dropout=cfg["dropout"],
                      use_bn=cfg["batch_norm"], gate_penalties=cfg["gate_penalties"])
        return AutoIFSAdapter(schema, mapped, cfg["source_root"])
    if method in ("M3OE_FIXED1", "M3OE_FIXED1_ADAPTED", "A0"):
        from experiments.mdmt_history_information_20261103.protocol import make_model, load_protocol
        dataset = cfg["dataset"]
        # The source factory owns the fixed1 initialization and transposed layout.
        return make_model(bundle, load_protocol()["datasets"][dataset]["source"])
    raise ValueError(f"Unknown published model: {method}")


def restore_model_meta(model, meta):
    """AutoIFS stage flags are absent from its tensor state_dict."""
    if not meta:
        return
    model.network.ticket = bool(meta["ticket"])
    model.network.temp = float(meta["temp"])
    if meta["gate_frozen"]:
        model.freeze_gate()

