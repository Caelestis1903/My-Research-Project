"""Single source of truth for the closed Stage S0 formal protocol."""

SEEDS = (2022, 2023, 2024, 2025)
ROTATIONS = ("d0_large", "d1_large", "d2_large")
COMMON_EPOCHS = (9, 19, 29, 39, 49)
GAMMA_P = 0.9431818
GAMMA_EQ = 0.980453

ARM_SPECS = {
    "sample_adam": {"kind": "adam", "shared_beta1": 0.9, "gamma": None},
    "comp1_domain": {"kind": "ema_domain", "shared_beta1": 0.0, "gamma": None},
    "uniform_090": {"kind": "ema_uniform", "shared_beta1": 0.0, "gamma": 0.9},
    "uniform_p_tau": {"kind": "ema_uniform", "shared_beta1": 0.0, "gamma": GAMMA_P},
    "uniform_eq_tau": {"kind": "ema_uniform", "shared_beta1": 0.0, "gamma": GAMMA_EQ},
    "adam_p_tau": {"kind": "adam", "shared_beta1": GAMMA_P, "gamma": None},
    "adam_eq_tau": {"kind": "adam", "shared_beta1": GAMMA_EQ, "gamma": None},
}

FORMAL_PROTOCOL = {
    "device": "cuda:0",
    "role_sizes": "150000,34500,3000",
    "batch_size": 40960,
    "epoch": 50,
    "learning_rate": 0.01,
    "learning_rate_darts": 0.001,
    "weight_decay": 1e-5,
    "expert_num": 4,
    "shadow_patience": 10,
    "instrument": 1,
    "require_oracles": 1,
}


def protocol_mismatches(values):
    """Return exact critical-field mismatches without coercive tolerance."""
    mismatches = {}
    for field, expected in FORMAL_PROTOCOL.items():
        observed = values.get(field)
        if observed != expected:
            mismatches[field] = {"expected": expected, "observed": observed}
    return mismatches
