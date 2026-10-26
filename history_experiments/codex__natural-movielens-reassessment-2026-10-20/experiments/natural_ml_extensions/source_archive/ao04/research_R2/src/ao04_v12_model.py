"""Stage R2 (AO-04 v1.2) AO04M3OE model — five-arm fusion extension.

Implements five arms (B0..B4) by overriding the
``StageS1M3OEAdapter.forward`` body.  All arms share the same backbone
construction as Stage S1 (``StageS1M3OEAdapter``).  Only the per-cell
domain term R_dt (and, for B1/B2/B4, additional trainable scalars)
changes.  The model is the single source of truth for the per-arm fusion
math described in ``research_R2/R2_Design_Specification.md`` section 2.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

# Set up sys.path so the read-only Stage S1 modules and M3OE root resolve.
R2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(R2_ROOT)
EXPERIMENTS_ROOT = os.path.join(PROJECT_ROOT, "experiments")
M3OE_ROOT = os.path.join(PROJECT_ROOT, "m3oe")
for _path in (PROJECT_ROOT, EXPERIMENTS_ROOT, M3OE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from stage_S1_data import FeatureSchema  # noqa: E402
from stage_S1_models import (  # noqa: E402
    StageS1M3OEAdapter,
    build_model,
    initial_state_hash,
    is_architecture_parameter,
    parameter_manifest,
)

from ao04_v12_protocol import (  # noqa: E402
    ARCHITECTURE_PARAMETER_NAMES,
    ARCHITECTURE_RAW_INIT,
    B3_RHO_TABLE,
    DOMAIN_NUM,
    EXPERT_NUM,
    FCN_DIMS,
    JK_PARTNER,
    N_CELLS,
    TASK_NUM,
    arm_supported,
)


# ---------------------------------------------------------------------------
# Names of new buffers / parameters.
# ---------------------------------------------------------------------------

# Buffer name (B3): registered, included in state_dict but not in
# named_parameters; never enters any optimizer.
B3_BUFFER_NAME = "rho_fixed"

# Trainable parameter names per arm.  Keys must not start with "_weight_"
# or "architecture." so that the existing is_architecture_parameter
# classifier never silently absorbs them into the arch optimizer.
Q_RHO_NAME = "q_rho"
DELTA_BAL_D_NAME = "delta_bal_d"


_B3_RHO_BUFFER = torch.tensor(B3_RHO_TABLE, dtype=torch.float32)


class AO04M3OE(StageS1M3OEAdapter):
    """Five-arm M3OE model for Stage R2 (AO-04 v1.2).

    All arms share the Stage S1 backbone.  Only the per-cell domain term
    ``R_dt`` differs across arms:

        B0  -- historic Stage S1 (uniform other split).
        B1  -- per-cell learnable ratio q_dt -> rho_dt.
        B2  -- per-domain shared ratio q_d -> rho_d.
        B3  -- fixed non-learnable table from the specification.
        B4  -- per-cell own-coefficient shift delta_dt on top of bD.

    B1 / B2 add ``q_rho`` (trainable, main optimizer), B3 registers a
    buffer ``rho_fixed`` (no gradient), B4 adds ``delta_bal_d``
    (trainable, main optimizer).  No other parameter is modified.  The
    construction path consumes exactly the same RNG sequence as
    ``build_model`` (M3OE branch).  New tensors are zero/constant so they
    consume no RNG; preservation of the original RNG stream is part of
    the pairing contract.
    """

    def __init__(self, features: Sequence[object], domain_num: int = DOMAIN_NUM,
                 task_num: int = TASK_NUM, arm: str = "B0",
                 fcn_dims: Sequence[int] = FCN_DIMS,
                 expert_num: int = EXPERT_NUM,
                 exp_d: float = ARCHITECTURE_RAW_INIT,
                 exp_t: float = ARCHITECTURE_RAW_INIT,
                 bal_d: float = ARCHITECTURE_RAW_INIT,
                 bal_t: float = ARCHITECTURE_RAW_INIT,
                 **kwargs) -> None:
        if not arm_supported(arm):
            raise ValueError("unknown AO-04 arm: {}".format(arm))
        if domain_num != DOMAIN_NUM or task_num != TASK_NUM:
            raise ValueError(
                "AO04M3OE only supports D=3/T=2 (got D={}, T={})".format(domain_num, task_num)
            )
        super().__init__(features=features, domain_num=domain_num, task_num=task_num,
                         fcn_dims=list(fcn_dims), expert_num=expert_num,
                         exp_d=exp_d, exp_t=exp_t, bal_d=bal_d, bal_t=bal_t,
                         **kwargs)
        self.arm = str(arm)
        if self.arm == "B1":
            self.q_rho = nn.Parameter(torch.zeros(N_CELLS, dtype=torch.float32))
        elif self.arm == "B2":
            self.q_rho = nn.Parameter(torch.zeros(DOMAIN_NUM, dtype=torch.float32))
        elif self.arm == "B3":
            self.register_buffer(B3_BUFFER_NAME, _B3_RHO_BUFFER.clone())
        elif self.arm == "B4":
            self.delta_bal_d = nn.Parameter(torch.zeros(N_CELLS, dtype=torch.float32))
        # B0: no extras.

    # ------------------------------------------------------------------
    # Query helpers (used by logging, audits and decision logic).
    # ------------------------------------------------------------------

    def arm_parameter_names(self) -> List[str]:
        """Names of arm-specific trainable parameters (excluding base backbone).

        B3 has none (rho_fixed is a buffer, not a parameter).
        """

        if self.arm == "B1":
            return [Q_RHO_NAME]
        if self.arm == "B2":
            return [Q_RHO_NAME]
        if self.arm == "B4":
            return [DELTA_BAL_D_NAME]
        return []

    def architecture_parameter_names(self) -> List[str]:
        """Strict list of exactly the four architecture raw scalars."""

        return list(ARCHITECTURE_PARAMETER_NAMES)

    def new_tensors_as_state_dict(self) -> Dict[str, torch.Tensor]:
        """Snapshot of arm-specific tensors (parameters + B3 buffer)."""

        out: Dict[str, torch.Tensor] = {}
        if self.arm in ("B1", "B2"):
            out[Q_RHO_NAME] = self.q_rho.detach().cpu().clone()
        if self.arm == "B3":
            out[B3_BUFFER_NAME] = getattr(self, B3_BUFFER_NAME).detach().cpu().clone()
        if self.arm == "B4":
            out[DELTA_BAL_D_NAME] = self.delta_bal_d.detach().cpu().clone()
        return out

    @torch.no_grad()
    def fusion_coefficients(self) -> Dict[str, object]:
        """eval-mode snapshot of all own/other coefficients per cell."""

        device = self._weight_exp_d.deep_weights.device
        aD = torch.sigmoid(self._weight_exp_d.deep_weights).detach()
        aT = torch.sigmoid(self._weight_exp_t.deep_weights).detach()
        bD_raw = self._weight_bal_d.deep_weights.detach()
        bT_raw = self._weight_bal_t.deep_weights.detach()
        bD = torch.sigmoid(bD_raw)
        bT = torch.sigmoid(bT_raw)

        snapshot: Dict[str, object] = {
            "arm": self.arm,
            "raw": {
                "aD_raw": float(self._weight_exp_d.deep_weights.detach().cpu().item()),
                "aT_raw": float(self._weight_exp_t.deep_weights.detach().cpu().item()),
                "bD_raw": float(self._weight_bal_d.deep_weights.detach().cpu().item()),
                "bT_raw": float(self._weight_bal_t.deep_weights.detach().cpu().item()),
            },
            "sigmoid": {
                "aD": float(aD.cpu().item()),
                "aT": float(aT.cpu().item()),
                "bD": float(bD.cpu().item()),
                "bT": float(bT.cpu().item()),
            },
        }
        if self.arm == "B1":
            rho = torch.sigmoid(self.q_rho).detach()
            snapshot["q_rho"] = self.q_rho.detach().cpu().clone().tolist()
            snapshot["rho_dt"] = rho.cpu().clone().tolist()
        elif self.arm == "B2":
            rho_per_d = torch.sigmoid(self.q_rho).detach()
            snapshot["q_rho_by_domain"] = self.q_rho.detach().cpu().clone().tolist()
            snapshot["rho_per_domain"] = rho_per_d.cpu().clone().tolist()
            rho_full = torch.stack([rho_per_d[d] for t in range(TASK_NUM)
                                    for d in range(DOMAIN_NUM)])
            snapshot["rho_dt"] = rho_full.cpu().clone().tolist()
        elif self.arm == "B3":
            rho_full = getattr(self, B3_BUFFER_NAME).detach()
            snapshot["rho_dt"] = rho_full.cpu().clone().tolist()
        elif self.arm == "B4":
            b_dt = torch.sigmoid(self.delta_bal_d + bD_raw).detach()
            snapshot["delta_dt"] = self.delta_bal_d.detach().cpu().clone().tolist()
            snapshot["b_dt"] = b_dt.cpu().clone().tolist()

        # Per-cell own/other coefficients and snapshot of own+other=1 check.
        per_cell: List[Dict[str, float]] = []
        for cell in range(N_CELLS):
            t = cell // DOMAIN_NUM
            d = cell % DOMAIN_NUM
            j, k = JK_PARTNER[d]
            entry = {"cell": cell, "task": t, "domain": d, "j": j, "k": k}
            if self.arm in ("B0",):
                own = float(aD.cpu().item()) * float(bD.cpu().item())
                other = float(aD.cpu().item()) * (1.0 - float(bD.cpu().item())) / 2.0
                entry.update({"rho_dt": 0.5, "own": own, "other_total": other * 2.0,
                              "other_per": other})
            elif self.arm == "B1":
                rho_v = float(rho[cell])
                own = float(aD.cpu().item()) * float(bD.cpu().item())
                other_total = float(aD.cpu().item()) * (1.0 - float(bD.cpu().item()))
                entry.update({"rho_dt": rho_v, "own": own,
                              "other_total": other_total,
                              "other_per": [other_total * rho_v,
                                             other_total * (1.0 - rho_v)]})
            elif self.arm == "B2":
                rho_v = float(rho_per_d[d])
                own = float(aD.cpu().item()) * float(bD.cpu().item())
                other_total = float(aD.cpu().item()) * (1.0 - float(bD.cpu().item()))
                entry.update({"rho_dt": rho_v, "own": own,
                              "other_total": other_total,
                              "other_per": [other_total * rho_v,
                                             other_total * (1.0 - rho_v)]})
            elif self.arm == "B3":
                rho_v = float(rho_full[cell])
                own = float(aD.cpu().item()) * float(bD.cpu().item())
                other_total = float(aD.cpu().item()) * (1.0 - float(bD.cpu().item()))
                entry.update({"rho_dt": rho_v, "own": own,
                              "other_total": other_total,
                              "other_per": [other_total * rho_v,
                                             other_total * (1.0 - rho_v)]})
            elif self.arm == "B4":
                b_dt_v = float(b_dt[cell])
                own = float(aD.cpu().item()) * b_dt_v
                other_total = float(aD.cpu().item()) * (1.0 - b_dt_v)
                entry.update({"b_dt": b_dt_v, "own": own,
                              "other_total": other_total,
                              "other_per": other_total / 2.0})
            per_cell.append(entry)
        snapshot["per_cell"] = per_cell
        return snapshot

    # ------------------------------------------------------------------
    # Forward — overrides StageS1M3OEAdapter.forward to insert the
    # arm-specific R_dt computation.  Everything except the weighted
    # domain construction follows the parent forward verbatim.
    # ------------------------------------------------------------------

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

        weighted_domain = _ao04_weighted_domain(self, domain_features,
                                                emb.dtype, device)

        weighted_task = []
        for index in range(self.task_num):
            value = self._weight_bal_t() * task_features[:, index, :]
            for other in range(self.task_num):
                if other != index:
                    value = value + (1 - self._weight_bal_t()) / (self.task_num - 1) * task_features[:, other, :]
            weighted_task.append(value)

        # B0 emits one domain term per physical domain (3 entries, shared
        # across tasks); B1/B2/B3/B4 emit one term per cell (6 entries in
        # task-major order).  Index the weighted_domain list accordingly
        # so that the ``fused[index]`` formula below reads the correct
        # R_dt.  See Specification §2 + Pitfall §12.
        if self.arm == "B0":
            wd_indexer = lambda index: weighted_domain[index % self.domain_num]  # noqa: E731
        else:
            wd_indexer = lambda index: weighted_domain[index]  # noqa: E731

        fused = [
            torch.bmm(gate_value[index], expert_features).squeeze(1)
            + self._weight_exp_d() * wd_indexer(index)
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


def _ao04_weighted_domain(model: "AO04M3OE",
                          domain_features: torch.Tensor,
                          emb_dtype: torch.dtype,
                          device: torch.device) -> List[torch.Tensor]:
    """Per-cell domain-fusion term for the current arm.

    Reproduces section 2 of the design spec:

        B0 (uniform):
            R_d  = bD * E_d + (1 - bD) / 2 * (E_j + E_k)
        B1 (asymmetric):
            R_dt = bD * E_d + (1 - bD) * (rho_dt * E_j + (1 - rho_dt) * E_k)
        B2 (shared across tasks):
            R_dt = bD * E_d + (1 - bD) * (rho_d * E_j + (1 - rho_d) * E_k)
        B3 (fixed rho — not init-equivalent):
            Same as B1 but rho comes from the buffer; B3's R_dt is not
            used in the B0-equivalence gate (Section 4.1 spec).
        B4 (delta shift):
            R_dt = b_dt * E_d + (1 - b_dt) / 2 * (E_j + E_k)
    """

    arm = model.arm
    out: List[torch.Tensor] = []
    if arm == "B0":
        for index in range(model.domain_num):
            value = model._weight_bal_d() * domain_features[:, index, :]
            for other in range(model.domain_num):
                if other != index:
                    value = value + (1 - model._weight_bal_d()) / (model.domain_num - 1) * domain_features[:, other, :]
            out.append(value)
        return out

    if arm == "B3":
        # Use fixed rho values (not a differentiable parameter).  Apply the
        # same structural formula as B1 but with a frozen rho table.  Note
        # B3 is NOT init-equivalent and is explicitly excluded from G0-5.
        bD = model._weight_bal_d()  # still differentiable
        rho_buf = getattr(model, B3_BUFFER_NAME)
        # Iterate in task-major order so that the returned list aligns
        # with the fused index ``index`` used in the parent forward.
        for task in range(model.task_num):
            for index in range(model.domain_num):
                cell = task * model.domain_num + index
                j, k = JK_PARTNER[index]
                E_d = domain_features[:, index, :]
                E_j = domain_features[:, j, :]
                E_k = domain_features[:, k, :]
                rho_dt = rho_buf[cell]
                value = (bD * E_d
                         + (1.0 - bD) * (rho_dt * E_j + (1.0 - rho_dt) * E_k))
                out.append(value)
        return out

    if arm == "B1":
        bD = model._weight_bal_d()
        rho = torch.sigmoid(model.q_rho)  # [N_CELLS]
    elif arm == "B2":
        bD = model._weight_bal_d()
        rho_per_domain = torch.sigmoid(model.q_rho)  # [D]
        rho = torch.stack([rho_per_domain[d]
                           for _ in range(model.task_num)
                           for d in range(model.domain_num)])
    elif arm == "B4":
        # CRITICAL: use the raw unsigmoided weights to avoid double sigmoid.
        # See Specification §2 / Pitfall 2.  B4 has no rho — split the
        # other coefficient uniformly across j and k.
        bD_raw = model._weight_bal_d.deep_weights
        b_dt = torch.sigmoid(model.delta_bal_d + bD_raw)
        other_total_coef = 1.0 - b_dt
    else:
        raise ValueError("unexpected arm {}".format(arm))

    # task-major: iterate t outer, d inner.  Returned list length is
    # ``T*D`` and ``out[cell]`` is the domain term for cell ``cell``.
    for task in range(model.task_num):
        for index in range(model.domain_num):
            cell = task * model.domain_num + index
            j, k = JK_PARTNER[index]
            E_d = domain_features[:, index, :]
            E_j = domain_features[:, j, :]
            E_k = domain_features[:, k, :]
            if arm == "B1":
                rho_dt = rho[cell]
                value = bD * E_d + (1.0 - bD) * (rho_dt * E_j + (1.0 - rho_dt) * E_k)
            elif arm == "B2":
                rho_dt = rho[cell]
                value = bD * E_d + (1.0 - bD) * (rho_dt * E_j + (1.0 - rho_dt) * E_k)
            elif arm == "B4":
                value = b_dt[cell] * E_d + other_total_coef[cell] / 2.0 * (E_j + E_k)
            else:
                raise ValueError("unexpected arm {}".format(arm))
            out.append(value)
    return out


def build_ao04_model(features: Sequence[object],
                     arm: str,
                     *,
                     domain_num: int = DOMAIN_NUM,
                     task_num: int = TASK_NUM,
                     fcn_dims: Sequence[int] = FCN_DIMS,
                     expert_num: int = EXPERT_NUM,
                     seed: Optional[int] = None,
                     **kwargs) -> AO04M3OE:
    """Wrap Stage S1's M3OE seed-setting + construction for AO-04 arms.

    Notes:
        - ``seed`` (if not None) must be applied *before* the backbone is
          built, identical to ``build_model``.  New arm tensors are
          zeros / constants and consume no RNG, so the original RNG
          stream is preserved.
        - The seed-setting here is global.  Use a paired ``setup_seed``
          call upstream to coordinate with the rest of the pipeline.
    """

    if seed is not None:
        torch.manual_seed(int(seed))
    if not arm_supported(arm):
        raise ValueError("unknown AO-04 arm: {}".format(arm))
    return AO04M3OE(
        features=features, domain_num=domain_num, task_num=task_num,
        arm=arm, fcn_dims=fcn_dims, expert_num=expert_num,
        exp_d=ARCHITECTURE_RAW_INIT, exp_t=ARCHITECTURE_RAW_INIT,
        bal_d=ARCHITECTURE_RAW_INIT, bal_t=ARCHITECTURE_RAW_INIT,
        **kwargs,
    ).to(dtype=torch.float32)


def ao04_seeds() -> Tuple[int, ...]:
    """The two Stage R2 formal seeds."""

    return (2022, 2023)


def is_ao04_architecture_parameter(name: str) -> bool:
    """Strict predicate — only the four raw scalars are architecture params."""

    return name in ARCHITECTURE_PARAMETER_NAMES


def is_ao04_buffer(name: str) -> bool:
    """True iff the given state-dict key belongs to the AO-04 B3 buffer."""

    return name == B3_BUFFER_NAME or name.endswith("." + B3_BUFFER_NAME)


__all__ = [
    "AO04M3OE",
    "B3_BUFFER_NAME",
    "Q_RHO_NAME",
    "DELTA_BAL_D_NAME",
    "build_ao04_model",
    "ao04_seeds",
    "is_ao04_architecture_parameter",
    "is_ao04_buffer",
]
