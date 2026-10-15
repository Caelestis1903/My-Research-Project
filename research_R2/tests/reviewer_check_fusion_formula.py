"""Reviewer-added (Independent Implementation Reviewer, k3) formula-level probes.

These checks go beyond the coder's gradient-based "asymmetric probe": they
numerically verify the per-cell fusion formula of every arm against an
independent re-computation that uses the spec's explicit j/k mapping and
task-major cell indexing.  A j/k swap or a task-major indexing error is
invisible to the coder's probe (symmetric at q=0) but caught here.

Checks:
  P1. B0 forward == parent StageS1M3OEAdapter forward (torch.equal).
  P2. B1 with non-uniform q: predictions match an independent forward that
      applies R_dt = bD*E_d + (1-bD)*(rho*E_j + (1-rho)*E_k) with
      j/k = {0:(1,2), 1:(0,2), 2:(0,1)}, i = t*3+d.
  P3. B2 with non-uniform q_d (two tasks share rho_d).
  P4. B3 fixed table: per-cell rho actually applied equals the spec table
      in task-major order (catches the index%D wiring bug).
  P5. B4 with non-uniform delta: b_dt = sigmoid(raw_bD + delta_dt), no
      double sigmoid; delta=0 equals B0.

Run from project root:
    venv/Scripts/python.exe research_R2/tests/reviewer_check_fusion_formula.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R2_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(R2_ROOT)
sys.path.insert(0, os.path.join(R2_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "m3oe"))

from MDMTRec.basic.features import SparseFeature  # noqa
from stage_S1_models import StageS1M3OEAdapter  # noqa

from ao04_v12_model import B3_BUFFER_NAME, build_ao04_model  # noqa
from ao04_v12_protocol import B3_RHO_TABLE, JK_PARTNER  # noqa
from ao04_v12_runner import setup_seed  # noqa

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _make_feats(n_features: int = 5, vocab_size: int = 4, embed_dim: int = 16):
    feats = [SparseFeature(name=i, vocab_size=vocab_size, embed_dim=embed_dim)
             for i in range(n_features)]
    feats.append(SparseFeature(name=n_features, vocab_size=3, embed_dim=embed_dim))
    return feats


def _make_batch(n_cols: int, batch_size: int = 96) -> torch.Tensor:
    rng = np.random.RandomState(123)
    arr = rng.randint(0, 4, size=(batch_size, n_cols)).astype(np.float32)
    arr[:, -1] = np.repeat(np.arange(3, dtype=np.float32), batch_size // 3)
    return torch.from_numpy(arr).T.to(device=DEVICE, dtype=torch.float32)


def _reference_forward(model, x, domain_term_fn):
    """Independent recomputation of the forward with an explicit R_dt.

    ``domain_term_fn(t, d, domain_features)`` returns R_dt for cell i=t*3+d.
    Mirrors the parent forward exactly except for the domain term, so any
    indexing discrepancy inside AO04M3OE shows up as a numeric mismatch.
    """
    domain_num, task_num = 3, 2
    domain_id = x[-1, :].clone().detach().long()
    masks = [(domain_id == d) for d in range(domain_num)]
    input_emb = model.embedding(x, model.features, squeeze_dim=True)
    skip = model.skip_conn(input_emb)
    emb = torch.zeros((input_emb.shape[0], model.star_dim[1]),
                      device=x.device, dtype=input_emb.dtype)
    for index, (weight, bias) in enumerate(zip(model.slot_weight, model.slot_bias)):
        output = torch.matmul(input_emb, torch.multiply(weight, model.shared_weight)) \
            + bias + model.shared_bias
        emb = torch.where(masks[index].unsqueeze(1), output, emb)
    emb = model.star_mlp(emb) + skip
    gate_value = [model.gate[i](emb.detach()).unsqueeze(1)
                  for i in range(task_num * domain_num)]
    shared_outputs = [model.expert[i](emb) for i in range(model.expert_num)]
    domain_outputs = [model.domain_expert[i](emb) for i in range(domain_num)]
    task_outputs = [model.task_expert[i](emb) for i in range(task_num)]
    expert_features = torch.cat([v.unsqueeze(1) for v in shared_outputs], dim=1)
    domain_features = torch.cat([v.unsqueeze(1) for v in domain_outputs], dim=1)
    task_features = torch.cat([v.unsqueeze(1) for v in task_outputs], dim=1)
    weighted_task = []
    for index in range(task_num):
        value = model._weight_bal_t() * task_features[:, index, :]
        for other in range(task_num):
            if other != index:
                value = value + (1 - model._weight_bal_t()) / (task_num - 1) * task_features[:, other, :]
        weighted_task.append(value)
    fused = []
    for t in range(task_num):
        for d in range(domain_num):
            index = t * domain_num + d
            r_dt = domain_term_fn(t, d, domain_features)
            fused.append(torch.bmm(gate_value[index], expert_features).squeeze(1)
                         + model._weight_exp_d() * r_dt
                         + model._weight_exp_t() * weighted_task[index // domain_num])
    results = [torch.sigmoid(model.tower[i](fused[i]).squeeze(1))
               for i in range(task_num * domain_num)]
    cell_output = torch.stack(results, dim=1)
    result = torch.zeros((cell_output.shape[0], task_num), device=x.device,
                         dtype=cell_output.dtype)
    for t in range(task_num):
        for d in range(domain_num):
            cell = t * domain_num + d
            result[:, t] = torch.where(masks[d], cell_output[:, cell], result[:, t])
    return result


def _b0_domain_term(model):
    def fn(t, d, domain_features):
        bD = model._weight_bal_d()
        j, k = JK_PARTNER[d]
        return bD * domain_features[:, d, :] \
            + (1 - bD) / 2.0 * (domain_features[:, j, :] + domain_features[:, k, :])
    return fn


def _report(name: str, y_model: torch.Tensor, y_ref: torch.Tensor) -> bool:
    diff = (y_model - y_ref).abs().max().detach().cpu().item()
    ok = bool(diff == 0.0 or diff <= 1e-6)
    print("  {}: max_abs={:.3e} exact_equal={} pass={}".format(
        name, diff, bool(torch.equal(y_model, y_ref)), ok))
    return ok


def main() -> int:
    ok = True
    feats = _make_feats()
    n_cols = len(feats)
    x = _make_batch(n_cols)

    # --- P1: B0 equals the parent class forward bit-exactly ---
    print("=== P1: B0 == StageS1M3OEAdapter ===")
    setup_seed(2022)
    torch.manual_seed(2022)
    parent = StageS1M3OEAdapter(feats, domain_num=3, task_num=2,
                                fcn_dims=[512, 256, 256, 64], expert_num=4,
                                exp_d=1, exp_t=1, bal_d=1, bal_t=1).to(dtype=torch.float32)
    setup_seed(2022)
    b0 = build_ao04_model(feats, "B0", seed=2022)
    b0.load_state_dict(parent.state_dict(), strict=True)
    parent = parent.to(DEVICE).eval()
    b0 = b0.to(DEVICE).eval()
    with torch.no_grad():
        y_parent = parent(x)
        y_b0 = b0(x)
    same = bool(torch.equal(y_parent, y_b0))
    print("  torch.equal:", same, "max_abs:",
          float((y_parent - y_b0).abs().max()))
    ok = ok and same

    # --- P2: B1 non-uniform q probe ---
    print("=== P2: B1 non-uniform q formula probe ===")
    setup_seed(2022)
    b1 = build_ao04_model(feats, "B1", seed=2022)
    b1.load_state_dict(parent.state_dict(), strict=False)
    q_vals = torch.tensor([1.5, -0.7, 0.2, 0.9, -1.0, 0.3], dtype=torch.float32)
    with torch.no_grad():
        b1.q_rho.copy_(q_vals)
    b1 = b1.to(DEVICE).eval()
    rho = torch.sigmoid(q_vals).to(DEVICE)

    def b1_term(t, d, domain_features):
        cell = t * 3 + d
        j, k = JK_PARTNER[d]
        bD = b1._weight_bal_d()
        return bD * domain_features[:, d, :] + (1 - bD) * (
            rho[cell] * domain_features[:, j, :]
            + (1 - rho[cell]) * domain_features[:, k, :])

    with torch.no_grad():
        y_model = b1(x)
        y_ref = _reference_forward(b1, x, b1_term)
    ok = _report("B1", y_model, y_ref) and ok

    # Also verify j/k sensitivity: swapping j and k must CHANGE the output
    # (this is what the coder's q=0-symmetric probe cannot see).
    def b1_term_swapped(t, d, domain_features):
        cell = t * 3 + d
        j, k = JK_PARTNER[d]
        bD = b1._weight_bal_d()
        return bD * domain_features[:, d, :] + (1 - bD) * (
            rho[cell] * domain_features[:, k, :]
            + (1 - rho[cell]) * domain_features[:, j, :])

    with torch.no_grad():
        y_swap = _reference_forward(b1, x, b1_term_swapped)
    differs = float((y_model - y_swap).abs().max()) > 1e-4
    print("  j/k swap changes output (probe has power):", differs)
    ok = ok and differs

    # --- P3: B2 non-uniform q_d probe (two tasks share rho_d) ---
    print("=== P3: B2 non-uniform q_d formula probe ===")
    setup_seed(2022)
    b2 = build_ao04_model(feats, "B2", seed=2022)
    b2.load_state_dict(parent.state_dict(), strict=False)
    qd = torch.tensor([0.8, -1.2, 0.4], dtype=torch.float32)
    with torch.no_grad():
        b2.q_rho.copy_(qd)
    b2 = b2.to(DEVICE).eval()
    rho_d = torch.sigmoid(qd).to(DEVICE)

    def b2_term(t, d, domain_features):
        j, k = JK_PARTNER[d]
        bD = b2._weight_bal_d()
        return bD * domain_features[:, d, :] + (1 - bD) * (
            rho_d[d] * domain_features[:, j, :]
            + (1 - rho_d[d]) * domain_features[:, k, :])

    with torch.no_grad():
        y_model = b2(x)
        y_ref = _reference_forward(b2, x, b2_term)
    ok = _report("B2", y_model, y_ref) and ok

    # --- P4: B3 fixed table task-major wiring probe ---
    print("=== P4: B3 fixed-table task-major probe ===")
    setup_seed(2022)
    b3 = build_ao04_model(feats, "B3", seed=2022)
    b3.load_state_dict(parent.state_dict(), strict=False)
    b3 = b3.to(DEVICE).eval()
    table = torch.tensor(B3_RHO_TABLE, dtype=torch.float32, device=DEVICE)

    def b3_term(t, d, domain_features):
        cell = t * 3 + d
        j, k = JK_PARTNER[d]
        bD = b3._weight_bal_d()
        return bD * domain_features[:, d, :] + (1 - bD) * (
            table[cell] * domain_features[:, j, :]
            + (1 - table[cell]) * domain_features[:, k, :])

    with torch.no_grad():
        y_model = b3(x)
        y_ref = _reference_forward(b3, x, b3_term)
    ok = _report("B3(task-major table)", y_model, y_ref) and ok

    # --- P5: B4 non-uniform delta probe (no double sigmoid) ---
    print("=== P5: B4 non-uniform delta formula probe ===")
    setup_seed(2022)
    b4 = build_ao04_model(feats, "B4", seed=2022)
    b4.load_state_dict(parent.state_dict(), strict=False)
    delta = torch.tensor([0.4, -0.9, 0.1, 1.1, -0.3, 0.7], dtype=torch.float32)
    with torch.no_grad():
        b4.delta_bal_d.copy_(delta)
    b4 = b4.to(DEVICE).eval()

    def b4_term(t, d, domain_features):
        cell = t * 3 + d
        j, k = JK_PARTNER[d]
        raw_bD = b4._weight_bal_d.deep_weights
        b_dt = torch.sigmoid(raw_bD + delta.to(DEVICE))[cell]
        return b_dt * domain_features[:, d, :] \
            + (1 - b_dt) / 2.0 * (domain_features[:, j, :] + domain_features[:, k, :])

    with torch.no_grad():
        y_model = b4(x)
        y_ref = _reference_forward(b4, x, b4_term)
    ok = _report("B4", y_model, y_ref) and ok

    # B4 with delta=0 must equal B0 (double-sigmoid would break this).
    with torch.no_grad():
        b4.delta_bal_d.zero_()
        y_b4_zero = b4(x)
    _report("B4(delta=0)==B0", y_b4_zero, y_b0)
    ok = ok and bool((y_b4_zero - y_b0).abs().max() <= 1e-6)

    print("=== OVERALL:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
