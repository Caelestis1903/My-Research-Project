"""Same-old-state finite Adam responses, computed without writing parameter grads."""
import time
import torch
from experiments.overnight_ab_20261103.state import state_digest
from .adam_counterfactual import predict_displacement
from .subspace import orthonormal_basis


def extract_aug_grads(terms, params):
    return {cell: torch.autograd.grad(value, params, create_graph=False,
        retain_graph=True, allow_unused=True) for cell, value in terms.aug_by_cell.items()}


def response_basis(snapshot, grads, aug_by_cell):
    start = time.perf_counter()
    active = [g is not None for g in grads]
    old_hash = state_digest(snapshot)
    full = predict_displacement(snapshot['theta'], grads, snapshot, active)
    removed = []
    for cells in (((0, 0),), ((0, 1),), ((0, 0), (0, 1))):
        z = [None if g is None else g.detach().clone() for g in grads]
        for cell in cells:
            for i, a in enumerate(aug_by_cell[cell]):
                if z[i] is not None and a is not None:
                    z[i] = z[i]-a
        removed.append(predict_displacement(snapshot['theta'], z, snapshot, active))
    qd, md = orthonormal_basis(torch.stack([full-removed[0], full-removed[1]], dim=1))
    qm, mm = orthonormal_basis((full-removed[2])[:, None])
    after_hash = state_digest(snapshot)
    if after_hash != old_hash:
        raise RuntimeError('Read-only Adam snapshot mutated')
    return dict(D=qd, M=qm, full_displacement=full, metadata=dict(D=md, M=mm,
        state_hash_before=old_hash, state_hash_after=after_hash,
        active_mask_hash=state_digest(torch.tensor(active)), response_seconds=time.perf_counter()-start))

