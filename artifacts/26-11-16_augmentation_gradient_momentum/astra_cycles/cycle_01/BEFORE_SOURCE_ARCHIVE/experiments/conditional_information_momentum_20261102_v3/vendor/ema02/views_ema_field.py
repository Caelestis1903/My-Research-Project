"""Joint teacher-response nullspace and registered energy controls."""
import torch
from .rng import normal, signs
from .input_probe import embedding, sparse_fields, gain_gradients, probe_logits


def null_basis(jacobian):
    j = jacobian.double()
    lengths = j.norm(dim=2, keepdim=True)
    j = torch.where(lengths > 1e-12, j/lengths.clamp_min(1e-12), torch.zeros_like(j))
    # Tiny FP64 matrices use CPU SVD; CUDA batched SVD dominates the measured step cost.
    _, singular, vh = torch.linalg.svd(j.cpu(), full_matrices=False)
    keep = (singular > singular[:, :1]*1e-6) & (singular[:, :1] > 1e-12)
    return (vh*keep[:, :, None]).to(j.device), j, keep.sum(1).to(j.device)


def project(basis, value):
    value = value.double()
    return value-torch.einsum('nrf,nr->nf', basis, torch.einsum('nrf,nf->nr', basis, value))


def scale_gain(h, direction, kappa):
    h = h.double(); direction = direction.double()
    weights = h.square().sum(2); total = weights.sum(1); weighted = (weights*direction.square()).sum(1)
    active = (total > 1e-24) & (weighted > 1e-24)
    a0 = kappa*total.sqrt()[:, None]*direction/weighted.clamp_min(1e-24).sqrt()[:, None]
    a0 = torch.where(active[:, None], a0, torch.zeros_like(a0))
    multiplier = (.25/a0.abs().amax(1).clamp_min(1e-24)).clamp(max=1)
    return a0*multiplier[:, None], multiplier < 1


def field_delta(model, ema, x, y, row_ids, config, context, diagnostic=False):
    arm = config['arm']; h = sparse_fields(model, embedding(model, x))
    if arm != 'F0' and ema is None:
        raise ValueError('Teacher state is required for '+arm)
    counts = dict(probe_forwards=0, input_gradient_calls=0, teacher_forwards=0,
        diagnostic_teacher_forwards=0)
    g = None
    if arm != 'F_PR':
        g, _ = gain_gradients(model, x, y)
        counts['probe_forwards'] = 1; counts['input_gradient_calls'] += 1
    basis = j = rank = old_logits = None
    if arm != 'F0':
        jacobian, old_logits = gain_gradients(ema, x)
        basis, j, rank = null_basis(jacobian)
        counts['teacher_forwards'] = 1; counts['input_gradient_calls'] += 2
    if arm == 'F_PR':
        direction = project(basis, normal(context, row_ids, (h.shape[1],), 'F|fallback', h))
        fallback = torch.ones(len(h), dtype=torch.bool, device=h.device)
        ratio = torch.zeros(len(h), device=h.device, dtype=torch.float64)
    else:
        direction = g.double() if arm == 'F0' else project(basis, g)
        gn = g.double().norm(dim=1)
        ratio = direction.norm(dim=1)/gn.clamp_min(1e-12)
        fallback = (gn <= 1e-12) | ((ratio < 1e-3) if arm != 'F0' else torch.zeros_like(gn, dtype=torch.bool))
        if bool(fallback.any()):
            random = normal(context, row_ids, (h.shape[1],), 'F|fallback', h).double()
            if basis is not None: random = project(basis, random)
            direction = torch.where(fallback[:, None], random, direction)
    direction = torch.where((direction.norm(dim=1) > 1e-12)[:, None], direction, torch.zeros_like(direction))
    gain, capped = scale_gain(h, direction, config['kappa'])
    counterfactual = (gain[:, :, None]*h.double()).detach()
    counter_energy = counterfactual.square().sum((1, 2))
    if arm == 'FR':
        total = h.double().square().sum((1, 2))
        amplitude = torch.where(total > 1e-24, (counter_energy/total.clamp_min(1e-24)).sqrt(), 0.)
        gain = amplitude[:, None]*signs(context, row_ids, (h.shape[1],), 'FR|signs', h).double()
    delta = (gain[:, :, None]*h.double()).to(h.dtype).detach()
    residual = torch.einsum('ntf,nf->nt', j, gain).norm(dim=1) if j is not None else None
    diagnostics = dict(fallback_rows=int(fallback.sum()), field_rows=len(h),
        projection_ratio_sum=float(ratio.sum()), capped_rows=int(capped.sum()),
        gain_abs_max=float(gain.abs().max()) if len(h) else 0.,
        teacher_residual_max=float(residual.max()) if residual is not None and len(h) else None)
    if rank is not None:
        diagnostics.update(rank0_rows=int((rank == 0).sum()), rank1_rows=int((rank == 1).sum()), rank2_rows=int((rank == 2).sum()))
    if arm == 'FR':
        actual = delta.double().square().sum((1, 2))
        diagnostics['fr_energy_relative_error_max'] = float(((actual-counter_energy).abs()/counter_energy.clamp_min(1e-24)).max())
    if diagnostic and ema is not None and arm != 'F0':
        with torch.no_grad():
            finite_logits = probe_logits(ema, x, gains=gain.to(h.dtype))
        diagnostics['teacher_logit_change_abs_mean'] = float((finite_logits-old_logits).abs().mean())
        diagnostics['teacher_logit_change_abs_max'] = float((finite_logits-old_logits).abs().max())
        counts['diagnostic_teacher_forwards'] = 1
    return delta, diagnostics, counts
