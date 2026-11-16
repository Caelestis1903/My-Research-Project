"""Source Sample budget, fully detached views, one ordinary Adam/EMA update."""
import torch
from torch.nn import functional as F
from .vendor.gradient_aug_20261031_v2.objectives import losses as source_losses
from .rng import normal
from .input_probe import add_delta, embedding, sparse_fields
from .views import build_delta
from .protocol import update_ema


def counts(batch, target):
    return dict(original_rows=batch, target_rows=target, extra_view_rows=0,
        training_forwards=1, training_forward_rows=batch, probe_forwards=0,
        probe_forward_rows=0, teacher_forwards=0, teacher_forward_rows=0,
        input_gradient_calls=0, parameter_backward_calls=1, optimizer_steps=1,
        ema_updates=1, diagnostic_teacher_forwards=0, diagnostic_teacher_forward_rows=0)


def source_config(active):
    return dict(num_views=int(active), view_family='gaussian' if active else 'none', pairing='iid')


def losses(model, ema, x, y, row_ids, config, context, *, diagnostic=False):
    arm = config['arm']; mask = x[-1] == config['target_domain']; n = int(mask.sum())
    count = counts(len(y), n); info = {}
    active = arm != 'A0' and config['lambda_aug'] != 0 and config['kappa'] != 0 and n > 0
    if not active or arm == 'A1':
        explicit = None
        if active:
            ids = row_ids[mask.detach().cpu().numpy()]
            noise = normal(context, ids, (len(model.perturb_slices), 16), 'A1|gaussian', x)
            explicit = [dict(family='gaussian', noise=noise)]
            h = sparse_fields(model, embedding(model, x[:, mask]))
            delta = config['kappa']*h.square().mean(2, keepdim=True).sqrt()*noise
            info.update(perturbation_energy=float(delta.double().square().sum()),
                nonzero_view_rows=int((delta.flatten(1).norm(dim=1) > 0).sum()))
        scalar, source = source_losses(model, x, y, config['target_domain'], source_config(active),
            context['dataset'], context['seed'], context['global_step'],
            mix_lambda=config['lambda_aug'], kappa=config['kappa'], explicit_views=explicit)
        count.update(extra_view_rows=n*int(active), training_forwards=1+int(active),
            training_forward_rows=len(y)+n*int(active))
        return scalar, source, info, count
    ids = row_ids[mask.detach().cpu().numpy()]
    deltas, info, work = build_delta(model, ema, x[:, mask], y[mask], ids, config, context, diagnostic)
    count.update(work)
    count['probe_forward_rows'] = n*count['probe_forwards']
    count['teacher_forward_rows'] = n*count['teacher_forwards']
    count['diagnostic_teacher_forward_rows'] = n*count['diagnostic_teacher_forwards']
    info['perturbation_energy'] = sum(float(delta.double().square().sum()) for delta in deltas)
    info['nonzero_view_rows'] = sum(int((delta.flatten(1).norm(dim=1) > 0).sum()) for delta in deltas)
    if not any(bool(torch.count_nonzero(delta)) for delta in deltas):
        scalar, source = source_losses(model, x, y, config['target_domain'], source_config(False),
            context['dataset'], context['seed'], context['global_step'])
        return scalar, source, info, count
    prediction = model(x); real = F.binary_cross_entropy(prediction, y, reduction='none')
    augmented = []
    for delta in deltas:
        handle = model.embedding.register_forward_hook(lambda module, inputs, output, delta=delta:
            add_delta(model, output, delta))
        try:
            augmented.append(F.binary_cross_entropy(model(x[:, mask]), y[mask], reduction='none'))
        finally:
            handle.remove()
    if arm in ('TG', 'RG'):
        aug_total = augmented[0][:, 0].sum()+augmented[1][:, 1].sum()
    else:
        aug_total = augmented[0].sum()
    scalar = (real.sum()+config['lambda_aug']*(aug_total-real[mask].sum()))/real.numel()
    count.update(extra_view_rows=n*len(deltas), training_forwards=1+len(deltas),
        training_forward_rows=len(y)+n*len(deltas))
    return scalar, dict(pred=prediction, real=real, mask=mask, augmented=augmented), info, count


def train_step(model, optimizer, ema, x, y, row_ids, config, context, *, diagnostic=False):
    if not bool(torch.isfinite(y).all()) or not bool(((y == 0) | (y == 1)).all()):
        raise ValueError('Expected complete finite binary source labels')
    if ema is None:
        raise ValueError('The registered protocol requires its fixed EMA state')
    optimizer.zero_grad(set_to_none=True)
    scalar, source, diagnostic_info, work = losses(model, ema, x, y, row_ids, config, context,
        diagnostic=diagnostic)
    if not bool(torch.isfinite(scalar)):
        raise FloatingPointError('Nonfinite supervised loss')
    scalar.backward()
    optimizer.step()
    update_ema(ema, model, config['rho'])
    return dict(loss=float(scalar.detach()), real_loss=float(source['real'].detach().mean()),
        pred=source['pred'].detach(), diagnostics=diagnostic_info, counts=work)
