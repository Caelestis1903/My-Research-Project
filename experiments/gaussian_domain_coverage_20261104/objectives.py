"""Replace half of each selected original sample's two-task loss."""
import torch
from torch.nn import functional as F
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.objectives import losses as source_losses
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.views import draw_view
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.protocol import view_key


def losses(model, x, y, target_domains=None, *, target_domain=None, seed, step,
           mix_lambda=.5, kappa=.1):
    domains = tuple(target_domains) if target_domains is not None else (target_domain,)
    if not domains or len(set(domains)) != len(domains) or any(d not in (0, 2) for d in domains):
        raise ValueError('Coverage must contain distinct domains from (0, 2)')
    if not 0 <= mix_lambda <= 1 or kappa < 0:
        raise ValueError('Invalid augmentation weight or noise scale')
    if len(domains) == 1:
        scalar, info = source_losses(model, x, y, domains[0],
            dict(num_views=1, view_family='gaussian', pairing='iid'), 'kuairand', seed, step,
            mix_lambda=mix_lambda, kappa=kappa)
        info['domain_views'] = [dict(domain=domains[0], rows=int(info['mask'].sum()), **v)
                                for v in info['views']]
        return scalar, info
    prediction = model(x)
    real = F.binary_cross_entropy(prediction, y, reduction='none')
    scalar = real.mean(1).mean()
    augmented, views, masks = [], [], []
    if mix_lambda != 0 and kappa != 0:
        for domain in domains:
            mask = x[-1] == domain
            if not bool(mask.any()):
                continue
            view = draw_view(model, int(mask.sum()), 'gaussian',
                             view_key('kuairand', seed, 'train', step, 'gaussian', 1), x.device)
            augmented.append(F.binary_cross_entropy(model(x[:, mask], view=view, kappa=kappa),
                                                    y[mask], reduction='none'))
            masks.append(mask)
            views.append(dict(domain=domain, rows=int(mask.sum()), **view))
        if augmented:
            delta = sum(a.sum() - real[m].sum() for a, m in zip(augmented, masks))
            scalar = (real.sum() + mix_lambda * delta) / real.numel()
    mask = sum((x[-1] == d).to(torch.int8) for d in domains).bool()
    return scalar, dict(pred=prediction, real=real, mask=mask, augmented=augmented,
        views=views, domain_views=views, extra_views=sum(v['rows'] for v in views),
        forwards=1 + len(augmented))


@torch.no_grad()
def update_ema(ema, model, rho):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.mul_(rho).add_(source, alpha=1-rho)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def train_step(model, optimizer, ema, x, y, row_ids, config, seed, step):
    optimizer.zero_grad(set_to_none=True)
    scalar, info = losses(model, x, y, config['target_domains'], seed=seed, step=step,
                          mix_lambda=config['lambda_aug'], kappa=config['kappa'])
    if not bool(torch.isfinite(scalar)):
        raise FloatingPointError(f'Nonfinite loss at step {step}')
    scalar.backward()
    optimizer.step()
    update_ema(ema, model, config['rho'])
    views = [dict(domain=v['domain'], rows=v['rows'], key=v['key'], seed=v['seed'],
                  noise_hash=v['hash']) for v in info['domain_views']]
    return dict(loss=float(scalar.detach()), real_loss=float(info['real'].detach().mean()),
        views=views, extra_views=info['extra_views'], forwards=info['forwards'],
        optimizer_steps=1, ema_updates=1, backward_calls=1)
