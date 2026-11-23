"""The frozen Gaussian objective, with explicit full-real and target-delta gradients."""
import torch
from experiments.gradient_aug_20261031_v2.objectives import losses as source_losses
from .optimizer import SADMomentum


def losses(model, x, y, target, c, dataset, seed, step, explicit_views=None):
    enabled = c['use_augmented_view']
    source = dict(num_views=1 if enabled else 0, view_family='gaussian', pairing='iid')
    return source_losses(model, x, y, target, source, dataset, seed, step,
                         mix_lambda=c['lambda_aug'], kappa=c['noise_scale'], explicit_views=explicit_views)


def split_gradients(info, parameters):
    active = bool(info['augmented'])
    real = torch.autograd.grad(info['real'].mean(1).mean(), parameters,
                               retain_graph=active, allow_unused=True)
    if active:
        target_delta = (info['augmented'][0].sum()-info['real'][info['mask']].sum())/info['real'].numel()
        delta = torch.autograd.grad(target_delta, parameters, allow_unused=True)
    else:
        delta = tuple(None for _ in parameters)
    return real, delta, 2 if active else 1


def train_step(model, optimizer, x, y, target, c, dataset, seed, step, diagnose=False):
    optimizer.zero_grad(set_to_none=True)
    scalar, info = losses(model, x, y, target, c, dataset, seed, step)
    if not bool(torch.isfinite(scalar)):
        raise FloatingPointError(f'Nonfinite loss at step {step}')
    if isinstance(optimizer, SADMomentum):
        parameters = tuple(model.parameters())
        real, delta, gradient_calls = split_gradients(info, parameters)
        diagnostic = optimizer.step(real, delta, diagnose=diagnose)
    else:
        scalar.backward(); gradient_calls = 1; optimizer.step()
        diagnostic = dict(measurement='last_actual_batch', separated_state=False,
            q_delta_norm=None, m_real_norm=None, injection_ratio=None, q_delta_cosine=None,
            cosine_reason='ordinary_Adam_no_separated_state', local_adam_update_cosine=None,
            local_adam_relative_difference=None, local_reference='ordinary_Adam') if diagnose else None
    return scalar.detach(), info, diagnostic, gradient_calls
