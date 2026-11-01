"""The unchanged Gaussian objective and explicit clean/delta channels."""
import torch
from experiments.gradient_aug_momentum_20261101.objectives import losses as source_losses, split_gradients
from experiments.gradient_aug_momentum_20261101.optimizer import SADMomentum
from .optimizer import AIGMomentum, innovation, parameter_groups


def losses(model, x, y, target, c, dataset, seed, step, explicit_views=None):
    strength = c['lambda_max'] if c['use_aig_m'] else c['lambda_aug']
    old = dict(use_augmented_view=c['use_augmentation'], lambda_aug=strength, noise_scale=c['noise_scale'])
    return source_losses(model,x,y,target,old,dataset,seed,step,explicit_views=explicit_views)


def train_step(model, optimizer, x, y, target, c, dataset, seed, step, diagnose=False, calibrate=False):
    optimizer.zero_grad(set_to_none=True)
    scalar, info = losses(model,x,y,target,c,dataset,seed,step)
    if not bool(torch.isfinite(scalar)):
        raise FloatingPointError(f'Nonfinite reference loss at step {step}')
    records = []
    if isinstance(optimizer,(SADMomentum,AIGMomentum)):
        parameters = tuple(model.parameters())
        real, delta, calls = split_gradients(info,parameters)
        if isinstance(optimizer,AIGMomentum):
            records = optimizer.step(real,delta,diagnose=diagnose)
        else:
            if calibrate or diagnose:
                u = innovation(optimizer,parameters,parameter_groups(model,target),real,delta)
                records = [dict(group=g,u=v,gate=None,lambda_t=c['lambda_aug']) for g,v in u.items()]
            diagnostic = optimizer.step(real,delta,diagnose=diagnose)
            if diagnostic is not None:
                records.append(dict(group='ALL',u=None,gate=None,lambda_t=c['lambda_aug'],
                    q_norm=diagnostic['q_delta_norm'],m_norm=diagnostic['m_real_norm'],
                    injection_ratio=diagnostic['injection_ratio']))
    else:
        scalar.backward(); calls = 1; optimizer.step()
    return scalar.detach(),info,records,calls
