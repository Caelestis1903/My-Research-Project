"""Read-only ordinary Adam predictions using PyTorch's exact single-tensor kernel."""
import copy
import torch
from torch.optim import _functional


def snapshot_optimizer(model, optimizer):
    parameters = list(model.parameters())
    lookup = {id(p): i for i, p in enumerate(parameters)}
    groups = []
    for group in optimizer.param_groups:
        for key in ('amsgrad', 'maximize', 'capturable', 'differentiable', 'foreach', 'fused', 'decoupled_weight_decay'):
            if group.get(key, False):
                raise ValueError('Unsupported Adam setting: '+key)
        groups.append(dict(indices=[lookup[id(p)] for p in group['params']],
            lr=group['lr'], betas=group['betas'], eps=group['eps'], weight_decay=group['weight_decay']))
    states = []
    for p in parameters:
        old = optimizer.state.get(p, {})
        states.append({k: v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)
                       for k, v in old.items()})
    return dict(theta=[p.detach().clone() for p in parameters], states=states, groups=groups)


@torch.no_grad()
def predict_displacement(theta, grads, optimizer_snapshot, active_mask):
    if len(theta) != len(grads) or len(theta) != len(active_mask):
        raise ValueError('Adam coordinate lists differ')
    new = [p.detach().clone() for p in theta]
    for group in optimizer_snapshot['groups']:
        indices = [i for i in group['indices'] if active_mask[i]]
        if not indices:
            continue
        if any(grads[i] is None for i in indices):
            raise ValueError('Active gradient cannot be None')
        moments, variances, steps = [], [], []
        for i in indices:
            state = optimizer_snapshot['states'][i]
            moments.append(state['exp_avg'].clone() if state else torch.zeros_like(theta[i]))
            variances.append(state['exp_avg_sq'].clone() if state else torch.zeros_like(theta[i]))
            steps.append(state['step'].clone() if state else torch.tensor(0.))
        _functional.adam([new[i] for i in indices], [grads[i] for i in indices],
            moments, variances, [], steps, foreach=False, capturable=False,
            differentiable=False, fused=False, amsgrad=False,
            beta1=group['betas'][0], beta2=group['betas'][1], lr=group['lr'],
            weight_decay=group['weight_decay'], eps=group['eps'], maximize=False)
    return torch.cat([(a-b).reshape(-1) for a, b in zip(new, theta)])

