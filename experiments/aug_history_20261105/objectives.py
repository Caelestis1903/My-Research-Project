"""Keep the parent scalar's exact operation order and expose its two cell terms."""
from dataclasses import dataclass
import torch
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.objectives import losses as source_losses


@dataclass
class LossTerms:
    total: torch.Tensor
    clean: torch.Tensor
    aug_by_cell: dict
    metadata: dict


def loss_terms(model, x, y, view):
    if x.ndim != 2 or y.shape != (x.shape[1], 2):
        raise ValueError('Expected x:[F,N], y:[N,2]')
    total, info = source_losses(model, x, y, 0,
        dict(num_views=1, view_family='gaussian', pairing='iid'),
        view['dataset'], view['seed'], view['step'],
        mix_lambda=view['lambda_aug'], kappa=view['kappa'])
    clean = info['real'].mean(1).mean()
    terms = {(0, t): view['lambda_aug'] *
        (info['augmented'][0][:, t]-info['real'][info['mask'], t]).sum()/y.numel()
        if info['augmented'] else clean*0 for t in range(2)}
    return LossTerms(total, clean, terms, info)

