"""One weight budget per original row, independent augmentation RNG."""
import torch
from torch.nn import functional as F


def losses(model, x, y, target_domain, noise_seed, relative_std=.1, mix_lambda=.5, enabled=True):
    pred = model(x)
    real = F.binary_cross_entropy(pred, y, reduction='none')
    mask = x[-1] == target_domain
    augmented = None
    view = real.mean(1).mean()
    # Identical zero-noise views collapse exactly, including Adam's epsilon-sensitive coordinates.
    if enabled and mix_lambda != 0 and relative_std != 0 and bool(mask.any()):
        generator = torch.Generator(device=x.device).manual_seed(noise_seed)
        aug_pred = model(x[:, mask], relative_std=relative_std, generator=generator)
        augmented = F.binary_cross_entropy(aug_pred, y[mask], reduction='none')
        view = (real.sum() + mix_lambda * (augmented.sum() - real[mask].sum())) / real.numel()
    return view, dict(pred=pred, real=real, augmented=augmented, mask=mask,
                      extra_views=int(mask.sum()) if augmented is not None else 0,
                      forwards=1 + int(augmented is not None))
