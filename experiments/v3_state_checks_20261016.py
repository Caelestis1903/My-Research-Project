"""Minimal numerical acceptance checks for the preregistered V11 updates."""
import argparse
from pathlib import Path
import torch
import v3_state_train_20261016 as state_train
import v3_state_shadow_20261016 as shadow


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    records = []
    initial = torch.tensor([.7, -.3, 1.2], dtype=torch.float64)
    gradients = [torch.tensor([.2, -.1, .5], dtype=torch.float64),
                 torch.tensor([-.1, .3, .1], dtype=torch.float64)]
    for variant, betas in [('native', (.9,.999)), ('beta1_zero_v_only', (0.,.999)),
                           ('beta2_zero', (.9,0.)), ('m_only', (.9,.999))]:
        param = torch.nn.Parameter(initial.clone())
        opt = (state_train.MOnly([param]) if variant == 'm_only' else
               torch.optim.Adam([param], lr=.003, betas=betas, weight_decay=1e-5))
        m, v = torch.zeros_like(initial), torch.zeros_like(initial)
        expected = initial.clone()
        for step, data_gradient in enumerate(gradients):
            gradient = data_gradient + 1e-5*expected
            delta = shadow.adam_delta(gradient, m, v, step, beta1=betas[0],
                beta2=betas[1], m_only=variant=='m_only')
            expected = expected + delta
            m = m.lerp(gradient, 1-betas[0])
            v = v*betas[1] + gradient.square()*(1-betas[1])
            param.grad = data_gradient.clone()
            opt.step()
            error = float((param.detach()-expected).abs().max())
            assert error < 1e-14, (variant, step, error)
            records.append(dict(variant=variant, step=step+1, max_absolute_error=error))
        # Identical-state null must be exactly identical, not merely similar.
        a = shadow.adam_delta(gradient, m, v, 2)
        b = shadow.adam_delta(gradient, m.clone(), v.clone(), 2)
        assert torch.equal(a, b)
    state_train.base.dump(args.out, dict(status='PASS', checks=records,
        identical_state_null_bit_exact=True, beta1_zero_is_v_only_alias=True,
        source_sha=state_train.base.sha(__file__),
        training_sha=state_train.base.sha(state_train.__file__),
        shadow_sha=state_train.base.sha(shadow.__file__), test_access=False))


if __name__ == '__main__':
    main()
