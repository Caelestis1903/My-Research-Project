"""Preregistered V11 update ablations using the immutable v3 training loop."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import traceback
import torch
import v3_train_20261016 as train

base = train.base
VARIANTS = ('beta1_zero_v_only', 'beta2_zero', 'm_only')


class MOnly(torch.optim.Optimizer):
    """Bias-corrected first moment with denominator one and coupled L2."""
    def __init__(self, params, lr=.003, beta=.9, weight_decay=1e-5):
        super().__init__(params, dict(lr=lr, beta=beta, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None
        for group in self.param_groups:
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                gradient = p.grad.add(p, alpha=group['weight_decay'])
                state = self.state[p]
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                state['step'] += 1
                state['exp_avg'].lerp_(gradient, 1 - beta)
                p.add_(state['exp_avg'], alpha=-group['lr'] / (1 - beta ** state['step']))


class StateRunner(train.Runner):
    def __init__(self, args):
        if args.resume:
            original = json.loads((args.run_dir / 'config.json').read_text())
            assert original['state_variant'] == args.state_variant
        adam_factory = torch.optim.Adam
        source_factory = base.sources

        def variant_factory(params, lr, weight_decay):
            if args.state_variant == 'm_only':
                return MOnly(params, lr=lr, weight_decay=weight_decay)
            betas = (0., .999) if args.state_variant == 'beta1_zero_v_only' else (.9, 0.)
            return adam_factory(params, lr=lr, betas=betas, weight_decay=weight_decay)

        def sources():
            return {**source_factory(), 'experiments/v3_state_train_20261016.py': base.sha(__file__)}

        # Process-local constructor injection lets the original initialization,
        # RNG and complete continuation path execute unchanged. Restore factories
        # before fit; no frozen source file is modified.
        torch.optim.Adam = variant_factory
        base.sources = sources
        try:
            super().__init__(args)
        finally:
            torch.optim.Adam = adam_factory
            base.sources = source_factory
        if not args.resume:
            self.config.update(optimizer_state_definition='V11_state_decomposition_design.json',
                state_ablation_lr_scope='Fixed calibrated Adam LR .003; no variant retuning')
            base.dump(self.dir / 'config.json', self.config)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='kuairand', choices=['kuairand'])
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--arm', choices=['SN', 'SB'], required=True)
    p.add_argument('--state-variant', choices=VARIANTS, required=True)
    p.add_argument('--optimizer', default='adam', choices=['adam'])
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--batch-size', type=int, default=40960)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--replacement', default='WR', choices=['WR'])
    p.add_argument('--order', default='ORIGINAL', choices=['ORIGINAL'])
    p.add_argument('--duplication', type=int, default=1, choices=[1])
    p.add_argument('--corruption', type=float, default=0., choices=[0.])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    try:
        StateRunner(args).fit()
    except Exception:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        base.dump(args.run_dir / 'failure.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
