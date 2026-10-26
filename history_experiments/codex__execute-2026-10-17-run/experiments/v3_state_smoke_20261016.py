"""Mechanical fresh-versus-resumed two-epoch acceptance of V11 wrappers."""
import argparse
from pathlib import Path
import subprocess
import sys
import torch
import v3_state_train_20261016 as state_train


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    records = []
    for variant in state_train.VARIANTS:
        directories = [a.out / (variant + '_full2'), a.out / (variant + '_resumed2')]
        command = [sys.executable, 'experiments/v3_state_train_20261016.py', '--seed', '2022',
            '--arm', 'SN', '--state-variant', variant, '--device', a.device, '--smoke']
        for directory, epochs, resume in [(directories[0],2,False),
                                         (directories[1],1,False), (directories[1],2,True)]:
            subprocess.run(command + ['--run-dir',str(directory),'--epochs',str(epochs)] +
                (['--resume'] if resume else []), check=True)
        states = [torch.load(d/'last_resume.pt', map_location='cpu', weights_only=False) for d in directories]
        for key in ('model', 'optimizer', 'sampler_rng', 'sampler_cycle_rng', 'steps',
                    'best', 'selected', 'history', 'batch_chain'):
            assert equal(states[0][key], states[1][key]), (variant,key)
        records.append(dict(variant=variant, fresh_and_resumed_model_optimizer_sampler_exact=True,
            directories=list(map(str,directories))))
    state_train.base.dump(a.out/'receipt.json', dict(status='PASS', runs=records,
        source_sha=state_train.base.sha(__file__), training_sha=state_train.base.sha(state_train.__file__),
        research_results=False, test_access=False))


if __name__ == '__main__':
    main()
