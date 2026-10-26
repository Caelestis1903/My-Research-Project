"""Equivalent CUDA float64 row-bootstrap arithmetic; frozen pair serializer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import v3_readout_20261016 as readout

FROZEN_SHA = '655e284c7c32052c2aea951ccf6005befc2dc8561c302736e775a9007c0dfe7c'
base = readout.base


def ordered_auc(order, labels, ends, weights):
    w = weights.index_select(1, order)
    positive = (w*labels).cumsum(1).index_select(1, ends)
    negative = (w*(1-labels)).cumsum(1).index_select(1, ends)
    pos = positive - torch.cat((torch.zeros_like(positive[:, :1]), positive[:, :-1]), 1)
    neg = negative - torch.cat((torch.zeros_like(negative[:, :1]), negative[:, :-1]), 1)
    return (pos*(negative-neg+.5*neg)).sum(1) / (positive[:, -1]*negative[:, -1])


def bootstrap_curves(labels, domains, predictions, subset, k, seed, progress=None, device='cuda:0'):
    arms, epochs, _, _ = predictions.shape
    result = torch.zeros((k, arms, epochs, 2), dtype=torch.float64, device=device)
    rng = np.random.default_rng(seed)
    for domain in range(3):
        rows = subset[domains[subset] == domain]
        n = len(rows)
        assert n*n < 2**52, 'Exact half-integer AUC count arithmetic bound'
        y = labels[rows]
        prepared = {}
        for arm in range(arms):
            for epoch in range(epochs):
                for task in range(2):
                    scores = predictions[arm, epoch, rows, task]
                    order, yy, starts = readout.auc_order(y[:, task], scores)
                    ends = np.r_[starts[1:]-1, n-1]
                    p = np.clip(scores.astype(float), 1e-12, 1-1e-12)
                    loss = -(y[:, task]*np.log(p)+(1-y[:, task])*np.log(1-p))
                    prepared[(arm, epoch, task)] = (order, yy.astype(np.float64), ends, loss)
        for offset in range(0, k, 16):
            m = min(16, k-offset)
            counts = rng.multinomial(n, np.full(n, 1/n), size=m)
            weights = torch.as_tensor(counts, device=device, dtype=torch.float64)
            for (arm, epoch, task), data in prepared.items():
                order, yy, ends, loss = (torch.as_tensor(value, device=device) for value in data)
                auc = ordered_auc(order, yy, ends, weights)
                ll = (weights @ loss)/n
                result[offset:offset+m, arm, epoch, 0].add_(auc/6)
                result[offset:offset+m, arm, epoch, 1].add_(ll/6)
            torch.cuda.synchronize(device)
            if progress:
                base.dump(progress, dict(status='BOOTSTRAP', domain=domain,
                    replicates_completed=offset+m, k=k, time=time.time(), backend='CUDA_float64'))
    actual = result.cpu().numpy()
    assert np.isfinite(actual).all()
    return actual


def check(device, out):
    labels = np.array([0,1,0,1,1])
    scores = np.array([.1,.4,.4,.8,.8])
    counts = np.array([[1,1,1,1,1], [2,1,2,3,1]])
    order, y, starts = readout.auc_order(labels, scores)
    ends = np.r_[starts[1:]-1, len(labels)-1]
    actual = ordered_auc(torch.as_tensor(order, device=device),
        torch.as_tensor(y, dtype=torch.float64, device=device),
        torch.as_tensor(ends, device=device),
        torch.as_tensor(counts, dtype=torch.float64, device=device)).cpu().numpy()
    expected = readout.weighted_auc(labels, scores, counts)
    assert np.array_equal(actual, expected)
    base.dump(out, dict(status='PASS', tie_weighted_auc_bit_exact=True,
        source_sha=base.sha(__file__), frozen_readout_sha=FROZEN_SHA))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['pair','check'], required=True)
    p.add_argument('--natural', type=Path)
    p.add_argument('--balanced', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--bootstraps', type=int, default=1000, choices=[1000])
    args = p.parse_args()
    assert base.sha(readout.__file__) == FROZEN_SHA
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    if args.mode == 'check':
        check(args.device, args.out)
        return
    started = time.time()
    old_file, old_function, old_dump = readout.__file__, readout.bootstrap_curves, base.dump

    def gpu_bootstrap(*positional, **keyword):
        return bootstrap_curves(*positional, **keyword, device=args.device)

    def dump(path, value):
        if Path(path) == args.out/'receipt.json' and value.get('status') == 'COMPLETE':
            value.update(original_pair_serializer_sha=FROZEN_SHA, compute_backend='CUDA_float64',
                bootstrap_design='GPU_bootstrap_equivalent_arithmetic_design.json',
                numerical_device=args.device, elapsed_seconds=time.time()-started,
                peak_reserved_mib=torch.cuda.max_memory_reserved(args.device)/1024**2)
        return old_dump(path, value)

    # The pinned original pair retains all producer checks, point metrics, RNG
    # seeds, selection/diagnostic rules and output serialization.
    readout.__file__ = __file__
    readout.bootstrap_curves = gpu_bootstrap
    base.dump = dump
    try:
        readout.pair(args)
    finally:
        readout.__file__, readout.bootstrap_curves, base.dump = old_file, old_function, old_dump


if __name__ == '__main__':
    main()
