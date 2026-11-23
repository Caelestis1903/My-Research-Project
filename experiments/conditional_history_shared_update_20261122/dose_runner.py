"""Matched-dose adapters for the pinned static PAIR objective."""
from .history_controls import history_config, make_history_optimizer
from .protocol import v3
from .results import DOSE


def dose_config(arm, dataset, lam=.025):
    if arm not in DOSE or lam not in (0., .025):
        raise ValueError((arm, lam))
    if lam == 0:
        return v3.config('A0', dataset, 0.)
    parent = history_config('A21', dataset) if arm == DOSE[0] else v3.config('P00', dataset, .05)
    return dict(parent, lambda_pair=lam)


def make_optimizer(model, cfg):
    if cfg['mode'] == 'collapsed':
        return make_history_optimizer(model, cfg)
    return v3.make_optimizer(model, cfg)


def worker(attempt, job, device, authority, stop):
    if job['arm'] == DOSE[0]:
        from .history_runner import worker as history_worker
        return history_worker(attempt, job, device, authority, stop)
    if job['arm'] != DOSE[1]:
        raise ValueError(job['arm'])
    # The frozen core worker labels every non-A_CORE endpoint as reconstruction.
    # Its science endpoint adapter preserves this job's true stage and trajectory.
    # Each invocation runs in a dedicated worker process.
    from . import runner
    from .history_runner import science_endpoint
    ordinary_endpoint = runner.endpoint
    runner.endpoint = science_endpoint
    try:
        return runner.worker(attempt, job, device, authority, stop)
    finally:
        runner.endpoint = ordinary_endpoint
