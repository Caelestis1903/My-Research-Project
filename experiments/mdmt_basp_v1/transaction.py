"""A finite four-arm experiment from one complete RAM state."""
from __future__ import annotations
import math
from .permissions import add_new_parameters


def _cell_losses(metrics):
    cells = metrics.get('cells', [])
    if isinstance(cells, dict):
        cells = list(cells.values())
    return [cell.get('logloss', cell.get('log_loss')) for cell in cells]


def decide(metrics, cfg):
    keep = metrics['KEEP']
    rules = cfg['split']
    def safe(name):
        value = metrics[name]
        if value.get('local_logloss') is None or keep.get('local_logloss') is None:
            return False
        if value['macro_logloss']-keep['macro_logloss'] > rules['max_macro_ll_harm']:
            return False
        cells, base = _cell_losses(value), _cell_losses(keep)
        if any(a is not None and b is not None and a-b > rules['max_cell_ll_harm'] for a, b in zip(cells, base)):
            return False
        return True
    split = metrics['SPLIT']
    wide = metrics['WIDE']
    reader = metrics['READER']
    split_reasons = []
    if not wide.get('capacity_match', False):
        split_reasons.append('CAPACITY_MATCH_UNAVAILABLE')
    if not safe('SPLIT'):
        split_reasons.append('RISK_SAFETY_OR_SUPPORT')
    if split.get('local_logloss') is None:
        split_reasons.append('NO_LOCAL_SUPPORT')
    else:
        if keep['local_logloss']-split['local_logloss'] < rules['local_ll_gain_vs_keep']:
            split_reasons.append('NO_INCREMENT_VS_KEEP')
        if wide.get('local_logloss') is None or wide['local_logloss']-split['local_logloss'] < rules['local_ll_gain_vs_wide']:
            split_reasons.append('NO_INCREMENT_VS_WIDE')
        if (keep['local_logloss']-split['local_logloss'])*split.get('local_mass', 0.) < rules['min_natural_weighted_local_gain']:
            split_reasons.append('INSUFFICIENT_NATURAL_CONTRIBUTION')
        if (reader.get('local_logloss') is not None and
            split['local_logloss']-reader['local_logloss'] > rules['max_loss_to_cheaper_reader'] and
            reader.get('added_parameters', math.inf) < split.get('added_parameters', 0)):
            split_reasons.append('CHEAPER_READER_SUFFICIENT')
    if not split_reasons:
        winner = 'SPLIT'
    else:
        eligible = ['KEEP']+[name for name in ('READER', 'WIDE') if safe(name)]
        winner = min(eligible, key=lambda name: (metrics[name]['macro_logloss'], metrics[name].get('added_parameters', 0)))
    return dict(winner=winner, split_accepted=winner == 'SPLIT', split_rejection_reasons=split_reasons,
                interpretation='STRUCTURE_INCREMENT' if winner == 'SPLIT' else
                'ORDINARY_CAPACITY_INCREMENT' if winner == 'WIDE' else 'READER_INCREMENT' if winner == 'READER' else 'KEEP')


def run_transaction(snapshot, proposal, callbacks, cfg):
    """callbacks: fork(snapshot,variant,proposal), train(arm,n), project(arm),
    evaluate(all_arms), capture(arm). evaluate performs the sole S-window read.
    """
    branches = {}
    for variant in cfg['split']['arms']:
        arm = callbacks['fork'](snapshot, variant, proposal)
        callbacks['train'](arm, cfg['split']['continuation_steps'])
        callbacks['project'](arm)
        branches[variant] = arm
    metrics = callbacks['evaluate'](branches)
    decision = decide(metrics, cfg)
    winner = decision['winner']
    selected = callbacks['capture'](branches[winner])
    records = [dict(variant=name, selected=name == winner, **value) for name, value in metrics.items()]
    return dict(winner=winner, snapshot=selected, records=records, decision=decision)
