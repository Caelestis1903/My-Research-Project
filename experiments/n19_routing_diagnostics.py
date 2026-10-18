"""Actual checkpoint routing weights and subspace margins; no data/test forwards."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from n19_data import OUT, population, dump
from n19_models import Model


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    rows = []
    margins = []
    features = {}
    for run in sorted((OUT/'runs').glob('natural__*')):
        if not (run/'receipt.json').exists():
            continue
        c = json.loads((run/'config.json').read_text())
        if c['engineering'] or c['method'] not in ['domain_mix', 'axis_mix', 'full_mix', 'interaction', 'reliability_interaction', 'utility_router', 'regret_router', 'coalition_router']:
            continue
        if c['dataset'] not in features:
            features[c['dataset']] = population(c['dataset'], 'natural').features
        model = Model(features[c['dataset']], c['seed'], c['method']).eval()
        for stage, name in [('selected', 'selected.pt'), ('final', 'last.pt')]:
            state = torch.load(Path(c['state_directory'])/name, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'])
            v = model.router_logits()
            base = dict(run_id=run.name, dataset=c['dataset'], method=c['method'], seed=c['seed'], stage=stage, epoch=state['epoch'])
            margins.append(dict(**base, domain_margin_max=float(v.mean(0).abs().max()),
                task_margin_max=float(v.mean(1).abs().max()), effective_logit_norm=float(v.norm())))
            own = model.base._weight_bal_d()
            for d in range(3):
                prior = torch.ones(3)*(1-own)/2
                prior[d] = own
                for task in range(2):
                    w = torch.softmax(prior.log()+v[d, task], 0)
                    entropy = float(-(w*w.log()).sum())
                    for source in range(3):
                        rows.append(dict(**base, domain=d, task=task, source=source,
                            raw_logit=float(model.mix_logits[d, task, source]),
                            effective_logit=float(v[d, task, source]), prior_weight=float(prior[source]),
                            source_weight=float(w[source]), entropy=entropy,
                            confidence=float(model.interaction_confidence[d, task, source])))
    target = OUT/'dynamics'
    target.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(target/'router_weights.csv', index=False)
    pd.DataFrame(margins).to_csv(target/'router_subspace_margins.csv', index=False)
    dump(target/'routing_scope.json', dict(optimization_steps=0, test_read=False,
        method='Use implemented Model.router_logits and actual selected/final checkpoint; fixed prior and softmax as training forward.',
        limitations='Centered logits do not imply centered probabilities. Entropy/source preference alone is not evidence of useful expert credit.'))
    print('Routing checkpoint cell/source records:', len(rows))


if __name__ == '__main__':
    main()
