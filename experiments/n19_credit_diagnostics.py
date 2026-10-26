"""Conditional finite credit versus local coordinate preference; no optimization."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from n19_data import OUT, population, dump
from n19_models import Model


def entropy(cost, sign=-1):
    q = torch.softmax(sign*cost/.1, 0)
    return float(-(q*q.clamp_min(1e-8).log()).sum())


def main():
    torch.set_num_threads(2)
    rows = []
    samples = {}
    for dataset in ['movielens', 'kuairand']:
        data = population(dataset, 'natural')
        rng = np.random.default_rng(191905)
        halves = [[], []]
        for d in range(3):
            pool = rng.permutation(data.meta[data.train_x[data.meta, -1] == d])
            n = min(256, len(pool)//2)
            halves[0].extend(pool[:n]); halves[1].extend(pool[n:2*n])
        halves = [np.asarray(h) for h in halves]
        samples[dataset] = [dict(n=len(h), per_domain=np.bincount(data.train_x[h, -1].astype(int), minlength=3).tolist(),
            row_ids_sha256=hashlib.sha256(np.asarray(data.train_ids[h]).tobytes()).hexdigest()) for h in halves]
        x = [torch.tensor(data.train_x[h], dtype=torch.float32).T for h in halves]
        y = [torch.tensor(data.train_y[h]) for h in halves]
        domains = [data.train_x[h, -1].astype(int) for h in halves]
        for method in ['full_mix', 'utility_router', 'regret_router', 'coalition_router', 'entropy_router']:
            for seed in [2022, 2023, 2024, 2025]:
                run = OUT/'runs'/f'natural__{dataset}__{method}__seed{seed}'
                if not (run/'receipt.json').exists():
                    continue
                c = json.loads((run/'config.json').read_text())
                # Same full-router state; measurement mode controls finite actions only.
                model = Model(data.features, seed, 'regret_router').eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                model.mix_logits.requires_grad_(True)
                for stage, name in [('selected', 'selected.pt'), ('final', 'last.pt')]:
                    state = torch.load(Path(c['state_directory'])/name, map_location='cpu', weights_only=False)
                    model.load_state_dict(state['model'])
                    individual = [{}, {}]; removal = [{}, {}]; mixed = [{}, {}]
                    weights = {}
                    with torch.no_grad():
                        for mode, target in [('regret_router', individual), ('coalition_router', removal)]:
                            model.method = mode
                            for half in range(2):
                                p, info = model(x[half], details=True)
                                for (d,t,where,alternatives), (_,_,w) in zip(info['source_probs'], info['router_weights']):
                                    target[half][d,t] = F.binary_cross_entropy(alternatives, y[half][where,t,None].expand_as(alternatives), reduction='none').mean(0)
                                    mixed[half][d,t] = float(F.binary_cross_entropy(p[where,t], y[half][where,t]))
                                    weights[d,t] = w.detach().tolist()
                    model.method = 'regret_router'
                    p = model(x[0])
                    for d in range(3):
                        where = domains[0] == d
                        for task in range(2):
                            loss = F.binary_cross_entropy(p[where,task], y[0][where,task])
                            slope = torch.autograd.grad(loss, model.mix_logits, retain_graph=True)[0][d,task].detach()
                            ca,cb = individual[0][d,task],individual[1][d,task]
                            ra,rb = removal[0][d,task],removal[1][d,task]
                            finite = int(ca.argmin()); local = int(slope.argmin()); preserve = int(ra.argmax())
                            rows.append(dict(run_id=run.name,dataset=dataset,method=method,seed=seed,stage=stage,epoch=state['epoch'],domain=d,task=task,
                                n_halfA=int(where.sum()),n_halfB=int((domains[1]==d).sum()),train_n=int((data.train_x[data.main,-1]==d).sum()),
                                finite_choice_A=finite,finite_choice_B=int(cb.argmin()),local_increase_choice_A=local,
                                preserve_choice_A=preserve,preserve_choice_B=int(rb.argmax()),
                                finite_local_agree=finite==local,finite_choice_replicates=finite==int(cb.argmin()),preserve_choice_replicates=preserve==int(rb.argmax()),
                                finite_minus_local_individual_risk_B=float(cb[finite]-cb[local]),
                                finite_minus_mixture_B=float(cb[finite]-mixed[1][d,task]),
                                oracle_B_minus_mixture_B=float(cb.min()-mixed[1][d,task]),
                                chosen_finite_excess_vs_B_min=float(cb[finite]-cb.min()),
                                mixture_loss_A=mixed[0][d,task],mixture_loss_B=mixed[1][d,task],
                                finite_target_entropy_A=entropy(ca),preserve_target_entropy_A=entropy(ra,1),
                                finite_cost_range_A=float(ca.max()-ca.min()),local_derivative_maxabs_A=float(slope.abs().max()),
                                individual_cost_A=json.dumps(ca.tolist()),individual_cost_B=json.dumps(cb.tolist()),
                                removal_cost_A=json.dumps(ra.tolist()),removal_cost_B=json.dumps(rb.tolist()),
                                local_logit_derivative_A=json.dumps(slope.tolist()),source_weights=json.dumps(weights[d,task])))
    target = OUT/'dynamics'
    target.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(target/'credit_decisions.csv', index=False)
    columns=['finite_local_agree','finite_choice_replicates','preserve_choice_replicates',
        'finite_minus_local_individual_risk_B','finite_minus_mixture_B','oracle_B_minus_mixture_B',
        'finite_target_entropy_A','finite_cost_range_A','local_derivative_maxabs_A']
    seedmeans=frame.groupby(['dataset','method','stage','seed'])[columns].mean()
    seedmeans.to_csv(target/'credit_seed_summaries.csv')
    seedmeans.groupby(['dataset','method','stage']).mean().to_csv(target/'credit_decision_summary.csv')
    dump(target/'credit_decision_scope.json', dict(samples=samples,optimization_steps=0,test_performance_used=False,
        limitation='Conditional same-tower finite actions and local coordinate preferences; not separately trained experts or the full utility update. Train-meta previously used by utility; cells/seeds share fixed populations.'))
    print('Credit decision cell measurements:', len(rows))


if __name__ == '__main__':
    main()
