"""Finite selected/final slot removal on fixed train-meta rows; no optimization."""
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
    for dataset in ['movielens', 'kuairand']:
        data = population(dataset, 'natural')
        rng = np.random.default_rng(191903)
        ix = np.concatenate([rng.permutation(data.meta[data.train_x[data.meta, -1] == d])[:512] for d in range(3)])
        x = torch.tensor(data.train_x[ix], dtype=torch.float32).T
        y = torch.tensor(data.train_y[ix])
        domains = data.train_x[ix, -1].astype(int)
        for method in ['fixed', 'rank_uniform', 'rank_protected', 'domain_codebook', 'cell_codebook']:
            for run in sorted((OUT/'runs').glob(f'natural__{dataset}__{method}__seed*')):
                if not (run/'receipt.json').exists():
                    continue
                config = json.loads((run/'config.json').read_text())
                if config['engineering']:
                    continue
                seed = config['seed']
                tag = run.name.split('__seed')[1].split('__', 1)[1] if '__' in run.name.split('__seed')[1] else ''
                model = Model(data.features, seed, method).eval()
                for stage, name in [('epoch1', 'epoch001.pt'), ('epoch5', 'epoch005.pt'), ('epoch10', 'epoch010.pt'), ('selected', 'selected.pt'), ('final', 'last.pt')]:
                    state = torch.load(Path(config['state_directory'])/name, map_location='cpu', weights_only=False)
                    model.load_state_dict(state['model'])
                    p = model(x)
                    slots = model.slots()
                    if isinstance(slots, (list, torch.nn.ParameterList)):
                        slots = torch.stack(list(slots))
                    slot_norm = float(slots.double().norm())
                    shared_norm = float(model.base.shared_weight.double().norm())
                    original_slots = model.slots
                    model.slots = lambda masks=None: torch.zeros_like(slots)
                    removed = model(x)
                    model.slots = original_slots
                    loss = torch.nn.functional.binary_cross_entropy(p, y, reduction='none')
                    loss_removed = torch.nn.functional.binary_cross_entropy(removed, y, reduction='none')
                    for d in range(3):
                        for task in range(2):
                            where = domains == d
                            rows.append(dict(run_id=run.name, dataset=dataset, method=method, tag=tag, seed=seed, stage=stage,
                                epoch=state['epoch'], domain=d, task=task, n=int(where.sum()),
                                slot_frobenius_norm_float64=slot_norm, shared_weight_frobenius_norm_float64=shared_norm,
                                baseline_logloss=float(loss[where, task].mean()),
                                all_slots_removed_loss_change=float((loss_removed-loss)[where, task].mean()),
                                probability_absolute_change=float((removed-p).abs()[where, task].mean())))
    target = OUT/'dynamics'
    target.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(target/'rank_finite_removal.csv', index=False)
    dump(target/'rank_finite_removal_scope.json', dict(
        data='Fixed train-meta rows, up to512/domain, seed191903, shared across all arms/seeds',
        intervention='Set generated slot matrices to zero; shared matrix, skip, biases and all other modules retained',
        optimization_steps=0, test_read=False,
        limitation='Conditional checkpoint sensitivity, not a trained no-slot ablation or proof of causal importance. Train-meta is not an untouched generalization sample.'))
    print('Finite slot-removal cell measurements:', len(rows))


if __name__ == '__main__':
    main()
