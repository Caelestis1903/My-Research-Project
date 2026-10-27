import json
import numpy as np
import pytest
from experiments.mdmt_relation_basis_v2.config import METHODS, DATASETS, SEEDS, load_config
from experiments.mdmt_relation_basis_v2.data import Data
from experiments.mdmt_relation_basis_v2.report import collect, freeze_manifest, release_evaluation
from experiments.mdmt_relation_basis_v2.state import write_json, source_identity


def test_seed_summary_and_full_freeze_quorum(tmp_path):
    cfg = load_config()
    values = dict(zip(METHODS, [.50, .49, .48, .47, .469, .468]))
    identities = {}
    source_hash = source_identity()['source_hash']
    for dataset in DATASETS:
        for seed in SEEDS:
            selected = {method:f'{dataset}_{seed}_{method}' for method in METHODS}
            identity = dict(dataset=dataset, seed=seed, config_hash=cfg['config_hash'],
                            source_hash=source_hash, selected=selected)
            identities[f'{dataset}/seed{seed}'] = identity
            write_json(tmp_path/'workers'/dataset/f'seed{seed}'/'READY.json', dict(status='READY_FOR_GLOBAL_FREEZE', identity=identity))
            for method in METHODS:
                root = tmp_path/'runs'/dataset/method/f'seed{seed}'
                ll = values[method]
                metrics = dict(macro_logloss=ll, natural_logloss=ll, macro_auc=.7,
                               pooled_auc_by_task=[.7, .7], valid_requests=24,
                               cells=[dict(cell=3*t+d, domain=d, task=t, n=4, positive=2, negative=2,
                                           auc=.7, logloss=ll) for t in range(2) for d in range(3)])
                write_json(root/'run_receipt.json', dict(scores={'Q_selected':metrics}, training_seconds=1.))
                for state in ('selected', 'final'):
                    write_json(root/f'Q_{state}_metrics.json', dict(identity=dict(epoch=5 if state == 'selected' else 20,
                                                                              model_hash=selected[method]), metrics=metrics))
                write_json(root/'diagnostics.json', dict(formation_evidence=method == 'LOWRANK_RELATION'))
    report = collect(tmp_path)
    assert report['completed_adaptation_runs'] == 48
    assert report['development_gates']['movielens']['Cstar'] == 'TARGET_ONLY'
    assert report['development_gates']['movielens']['GUARD']
    assert report['development_gates']['movielens']['PRACTICAL']
    main = next(x for x in report['paired'] if x['dataset'] == 'movielens' and x['method'] == 'LOWRANK_RELATION' and x['state'] == 'selected')
    assert main['statistics']['macro_logloss']['n'] == 4
    assert abs(main['statistics']['macro_logloss']['mean']+.001) < 1e-12
    decision_path = tmp_path/'decision.json'
    write_json(decision_path, dict(status='FINAL_METHODS_FROZEN', stage2={d:'NOT_TRIGGERED' for d in DATASETS}))
    missing = tmp_path/'workers'/'kuairand'/'seed2025'/'READY.json'
    ready = missing.read_text()
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        freeze_manifest(tmp_path, cfg, decision_path)
    assert not (tmp_path/'GLOBAL_FREEZE.json').exists()
    missing.write_text(ready)
    freeze_manifest(tmp_path, cfg, decision_path)
    manifest = json.loads((tmp_path/'GLOBAL_FREEZE.json').read_text())
    data = Data.__new__(Data)
    data.dataset = 'movielens'
    data.authorize_final(manifest, identities['movielens/seed2022'])
    assert data.outer_frozen is not None
    release_evaluation(tmp_path, 'movielens', 2022)
    assert (tmp_path/'workers'/'movielens'/'seed2022'/'EVALUATE.json').exists()

