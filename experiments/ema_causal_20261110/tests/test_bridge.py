import copy

import numpy as np

from experiments.ema_causal_20261110.bridge import (
    SEEDS, _ridge, action_order, choose, run_bridge, seed_summary,
)


def sample_records():
    records = []
    for menu in ('rho7', 'mechanism7'):
        for seed in SEEDS:
            for epoch in range(5, 51, 5):
                for i, action in enumerate(action_order(menu)):
                    utility = 0. if i == 0 else (.001 if i == 1 else -.001*i)
                    records.append(dict(dataset='movielens', objective='clean', seed=seed,
                        epoch=epoch, menu=menu, action_id=action,
                        metrics={p: [dict(auc=.7+utility+c*.001, logloss=.5-utility) for c in range(6)]
                                 for p in ('P_A', 'P_B', 'V_A', 'V_B', 'V_mech')},
                        geometry=dict(r_rms=.01, delta_rms=i*.001, cosine=.2,
                                      is_zero_R=0, is_zero_delta=int(i==0),
                                      mean_cell_rms_zdiff=i*.002, group_rank_gradient_rms=.03)))
    return records


def test_ridge_intercept_and_penalty_are_exact():
    x = np.array([[0., 0.], [1., 1.], [0., 2.], [1., 3.]])
    y = np.array([1., 2., 4., 8.])
    model = _ridge(x, y, [1])
    z = np.column_stack((np.ones(4), (x-model['mean'])/model['scale']))
    expected = np.linalg.solve(z.T@z+np.diag([0., 1., 1.]), z.T@y)
    np.testing.assert_allclose(model['beta'], expected)
    constant = _ridge([[0.], [1.], [2.]], [3., 3., 3.], [0])
    np.testing.assert_allclose(constant['beta'], [3., 0.], atol=1e-15)
    assert choose({'F':0., 'x':0., 'y':-1.}, ('F','x','y')) == 'F'
    assert choose({'F':0., 'x':1., 'y':1.}, ('F','x','y')) == 'x'
    assert choose({'F':0., 'x':None}, ('F','x')) is None


def test_loso_clock_values_and_seed_statistics():
    result = run_bridge(sample_records())
    assert len(result['BRIDGE_FOLDS']) == 20
    assert all(f['training_rows']==72 and f['temporal_units']==12 for f in result['BRIDGE_FOLDS'])
    assert all(f['train_targets']==[10,15,20] and f['evaluation_origins']==[25,30,35,40,45]
               for f in result['BRIDGE_FOLDS'])
    direct = [r for r in result['BRIDGE_VALUE'] if r['selector']=='C_carry_direct']
    assert all(r['fully_static'] and r['action_entropy']==0. for r in direct)
    assert all(abs(r['utility']-.001)<1e-14 and r['delta_static']==0. for r in direct)
    assert all(r['status']=='NOT_TRIGGERED' for r in result['H2_TRIGGER'])
    summary = seed_summary({s:i for i,s in enumerate(SEEDS)})
    assert summary['mean']==2 and summary['n']==5
    assert abs(summary['sd']-np.std(range(5),ddof=1))<1e-14
    assert seed_summary({2022:1})['status']=='INCOMPLETE'


def test_heldout_and_validation_never_fit_or_scale():
    records = sample_records(); before = run_bridge(records)
    changed = copy.deepcopy(records)
    for row in changed:
        # The held-out seed and every validation population may change freely.
        for population, cells in row['metrics'].items():
            if row['seed']==2022 or population!='P_A':
                for c in cells:
                    c['auc'] += .01 if row['action_id']=='F' else -.02
        if row['seed']==2022:
            row['geometry']['r_rms'] *= 100
    after = run_bridge(changed)
    a = [f for f in before['BRIDGE_FOLDS'] if f['holdout_seed']==2022]
    b = [f for f in after['BRIDGE_FOLDS'] if f['holdout_seed']==2022]
    assert a == b
    # Missing current features cannot be imputed or select a reduced menu.
    missing = sample_records()
    for row in missing:
        if row['seed']==2022 and row['epoch']==25 and row['action_id']=='all/fast':
            row['geometry']['r_rms'] = None
    result = run_bridge(missing)
    rows = [r for r in result['BRIDGE_PREDICTIONS'] if r['holdout_seed']==2022
            and r['origin_epoch']==25 and r['menu']=='rho7' and r['predictor']=='C_simple_adapted']
    assert rows and all(r['selected_action'] is None for r in rows)
