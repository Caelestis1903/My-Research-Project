"""Synthetic arithmetic and gate checks; no data loads, training or model evaluation."""
import csv

import pytest

from experiments.conditional_history_shared_update_20261122 import results as r


def records(gains):
    rows = []
    for d in r.DATASETS:
        for arm, offsets in gains.items():
            for seed in r.SEEDS:
                key = f'{d}/{arm}/{seed}'
                row = dict(dataset=d, arm=arm, seed=seed, split='validation', readout='fixed_ema',
                           checkpoint_rule='selected', epoch=15, model_hash=r.digest(key),
                           trajectory_id=key, source_ref='synthetic', reuse_status='NEW_TRAIN')
                for i,c in enumerate(r.CELLS):
                    row[c+'_auc'] = .7 + (offsets if isinstance(offsets,float) else offsets[i])
                    row[c+'_logloss'] = .3
                    row[c+'_count'] = 2+i//2
                rows.append(row)
    return rows


def test_paired_units_and_sealing():
    l = dict(zip(r.SEEDS,(-1,0,1,2,3)))
    z = dict.fromkeys(r.SEEDS,0)
    stats = r.paired_stats(l,z)
    assert (stats['mean_delta'],stats['positive'],stats['negative'],stats['zero']) == (1,3,1,1)
    assert stats['ci95_low'] == pytest.approx(1 - 2.776445 / 2**.5)
    with pytest.raises(ValueError):
        r.paired_stats({2022:1},z)
    rows = records(dict.fromkeys(r.CORE,0.))
    with pytest.raises(ValueError,match='population'):
        r.index_rows([{**rows[0],'split':'test'}])
    with pytest.raises(ValueError,match='Duplicate'):
        r.index_rows([rows[0],rows[0]])
    decision = r.stage_decision('A_CORE',r.index_rows(rows[:-1]))
    assert decision['status']=='TECHNICAL_BLOCKED' and not decision['next_stages']
    assert r.stage_decision('A_CORE',r.index_rows(rows))['datasets']['movielens']['signal']=='NO_CONTINUATION_SIGNAL'


def test_guard_selection_dose_scope_and_render(tmp_path):
    gains = dict(A00=0.,A10=.001,A20=.006,A30=.001,A22=.00595,A21=.00599)
    rows = records(gains)
    core = r.write_stage('A_CORE',rows,tmp_path/'core')
    assert set(core['next_stages'])=={'A_HISTORY'}
    assert (tmp_path/'core/paired_macro_auc.png').read_bytes().startswith(b'\x89PNG')
    with (tmp_path/'core/paired_seed.csv').open() as f:
        pairs=list(csv.DictReader(f))
    seq = [p for p in pairs if p['dataset']=='movielens' and p['comparison_id']=='A20-A10'
           and p['metric']=='auc' and p['cell']=='macro']
    assert len(seq)==5 and [float(p['delta']) for p in seq] == pytest.approx([.005]*5)
    assert not any(p['cell']=='pooled_task_mean' for p in pairs)
    history = r.stage_decision('A_HISTORY',r.index_rows(rows),core)
    assert history['history_form']=='A22' and not history['next_stages']
    assert history['target_dataset']=='kuairand'  # lexical tie
    gains.update(A22=.001,A21=.001,A20=[-.002,.01,.01,.01,.01,.01])
    # No protected history superior to protected Adam. An unprotected high macro cannot supersede it.
    assert 'history_form' not in r.stage_decision('A_HISTORY',r.index_rows(records(gains)),core)
    gains['A10']=[-.002,.003,.003,.003,.003,.003]
    gains['A21']=gains['A22']=gains['A10']
    history = r.stage_decision('A_HISTORY',r.index_rows(records(gains)),core)
    assert history['history_form']=='A20' and history['next_stages']==['A_DOSE']
    gains[r.DOSE[0]]=[-.0015,.012,.012,.012,.012,.012]
    gains[r.DOSE[1]]=gains['A10']
    index = r.index_rows(records(gains))
    pending = r.stage_decision('A_DOSE',index,history)
    assert pending['status']=='TECHNICAL_BLOCKED' and not pending['next_stages']
    localization = [
        dict(dataset=d, arm=r.DOSE[0],history_form='A20', split='validation',readout='fixed_ema',
             action='whole_group_to_adam',group=group,seed=s,anchor_epoch=e,scale=scale,source_ref='synthetic',
             **{c+'_delta_auc':.001 for c in r.CELLS})
        for d in r.DATASETS for group in ('STAR_SKIP','EXPERT') for s in r.SEEDS
        for e in (5,20,35,50) for scale in ('raw','norm_matched')]
    dose = r.stage_decision('A_DOSE',index,history,localization)
    assert dose['next_stages']==['A_SCOPE'] and dose['datasets']['movielens']['matched_adam']==r.DOSE[1]
    gains.update({r.SCOPE[0]:.015,r.SCOPE[1]:.006})
    scope = r.stage_decision('A_SCOPE',r.index_rows(records(gains)),dose)
    assert scope['scope']=='STAR_SKIP' and scope['target_dataset']==history['target_dataset']
    gains['A10']=.00595
    gains['A20']=.006
    gains['A21']=gains['A22']=.002
    no_h = r.stage_decision('A_HISTORY',r.index_rows(records(gains)),core)
    assert all(x['preferred']=='A10' for x in no_h['datasets'].values())


def test_freeze_identity_resources():
    rows = records(dict(A10=.001,A00=.0,PUB=.0))
    roles = {d:dict(candidate=('A10','fixed_ema'),strong_internal=('A00','fixed_ema'),
                    strongest_published=('PUB','fixed_ema')) for d in r.DATASETS}
    comp={d:[('candidate','strong_internal'),('candidate','strongest_published')] for d in r.DATASETS}
    packet = r.freeze_identities(rows,roles,comp,{d:{'lambda_pair':.05} for d in r.DATASETS},[__file__])
    assert len(packet['selected'])==30 and len(packet['identity_sha256'])==64
    assert {x['model_hash'] for x in packet['selected']}=={x['model_hash'] for x in rows}
