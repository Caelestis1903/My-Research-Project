"""Small deterministic checks for rules that change scientific selection."""
import json
import pytest
from .. import decisions as d
from ..protocol import config, write_json


def record(dataset,arm,seed,value=.7,**params):
    metrics={k:value for k in d.HARM};metrics.update(macro_auc=value,macro_logloss=.5)
    return dict(job=dict(dataset=dataset,seed=seed,config=config(arm,dataset,**params)),selected_validation=dict(fixed_ema=metrics))


def save(path,stage,decision):write_json(path/'decisions'/f'{stage}.json',decision)


def test_d0_separate_datasets_and_fixed_grid(tmp_path):
    rs=[]
    for dataset in ('movielens','kuairand'):
        for arm in d.load_protocol()['stages']['D0']['arms']:
            rs.append(record(dataset,arm,2022,.701 if dataset=='movielens' and arm=='M' else .7))
    decision=d.decide(tmp_path,'D0',rs)
    assert decision['datasets']['movielens']['family']=='M'
    assert decision['datasets']['kuairand']['next_stage'] is None
    save(tmp_path,'D0',decision);jobs=d.stage_jobs(tmp_path,'D1')
    assert len(jobs)==12 and {j['dataset'] for j in jobs}=={'movielens'}
    assert all(len(g['configs'])==4 for g in decision['datasets']['movielens']['groups'])


def test_2024_does_not_change_frozen_winners(tmp_path):
    winners=[dict(arm='M',lambda_mix=.1,tau=.05),dict(arm='U_MIX',lambda_mix=.1,tau=.2),dict(arm='A0f',lr=.005)]
    prior=dict(datasets=dict(movielens=dict(next_stage='REPLICATE',family='M',winners=winners),kuairand=dict(next_stage=None,status='NO_GAIN')))
    save(tmp_path,'D1',prior)
    rs=[]
    for seed in (2022,2024):
        rs.append(record('movielens','A0f',seed))
        for r in winners:rs.append(record('movielens',r['arm'],seed,.702 if r['arm']!='A0f' else .7005,**{k:v for k,v in r.items() if k!='arm'}))
    decided=d.decide(tmp_path,'REPLICATE',rs)
    assert decided['datasets']['movielens']['winners']==winners
    assert decided['datasets']['movielens']['next_stage']=='D2'
    assert {j['seed'] for j in d.stage_jobs(tmp_path,'REPLICATE')}=={2024}


def test_complex_gate_selects_simple_neighbor(tmp_path):
    main=dict(arm='H_MIX',tau=.1,kappa=1.);simple=dict(arm='H_REAL',kappa=1.);a0=dict(arm='A0f')
    controls=d.matched_controls('H',main,simple)
    prior=dict(datasets=dict(movielens=dict(next_stage='D2',family='H',winners=[main,simple,a0],controls=controls),kuairand=dict(next_stage=None,status='NO_GAIN')))
    save(tmp_path,'REPLICATE',prior);rs=[]
    seen=set()
    for r in [main,simple,a0]+controls:
        for seed in (2022,2024):
            c=config(r['arm'],'movielens',**{k:v for k,v in r.items() if k!='arm'});key=(d.sig(c),seed)
            if key in seen:continue
            seen.add(key);rs.append(record('movielens',r['arm'],seed,.7 if r['arm']=='A0f' else .701,**{k:v for k,v in r.items() if k!='arm'}))
    decision=d.decide(tmp_path,'D2',rs)['datasets']['movielens']
    assert decision['chosen']['arm']=='SCALE'
    assert next(v for v in decision['candidates'] if v['recipe']['arm']=='H_MIX')['joint'] is False
    failed=next(r for r in rs if r['job']['config']['arm']=='H_U' and r['job']['seed']==2024)
    blocked={('movielens',d.sig(failed['job']['config']),2024)}
    decision=d.decide(tmp_path,'D2',[r for r in rs if r is not failed],blocked)['datasets']['movielens']
    assert decision['chosen']['arm']=='SCALE'
    focal=next(v for v in decision['candidates'] if v['recipe']['arm']=='H_MIX')
    assert not focal['joint'] and any(x.get('status')=='IMPLEMENTATION_BLOCKED' for x in focal['components'])


def test_evaluate_rejects_before_freeze(tmp_path):
    from ..run import final_evaluate
    with pytest.raises(FileNotFoundError):final_evaluate(tmp_path,[0])


def test_cost_block_is_exact_configuration_not_family(tmp_path):
    blocked={('movielens','H_MIX')};rs=[]
    for dataset in ('movielens','kuairand'):
        for arm in d.load_protocol()['stages']['D0']['arms']:
            if (dataset,arm) in blocked:continue
            rs.append(record(dataset,arm,2022,.701 if dataset=='movielens' and arm=='H_REAL' else .7))
    decision=d.decide(tmp_path,'D0',rs,blocked)
    assert decision['datasets']['movielens']['family']=='H'
    save(tmp_path,'D0',decision)
    measured={ (r['job']['dataset'],d.sig(r['job']['config'])):r for r in rs }
    for group in decision['datasets']['movielens']['groups']:
        for r in group['configs']:
            if d.blocked_recipe('movielens',r,blocked):continue
            v=record('movielens',r['arm'],2022,.701,**{k:x for k,x in r.items() if k!='arm'})
            measured[('movielens',d.sig(v['job']['config']))]=v
    chosen=d.decide(tmp_path,'D1',list(measured.values()),blocked)['datasets']['movielens']
    assert len(chosen['groups'][0]['group']['configs'])==4
    assert len(chosen['groups'][0]['cost_blocked'])==1
    assert chosen['winners'][0]['tau']==.05
