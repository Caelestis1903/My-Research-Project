from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import torch
from experiments.ema_causal_20261110.block import _selection, _time_groups, _descriptor, _contrast_candidates


def test_five_seed_budget_selection_uses_late_V_A_only():
    tables=defaultdict(list)
    for seed in range(2022,2027):
        for epoch in ('E35','E50'):
            for name,value in (('F',.75),('B/N',.7497),('B/S',.7499)):
                for cell in range(6): tables['INTERVENTION_METRICS'].append(dict(seed=seed,anchor=epoch,descriptor_id=name,population='V_A',auc=value))
            for cell in range(6): tables['FACTORIAL_EFFECTS'].append(dict(seed=seed,anchor=epoch,contrast_id='B/SxN',population='V_A',metric='auc',interaction=.00001))
    result=_selection(tables)
    assert result['selected']==['S']
    assert result['candidates']==['EMBED','STAR_SKIP','EXPERT','N']
    for row in tables['FACTORIAL_EFFECTS']: row['interaction']=.0003
    assert _selection(tables)['selected']==['S','N']
    assert 'B/N' in _contrast_candidates(['EMBED','N'],result)
    tables['INTERVENTION_METRICS']=[r for r in tables['INTERVENTION_METRICS'] if r['seed']!=2026]
    try: _selection(tables)
    except RuntimeError: pass
    else: raise AssertionError('Missing seed must block budget decisions')


def test_group_time_two_backgrounds_use_same_actual_correction(monkeypatch):
    monkeypatch.setattr('experiments.ema_causal_20261110.block._factorial_function',lambda *args:None)
    online=torch.tensor([1.,2.,3.,4.]);fixed=torch.tensor([2.,4.,6.,8.]);r=fixed.double()-online.double()
    old=r*.25;mask=torch.tensor([True,False,True,False]);seen={}
    anchor=SimpleNamespace(job={'dataset':'movielens'},packet=dict(step=100,online_fp32=online,F_fp32=fixed,R_fp64=r,old_R_fp64_by_effective_h={40:old}),masks={'S':mask})
    anchor.evaluate=lambda name,vector,descriptor:seen.update({name:vector})
    anchor.factorial=lambda *args,**kwargs:None
    _time_groups(anchor,['S'])
    assert torch.equal(seen['C2/S/O/zero'],online)
    assert torch.equal(seen['C2/S/F/full'],fixed)
    for background in ('O','F'):
        a,b,c,d=[seen[f'C2/S/{background}/{component}'].double() for component in ('zero','recent','old','full')]
        torch.testing.assert_close(d-a,(b-a)+(c-a),rtol=0,atol=0)
        assert torch.equal(a[~mask],online[~mask] if background=='O' else fixed[~mask])


def test_discovery_bounds_reused_by_confirmation():
    from experiments.ema_causal_20261110.block import _discovery_bounds
    y=np.tile(np.array([1,1,0,0]),(6,1))
    panels={'P_A':{'rank_y':y},'P_B':{'rank_y':y}}
    reference={'P_A':{'rank_p':np.tile([.8,.6,.3,.2],(6,1)),'rank_z':np.zeros((6,4))},
               'P_B':{'rank_p':np.tile([.9,.9,.8,.8],(6,1)),'rank_z':np.zeros((6,4))}}
    job={'dataset':'movielens','seed':2022}
    a,aid=_discovery_bounds(job,panels,reference,'P_A',0,0)
    b,bid=_discovery_bounds(job,panels,reference,'P_B',0,0)
    assert a==b and aid==bid and aid['discovery_population']=='P_A'
    reference['P_B']['rank_p'][:]=0
    assert _discovery_bounds(job,panels,reference,'P_B',0,0)==(a,aid)
