"""S5 checks use scalar fixtures and small in-memory tensors only."""
import csv

import pytest
import torch

from experiments.conditional_history_shared_update_20261122 import final_evaluation as f


def metrics(value=.7):
    return {c+k:v for c in f.results.CELLS for k,v in
            [('_auc',value),('_count',10),('_positive',4),('_logloss',.4)]} | {'macro_auc':value}


def row(arm='A10',seed=2022,d='movielens'):
    return dict(dataset=d,arm=arm,seed=seed,readout='online' if arm in f.PUBLISHED else 'fixed_ema',
                epoch=7,model_hash=f.digest([d,arm,seed]),trajectory_id=f.digest([arm,d,seed]),
                source_hash='src',config_hash='cfg',data_hash='data',reuse_status='NEW_TRAIN',
                source_ref='fixture',split='validation',checkpoint_rule='selected',**metrics())


def selection(candidate='A21'):
    branches={s:{'status':'COMPLETE' if s=='A_HISTORY' else 'NOT_TRIGGERED'} for s in f.BRANCHES}
    return dict(status='FINAL_CANDIDATE',branches=branches,datasets={d:dict(candidate=candidate,
        parameters={'lambda_pair':.05},matched_adam='A10',strong_internal='A00',
        strongest_published=f.STRONG[d]) for d in f.DATASETS})


def test_immutable_once_and_ambiguous_access(tmp_path):
    r=row()
    calls=[]
    def score(mark):
        mark()
        calls.append(1)
        return metrics(),{'test_evaluations':1}
    first=f.once(tmp_path,'freeze',r,score)
    assert f.once(tmp_path,'freeze',r,score)==first and len(calls)==1
    with pytest.raises(ValueError,match='identity'):
        f.once(tmp_path,'another',r,score)
    def crash(mark):
        mark()
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        f.once(tmp_path,'freeze',row('A22'),crash)
    with pytest.raises(ValueError,match='Indeterminate'):
        f.once(tmp_path,'freeze',row('A22'),score)
    with pytest.raises(ValueError,match='Immutable'):
        f.immutable(tmp_path/(f.digest(f.state_id(r))+'.json'),{'other':1})


def test_earliest_selection_and_current_guardian(tmp_path,monkeypatch):
    from experiments.conditional_history_shared_update_20261122 import state as guardian_state
    from experiments.conditional_history_shared_update_20261122.protocol import write
    r=row()
    key=r['trajectory_id']
    attempt=tmp_path/'attempt1'
    path=attempt/'runs'/key/'result.json'
    write(attempt/'guardian.json',dict(address='fixture',pid=123))
    owner=f.digest(['fixture',123])
    ident=owner[:10]+'-attempt1'
    config={'lambda_pair':.05}
    r['config_hash']=f.digest(config)
    weights={'p':torch.tensor([1.,3.])}
    r['model_hash']=f.v3.state_digest(weights)
    job={'dataset':r['dataset'],'arm':r['arm'],'seed':r['seed'],'key':key,'source_hash':'src','config':config}
    chosen={'fixed_ema':r['epoch']}
    res=dict(job=job,epochs=50,cost_class='NEW_SCIENTIFIC_TRAIN',owner_generation=owner,attempt_id=ident,
             rows=[r],selected_epochs=chosen,selected_hashes={'fixed_ema':r['model_hash']},
             data_hash='data',group_hash='groups')
    write(path,res)
    write(path.with_name('COMPLETE_VERIFIED.json'),dict(result_hash=f.digest(res),owner_generation=owner))
    for i in range(1,51):
        value=.7 if i in (7,43) else .68
        d=path.parent/ident
        write(d/f'epoch_{i:02}.json',dict(epoch=i,validation={'fixed_ema':metrics(value)},ram_state_hash=str(i)))
        write(d/f'RAM_{i:02}.json',dict(state_hash=str(i)))
    out=f.accepted_result(path,r['dataset'],r['arm'],r['seed'])
    state=dict(selected_epochs=chosen,selected_validation={'fixed_ema':metrics()},selected_models={'fixed_ema':weights})
    registry={key:dict(job=job,epoch=50,status='COMPLETE',result=res)}
    def request(location,cmd,**kwargs):
        assert location==attempt
        if cmd=='status':
            return dict(pid=123,jobs=registry)
        assert cmd=='get' and kwargs=={'key':key}
        return dict(result=res,state=state)
    monkeypatch.setattr(guardian_state,'request',request)
    monkeypatch.setattr(f,'request',request)
    assert torch.equal(f.selected_ram(out)['p'],weights['p'])
    weights['p'][0]=4
    with pytest.raises(ValueError,match='RAM hash'): f.selected_ram(out)
    write(path.parent/ident/'epoch_03.json',dict(epoch=3,validation={'fixed_ema':metrics()},ram_state_hash='3'))
    with pytest.raises(ValueError,match='earliest'): f.accepted_result(path,r['dataset'],r['arm'],r['seed'])


def test_matrix_and_report(tmp_path,monkeypatch):
    proposal=selection()
    packet=dict(status='FINAL_CANDIDATE',selection=proposal,selected=[],comparisons={})
    for d in f.DATASETS:
        arms,pairs=f.matrix(proposal,d)
        assert set(arms)>=set((*f.PUBLISHED,*f.results.CORE,'A21','A22'))
        assert ('A21','A10') in pairs and ('A21',f.STRONG[d]) in pairs
        packet['comparisons'][d]=pairs
        for a in arms:
            for s in f.SEEDS:
                packet['selected'].append(row(a,s,d))
    folder=tmp_path/'S5'
    h=f.digest(packet)
    receipts={}
    for r in packet['selected']:
        def score(mark):
            mark()
            return metrics(.705 if r['arm']=='A21' else .7),dict(test_evaluations=1)
        rec=f.once(folder,h,r,score)
        receipts[f.digest(f.state_id(r))]=f.digest(rec)
    f.immutable(folder/'EVALUATED.json',dict(freeze_sha256=h,receipts=receipts))
    monkeypatch.setattr(f,'frozen',lambda art:packet)
    out=f.report(tmp_path)
    assert all(v['status']=='FINAL_STRONG' for v in out['datasets'].values())
    assert (folder/'paired_test_macro_auc.png').is_file()
    with (folder/'paired_summary.csv').open() as fp:
        stats=next(r for r in csv.DictReader(fp) if r['split']=='test' and r['left']=='A21'
                   and r['right']==f.PUBLISHED[0] and r['cell']=='macro' and r['metric']=='auc')
    assert stats['n']=='5' and float(stats['mean_delta'])==pytest.approx(.005) and stats['right_readout']=='online'
    assert stats['positive']=='5' and float(stats['delta_2026'])==pytest.approx(.005)
    proposal['datasets']['movielens']['parameters']['lambda_pair']=.025
    with pytest.raises(ValueError,match='dose'): f.matrix(proposal,'movielens')


def test_closed_gate_and_no_candidate(tmp_path,monkeypatch):
    with pytest.raises(FileNotFoundError): f.freeze(tmp_path)
    packet={'status':'NO_CANDIDATE','selection':{'reason':'fixture'},'evidence':{}}
    monkeypatch.setattr(f,'frozen',lambda p:packet)
    monkeypatch.setattr(f,'historical',lambda *args:[dict(row(),test_exposure='HISTORICAL_EXPOSED')])
    monkeypatch.setattr(f,'selected_ram',lambda *args:pytest.fail('Should not read selected state'))
    assert f.report(tmp_path)['new_test_evaluations']==0
    with pytest.raises(ValueError,match='No candidate'): f.evaluate(tmp_path,torch.device('cpu'))


def test_reviewed_gaussian_control_gap_only_allows_the_unaugmented_selection(tmp_path,monkeypatch):
    from experiments.conditional_history_shared_update_20261122.protocol import write
    report=tmp_path/'report'
    monkeypatch.setattr(f,'REPORT',report)
    trigger=tmp_path/'G_FROZEN_TRIGGER.json'
    audit=tmp_path/'SOURCE_G_CONTROL_AUDIT.json'
    receipt=report/'G_CONTROL_REVIEW_RECEIPT.json'
    write(trigger,dict(status='TRIGGERED',split='validation',test_used_for_selection=False,unaugmented='A21'))
    control=dict(stage='G_FROZEN',status='TECHNICAL_BLOCKED',reason='Unmatched control',
                 author='auditor',trigger_sha256=f.sha(trigger),
                 exact_control_seeds={'movielens':[2022,2024],'kuairand':[]})
    write(audit,control)
    review=dict(stage='G_CONTROL',verdict='PASS',blocking_issues=[],reviewer='reviewer',audit_sha256=f.sha(audit))
    write(receipt,review)
    decision=dict(status='TECHNICAL_BLOCKED',reason=control['reason'],
                  evidence={str(p):f.sha(p) for p in (audit,receipt)})
    proposal=selection()
    evidence={}
    f.blocked_gaussian(tmp_path,proposal,decision,evidence)
    assert set(evidence)=={str(p) for p in (audit,receipt,trigger)}
    with pytest.raises(ValueError,match='stay unaugmented'):
        f.blocked_gaussian(tmp_path,selection('G_FROZEN'),decision,{})
    with pytest.raises(ValueError,match='Missing Gaussian control evidence'):
        f.blocked_gaussian(tmp_path,proposal,dict(decision,evidence={}),{})
    write(receipt,dict(review,reviewer=control['author']))
    decision['evidence'][str(receipt)]=f.sha(receipt)
    with pytest.raises(ValueError,match='audit/review mismatch'):
        f.blocked_gaussian(tmp_path,proposal,decision,{})
    write(audit,dict(control,exact_control_seeds={d:list(f.SEEDS) for d in f.DATASETS}))
    write(receipt,dict(review,audit_sha256=f.sha(audit)))
    decision['evidence']={str(p):f.sha(p) for p in (audit,receipt)}
    with pytest.raises(ValueError,match='Missing Gaussian control gap'):
        f.blocked_gaussian(tmp_path,proposal,decision,{})


def test_evaluate_reuses_archive_and_only_scores_unconsumed_ram(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    model=torch.nn.Linear(1,1)
    weights=model.state_dict()
    new=row()
    new.update(data_hash=f.digest('synthetic population'),model_hash=f.v3.state_digest(weights))
    archive=dict(row('A20'),reuse_status='REUSE_NUMERIC',selection_id='original',source_file_sha256='pinned')
    packet=dict(status='FINAL_CANDIDATE',selected=[archive,new])
    monkeypatch.setattr(f,'frozen',lambda _:packet)
    monkeypatch.setattr(f,'historical',lambda *args:[dict(archive,split='test')])
    monkeypatch.setattr(f,'selected_ram',lambda _:weights)
    monkeypatch.setattr(f.v3,'load_protocol',lambda:{'datasets':{'movielens':{'source':'fixture'}}})
    monkeypatch.setattr(f.v3,'load_dataset',lambda *args,**kwargs:(
        SimpleNamespace(test_x=np.ones((3,1)),test_y=np.zeros((3,2))),'synthetic population'))
    monkeypatch.setattr(f.v3,'setup',lambda *args:None)
    monkeypatch.setattr(f.v3,'make_model',lambda *args:model)
    calls=[]
    def mock_eval(*args):
        calls.append(1)
        return metrics(),dict(rows=3,forwards=1,seconds=.1)
    monkeypatch.setattr(f.v3,'evaluate',mock_eval)
    out=f.evaluate(tmp_path,torch.device('cpu'))
    assert len(out['receipts'])==2 and calls==[1]
    assert f.evaluate(tmp_path,torch.device('cpu'))==out and calls==[1]
    recpath=tmp_path/'S5'/(f.digest(f.state_id(new))+'.json')
    rec=f.payload(recpath)
    rec['identity']['seed']=2026
    recpath.unlink()
    f.immutable(recpath,rec)
    with pytest.raises(ValueError,match='changed'):f.evaluate(tmp_path,torch.device('cpu'))
