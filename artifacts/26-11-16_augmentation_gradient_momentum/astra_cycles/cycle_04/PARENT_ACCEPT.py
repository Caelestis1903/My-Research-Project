"""Direct parent acceptance of all completed Cycle04 selected RAM pairs."""
from datetime import datetime,timezone
import json
from pathlib import Path
from experiments.completion_weight_attribution_20261117 import protocol as p,runner as r


def main():
    n=p.ATTEMPT;m=p.validate_attempt(n,True);status=p.request(n,'status')
    assert m['source_hash']==p.source_manifest()['sha256']
    jobs=[p.job(d,a,s,m['source_hash']) for d in p.DATASETS for a in p.ARMS for s in p.SEEDS]
    assert len(jobs)==20
    rows=[];epochs=updates=0
    for j in jobs:
        key=j['key'];record=status['jobs'][key]
        assert record['status']=='COMPLETE' and record['epoch']==50 and not r._alive(record.get('worker_pid'))
        got=p.request(n,'get',key=key);state=got['state'];result=got['result']
        assert set(state)=={'selected_models','selected_epochs','selected_validation'}
        assert result==p.read_json(n/'runs'/key/'result.json') and result['job']==j and result['epochs']==50
        hashes={kind:p.state_digest(value) for kind,value in state['selected_models'].items()}
        assert set(hashes)=={'online','fixed_ema'} and hashes==result['selected_hashes']
        assert state['selected_epochs']==result['selected_epochs'] and state['selected_validation']==result['selected_validation']
        records=p.epoch_files(n,result);assert set(records)==set(range(1,51));p.verify_selection(result,records)
        assert result['counts']==dict(optimizer_updates=50*p.STEPS[j['dataset']],ema_updates=50*p.STEPS[j['dataset']])
        epochs+=50;updates+=result['global_step']
        rows.append(dict(key=key,dataset=j['dataset'],arm=j['arm'],seed=j['seed'],selected_hashes=hashes,
                         selected_epochs=result['selected_epochs'],selected_validation=result['selected_validation'],
                         result_sha256=p.sha_file(n/'runs'/key/'result.json'),verified_epoch_ACKs=50))
        del got,state,result,records
    assert epochs==1000 and updates==34500
    handoff=p.read_json(n/'RUNNER_HANDOFF.json')
    assert handoff['source_hash']==m['source_hash'] and handoff['test_accessed'] is False
    physical=[json.loads(line) for line in (n/'physical_attempts.jsonl').read_text().splitlines()]
    pids={x['pid'] for x in physical if x.get('purpose')=='formal' and x['status']=='STARTED'}
    assert all(not r._alive(pid) for pid in pids)
    assert status['retained_states']==20 and status['retained_bytes']==191017920
    acceptance=dict(status='ACCEPTED',source_hash=m['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
        owner=p.owner_binding(n),rows=rows,verified_selected_models=40,verified_epoch_ACKs=epochs,
        useful_adam_steps=updates,useful_ema_steps=updates,retained_states=status['retained_states'],
        retained_bytes=status['retained_bytes'],all_formal_workers_exited=True,
        handoff_sha256=p.sha_file(n/'RUNNER_HANDOFF.json'),time_utc=datetime.now(timezone.utc).isoformat(),test_accessed=False)
    p.write_json(n/'PARENT_STORAGE_ACCEPTANCE.json',acceptance)
    print(json.dumps({k:acceptance[k] for k in ['status','verified_selected_models','verified_epoch_ACKs','useful_adam_steps','retained_states','retained_bytes','test_accessed']}))


if __name__=='__main__':main()
