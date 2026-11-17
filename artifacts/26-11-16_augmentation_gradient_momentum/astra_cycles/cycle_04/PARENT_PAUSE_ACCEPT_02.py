"""Direct parent verification of Cycle04's second quiescent pause."""
from datetime import datetime,timezone
import json
from experiments.completion_weight_attribution_20261117 import protocol as p,runner as r


def main():
    n=p.ATTEMPT
    assert not p.stop_requested() and (n/'PAUSE_TRAINING').exists()
    m=p.validate_attempt(n,True);status=p.request(n,'status');owner=p.owner_binding(n)
    assert owner['pid']==2476891 and owner['tool_session']=='67406'
    physical=[json.loads(line) for line in (n/'physical_attempts.jsonl').read_text().splitlines()]
    pids={x['pid'] for x in physical if x.get('purpose')=='formal' and x['status']=='STARTED'}
    assert all(not r._alive(pid) for pid in pids) and not r._alive(2611647)
    rows=[];epochs=updates=completed=0
    for dataset in p.DATASETS:
        for arm in p.ARMS:
            for seed in p.SEEDS:
                j=p.job(dataset,arm,seed,m['source_hash']);key=j['key'];record=status['jobs'][key]
                assert record['job']==j and record['status'] in ('COMPLETE','PAUSED','FAILED')
                got=p.request(n,'get',key=key);state=got['state'];assert state is not None
                records=p.epoch_files(n,dict(job=j));e=record['epoch']
                assert set(records)==set(range(1,e+1))
                row=dict(key=key,dataset=dataset,arm=arm,seed=seed,status=record['status'],last_ack=e,
                    attempts=record['attempts'],verified_epoch_ACKs=len(records))
                if record['status']=='COMPLETE':
                    result=got['result'];assert e==50 and result==p.read_json(n/'runs'/key/'result.json')
                    assert set(state)=={'selected_models','selected_epochs','selected_validation'}
                    hashes={k:p.state_digest(v) for k,v in state['selected_models'].items()}
                    assert hashes==result['selected_hashes'] and state['selected_epochs']==result['selected_epochs']
                    assert state['selected_validation']==result['selected_validation'];p.verify_selection(result,records)
                    row.update(selected_hashes=hashes,selected_epochs=state['selected_epochs'],result_sha256=p.sha_file(n/'runs'/key/'result.json'))
                    completed+=1
                else:
                    r._verify_saved_ack(n,j,state)
                    assert state['epoch']==e and state['global_step']==e*p.STEPS[dataset]
                    row.update(continuation_hash=p.state_digest(state),scientific_identity_hash=p.scientific_identity(state),
                        counts=state['counts'],global_step=state['global_step'])
                last=p.read_json(n/'acks'/key/f'epoch_{e:02d}.json')
                row.update(last_ack_sha256=p.sha_file(n/'acks'/key/f'epoch_{e:02d}.json'),
                    epoch_file=last['epoch_file'],epoch_file_sha256=last['epoch_file_sha256'],
                    transaction_file=last['transaction_file'],transaction_sha256=last['transaction_sha256'])
                rows.append(row);epochs+=e;updates+=e*p.STEPS[dataset]
                print(json.dumps(dict(verified_key=key,epoch=e,status=record['status'])),flush=True)
                del got,state,records
    assert (len(rows),completed,epochs,updates)==(20,15,889,32251)
    assert status['retained_states']==20
    value=dict(status='ACCEPTED_QUIESCENT_PAUSE',source_hash=m['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
        owner=owner,pause_sha256=p.sha_file(n/'TECHNICAL_PAUSE.json'),
        preparation_review_sha256=p.sha_file(n/'RECOVERY_REVIEW_PREPARATION_02.json'),rows=rows,
        completed_trajectories=completed,verified_epoch_ACKs=epochs,useful_updates=updates,
        remaining_epochs=1000-epochs,remaining_updates=34500-updates,
        retained_states=status['retained_states'],retained_bytes=status['retained_bytes'],
        all_formal_workers_exited=True,controller_exited=True,time_utc=datetime.now(timezone.utc).isoformat(),
        test_accessed=False,hash_policy='allow',context_reuse=False)
    p.immutable(n/'PARENT_FORMAL_PAUSE_ACCEPTANCE_02.json',value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('rows','owner')}),flush=True)


if __name__=='__main__':main()
