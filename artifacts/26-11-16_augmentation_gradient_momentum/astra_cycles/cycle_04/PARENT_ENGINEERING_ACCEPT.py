"""Parent direct get acceptance of the four current-attempt engineering states."""
from datetime import datetime,timezone
import json
from experiments.completion_weight_attribution_20261117 import protocol as p,runner as r


def main():
    n=p.ATTEMPT;m=p.validate_attempt(n);status=p.request(n,'status');rows=[]
    for i,(device,dataset,arm) in enumerate(r.CASES,1):
        case_id=f'case{i}_gpu{device}_{dataset}'
        paths=list((n/'smoke'/case_id).glob('attempt*/case.json'));assert len(paths)==1
        path=paths[0];case=p.read_json(path);key=case['key'];j=case['job'];record=status['jobs'][key]
        assert (device,dataset,arm)==(case['device'],case['dataset'],case['arm'])
        assert case['source_hash']==m['source_hash'] and case['verified_plan_counts']==p.ENGINEERING_PLAN
        assert case['scientific_epochs_completed']==0 and record['status']=='PAUSED' and record['epoch']==2
        assert not r._alive(record.get('worker_pid'))
        got=p.request(n,'get',key=key);state=got['state'];r._verify_saved_ack(n,j,state)
        assert state['counts']==dict(optimizer_updates=4,ema_updates=4)
        assert state['scientific_epochs_completed']==0 and state['engineering_state_step']==4 and state['engineering_commit_ordinal']==2
        assert p.scientific_identity(state)==case['suffix_scientific_identity_hash']
        rows.append(dict(case_id=case_id,key=key,case_receipt=dict(path=str(path),sha256=p.sha_file(path)),
            continuation_hash=p.state_digest(state),scientific_identity_hash=p.scientific_identity(state),
            counts=state['counts'],engineering_state_step=4,engineering_commit_ordinal=2,scientific_epochs_completed=0))
        del got,state
    assert status['retained_states']==4 and status['retained_bytes']==114653192
    physical=[json.loads(line) for line in (n/'physical_attempts.jsonl').read_text().splitlines()]
    assert all(not r._alive(x['pid']) for x in physical if x.get('purpose')=='engineering_smoke' and x['status']=='STARTED')
    receipt=dict(status='ACCEPTED',source_hash=m['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
        owner=p.owner_binding(n),cases=rows,retained_engineering_keys=[x['key'] for x in rows],
        retained_states=status['retained_states'],retained_tensor_bytes=status['retained_bytes'],
        all_engineering_workers_exited=True,engineering_keys_released=False,resource_review_pending=True,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),test_accessed=False)
    p.write_json(n/'PARENT_ENGINEERING_SUPERVISION_ACCEPTANCE.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','retained_states','retained_tensor_bytes','source_hash','test_accessed']}))


if __name__=='__main__':main()
