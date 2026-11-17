"""Verify a fresh native mechanical runner's actual first-turn metadata before GPU work."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from experiments.completion_weight_attribution_20261117 import protocol as p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('agent_id');parser.add_argument('stage',choices=('SMOKE','FORMAL','RECOVERY_01','RECOVERY_02'))
    args=parser.parse_args();n=p.ATTEMPT;m=p.validate_attempt(n,args.stage!='SMOKE')
    review_path=n/f"RECOVERY_REVIEW_{args.stage.removeprefix('RECOVERY_')}.json" if args.stage.startswith('RECOVERY_') else None
    recovery=p.check_review(review_path,'recovery',m['source_hash']) if review_path else None
    if recovery:
        assert recovery['pause_sha256']==p.sha_file(n/'TECHNICAL_PAUSE.json')
        assert recovery['owner_endpoint_hash']==p.owner_binding(n)['endpoint_hash']
    paths=list(Path('/home/xzf/.codex-clean/sessions').rglob('*'+args.agent_id+'*.jsonl'));assert len(paths)==1
    meta=turn=None
    for line in paths[0].open():
        event=json.loads(line)
        if event['type']=='session_meta':meta=event['payload']
        if event['type']=='turn_context':turn=event['payload'];break
    parent='01a1dfb8-605b-7430-8462-5fd3da0b048f'
    assert meta['id']==args.agent_id and meta['parent_thread_id']==parent
    assert meta['source']['subagent']['thread_spawn']['parent_thread_id']==parent
    assert turn['model']=='gpt-6-luna' and turn['effort']=='high'
    assert m['seeds']==list(range(2022,2027))
    for root in (p.ROOT,Path('/home/xzf/Code/phase0-3')):
        a=root/p.REL_ARTIFACT;state=p.read_json(a/'AUTONOMOUS_STATE.json')
        assert not state.get('manual_stop_requested') and not (a/'STOP_AUTONOMOUS').exists()
    receipt=dict(status='PASS',agent_id=args.agent_id,parent_conversation_id=parent,
        actual_model=turn['model'],actual_effort=turn['effort'],fork_context=False,
        fork_context_evidence='explicit fresh native spawn request with fork_context=false',
        metadata_source=meta['source'],history_mode=meta.get('history_mode'),
        source_hash=m['source_hash'],design_bundle_sha256=p.DESIGN_HASH,
        code_review_sha256=m['code_review_sha256'],owner=p.owner_binding(n),
        engineering_review_sha256=p.sha_file(n/'ENGINEERING_SIGNOFF.json') if args.stage!='SMOKE' else None,
        recovery_review_sha256=p.sha_file(review_path) if recovery else None,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),test_accessed=False)
    p.write_json(n/f'RUNNER_DISPATCH_{args.stage}.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','agent_id','actual_model','actual_effort','source_hash','test_accessed']}))


if __name__=='__main__':main()
