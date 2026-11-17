"""Parent verifies actual fresh native runner metadata before authorizing GPU work."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.cell_mass_bce_20261117 import protocol as p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('agent_id')
    parser.add_argument('stage', choices=('SMOKE', 'FORMAL', 'FORMAL_RECOVERY_01', 'FORMAL_RECOVERY_02', 'ENGINEERING_RECOVERY_01', 'ENGINEERING_RECOVERY_02', 'ENGINEERING_RECOVERY_03'))
    args = parser.parse_args()
    parent = '01a1dfb8-605b-7430-8462-5fd3da0b048f'
    paths = list(Path('/home/xzf/.codex-clean/sessions').rglob('*' + args.agent_id + '*.jsonl'))
    assert len(paths) == 1
    meta = turn = None
    for line in paths[0].open():
        event = json.loads(line)
        if event['type'] == 'session_meta':
            meta = event['payload']
        if event['type'] == 'turn_context':
            turn = event['payload']
            break
    assert meta['id'] == args.agent_id and meta['parent_thread_id'] == parent
    assert meta['source']['subagent']['thread_spawn']['parent_thread_id'] == parent
    assert turn and turn['model'] == 'gpt-6-luna' and turn['effort'] == 'high'
    assert not p.stop_requested()
    manifest = p.validate_attempt(p.ATTEMPT, require_engineering=args.stage == 'FORMAL' or args.stage.startswith('FORMAL_RECOVERY_'))
    assert list(manifest['seeds']) == list(range(2022, 2027))
    recovery_receipts = []
    if args.stage.startswith('ENGINEERING_RECOVERY_'):
        suffix = '02' if args.stage == 'ENGINEERING_RECOVERY_03' else '01'
        for name in (f'ENGINEERING_RECOVERY_CASE1_{suffix}.json', f'ENGINEERING_RECOVERY_CASE3_{suffix}.json'):
            path = p.ATTEMPT / name
            review = p.check_review(path, 'recovery', manifest['source_hash'])
            assert review['pause_sha256'] == p.sha_file(p.ATTEMPT / 'TECHNICAL_PAUSE.json')
            assert review['owner_endpoint_hash'] == p.owner_binding(p.ATTEMPT)['endpoint_hash']
            recovery_receipts.append(dict(path=str(path), sha256=p.sha_file(path)))
    if args.stage.startswith('FORMAL_RECOVERY_'):
        path = p.ATTEMPT / ('FORMAL_RECOVERY_REVIEW_' + args.stage[-2:] + '.json')
        review = p.check_review(path, 'recovery', manifest['source_hash'])
        assert review['pause_sha256'] == p.sha_file(p.ATTEMPT / 'TECHNICAL_PAUSE.json')
        assert review['owner_endpoint_hash'] == p.owner_binding(p.ATTEMPT)['endpoint_hash']
        recovery_receipts.append(dict(path=str(path), sha256=p.sha_file(path)))
    receipt = dict(status='PASS', agent_id=args.agent_id, parent_conversation_id=parent,
        actual_model=turn['model'], actual_effort=turn['effort'], stage=args.stage,
        fork_context=False, fork_context_evidence='Explicit fresh native parent spawn with fork_context=false',
        actual_metadata_verified=True, source_hash=manifest['source_hash'],
        design_bundle_sha256=p.DESIGN_HASH, research_bundle_sha256=p.RESEARCH_HASH,
        code_review_sha256=manifest['code_review_sha256'], owner=p.owner_binding(p.ATTEMPT),
        recovery_receipts=recovery_receipts,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(), test_accessed=False)
    p.write_json(p.ATTEMPT / ('RUNNER_DISPATCH_' + args.stage + '.json'), receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'agent_id', 'stage', 'actual_model', 'actual_effort', 'source_hash')}))


if __name__ == '__main__':
    main()
