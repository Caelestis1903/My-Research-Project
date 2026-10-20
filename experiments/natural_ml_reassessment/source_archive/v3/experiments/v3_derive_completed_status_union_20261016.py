"""Lossless provenance union of completed actual queues; never a queue launch."""
import hashlib
import json
from pathlib import Path
import sys
import time


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'research_plans/2026-10-16_v3/execution_20261016'
    stem, *children = sys.argv[1:]
    assert Path(stem).name == stem and children
    template = out / (stem + '_manifest.json')
    destination = template.with_suffix('.status.json')
    assert not destination.exists()
    expected = {task['name']: task for task in load(template)['tasks']}
    assert len(expected) == len(load(template)['tasks'])
    records, sources = {}, []
    for child in children:
        assert Path(child).name == child
        manifest = out / (child + '_manifest.json')
        status = manifest.with_suffix('.status.json')
        launch = out / (child + '_queue_launch.json')
        m, s, l = load(manifest), load(status), load(launch)
        assert s['status'] == 'COMPLETE' and s['queue_pid'] == l['pid']
        assert l['manifest'] == str(manifest) and l['manifest_sha256'] == sha(manifest)
        assert l['status'] == str(status) and isinstance(l['pid'], int)
        declared = {task['name']: task for task in m['tasks']}
        assert set(declared) == set(s['tasks'])
        for name, actual in s['tasks'].items():
            assert name in expected and name not in records
            assert declared[name] == expected[name]
            assert actual['status'] == 'COMPLETE' and actual['exit_code'] == 0
            assert isinstance(actual['pid'], int) and actual['pid'] > 0
            command = expected[name]['command'].copy()
            device_index = command.index('--device') + 1
            assert command[device_index] == '{device}'
            command[device_index] = 'cuda:' + str(actual['gpu'])
            assert actual['command'] == command
            records[name] = actual
        sources.append(dict(manifest=str(manifest), manifest_sha256=sha(manifest),
                            status=str(status), status_sha256=sha(status),
                            launch=str(launch), launch_sha256=sha(launch),
                            actual_queue_pid=s['queue_pid'], tasks=list(s['tasks'])))
    assert set(records) == set(expected)
    result = dict(status='COMPLETE', queue_pid=None, time=time.time(),
                  provenance_kind='DERIVED_STATUS_UNION_NOT_AN_ACTUAL_QUEUE',
                  whole_template_was_launched=False, independent_review=False,
                  source_sha256=sha(Path(__file__)), template_sha256=sha(template),
                  derived_status_sources=sources, tasks=records, test_access=False)
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status='PASS', path=str(destination), actual_queues=len(sources),
                          actual_tasks=len(records), queue_pid=None)))


if __name__ == '__main__':
    main()
