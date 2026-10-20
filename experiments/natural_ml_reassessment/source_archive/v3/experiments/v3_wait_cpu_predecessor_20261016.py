"""Mechanical CPU dependency wait; no scientific computation or interpretation."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preceding-status', type=Path, action='append', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command, 'Missing downstream command'
    while True:
        ready = True
        for path in args.preceding_status:
            states = json.loads(path.read_text())['tasks']
            assert states, f'Empty predecessor: {path}'
            if any(state['status'] == 'FAILED' for state in states.values()):
                raise RuntimeError(f'Predecessor failed: {path}')
            ready = ready and all(state['status'] == 'COMPLETE' for state in states.values())
        if ready:
            break
        time.sleep(20)
    print(json.dumps(dict(status='CPU_PREDECESSORS_COMPLETE', time=time.time(),
                          predecessors=list(map(str, args.preceding_status)), command=command)), flush=True)
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
