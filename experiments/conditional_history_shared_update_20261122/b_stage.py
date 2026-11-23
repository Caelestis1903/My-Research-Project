"""Connect the ten accepted clean replays with dataset-specific B gates."""
from .protocol import DATASETS, jobs, read, sha
from .scheduler import verify_selected


def analyze_core(art,authority):
    from .b_gate import core_gate
    anchors = {d:[] for d in DATASETS}
    for job in jobs('B_CORE',authority['source_hash']):
        attempt = art/'attempt1'
        verify_selected(attempt,job)
        result = read(attempt/'runs'/job['key']/'result.json')
        for epoch in (5,20,35,50):
            record = result['b_outputs'][f'B_ANCHOR_E{epoch:02d}.json']
            assert sha(record['path'])==record['sha256']
            anchors[job['dataset']].append(record['path'])
        for epoch in (20,35):
            record = result['b_outputs'][f'B_WINDOW_E{epoch}.json']
            assert sha(record['path'])==record['sha256']
    return {d:core_gate(d,paths,art/'tables/B_CORE') for d,paths in anchors.items()}


def advance(art):
    folder = art/'tables/B_CORE'
    gates = [folder/f'B_GATE_{d}.json' for d in DATASETS]
    if not all(p.exists() for p in gates):
        return dict(next='analyze B_CORE',protocol='S3')
    values = [read(p) for p in gates]
    if any(v['B_PULSE']=='TRIGGERED' and v['B_SEARCH']=='AWAITING_PULSE' for v in values):
        return dict(next='python -m experiments.conditional_history_shared_update_20261122.b_stage pulse',protocol='S3')
    return dict(next='S4 conditional stage review',protocol='S4',
                B_SEARCH={v['dataset']:v['B_SEARCH'] for v in values})


def pulse_receipts(art, job, gate):
    """Return each accepted pulse locator, validating its parent and scalar payload."""
    from .protocol import digest
    base = art/'B_PULSE'/job['key']
    key = digest([job,gate['anchors'],gate['static_hash']])
    accepted = []
    for epoch in (20,35):
        for family in ('true','pseudo0'):
            path = base/f'B_PULSE_E{epoch}_{family}.json'
            ack = path.with_suffix('.receipt.json')
            if ack.exists():
                receipt = read(ack)
                if receipt['parent_hash']!=key or receipt['sha256']!=sha(path):
                    raise RuntimeError(f'Pulse receipt mismatch {ack}')
                accepted.append(path)
    return key,accepted


def run_seed(art, job, gate, device, stop):
    """Replay only the four frozen slots of one dataset/seed from its verified RAM parent."""
    import os
    import time
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    from .b_observer import BObserver
    from .b_pulse import run_pulse
    from .protocol import digest, v3, write
    from .scheduler import sample, residency
    from .state import generation, request
    if gate['dataset']!=job['dataset'] or gate['B_PULSE']!='TRIGGERED':
        raise RuntimeError('Dataset B_PULSE gate is closed')
    out = art/'B_PULSE'/job['key']
    out.mkdir(parents=True,exist_ok=True)
    parent_hash,done = pulse_receipts(art,job,gate)
    attempt = art/'attempt1'
    result = read(attempt/'runs'/job['key']/'result.json')
    assert result['job']==job and result['owner_generation']==generation(attempt)
    assert digest(result)==read(attempt/'runs'/job['key']/'COMPLETE_VERIFIED.json')['result_hash']
    branch = request(attempt,'get',key=result['branch_key'])['state']
    dataset = job['dataset']
    protocol = v3.load_protocol()
    bundle, identity = v3.load_dataset(dataset,p=protocol)
    assert digest(identity)==result['data_hash']
    v3.setup(job['seed'],device)
    model = v3.make_model(bundle,protocol['datasets'][dataset]['source']).to(device)
    observer = BObserver(bundle,dataset,job['seed'],out)
    observer.load_state_dict(branch['observer'])
    for anchor in observer.ram['anchors']:
        evidence = result['b_outputs'][f"B_ANCHOR_E{anchor['epoch']:02d}.json"]
        assert sha(evidence['path'])==evidence['sha256'] and read(evidence['path'])['identity']==anchor
    ack = out/'FULL_ALLOCATION.json'
    start = time.monotonic()
    def allocated(*_):
        if not ack.exists() or read(ack)['pid']!=os.getpid():
            torch.cuda.synchronize(device)
            telem = sample()
            write(ack,dict(pid=os.getpid(),device=str(device),bytes=residency(telem,os.getpid()),telemetry=telem))
    hook = register_optimizer_step_post_hook(allocated)
    try:
        for epoch in (20,35):
            for family in ('true','pseudo0'):
                path = out/f'B_PULSE_E{epoch}_{family}.json'
                if path in done:
                    continue
                if stop():
                    return
                t0 = time.monotonic()
                run_pulse(observer,model,job['config'],epoch,family,gate=gate)
                write(path.with_suffix('.receipt.json'),dict(parent_hash=parent_hash,sha256=sha(path),
                    pid=os.getpid(),device=str(device),worker_seconds=time.monotonic()-t0,
                    elapsed_seconds=time.monotonic()-start,owner_generation=result['owner_generation']))
    finally:
        hook.remove()


def pulses(art,authority):
    """Admit attached mechanical pulse workers; name every one of the twenty slots per gated dataset."""
    import fcntl
    import os
    import signal
    import subprocess
    import sys
    import time
    from .b_gate import search_gate
    from .protocol import ROOT, append, write
    from .scheduler import admit, sample, stopping
    folder = art/'tables/B_CORE'
    checks = read(art/'B_CORE_CHECK.json')
    gates = {d:read(folder/f'B_GATE_{d}.json') for d in DATASETS}
    eligible = [j for j in jobs('B_CORE',authority['source_hash']) if gates[j['dataset']]['B_PULSE']=='TRIGGERED']
    queue = [j for j in eligible if len(pulse_receipts(art,j,gates[j['dataset']])[1])!=4]
    lock = (art/'B_PULSE_SUPERVISION.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    live = {}
    profile = dict(checks['full_process_bytes'])
    pause = art/'B_PULSE_PAUSE'
    def signal_stop(_s,_frame):
        write(pause,dict(reason='SUPERVISOR_INTERRUPTED',time=time.time()))
    signals = {s:signal.signal(s,signal_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        while queue or live:
            telem = sample()
            pending = set()
            for key,(p,j,gpu,stream) in list(live.items()):
                rc = p.poll()
                if rc is not None:
                    stream.close()
                    del live[key]
                    if rc:
                        write(pause,dict(reason='FAILED_PULSE',key=key,pid=p.pid,returncode=rc))
                    continue
                alloc = art/'B_PULSE'/key/'FULL_ALLOCATION.json'
                if not alloc.exists() or read(alloc)['pid']!=p.pid:
                    pending.add(gpu)
                for row in telem['processes']:
                    if row['pid']==p.pid:
                        profile[j['dataset']]=max(profile[j['dataset']],row['bytes'])
            if stopping(art,'B_PULSE'):
                if not live:
                    return dict(status='PAUSED')
            else:
                for row in telem['devices']:
                    gpu = row['index']
                    if gpu in pending:
                        continue
                    job = next((j for j in queue if admit(row,telem['host_ram_used'],
                               checks['cuda_usable_total_bytes'][str(gpu)],profile[j['dataset']])),None)
                    if job is None:
                        continue
                    queue.remove(job)
                    log = (art/f"B_PULSE_{job['key']}.log").open('ab')
                    proc = subprocess.Popen([sys.executable,'-m',__package__+'.b_stage','_seed',
                        '--artifacts',str(art),'--key',job['key'],'--gpu',str(gpu),'--parent',str(os.getpid())],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    live[job['key']] = (proc,job,gpu,log)
                    append(art/'B_PULSE_admissions.jsonl',dict(pid=proc.pid,key=job['key'],telemetry=telem,
                                                              gpu=gpu,profile=profile))
                    break
            time.sleep(1)
    finally:
        if live:
            write(pause,dict(reason='SUPERVISOR_EXIT',time=time.time()))
        for proc,job,gpu,stream in live.values():
            proc.wait()
            stream.close()
        for s,handler in signals.items():
            signal.signal(s,handler)
        lock.close()
    for d,gate in gates.items():
        paths = [p for job in eligible if job['dataset']==d for p in pulse_receipts(art,job,gate)[1]]
        search_gate(gate,paths,folder)
    return dict(status='COMPLETE')


def main():
    import argparse
    import os
    from pathlib import Path
    from .cli import reviewed
    from .protocol import ART
    from .scheduler import stopping
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('pulse','_seed'))
    parser.add_argument('--artifacts',type=Path,default=ART)
    parser.add_argument('--key')
    parser.add_argument('--gpu',type=int)
    parser.add_argument('--parent',type=int)
    args = parser.parse_args()
    art = args.artifacts.resolve()
    authority = reviewed(art,'B_CORE')
    if args.action=='pulse':
        print(pulses(art,authority))
    else:
        if not args.parent or os.getppid()!=args.parent:
            raise RuntimeError('Pulse child requires active foreground supervision')
        job = next(j for j in jobs('B_CORE',authority['source_hash']) if j['key']==args.key)
        gate = read(art/'tables/B_CORE'/f"B_GATE_{job['dataset']}.json")
        run_seed(art,job,gate,f'cuda:{args.gpu}',lambda:os.getppid()!=args.parent or stopping(art,'B_PULSE'))


if __name__=='__main__':
    main()
