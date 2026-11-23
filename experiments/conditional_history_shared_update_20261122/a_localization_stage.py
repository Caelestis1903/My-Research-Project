"""Foreground A_TARGET diagnostics and the complete digest-bound A_DOSE gate input."""
import argparse
import fcntl
import itertools
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from experiments.conditional_information_momentum_20261102_v3.boundary_sampler import sample as draw_pair

from .a_localization import GROUPS, SOURCES, gate_rows, observe
from .b_observer import ALL, write_result
from .b_panel_identity import HASHES
from .b_panels import prepare
from .history_runner import ANCHORS
from .history_stage import stopping
from .protocol import ART, DATASETS, PACKAGE, REPORT, ROOT, SEEDS, append, digest, read, sha, v3, write
from .results import DOSE, stage_arms
from .scheduler import admit, residency, sample
from .state import generation, request

STAGE = 'A_TARGET'
ARMS = ('A21', DOSE[0])
FILES = ('a_localization.py','a_localization_stage.py','a_localization_checks.py',
         'history_controls.py','history_runner.py','b_panels.py','b_pairs.py','b_pairs_kernel.cpp',
         'b_build_panels.py','b_panel_identity.py','b_virtual.py','b_observer.py','results.py',
         'protocol.py','history_stage.py','scheduler.py','state.py')


def source_id():
    return digest({n:sha(PACKAGE/n) for n in FILES})


def reviewed(art):
    review = read(REPORT/'A_TARGET_REVIEW_RECEIPT.json')
    decision = read(art/'tables/A_HISTORY/decisions.json')
    stage_arms('A_DOSE', decision)
    if (review['verdict'] not in ('PASS','PASS WITH FIXES') or review['blocking_issues'] or
        review['source_hash']!=source_id() or review['design_sha256']!=sha(REPORT/'DESIGN_SPECIFICATION.md') or
        decision['history_form']!='A21' or decision['statistics_script_sha256']!=sha(PACKAGE/'results.py') or
        any(sha(art/'tables/A_HISTORY'/n)!=h for n,h in decision['evidence'].items())):
        raise RuntimeError('A_TARGET review or history gate identity mismatch')
    return review['source_hash']


def parents(art, arm):
    if arm=='A21':
        from . import history_stage as parent_stage
    else:
        from . import dose_stage as parent_stage
    stage = parent_stage.STAGE
    audit = parent_stage.reviewed(art)
    manifest = read(art/f'{stage}_SOURCE_MANIFEST.json')
    if manifest['jobs']!=parent_stage.jobs(stage,audit['source_hash']):
        raise ValueError('Parent jobs differ from the reviewed source')
    jobs = [j for j in manifest['jobs'] if j['arm']==arm]
    if {(j['dataset'],j['seed']) for j in jobs}!=set(itertools.product(DATASETS,SEEDS)) or len(jobs)!=10:
        raise ValueError('Incomplete frozen parent jobs')
    if (audit['source_hash']!=manifest['source_hash'] or audit['owner_generation']!=generation(art/'attempt1')
        or read(art/f'{stage}_RUN.json')['status']!='COMPLETE'):
        raise RuntimeError('Parent run incomplete or RAM owner changed')
    for j in jobs:
        cfg = j['config']
        if (j['source_hash']!=manifest['source_hash'] or j['key']!=digest({k:v for k,v in j.items() if k!='key'})
            or cfg['mode']!='collapsed' or cfg['dynamic'] or cfg['lambda_pair']!=(.05 if arm=='A21' else .025)):
            raise ValueError('Parent configuration differs')
    return jobs


def parent_result(art, job):
    base = art/'attempt1/runs'/job['key']
    result = read(base/'result.json')
    accepted = read(base/'COMPLETE_VERIFIED.json')
    if (result['job']!=job or result['owner_generation']!=accepted['owner_generation']
        or accepted['result_hash']!=digest(result) or result['epochs']!=50 or result['test_evaluations']!=0
        or set(result['anchors'])!=set(map(str,ANCHORS))):
        raise ValueError('Missing or changed authenticated parent')
    return result


def slot(art, job, epoch):
    return art/STAGE/job['key']/f'E{epoch:02}.json'


def accepted_slot(path, source_hash, parent):
    ack_path = path.with_suffix('.receipt.json')
    if not ack_path.is_file():
        return False
    r = read(ack_path)
    if r['sha256']!=sha(path) or r['source_hash']!=source_hash or r['anchor']!=parent:
        raise ValueError(f'Changed diagnostic or parent identity {path}')
    return True


def seed_run(art, job, source_hash, gpu, stop):
    seed, dataset = job['seed'], job['dataset']
    result = parent_result(art,job)
    owner = generation(art/'attempt1')
    if result['owner_generation']!=owner:
        raise RuntimeError('Parent generation changed')
    protocol = v3.load_protocol()
    bundle, ident = v3.load_dataset(dataset,p=protocol)
    if digest(ident)!=result['data_hash']:
        raise RuntimeError('Dataset identity changed')
    device = torch.device(f'cuda:{gpu}')
    v3.setup(seed,device)
    template = v3.make_model(bundle,protocol['datasets'][dataset]['source']).to(device)
    panels, panel_manifest = prepare(bundle,seed)
    for epoch in ANCHORS:
        if stop():
            return
        path = slot(art,job,epoch)
        parent = result['anchors'][str(epoch)]
        if accepted_slot(path,source_hash,parent):
            continue
        start = time.monotonic()
        packet = request(art/'attempt1','get',key=parent['key'])['state']['anchor']
        ctx = packet['context']
        cursor = packet['cursor']
        K = {'movielens':20, 'kuairand':49}[dataset]
        if (v3.state_digest(packet)!=parent['state_hash'] or parent['epoch']!=epoch
            or parent['owner_generation']!=packet['owner_generation'] or packet['owner_generation']!=owner
            or parent['actual_post']!=packet['actual_post'] or packet['job']!=job
            or packet['data_hash']!=result['data_hash'] or packet['group_hash']!=result['group_hash']
            or (ctx['dataset'],ctx['seed'],ctx['epoch'],ctx['global_step'])!=(dataset,seed,epoch,epoch*K)
            or not ctx['last_batch'] or cursor!=(K-1)*job['config']['batch_size']
            or packet['global_step_before']!=ctx['global_step']-1
            or not np.array_equal(packet['permutation'][cursor:], packet['batch_indices'])
            or not np.array_equal(bundle.train_row_ids[packet['batch_indices']], packet['row_ids'])):
            raise ValueError('RAM anchor identity, row order or step changed')
        ids = packet['batch_indices']
        expected_pair = draw_pair(bundle.train_x[ids,-1],bundle.train_y[ids],packet['optimizer']['demand'],
                                  False,ctx,packet['row_ids'])
        if not v3.equal(expected_pair,packet['pair']):
            raise ValueError('Reconstructed static pair differs from the captured draw')
        v3.restore_rng(packet['rng'],device)
        rng_before = v3.state_digest(v3.rng_state(device))
        raw = observe(template, panels, packet, device)
        if v3.state_digest(v3.rng_state(device))!=rng_before:
            raise ArithmeticError('Diagnostic changed RNG')
        torch.cuda.synchronize(device)
        telemetry = sample()
        proc_bytes = residency(telemetry,os.getpid())
        raw.update(stage=STAGE,dataset=dataset,arm=job['arm'],seed=seed,history_form='A21',anchor_epoch=epoch,
                   context=ctx,B=len(packet['row_ids']),row_id_hash=v3.state_digest(packet['row_ids']),
                   pair_hash=None if packet['pair'] is None else packet['pair']['pair_index_hash'],
                   source_hash=source_hash,anchor=parent,parent_job_key=job['key'],group_hash=result['group_hash'],
                   data_hash=result['data_hash'],panel_manifest=panel_manifest,
                   loader_hash=v3.state_digest(packet['loader']),rng_hash=rng_before,
                   worker_seconds=time.monotonic()-start,process_residency_bytes=proc_bytes,
                   peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(device))
        validate(raw,job,parent,source_hash)
        write_result(path,raw)
        write(path.with_suffix('.receipt.json'),dict(sha256=sha(path),source_hash=source_hash,anchor=parent))
        write(path.parent/'FULL_ALLOCATION.json',dict(pid=os.getpid(),gpu=gpu,bytes=proc_bytes))
        del packet,raw


def validate(raw, job, parent, source_hash):
    if ((raw['dataset'],raw['seed'],raw['arm'],raw['history_form'])!=(job['dataset'],job['seed'],job['arm'],'A21')
        or raw['anchor']!=parent or raw['anchor_epoch']!=parent['epoch'] or raw['parent_job_key']!=job['key']
        or raw['source_hash']!=source_hash or raw['audit']['actual_post']!=parent['actual_post']
        or raw['panel_manifest']['source_hashes']!=HASHES[job['dataset']][job['seed']]):
        raise ValueError('Unverified parent, action reference or panel')
    expected = set(itertools.product(GROUPS,SOURCES,('raw','norm_matched'))) | {('S_DENSE','ALL','raw')}
    got = set()
    for e in raw['effects']:
        got.add((e['group'],e['source'],e['scale']))
        if e['status']!='OK':
            if (e['status'],e['scale'],e['raw_alt_update_norm'])!=('UNMATCHABLE_ZERO_ALTERNATIVE','norm_matched',0):
                raise ValueError('Invalid missing measurement')
            continue
        for r in ('online','fixed_ema'):
            if set(e[r])!=set(ALL) or any([c['q'] for c in p['cells']]!=list(range(6)) for p in e[r].values()):
                raise ValueError('Missing panel or cell')
    if got!=expected or len(got)!=len(raw['effects']):
        raise ValueError('Missing or duplicate history intervention')


def combine(art, source_hash):
    records,rows = [],[]
    for arm in ARMS:
        for job in parents(art,arm):
            result = parent_result(art,job)
            for epoch in ANCHORS:
                path = slot(art,job,epoch)
                parent = result['anchors'][str(epoch)]
                if not accepted_slot(path,source_hash,parent):
                    raise ValueError(f'Pending {path}')
                raw = read(path)
                validate(raw,job,parent,source_hash)
                h = sha(path)
                records.append(dict(path=str(path),sha256=h,anchor=parent))
                rows.extend(gate_rows(raw, str(path)+'#sha256='+h))
    if len(records)!=80 or len(rows)!=320:
        raise RuntimeError('A_TARGET requires both doses and all 80 anchors')
    out = art/'tables'/STAGE/'localization.json'
    write(out,rows)
    receipt = dict(stage=STAGE,status='COMPLETE',arms=list(ARMS),anchor_count=len(records),
                   gate_rows=len(rows),source_hash=source_hash,localization_sha256=sha(out),inputs=records,
                   full_training_replays=0,test_evaluations=0)
    write(art/'A_TARGET_RUN.json',receipt)
    return receipt


def supervise(art, arm, source_hash):
    jobs = parents(art,arm)
    queue = []
    for job in jobs:
        result = parent_result(art,job)
        if any(not accepted_slot(slot(art,job,e),source_hash,result['anchors'][str(e)]) for e in ANCHORS):
            queue.append(job)
    check = read(art/'A_HISTORY_CHECK.json')
    # Same native models and RAM packets. Start with measured complete history-worker residency.
    profile = dict(read(art/'A_HISTORY_RUN.json')['full_process_bytes'])
    lock = (art/'A_TARGET_SUPERVISION.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    live = {}
    pause = art/'A_TARGET_PAUSE'
    def onstop(_sig,_frame):
        write(pause,dict(reason='SUPERVISOR_INTERRUPTED'))
    old = {s:signal.signal(s,onstop) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        while queue or live:
            telemetry = sample()
            pending = set()
            for key,(proc,j,gpu,log) in list(live.items()):
                rc = proc.poll()
                if rc is not None:
                    log.close()
                    del live[key]
                    if rc:
                        write(pause,dict(reason='FAILED_A_TARGET',job=key,returncode=rc))
                    continue
                alloc = slot(art,j,5).parent/'FULL_ALLOCATION.json'
                if not alloc.is_file() or read(alloc)['pid']!=proc.pid:
                    pending.add(gpu)
                for p in telemetry['processes']:
                    if p['pid']==proc.pid:
                        profile[j['dataset']]=max(profile[j['dataset']],p['bytes'])
            if stopping(art,STAGE):
                if not live:
                    return 'PAUSED'
            else:
                for card in telemetry['devices']:
                    gpu = card['index']
                    if gpu in pending:
                        continue
                    job = next((j for j in queue if admit(card,telemetry['host_ram_used'],
                               check['cuda_usable_total_bytes'][str(gpu)],profile[j['dataset']])),None)
                    if job is None:
                        continue
                    queue.remove(job)
                    log = (art/f"A_TARGET_{job['key']}.log").open('ab')
                    proc = subprocess.Popen([sys.executable,'-m',__package__+'.a_localization_stage','_seed',
                        '--artifacts',str(art),'--arm',arm,'--key',job['key'],'--gpu',str(gpu),'--parent',str(os.getpid())],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    live[job['key']]=(proc,job,gpu,log)
                    append(art/'A_TARGET_admissions.jsonl',dict(pid=proc.pid,job=job['key'],gpu=gpu,
                           physical_fit_bytes=profile[job['dataset']],telemetry=telemetry))
                    break
            time.sleep(1)
    finally:
        if live:
            write(pause,dict(reason='SUPERVISOR_EXIT'))
        for proc,_,_,log in live.values():
            proc.wait()
            log.close()
        for s,handler in old.items():
            signal.signal(s,handler)
        lock.close()
    for j in jobs:
        result = parent_result(art,j)
        if not all(accepted_slot(slot(art,j,e),source_hash,result['anchors'][str(e)]) for e in ANCHORS):
            raise RuntimeError('Worker exited with missing scalar receipts')
    return 'COMPLETE'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=('run','combine','_seed'))
    parser.add_argument('--artifacts',type=Path,default=ART)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--key')
    parser.add_argument('--gpu',type=int)
    parser.add_argument('--parent',type=int)
    args = parser.parse_args()
    art = args.artifacts.resolve()
    if stopping(art,STAGE):
        raise RuntimeError('Stop or pause control active')
    source_hash = reviewed(art)
    if args.action=='combine':
        print(combine(art,source_hash)['status'])
    elif args.action=='run':
        if args.arm is None:
            parser.error('run requires --arm')
        print(supervise(art,args.arm,source_hash))
    else:
        if not args.parent or os.getppid()!=args.parent:
            raise RuntimeError('A_TARGET worker requires foreground supervisor')
        job = next(j for j in parents(art,args.arm) if j['key']==args.key)
        seed_run(art,job,source_hash,args.gpu,lambda:os.getppid()!=args.parent or stopping(art,STAGE))


if __name__=='__main__':
    main()
