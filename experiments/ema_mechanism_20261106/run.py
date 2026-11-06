"""Finite EMA mechanism workflow and internal RAM-only process entrypoints."""
import argparse
import json
import os
import resource
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from .protocol import ROOT,HERE,resolve_study,read_json,write_json,implementation_identity
from .state import ensure_guardian,request,alive,append_json


def _study(args):
    return resolve_study(args.manifest,dict(manifest=str(Path(args.manifest).resolve()),out=str(Path(args.out).resolve())))


def worker_process(study,out,key,device):
    from .worker import create_session,restore_session,advance_epoch,export_session
    from .panels import load_panels
    from experiments.overnight_ab_20261103.data_bridge import cached_bundle
    job=next(j for j in study.jobs if j['job_id']==key)
    prepared=read_json(Path(out)/'PREPARED.json')
    if prepared['identity']['implementation_hash']!=implementation_identity()['sha256']: raise ValueError('Implementation changed after preflight')
    job=next(j for j in prepared['jobs'] if j['job_id']==key)
    registered=request(out,'register',key=key,job=job,pid=os.getpid(),device=device)
    if registered.get('complete'): return
    try:
        bundle=cached_bundle(prepared['cache_root'],job['dataset'],include_test=False)
        panels=load_panels(out,job['dataset'],job['seed'],bundle)
        session=restore_session(job,bundle,device,panels,registered['state']) if registered.get('state') is not None else create_session(job,bundle,device,panels)
        del registered
        while not session['complete']:
            record,predictions=advance_epoch(session)
            before=time.perf_counter()
            answer=request(out,'commit',key=key,pid=os.getpid(),state=export_session(session),record=record,predictions=predictions)
            ipc_seconds=time.perf_counter()-before
            append_json(Path(out)/'IPC_COSTS.jsonl',dict(job_id=key,epoch=session['epoch'],ipc_seconds=ipc_seconds))
            print(json.dumps(dict(job_id=key,epoch=session['epoch'],global_step=session['global_step'],status=answer['status'],ipc_seconds=ipc_seconds)),flush=True)
            del record,predictions
        request(out,'complete',key=key,pid=os.getpid())
    except Exception:
        error=traceback.format_exc()
        try: request(out,'failed',key=key,pid=os.getpid(),error=error)
        except Exception: pass
        raise


def resource_admission(devices):
    raw=Path('/proc/meminfo').read_text().splitlines();memory={line.split(':')[0]:int(line.split()[1]) for line in raw}
    if 1-memory['MemAvailable']/memory['MemTotal']>=.9: return []
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,power.draw,power.limit','--format=csv,noheader,nounits'],text=True)
    eligible=[]
    for line in output.strip().splitlines():
        index,used,total,power,limit=[float(x.strip()) for x in line.split(',')]
        if int(index) in devices and used/total<.85 and power/limit<.85: eligible.append(int(index))
    return eligible


def train(study,out,devices,resume):
    out=Path(out).resolve();prepared=read_json(out/'PREPARED.json');receipt=read_json(out/'VALIDATION_RECEIPT.json')
    if receipt['implementation_hash']!=implementation_identity()['sha256'] or prepared['identity']['implementation_hash']!=receipt['implementation_hash']: raise ValueError('Validation does not bind current code')
    review=out/'IMPLEMENTATION_REVIEW.md'
    if not review.exists() or 'PASS' not in review.read_text(): raise ValueError('Independent implementation review required')
    for ds in ('movielens','kuairand'):
        smoke=read_json(out/f'SMOKE_{ds.upper()}.json')
        if smoke['implementation_hash']!=receipt['implementation_hash'] or smoke['status']!='PASS': raise ValueError('Both full-batch smoke receipts required')
    if not torch.cuda.is_available(): raise MemoryError('CUDA unavailable')
    ensure_guardian(out);active={};jobs=prepared['jobs'];logs={};start=time.time()
    while True:
        status=request(out,'status')['jobs']
        for key,child in list(active.items()):
            if child.poll() is not None:
                logs.pop(key).close();active.pop(key)
                if child.returncode and status.get(key,{}).get('launches',0)>=2: raise RuntimeError('Worker restart exhausted; inspect runs/'+key+'/worker.log')
        completed={key for key,row in status.items() if row['status']=='COMPLETE'}
        if len(completed)==20: break
        for job in jobs:
            key=job['job_id'];row=status.get(key,{})
            if key in completed or key in active or alive(row.get('worker')): continue
            if row.get('launches',0)>=2: raise RuntimeError('Restart budget exhausted: '+key)
            if len(active)>=2: break
            admitted=resource_admission(devices)
            if not admitted: break
            device=admitted[0];directory=out/'runs'/key;directory.mkdir(parents=True,exist_ok=True)
            logs[key]=(directory/'worker.log').open('ab')
            cmd=[sys.executable,'-m','experiments.ema_mechanism_20261106.run','_worker','--manifest',str(study.paths['manifest']),'--out',str(out),'--key',key,'--device',f'cuda:{device}']
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            active[key]=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=logs[key],stderr=logs[key],start_new_session=True)
            time.sleep(3)
        time.sleep(3)
    write_json(out/'TRAINING_COMPLETE.json',dict(status='COMPLETE',logical_jobs=20,logical_epochs=1000,logical_adam_steps=34500,started_at=start,completed_at=time.time()))
    return dict(status='COMPLETE',logical_jobs=20)


def validate(study,out):
    result=subprocess.run([sys.executable,'-m','pytest',str(HERE/'tests'),'-q'],cwd=ROOT,text=True,capture_output=True)
    path=Path(out)/'validation_tests.log';path.write_text(result.stdout+'\n'+result.stderr)
    if result.returncode: raise FloatingPointError('Necessary validation failed; see validation_tests.log')
    receipt=dict(status='PASS',implementation_hash=implementation_identity()['sha256'],manifest_hash=study.manifest_hash,tests_log=str(path),time=time.time())
    write_json(Path(out)/'VALIDATION_RECEIPT.json',receipt);return receipt


def smoke(study,out,dataset,device):
    from .worker import create_session,train_batch,collect_observations,export_session,restore_session
    from .panels import load_panels
    from experiments.overnight_ab_20261103.data_bridge import cached_bundle
    from experiments.overnight_ab_20261103.state import state_digest
    from experiments.mdmt_history_information_20261103 import protocol as history
    prepared=read_json(Path(out)/'PREPARED.json');bundle=cached_bundle(prepared['cache_root'],dataset,include_test=False)
    panels=load_panels(out,dataset,2022,bundle);results=[];start=time.perf_counter()
    for objective in ('clean','gaussian_d0'):
        job=next(j for j in prepared['jobs'] if j['dataset']==dataset and j['objective']==objective and j['seed']==2022)
        observed=create_session(job,bundle,device,panels);baseline=create_session(job,bundle,device,panels)
        observed['pulse'].begin(observed['bank'].get('F'),1,0);S=job['steps_per_epoch'];batch=job['config']['batch_size']
        tail=len(bundle.train_y)-(S-1)*batch
        train_start=time.perf_counter()
        for step in range(S):
            ids=np.arange(batch) if step<S-1 else np.arange(len(bundle.train_y)-tail,len(bundle.train_y))
            x=torch.as_tensor(np.array(bundle.train_x[ids]),device=device).T;y=torch.as_tensor(np.array(bundle.train_y[ids]),device=device)
            train_batch(baseline,x,y,observe=False);train_batch(observed,x,y,observe=True)
        if str(device).startswith('cuda'): torch.cuda.synchronize(device)
        training_seconds=time.perf_counter()-train_start
        if state_digest(baseline['model'].state_dict())!=state_digest(observed['model'].state_dict()) or state_digest(baseline['optimizer'].state_dict())!=state_digest(observed['optimizer'].state_dict()): raise FloatingPointError('Observer changed online/Adam')
        if not torch.equal(baseline['bank'].get('F'),observed['bank'].get('F')): raise FloatingPointError('Source F changed')
        rng=history.rng_state(device);before_optimizer=state_digest(observed['optimizer'].state_dict());before_loader=observed['loader'].get_state().clone();before=state_digest(observed['model'].state_dict());before_grad=[p.grad.detach().clone() for p in observed['model'].parameters()]
        data,payload,cost=collect_observations(observed,1)
        if not history.equal(rng,history.rng_state(device)) or not torch.equal(before_loader,observed['loader'].get_state()) or before_optimizer!=state_digest(observed['optimizer'].state_dict()): raise FloatingPointError('Observer changed RNG/loader/Adam')
        history.restore_rng(rng,device)
        if before!=state_digest(observed['model'].state_dict()) or any(not torch.equal(g,p.grad) for g,p in zip(before_grad,observed['model'].parameters())): raise FloatingPointError('Scratch changed online/grad')
        snapshot=export_session(observed);restored=restore_session(job,bundle,device,panels,snapshot)
        if state_digest(restored['model'].state_dict())!=before: raise FloatingPointError('RAM restore changed parameters')
        results.append(dict(objective=objective,training_seconds=training_seconds,engineering_steps=2*S,batch_size=batch,tail_rows=tail,pulse_steps=S,full_validations=len(data['validation']),pulse_readouts=len(data['pulse_readouts']),cost=cost,
            model_hash=before,optimizer_equal=True,observer_pure=True,ram_restore_equal=True))
        del observed,baseline,restored,snapshot,data,payload,before_grad
        torch.cuda.empty_cache()
    receipt=dict(status='PASS',role='ENGINEERING_ONLY',dataset=dataset,seed=2022,results=results,
        implementation_hash=implementation_identity()['sha256'],manifest_hash=study.manifest_hash,elapsed_seconds=time.perf_counter()-start,peak_worker_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(device),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(Path(out)/f'SMOKE_{dataset.upper()}.json',receipt)
    other=Path(out)/f'SMOKE_{("kuairand" if dataset=="movielens" else "movielens").upper()}.json'
    if other.exists():
        write_json(Path(out)/'RESOURCE_ESTIMATE.json',dict(smokes=[read_json(other),receipt],max_concurrent_workers=2,estimated_worker_seconds=sum(250*(r['training_seconds']/2+r['cost']['validation_seconds']+.2*(r['cost']['pulse_seconds']+r['cost']['gradient_seconds'])) for smoke_receipt in (read_json(other),receipt) for r in smoke_receipt['results']),estimation_formula='Per objective 5 seeds * 50 epochs * (smoke train seconds / 2 + full V seconds + 0.2 * (pulse + gradient seconds))',logical_jobs=20,logical_epochs=1000,batch_size=40960,admission='per GPU memory and power <85%, host RAM <90%'))
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('audit','preflight','validate','smoke','train','analyze','controls','freeze','evaluate','report','release','_guardian','_worker'):
        p=sub.add_parser(name);p.add_argument('--manifest',required=name not in ('_guardian',));p.add_argument('--out',required=True)
        if name=='preflight':
            p.add_argument('--cache-root');p.add_argument('--ml-csv');p.add_argument('--kr-csv')
        if name=='smoke': p.add_argument('--dataset',required=True,choices=['movielens','kuairand']);p.add_argument('--device',required=True)
        if name=='train': p.add_argument('--devices',type=int,nargs='+',required=True);p.add_argument('--resume',action='store_true')
        if name=='analyze': p.add_argument('--stage',type=int,choices=[1,2,3,4],required=True)
        if name=='controls': p.add_argument('--decision',required=True)
        if name=='_worker': p.add_argument('--key',required=True);p.add_argument('--device',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='_guardian':
            from .state import serve
            serve(args.out);return 0
        study=_study(args);out=Path(args.out).resolve()
        if args.command=='audit':
            from .protocol import audit
            audit(study,out)
        elif args.command=='preflight':
            from .protocol import preflight
            preflight(study,out,args.cache_root,args.ml_csv,args.kr_csv)
        elif args.command=='validate': validate(study,out)
        elif args.command=='smoke': smoke(study,out,args.dataset,args.device)
        elif args.command=='train': train(study,out,args.devices,args.resume)
        elif args.command=='_worker': worker_process(study,out,args.key,args.device)
        elif args.command=='analyze':
            from .analysis import analyze_stage
            analyze_stage(study,out,args.stage)
        elif args.command=='controls':
            from .analysis import run_controls
            run_controls(study,out,args.decision)
        elif args.command=='freeze':
            from .evaluation import freeze_endpoints
            freeze_endpoints(study,out)
        elif args.command=='evaluate':
            from .evaluation import evaluate_frozen
            evaluate_frozen(study,out)
        elif args.command=='report':
            from .analysis import generate_report
            generate_report(study,out)
        elif args.command=='release': request(out,'release')
        return 0
    except MemoryError as exc: print(str(exc),file=sys.stderr);return 4
    except FloatingPointError as exc: print(str(exc),file=sys.stderr);return 3
    except (ValueError,FileNotFoundError) as exc: print(str(exc),file=sys.stderr);return 2
    except RuntimeError as exc: print(str(exc),file=sys.stderr);return 5

if __name__=='__main__': raise SystemExit(main())
