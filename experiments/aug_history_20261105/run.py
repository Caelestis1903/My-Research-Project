"""Review-gated B execution with RAM-only state and finite continuation commands."""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from experiments.overnight_ab_20261103 import queue
from experiments.overnight_ab_20261103.data_bridge import cached_bundle
from experiments.overnight_ab_20261103.state import read_json,write_json,append_json,sha_file,state_digest
from experiments.mdmt_history_information_20261103 import protocol as history
from . import guardian, protocol, worker, selection, report

PACKAGE='experiments.aug_history_20261105.run'
DEFAULT_MANIFEST=protocol.ROOT/'research_plans/26-11-5/experiment_manifest.json'


def prepared(output,manifest):
    p=read_json(Path(output)/'PREPARED.json')
    if p['identity']['manifest_hash']!=sha_file(manifest) or p['identity']['source_hash']!=protocol.source_manifest()['sha256']:
        raise ValueError('Implementation/manifest differs from prepared identity')
    return p


def review_gate(output,p):
    path=protocol.ROOT/'research_reports/2026-11-05_aug_history/REVIEW_ACCEPTANCE.json'
    if not path.exists():
        raise RuntimeError('Independent review acceptance is required before formal train')
    review=read_json(path)
    if review.get('status') not in ('PASS','PASS WITH FIXES','PASS_WITH_FIXES') or review.get('blocking_issues') or review.get('source_hash')!=p['identity']['source_hash']:
        raise RuntimeError('Independent review has blocking issues or another source hash')
    checks=read_json(Path(output)/'STAGE0_CHECKS.json')
    if checks['B']['status']!='PASS' or checks['source_hash']!=p['identity']['source_hash']:
        raise RuntimeError('B engineering checks must pass for this implementation')
    smoke=read_json(Path(output)/'GPU_SMOKE.json')
    if smoke.get('source_hash')!=p['identity']['source_hash'] or any(smoke['datasets'].get(ds,{}).get('status')!='PASS' for ds in ('movielens','kuairand')):
        raise RuntimeError('Both original-batch GPU smokes must pass')


def gpu_smoke(output,p,devices):
    from .model_adapter import build_model
    from .objectives import source_losses
    if not torch.cuda.is_available():
        raise CostBlocked('CUDA is unavailable')
    rows={}
    for ds in ('movielens','kuairand'):
        job=next(j for j in p['jobs'] if j['dataset']==ds and j['seed']==2022)
        free,total=queue.available_ram()
        resources=[queue.gpu_resources(device) for device in devices]
        eligible=[r for r in resources if r['memory_used_mib']/r['memory_total_mib']<.85 and
            r['power_draw_w']/r['power_limit_w']<.85]
        if not eligible or free/total<=.1:
            raise CostBlocked('GPU smoke awaits an eligible device/RAM admission')
        resource=min(eligible,key=lambda r:r['memory_used_mib'])
        device=f'cuda:{resource["device"]}'
        bundle=cached_bundle(p['cache_root'],ds,include_test=False)
        session=worker.create_session(job,bundle,device)
        reference=copy.deepcopy(session['model'])
        opt=torch.optim.Adam(reference.parameters(),lr=.01,betas=(.9,.999),eps=1e-8,
            weight_decay=job['config']['weight_decay'],foreach=False,fused=False,amsgrad=False)
        gen=torch.Generator().manual_seed(2022+11000)
        ids=torch.randperm(len(bundle.train_y),generator=gen)[:40960].numpy()
        x=torch.as_tensor(np.array(bundle.train_x[ids]),device=device).T
        y=torch.as_tensor(np.array(bundle.train_y[ids]),device=device)
        torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter(); result=worker.train_batch(session,x,y)
        opt.zero_grad(set_to_none=True)
        loss,_=source_losses(reference,x,y,0,dict(num_views=1,view_family='gaussian',pairing='iid'),
            ds,2022,1,mix_lambda=.5,kappa=job['config']['kappa'])
        loss.backward(); opt.step()
        if any(not torch.equal(a,b) for a,b in zip(reference.parameters(),session['model'].parameters())):
            raise RuntimeError('Original-batch online source equivalence failed')
        torch.cuda.synchronize(device)
        rows[ds]=dict(status='PASS',purpose='ENGINEERING_ONLY',device=device,batch_size=40960,
            peak_gpu_bytes=torch.cuda.max_memory_reserved(device),wall_seconds=time.perf_counter()-started,
            loss=result['loss'],online_model_hash=state_digest(session['model'].state_dict()),
            source_model_hash=state_digest(reference.state_dict()),optimizer_steps=1,shadow_updates=1,
            admission=resource)
        del session,reference,opt,x,y,bundle
        torch.cuda.empty_cache()
    result=dict(source_hash=p['identity']['source_hash'],datasets=rows)
    write_json(Path(output)/'GPU_SMOKE.json',result)
    return result


class CostBlocked(RuntimeError):
    pass


def validate(output,p,suite,gpu,devices):
    if suite in ('A','C'):
        result={suite:dict(status='DESIGN_COMPLETE_NOT_IMPLEMENTED')}
        write_json(Path(output)/f'{suite}_CHECKS.json',result)
        return result
    command=[sys.executable,'-m','pytest',str(Path(__file__).parent/'tests'),'-q']
    result=subprocess.run(command,cwd=protocol.ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (Path(output)/'validation_tests.log').write_text(result.stdout)
    checks=dict(source_hash=p['identity']['source_hash'],B=dict(status='PASS' if result.returncode==0 else 'FAIL',
        command=command,exit_code=result.returncode,log='validation_tests.log'),
        A=dict(status='DESIGN_COMPLETE_NOT_IMPLEMENTED'),C=dict(status='DESIGN_COMPLETE_NOT_IMPLEMENTED'))
    write_json(Path(output)/'STAGE0_CHECKS.json',checks)
    if result.returncode:
        raise RuntimeError('B validation failed; see validation_tests.log')
    if gpu:
        checks['GPU_SMOKE']=gpu_smoke(output,p,devices)
    return checks


def execute_job(output,p,key,device):
    review_gate(output,p)
    job=next(j for j in p['jobs'] if j['job_id']==key)
    output=Path(output); directory=output/'runs'/key; directory.mkdir(parents=True,exist_ok=True)
    registered=False
    try:
        previous=guardian.request(output,'register',key=key,job=job,pid=os.getpid(),device=device)['state']
        registered=True
        bundle=cached_bundle(p['cache_root'],job['dataset'],include_test=False)
        session=worker.restore_session(job,bundle,device,previous) if previous else worker.create_session(job,bundle,device)
        del previous
        write_json(directory/'ALLOCATED.json',dict(pid=os.getpid(),time=time.time(),device=device))
        if session['epoch']==0 and guardian.epoch_path(output,key,1).exists():
            append_json(output/'REPLAYS.jsonl',dict(job_id=key,time=time.time(),reason='RAM_STATE_LOST',from_epoch=0))
        while not session['complete']:
            record=worker.advance_epoch(session)
            started=time.perf_counter(); snapshot=worker.export_session(session)
            guardian.request(output,'commit',key=key,state=snapshot,record=record)
            append_json(output/'IPC_COSTS.jsonl',dict(job_id=key,epoch=record['epoch'],seconds=time.perf_counter()-started))
            del snapshot
            write_json(directory/'progress.json',dict(status='RUNNING',epoch=record['epoch'],
                global_step=record['global_step'],commit_index=record['commit_index'],device=device))
        guardian.request(output,'complete',key=key)
        write_json(directory/'TRAIN_COMPLETE.json',dict(status='TRAIN_COMPLETE',epoch=50,global_step=session['global_step']))
    except Exception:
        error=traceback.format_exc()
        failure='FAILURE.json' if registered else f'REGISTRATION_FAILURE_{os.getpid()}.json'
        write_json(directory/failure,dict(error=error,time=time.time()))
        if registered:
            try:
                guardian.request(output,'failed',key=key,pid=os.getpid(),error=error)
            except Exception:
                pass
        raise


def ready_jobs(output,jobs,status,running,resume,retried):
    ready=[]
    for job in jobs:
        key=job['job_id']; row=status.get(key,{})
        if key in running or row.get('status') in ('TRAIN_COMPLETE','RELEASED') or guardian.alive(row.get('worker')):
            continue
        if row.get('status') in ('ENGINEERING_BLOCKED','RUNNING'):
            if not resume or key in retried:
                raise RuntimeError(f'Job {key} failed or lost its worker; repair then invoke train --resume')
            guardian.request(output,'retry',key=key)
            retried.add(key)
        ready.append(job)
    return ready


def train(output,p,datasets,seeds,devices,resume=False):
    review_gate(output,p)
    if seeds!=protocol.SEEDS:
        raise ValueError('Scientific train requires all five frozen seeds')
    if not torch.cuda.is_available():
        raise CostBlocked('CUDA is unavailable')
    guardian.ensure_guardian(output)
    jobs=[j for j in p['jobs'] if j['dataset'] in datasets]
    running={}; retried=set(); output=Path(output)
    smoke=read_json(output/'GPU_SMOKE.json')['datasets']
    with queue.lock('aug-history-1105-coordinator',blocking=False):
        while True:
            for key,(process,stream,device) in list(running.items()):
                code=process.poll()
                if code is not None:
                    stream.close(); del running[key]
                    if code:
                        raise RuntimeError(f'Worker {key} failed; inspect runs/{key}/worker.log')
            status=guardian.request(output,'status')['jobs']
            if all(status.get(j['job_id'],{}).get('status') in ('TRAIN_COMPLETE','RELEASED') for j in jobs):
                return dict(status='TRAIN_COMPLETE',jobs=len(jobs))
            ready=ready_jobs(output,jobs,status,running,resume,retried)
            launched=False
            for device in devices:
                pending=[key for key,(_,_,d) in running.items() if d==device and not (output/'runs'/key/'ALLOCATED.json').exists()]
                if pending or not ready:
                    continue
                resource=queue.gpu_resources(device); free,total=queue.available_ram()
                job=ready[0]; forecast=job['config']['parameter_count']*4*70
                peak=smoke[job['dataset']]['peak_gpu_bytes']
                eligible=(resource['memory_used_mib']/resource['memory_total_mib']<.85 and
                    resource['power_draw_w']/resource['power_limit_w']<.85 and free/total>.1 and
                    (resource['memory_total_mib']-resource['memory_used_mib'])*(1<<20)>1.2*peak and free>2*forecast)
                if not eligible:
                    continue
                directory=output/'runs'/job['job_id']; directory.mkdir(parents=True,exist_ok=True)
                (directory/'ALLOCATED.json').unlink(missing_ok=True)
                stream=(directory/'worker.log').open('ab')
                process=subprocess.Popen([sys.executable,'-m',PACKAGE,'_worker','--manifest',p['manifest'],
                    '--output',str(output),'--job',job['job_id'],'--device',f'cuda:{device}'],cwd=protocol.ROOT,
                    stdout=stream,stderr=stream,start_new_session=True)
                running[job['job_id']]=(process,stream,device)
                append_json(output/'ADMISSIONS.jsonl',dict(job_id=job['job_id'],dataset=job['dataset'],seed=job['seed'],
                    pid=process.pid,device=device,resource=resource,ram_available=free,ram_total=total,time=time.time()))
                ready.pop(0); launched=True
            write_json(output/'progress.json',dict(status='RUNNING',active_jobs=list(running),
                completed_jobs=sum(status.get(j['job_id'],{}).get('status')=='TRAIN_COMPLETE' for j in jobs),time=time.time()))
            time.sleep(3 if launched else 10)


def continuation(output,decision,manifest):
    decision=read_json(decision)
    frozen=selection.require_frozen(output,Path(output)/'B/selection_frozen.json')
    if decision['selection_hash']!=frozen['selection_hash']:
        raise ValueError('Continuation decision does not match frozen selection')
    commands=[]
    for action in decision['next_actions']:
        route=action['route']
        commands.append(dict(action=action,status='DESIGN_COMPLETE_NOT_IMPLEMENTED',
            specification=str(protocol.ROOT/f'research_plans/26-11-5/route_{route}/04_CODEX_EXECUTION.md'),
            command=[sys.executable,'-m',PACKAGE,'train','--manifest',str(manifest),'--route',route,
                '--datasets',action['dataset'],'--seeds',*map(str,protocol.SEEDS),'--output',str(output),'--resume']))
    result=dict(status='AWAITING_TRIGGERED_IMPLEMENTATION' if commands else 'STOP_AND_REPORT',commands=commands)
    write_json(Path(output)/'CONTINUATION.json',result)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest='command',required=True)
    for name in ('preflight','validate','train','select','decide','evaluate','continue','report','_guardian','_worker','release'):
        sub=subs.add_parser(name)
        sub.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
        sub.add_argument('--output',type=Path,required=True)
        sub.add_argument('--route',choices=('A','B','C'),default='B')
        if name in ('validate','train'):
            sub.add_argument('--devices',type=int,nargs='+',default=[0,1])
        if name=='validate':
            sub.add_argument('--suite',choices=('all','B','A','C'),default='all')
            sub.add_argument('--gpu-smoke',action='store_true')
        if name=='train':
            sub.add_argument('--datasets',nargs='+',choices=('movielens','kuairand'),default=['movielens','kuairand'])
            sub.add_argument('--seeds',nargs='+',type=int,default=protocol.SEEDS)
            sub.add_argument('--resume',action='store_true')
        if name=='select': sub.add_argument('--split',choices=('validation',),default='validation')
        if name=='decide': sub.add_argument('--basis',choices=('validation',),default='validation')
        if name=='evaluate':
            sub.add_argument('--frozen',type=Path,required=True)
            sub.add_argument('--device',default='cuda:0')
        if name=='continue':
            sub.add_argument('--decision',type=Path,required=True)
            sub.add_argument('--resume',action='store_true')
        if name=='_worker':
            sub.add_argument('--job',required=True); sub.add_argument('--device',required=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv); args.output=args.output.resolve()
    try:
        if args.command=='_guardian':
            guardian.serve(args.output); return 0
        if args.command=='preflight':
            result=protocol.preflight(args.manifest,args.output)
            print(json.dumps(dict(status='PREFLIGHT_PASS',identity=result['identity']))); return 0
        p=prepared(args.output,args.manifest)
        if args.route!='B' and args.command not in ('validate',):
            raise RuntimeError(f'Route {args.route}: DESIGN_COMPLETE_NOT_IMPLEMENTED; follow its frozen specification and independent review')
        if args.command=='validate': result=validate(args.output,p,args.suite,args.gpu_smoke,args.devices)
        elif args.command=='train': result=train(args.output,p,args.datasets,args.seeds,args.devices,args.resume)
        elif args.command=='_worker': execute_job(args.output,p,args.job,args.device); return 0
        elif args.command=='select': result=selection.select(args.output)
        elif args.command=='decide': result=report.decide(args.output)
        elif args.command=='evaluate': result=report.evaluate(args.output,args.frozen,args.device)
        elif args.command=='continue': result=continuation(args.output,args.decision,args.manifest)
        elif args.command=='report': result=report.report(args.output)
        elif args.command=='release':
            result=guardian.request(args.output,'release'); guardian.request(args.output,'shutdown')
        print(json.dumps(result,ensure_ascii=False)); return 0
    except selection.Incomplete as error:
        print(str(error),file=sys.stderr); return 5
    except CostBlocked as error:
        print(str(error),file=sys.stderr); return 4
    except (ValueError,FloatingPointError) as error:
        print(str(error),file=sys.stderr); return 2
    except Exception as error:
        print(f'{type(error).__name__}: {error}',file=sys.stderr); return 3


if __name__=='__main__':
    raise SystemExit(main())
