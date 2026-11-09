"""Finite audited PUB-MDMT workflow. Training state never leaves process RAM."""
import argparse
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from .common import *
from .data import DEFAULT_CACHE,prepare_cache,load_bundle,model_for
from .training import run_job,job_path,WD
from .group_registry import build_path_groups,GROUPS
from .optimizer import PUBAdam,losses,gradients,candidate,CELLS
from .solver import CCPSolver

DEFAULT_OUT=ROOT/'artifacts/pub_mdmt_20261109/attempt1'
REPORT=ROOT/'research_reports/2026-11-09_pub_mdmt'

def require_review(out):
    review=REPORT/'REVIEW_REPORT.md'
    if not review.exists() or not any(x in review.read_text() for x in ('Verdict: PASS','VERDICT: PASS','判定：PASS','**PASS')): raise ValueError('Independent reviewer PASS required before scientific execution')
    checks=read_json(Path(out)/'CHECKS.json')
    if checks['status']!='PASS' or checks['implementation_hash']!=implementation_hash(): raise ValueError('Required checks stale or failed')
    for ds in DATASETS:
        smoke=read_json(Path(out)/f'SMOKE_{ds}.json')
        if smoke['implementation_hash']!=implementation_hash() or smoke['status']!='PASS': raise ValueError('Both current full-batch smokes required')

def audit(out,cache):
    data=prepare_cache(cache)
    environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas','scikit-learn','cvxpy','clarabel')})
    protocol=dict(datasets=list(DATASETS),seeds=list(SEEDS),epochs=50,batch=BATCH,lr=.01,betas=[.9,.999],eps=1e-8,weight_decay=WD,loss='natural mean(row,task) BCE',selection='validation macro six-cell AUC strict max earliest',formal_trajectories=70,formal_steps=120750,model_files_written=0,implementation_hash=implementation_hash())
    write_json(Path(out)/'AUDIT.json',dict(status='PASS',data=data,environment=environment,protocol=protocol,cache_root=str(Path(cache).resolve())))
    return protocol

def check(out):
    result=subprocess.run([sys.executable,'-m','pytest',str(HERE/'tests'),'-q'],cwd=ROOT,text=True,capture_output=True)
    Path(out).mkdir(parents=True,exist_ok=True); (Path(out)/'CHECKS.log').write_text(result.stdout+result.stderr)
    receipt=dict(status='PASS' if result.returncode==0 else 'FAIL',implementation_hash=implementation_hash(),exit_code=result.returncode)
    write_json(Path(out)/'CHECKS.json',receipt)
    if result.returncode: raise RuntimeError('Correctness checks failed: '+str(Path(out)/'CHECKS.log'))
    return receipt

def smoke(out,cache,device):
    import resource
    result=[]
    for ds in DATASETS:
        rows=[]
        for topology in (0,1,2):
            setup(2022,device); b=load_bundle(ds,cache); model=model_for(b).to(device); registry=build_path_groups(model,b)
            opt=PUBAdam(model,registry,topology,weight_decay=WD[ds],solver=CCPSolver(niter=200,tol=1e-4,ridge=1e-12))
            x=torch.as_tensor(np.array(b.train_x[:BATCH]),device=device).T; y=torch.as_tensor(np.array(b.train_y[:BATCH]),device=device)
            torch.cuda.reset_peak_memory_stats(device); start=time.perf_counter()
            terms,loss,active=losses(model,x,y,topology); gs=gradients(terms,model); trace,diags,_=opt.apply(gs,active,diagnostics=True)
            torch.cuda.synchronize(device)
            if not torch.isfinite(registry.layout.flatten(model.parameters())).all(): raise FloatingPointError('Nonfinite smoke update')
            rows.append(dict(topology=topology,batch=BATCH,seconds=time.perf_counter()-start,loss=float(loss.detach()),peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device),max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,groups=diags))
            del x,y,gs,model,opt; torch.cuda.empty_cache()
        receipt=dict(status='PASS',dataset=ds,device=device,implementation_hash=implementation_hash(),observations=rows)
        write_json(Path(out)/f'SMOKE_{ds}.json',receipt); result.append(receipt)
    return result

def collect_grams(ds,seed,device,cache):
    setup(seed,device); bundle=load_bundle(ds,cache); model=model_for(bundle).to(device); registry=build_path_groups(model,bundle); layout=registry.layout
    native=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=WD[ds],foreach=False,fused=False)
    order=torch.randperm(len(bundle.train_y),generator=torch.Generator().manual_seed(seed+11000)); records=[]
    for step in range(2):
        ids=order[step*BATCH:(step+1)*BATCH].numpy(); x=torch.as_tensor(np.array(bundle.train_x[ids]),device=device).T; y=torch.as_tensor(np.array(bundle.train_y[ids]),device=device)
        terms,loss,active=losses(model,x,y,1); gs=gradients(terms,model)
        m=layout.flatten([native.state[p].get('exp_avg',torch.zeros_like(p)) for p in model.parameters()]); v=layout.flatten([native.state[p].get('exp_avg_sq',torch.zeros_like(p)) for p in model.parameters()])
        for topology in (0,2):
            opt=PUBAdam(model,registry,topology); opt.m,opt.v,opt.step=m,v,step
            matrix=torch.stack([gs[::2].sum(0),gs[1::2].sum(0)]) if topology==0 else gs
            for group in GROUPS:
                members=opt.members(group,active)
                if len(members)<2: continue
                idx=opt.indices[group]; u=candidate(matrix[members][:,idx],m[idx],v[idx],step+1).double(); G=(u@u.T).cpu().numpy()
                eig=np.linalg.eigvalsh(G); degenerate=bool(eig[0]<=1e-12*max(eig[-1],1e-30))
                records.append(dict(dataset=ds,seed=seed,ordinary_adam_steps=step,topology=topology,group=group,gram=G.tolist(),numerically_degenerate=degenerate))
        native.zero_grad(set_to_none=True)
        for p,g in zip(model.parameters(),gs.sum(0).split([p.numel() for p in model.parameters()])): p.grad=g.reshape(p.shape)
        native.step()
    return records

def calibrate(out,cache,device):
    require_review(out); out=Path(out)
    if (out/'SOLVER_DECISION.json').exists(): return read_json(out/'SOLVER_DECISION.json')
    records=[]
    for ds in DATASETS:
        for seed in SEEDS: records.extend(collect_grams(ds,seed,device,cache))
    write_json(out/'CALIBRATION_GRAMS.json',records); scored=[]
    for niter,tol,ridge in itertools.product((50,100,200),(1e-4,1e-6),(1e-12,1e-10,1e-8)):
        cfg=dict(niter=niter,tol=tol,ridge=ridge); solver=CCPSolver(**cfg); observations=[]
        for row in records:
            alpha,diag=solver.solve(row['gram']); record=dict(**row,solver=cfg,diagnostics=diag)
            append_json(out/'CALIBRATION.jsonl',record); observations.append((row,diag))
        success=all(d['success'] for r,d in observations if not r['numerically_degenerate'])
        scored.append(dict(solver=cfg,eligible=success,median_seconds=float(np.median([d['seconds'] for _,d in observations])),successful=sum(d['success'] for _,d in observations),total=len(observations)))
    eligible=[r for r in scored if r['eligible']]
    if not eligible: raise RuntimeError('No solver configuration passed real-Gram numerical calibration; inspect CALIBRATION.jsonl')
    best=min(eligible,key=lambda r:(r['median_seconds'],r['solver']['ridge'],r['solver']['niter'],r['solver']['tol']))
    decision=dict(status='FROZEN',solver=best['solver'],rule='all nondegenerate Grams pass; minimum median wall time; ties ridge/niter/tol ascending',scores=scored,implementation_hash=implementation_hash(),input_hash=digest(records))
    write_json(out/'SOLVER_DECISION.json',decision); return decision

def resource_admission(devices,required_bytes):
    info={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    if 1-info['MemAvailable']/info['MemTotal']>=.9: return []
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,power.draw,power.limit','--format=csv,noheader,nounits'],text=True)
    good=[]
    for line in raw.splitlines():
        i,used,total,power,limit=[float(x.strip()) for x in line.split(',')]
        if int(i) in devices and used/total<.85 and power/limit<.85 and (total-used)*1024**2>required_bytes:
            good.append((used/total,int(i)))
    return [i for _,i in sorted(good)]

def dispatch(jobs,out,cache,devices):
    out=Path(out); active={}; pending=list(jobs); failed=[]
    required=max(r['peak_reserved'] for ds in DATASETS for r in read_json(out/f'SMOKE_{ds}.json')['observations'])+768*1024**2
    while pending or active:
        for key,(proc,log,job) in list(active.items()):
            if proc.poll() is not None:
                log.close(); del active[key]
                if proc.returncode: failed.append((key,proc.returncode))
        if failed:
            pending=[]
            if not active: raise RuntimeError('Worker failures: '+repr(failed))
        for job in list(pending):
            path=job_path(out,job['phase'],job['dataset'],job['arm'],job['seed'],job.get('frequency',10))
            if (path/'RESULT.json').exists():
                row=read_json(path/'RESULT.json')
                if row['identity_hash']!=digest(dict(job=job,implementation_hash=implementation_hash())): raise ValueError('Existing job config differs: '+str(path))
                pending.remove(job); continue
            if job['arm'].startswith('G') and not (job_path(out,'formal',job['dataset'],'C'+job['arm'][1],job['seed'])/'RESULT.json').exists(): continue
            eligible=resource_admission(devices,required)
            if not eligible: break
            device=eligible[0]; path.mkdir(parents=True,exist_ok=True); log=(path/'worker.log').open('ab')
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            proc=subprocess.Popen([sys.executable,'-m','experiments.pub_mdmt_20261109.run','_worker','--out',str(out),'--cache',str(cache),'--device',f'cuda:{device}','--job',json.dumps(job)],cwd=ROOT,env=env,stdout=log,stderr=log,stdin=subprocess.DEVNULL)
            key=str(path.name); active[key]=(proc,log,job); pending.remove(job)
            append_json(out/'SCHEDULER.jsonl',dict(time=time.time(),event='launch',job=job,pid=proc.pid,device=device))
            time.sleep(5)
        if pending or active: time.sleep(2)
    return [read_json(job_path(out,j['phase'],j['dataset'],j['arm'],j['seed'],j.get('frequency',10))/'RESULT.json') for j in jobs]

def pilot(out,cache,devices):
    require_review(out); out=Path(out); solver=read_json(out/'SOLVER_DECISION.json')['solver']
    jobs=[dict(phase='pilot',dataset=ds,seed=s,arm=arm,frequency=f,solver=solver,epochs=2) for ds,s,arm,f in itertools.product(DATASETS,SEEDS,('C0','C1'),(1,10))]
    results=dispatch(jobs,out,cache,devices); observations=[dict(dataset=r['job']['dataset'],seed=r['job']['seed'],arm=r['job']['arm'],frequency=r['job']['frequency'],metrics=r['last_validation'][r['job']['arm']]) for r in results]
    scores={f:float(np.mean([r['metrics']['macro_auc'] for r in observations if r['frequency']==f])) for f in (1,10)}
    frequency=10 if scores[1]<=scores[10]+1e-12 else 1
    decision=dict(status='FROZEN',frequency=frequency,solver=solver,scores=scores,validation_inputs=observations,implementation_hash=implementation_hash(),input_hash=digest(observations),rule='epoch2 full Validation; equally average dataset x topology x seed; ties choose10')
    write_json(out/'SEARCH_DECISION.json',decision); return decision

def train(out,cache,devices):
    require_review(out); out=Path(out); search=read_json(out/'SEARCH_DECISION.json')
    if search['implementation_hash']!=implementation_hash(): raise ValueError('Search decision stale')
    jobs=[dict(phase='formal',dataset=ds,seed=s,arm=arm,frequency=search['frequency'],solver=search['solver'],epochs=50) for arm,ds,s in itertools.product(('A','C0','C1','G0','G1','C2','G2'),DATASETS,SEEDS)]
    dispatch([j for j in jobs if j['arm'] not in ('C2','G2')],out,cache,devices)
    dispatch([j for j in jobs if j['arm'] in ('C2','G2')],out,cache,devices)
    write_json(out/'FORMAL_COMPLETE.json',dict(status='COMPLETE',trajectories=70,epochs=3500,steps=120750,implementation_hash=implementation_hash()))
    return combination(out,cache,devices)

def combination(out,cache,devices):
    # Selection reads only frozen Validation metadata files, never RESULT/Test.
    out=Path(out); choices={}; observations={}
    for ds in DATASETS:
        eligible=[]; observations[ds]={}
        for arm in ('C0','C1','C2'):
            rows=[]
            for seed in SEEDS:
                base=read_json(job_path(out,'formal',ds,'A',seed)/'SELECTION_FROZEN.json')['selections']['A']['metrics']
                method=read_json(job_path(out,'formal',ds,arm,seed)/'SELECTION_FROZEN.json')['selections'][arm]['metrics']
                rows.append({k:method[k]-base[k] for k in ['macro_auc']+[f'd{d}t{t}_auc' for d,t in CELLS]})
            mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}; ok=mean['macro_auc']>=.003 and all(mean[f'd{d}t{t}_auc']>=-.001 for d,t in CELLS)
            observations[ds][arm]=dict(seed_deltas=rows,means=mean,eligible=ok)
            if ok: eligible.append((mean['macro_auc'],arm))
        choices[ds]=max(eligible,key=lambda x:(x[0],-int(x[1][-1])))[1] if eligible else None
    decision=dict(status='FROZEN',choices=choices,validation_inputs=observations,input_hash=digest(observations),test_inputs=False)
    write_json(out/'COMBINATION_DECISION.json',decision)
    search=read_json(out/'SEARCH_DECISION.json'); jobs=[dict(phase='combination',dataset=ds,seed=s,arm='E'+arm[-1],frequency=search['frequency'],solver=search['solver'],epochs=50) for ds,arm in choices.items() if arm for s in SEEDS]
    if jobs: dispatch(jobs,out,cache,devices)
    write_json(out/'EXPERIMENT_COMPLETE.json',dict(status='COMPLETE',formal_trajectories=70,combination_trajectories=len(jobs),choices=choices,implementation_hash=implementation_hash()))
    return decision

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['audit','check','smoke','calibrate','pilot','train','report','_worker']); p.add_argument('--out',type=Path,default=DEFAULT_OUT); p.add_argument('--cache',default=DEFAULT_CACHE); p.add_argument('--device',default='cuda:0'); p.add_argument('--devices',default='0,1'); p.add_argument('--job'); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); devices=[int(i) for i in a.devices.split(',')]
    if a.command=='audit': result=audit(a.out,a.cache)
    elif a.command=='check': result=check(a.out)
    elif a.command=='smoke': result=smoke(a.out,a.cache,a.device)
    elif a.command=='calibrate': result=calibrate(a.out,a.cache,a.device)
    elif a.command=='pilot': result=pilot(a.out,a.cache,devices)
    elif a.command=='train': result=train(a.out,a.cache,devices)
    elif a.command=='_worker': result=run_job(json.loads(a.job),a.out,a.device,a.cache)
    else:
        from .report import report
        result=report(a.out)
    print(json.dumps(result,default=str))

if __name__=='__main__': main()
