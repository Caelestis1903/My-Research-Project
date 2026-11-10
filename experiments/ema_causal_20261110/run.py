"""Finite EMA causal experiment; all model state remains in the guardian RAM."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from .protocol import ROOT,HERE,resolve,audit,prepare,read_json,write_json,implementation_identity,CONDITIONS,peak_active_bytes
from .state import request,ensure_guardian,serve,alive
from experiments.ema_mechanism_20261106.run import resource_admission


def worker(out,key,device):
    from .trajectory import create_session,restore_session,advance_epoch,export_session
    from experiments.ema_mechanism_20261106.panels import load_panels
    from experiments.overnight_ab_20261103.data_bridge import cached_bundle
    prepared=read_json(out/'PREPARED.json');job=next(j for j in prepared['jobs'] if j['job_id']==key)
    if prepared['identity']['implementation_hash']!=implementation_identity()['sha256']: raise ValueError('Implementation changed since prepare')
    answer=request(out,'register',key=key,job=job,pid=os.getpid())
    if answer.get('complete'): return
    try:
        bundle=cached_bundle(prepared['cache_root'],job['dataset'],include_test=False);panels=load_panels(out,job['dataset'],job['seed'],bundle)
        s=restore_session(job,bundle,device,panels,answer['state']) if answer.get('state') else create_session(job,bundle,device,panels);del answer
        while not s['complete']:
            record=advance_epoch(s);state=export_session(s)
            result=request(out,'commit',key=key,pid=os.getpid(),state=state,packets=s['pending_packets'],record=record)
            s['pending_packets'].clear();del state
            print(json.dumps(dict(job_id=key,epoch=s['epoch'],global_step=s['global_step'],**result)),flush=True)
        request(out,'complete',key=key,pid=os.getpid())
    except Exception:
        request(out,'failed',key=key,pid=os.getpid(),error=traceback.format_exc());raise


def train(study,out,devices,max_workers):
    from .block import finalize_block
    prepared=read_json(out/'PREPARED.json');identity=implementation_identity()['sha256']
    if prepared['identity']['implementation_hash']!=identity: raise ValueError('Prepared code identity changed')
    review=out/'IMPLEMENTATION_REVIEW.md'
    if not review.exists() or not any(line.strip() in ('PASS','Verdict: PASS','**Verdict: PASS**','PASS WITH FIXES','Verdict: PASS WITH FIXES') for line in review.read_text().splitlines()): raise RuntimeError('Independent review PASS required')
    for ds in study.configs:
        receipt=read_json(out/f'SMOKE_{ds.upper()}.json')
        if receipt['status']!='PASS' or receipt['implementation_hash']!=identity: raise ValueError('Current smoke required: '+ds)
    ensure_guardian(out);active={};logs={}
    for ds,obj in CONDITIONS:
        block=ds+'__'+obj;jobs=[j for j in prepared['jobs'] if j['dataset']==ds and j['objective']==obj]
        if block in request(out,'status')['blocks']:continue
        while True:
            guardian_status=request(out,'status');status=guardian_status['jobs']
            for key,process in list(active.items()):
                if process.poll() is not None:
                    logs.pop(key).close();active.pop(key)
                    if process.returncode and status.get(key,{}).get('launches',0)>=2: raise RuntimeError('Worker restart exhausted: '+key)
            completed={k for k,r in status.items() if r['status'] in ('COMPLETE','FINALIZED')}
            if all(j['job_id'] in completed for j in jobs):break
            for j in jobs:
                key=j['job_id'];row=status.get(key,{})
                if key in completed or key in active or alive(row.get('worker')):continue
                if row.get('launches',0)>=2:raise RuntimeError('Restart budget exhausted: '+key)
                if len(active)>=max_workers:break
                forecast=peak_active_bytes(ds,j['config']['parameter_count'])
                if guardian_status['payload_bytes']+(2*len(active)+3)*forecast>(4<<30):break
                admitted=resource_admission(devices)
                if not admitted:break
                device=admitted[0];directory=out/'runs'/key;directory.mkdir(parents=True,exist_ok=True);logs[key]=(directory/'worker.log').open('ab')
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                active[key]=subprocess.Popen([sys.executable,'-m','experiments.ema_causal_20261110.run','_worker','--out',str(out),'--key',key,'--device',f'cuda:{device}'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=logs[key],stderr=logs[key],start_new_session=True)
                time.sleep(3)
            time.sleep(3)
        while not (admitted:=resource_admission(devices)):time.sleep(3)
        finalize_block(out,jobs,f'cuda:{admitted[0]}')
        request(out,'finalize',block=block,keys=[j['job_id'] for j in jobs])
    write_json(out/'TRAINING_COMPLETE.json',dict(status='COMPLETE',jobs=20,epochs=1000,real_adam_steps=34500,time=time.time()))


def _fault_holder(conn):
    from experiments.mdmt_history_information_20261103.protocol import send_memory,receive_memory
    snapshot=receive_memory(conn)
    send_memory(conn,dict(pid=os.getpid(),state=snapshot))
    conn.recv()
    os._exit(37)


def fault_roundtrip(out,job,snapshot):
    import multiprocessing as mp
    from .state import Guardian
    from experiments.mdmt_history_information_20261103.protocol import send_memory,receive_memory
    parent,child=mp.get_context('spawn').Pipe()
    process=mp.get_context('spawn').Process(target=_fault_holder,args=(child,));process.start()
    send_memory(parent,snapshot);message=receive_memory(parent);pid=message['pid']
    guard=Guardian(out/'fault_injection'/job['job_id']);key=job['job_id']
    guard.handle(dict(command='register',key=key,job=job,pid=pid))
    guard.handle(dict(command='commit',key=key,pid=pid,state=message['state'],packets={},record=dict(epoch=1,cost={},role='ENGINEERING_ONLY')))
    parent.send('epoch_committed');process.join(30)
    if process.exitcode!=37:raise RuntimeError('Fault injection child did not exit as specified')
    result=guard.handle(dict(command='register',key=key,job=job,pid=os.getpid()))['state']
    parent.close();return result


def smoke(study,out,ds,device):
    from .trajectory import create_session,advance_epoch,train_batch,export_session,restore_session,full_prediction,pulse_observations,rounding_observation
    from .pairs import exact_transition_table,transition_metrics
    from experiments.ema_mechanism_20261106.worker import train_batch as source_batch
    from experiments.ema_mechanism_20261106.panels import load_panels
    from experiments.overnight_ab_20261103.data_bridge import cached_bundle
    from experiments.overnight_ab_20261103.state import state_digest,tensor_bytes
    from experiments.mdmt_history_information_20261103 import protocol as src
    prepared=read_json(out/'PREPARED.json');bundle=cached_bundle(prepared['cache_root'],ds,include_test=False);panels=load_panels(out,ds,2022,bundle);results=[]
    for obj in ('clean','gaussian_d0'):
        job=next(j for j in prepared['jobs'] if j['dataset']==ds and j['objective']==obj and j['seed']==2022)
        baseline=create_session(job,bundle,device,panels);observed=create_session(job,bundle,device,panels)
        observed['pulse'].begin(observed['bank'].get('F'),1,0)
        order=torch.randperm(len(bundle.train_y),generator=baseline['loader']).numpy();batch=job['config']['batch_size'];start=time.perf_counter()
        for offset in range(0,len(order),batch):
            ids=order[offset:offset+batch];x=torch.as_tensor(np.array(bundle.train_x[ids]),device=device).T;y=torch.as_tensor(np.array(bundle.train_y[ids]),device=device)
            source_batch(baseline,x,y,observe=False)
        baseline_rng=src.rng_state(device)
        record=advance_epoch(observed)
        if not src.equal(baseline_rng,observed['rng']):raise FloatingPointError('Observer changed RNG')
        rounding_rows=rounding_observation(observed)
        if len(rounding_rows)!=18 or not src.equal(observed['rng'],src.rng_state(device)):raise FloatingPointError('Rounding observer purity/count failed')
        pulse_start=time.perf_counter();pulse_rows=pulse_observations(observed,observed['selected_packet']);pulse_seconds=time.perf_counter()-pulse_start
        if len(pulse_rows)!=14:raise ValueError('Both seven-action menus required')
        for key in ('model','optimizer'):
            if state_digest(baseline[key].state_dict())!=state_digest(observed[key].state_dict()): raise FloatingPointError('Observer changed '+key)
        if not torch.equal(baseline['bank'].get('F'),observed['bank'].get('F')) or not torch.equal(baseline['loader'].get_state(),observed['loader'].get_state()):raise FloatingPointError('Observer changed F/loader')
        if any(not torch.equal(a.grad,b.grad) for a,b in zip(baseline['model'].parameters(),observed['model'].parameters())):raise FloatingPointError('Observer changed gradient')
        snapshot=export_session(observed);recovered=fault_roundtrip(out,job,snapshot)
        restored=restore_session(job,bundle,device,panels,recovered);del recovered
        train_batch(observed,x,y);train_batch(restored,x,y)
        if state_digest(observed['model'].state_dict())!=state_digest(restored['model'].state_dict()) or state_digest(observed['optimizer'].state_dict())!=state_digest(restored['optimizer'].state_dict()):raise FloatingPointError('RAM recovery changed next update')
        _,p,_=full_prediction(observed['scratch'],bundle);_,p0,_=full_prediction(baseline['scratch'],bundle);tic=time.perf_counter()
        counts=[]
        for d in range(3):
            mask=bundle.val_x[:,-1]==d
            for t in range(2):counts.append(transition_metrics(exact_transition_table(bundle.val_y[mask,t],p0[mask,t],p[mask,t])))
        results.append(dict(objective=obj,observer_pure=True,ram_recovery_next_step_equal=True,worker_fault_exit_code=37,rounding_observer_cells=len(rounding_rows),epoch_seconds=record['cost']['wall_seconds'],pulse_seconds=pulse_seconds,smoke_seconds=time.perf_counter()-start,pair_population_seconds=time.perf_counter()-tic,pair_counts=counts,active_payload_bytes=tensor_bytes(snapshot),batch_size=batch,tail_rows=len(order)%batch))
        del observed,baseline,restored,snapshot;torch.cuda.empty_cache()
    from .block import engineering_costs
    block=engineering_costs(out,ds,device)
    block['implementation_hash']=implementation_identity()['sha256'];write_json(out/f'BLOCK_SMOKE_{ds.upper()}.json',block)
    result=dict(status='PASS',role='ENGINEERING_ONLY',dataset=ds,seed=2022,implementation_hash=implementation_identity()['sha256'],results=results,block_calibration=block)
    write_json(out/f'SMOKE_{ds.upper()}.json',result)
    forecasts=[]
    for path in out.glob('SMOKE_*.json'):
        receipt=read_json(path);b=receipt.get('block_calibration')
        if b is None:continue
        train_seconds=sum(250*(r['epoch_seconds']+.2*r['pulse_seconds']) for r in receipt['results'])
        causal_seconds=10*(180*b['core_state_and_exact_pair_seconds']+34*b['cut_mixed_seconds']+12*b['fusion_delete_seconds']+8*b['jvp_seconds']+40*b['endpoint_pair_seconds'])
        forecasts.append(dict(dataset=receipt['dataset'],training_observer_worker_seconds=train_seconds,causal_worker_seconds_upper=causal_seconds,active_recovery_payload_bound=peak_active_bytes(receipt['dataset'],study.configs[receipt['dataset']]['parameter_count'])))
    write_json(out/'RESOURCE_ESTIMATE.json',dict(implementation_hash=implementation_identity()['sha256'],calibrations=forecasts,payload_cap_bytes=4<<30,observation_cap_bytes=2<<30,estimated_total_worker_seconds_upper=sum(r['training_observer_worker_seconds']+r['causal_worker_seconds_upper'] for r in forecasts),causal_formula_per_seed='180 core + 34 mixed cut + 12 fusion delete + 8 path integrals + 40 endpoint pairs; ten jobs per dataset',resource_rule='per-device memory and power below 85%; host RAM below 90%; active snapshot/worker/transaction payload included',state_budget='one condition block of anchors, three FP32 endpoints per job'))

    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('audit','prepare','smoke','run','resume','status','analyze','freeze','evaluate','report','release','_guardian','_worker'):
        p=sub.add_parser(command);p.add_argument('--plan-dir',default='research_plans/26-11-10_ema_causal_attribution');p.add_argument('--out',required=True)
        if command=='prepare':p.add_argument('--cache-root');p.add_argument('--ml-csv');p.add_argument('--kr-csv')
        if command=='smoke':p.add_argument('--dataset',required=True,choices=['movielens','kuairand']);p.add_argument('--device',default='cuda:1')
        if command in ('run','resume'):p.add_argument('--devices',nargs='+',type=int,required=True);p.add_argument('--max-workers',type=int,default=2)
        if command=='analyze':p.add_argument('--stage',type=int,choices=[1,2,3,4],required=True)
        if command=='evaluate':p.add_argument('--device',default='cuda:1')
        if command=='_worker':p.add_argument('--key',required=True);p.add_argument('--device',required=True)
    args=parser.parse_args(argv);out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    try:
        if args.command=='_guardian':serve(out);return 0
        if args.command=='_worker':worker(out,args.key,args.device);return 0
        if args.command=='status':print(json.dumps(request(out,'status'),indent=2));return 0
        study=resolve(args.plan_dir)
        if args.command=='audit':result=audit(study,out)
        elif args.command=='prepare':result=prepare(study,out,args.cache_root,args.ml_csv,args.kr_csv)
        elif args.command=='smoke':result=smoke(study,out,args.dataset,args.device)
        elif args.command in ('run','resume'):result=train(study,out,args.devices,args.max_workers)
        elif args.command=='release':result=request(out,'release')
        else:
            from . import analysis
            result=getattr(analysis,args.command)(study,out,**({'stage':args.stage} if args.command=='analyze' else {'device':args.device} if args.command=='evaluate' else {}))
        print(json.dumps(dict(command=args.command,status=(result or {}).get('status','COMPLETE'),out=str(out)),default=str));return 0
    except MemoryError as e: print(str(e),file=sys.stderr);return 4
    except FloatingPointError as e: print(str(e),file=sys.stderr);return 3
    except (ValueError,FileNotFoundError) as e: print(str(e),file=sys.stderr);return 2
    except RuntimeError as e: print(str(e),file=sys.stderr);return 5
if __name__=='__main__':raise SystemExit(main())
