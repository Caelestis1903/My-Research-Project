"""Audit, prepare, smoke, execute and inspect the finite RAM-only EMA study."""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .protocol import ROOT,resolve_plan,audit_source,prepare,implementation_identity,make_job,parent_id
from .state import request,serve,ensure_guardian,unique_bytes
from experiments.overnight_ab_20261103.state import read_json,write_json,state_digest


def worker(out,key,device):
    from .trajectory import create_parent,restore_session,fork_at_epoch,advance_epoch,export_session,reduce_session
    from .distillation import create_student
    out=Path(out);prepared=read_json(out/'PREPARED.json');job=read_json(out/'runs'/key/'JOB.json')
    if prepared['identity']['implementation_hash']!=implementation_identity()['sha256']:raise ValueError('Implementation changed since prepare')
    answer=request(out,'register',key=key,job=job,pid=os.getpid())
    if answer.get('complete'):return
    try:
        if answer['state']:
            s=restore_session(job,prepared,device,answer['state'],answer['objects'])
        elif job['family']=='PARENT':s=create_parent(job,prepared,device)
        else:
            parent=request(out,'get',key=job['parent_id']);state=parent['row']['state'];objects=parent['objects']
            if job['family'] in ('B_DISTILL','KD_LABEL','KD_FTEACHER'):
                teacher=objects[state['selected'][job['teacher_readout']]['packet_id']]
                s=create_student(job,prepared,device,teacher,objects)
            elif job.get('continuation'):s=fork_at_epoch(job,prepared,device,state,objects)
            else:
                snapshot=objects[state['anchors'][str(job.get('anchor',job['start_epoch']))]]
                s=fork_at_epoch(job,prepared,device,snapshot,objects)
            del parent,objects,state
        del answer
        if s.get('initial_record') is not None:
            path=out/'runs'/key/'epochs'/'E000.json';initial=s['initial_record']
            if path.exists() and read_json(path)!=initial:raise ValueError('KD e0 replay mismatch')
            write_json(path,initial)
        while s['epoch']<job['end_epoch']:
            record,objects=advance_epoch(s);state=export_session(s)
            result=request(out,'commit',key=key,pid=os.getpid(),state=state,new_objects=objects,record=record)
            del objects,state;reduce_session(s)
            print(json.dumps(dict(job_id=key,epoch=s['epoch'],**result)),flush=True)
        keep_live=job['family'] in ('A_MATCH','A_RAW') and job['end_epoch']==26
        request(out,'complete',key=key,pid=os.getpid(),keep_live=keep_live)
        if job['family'] in ('A_MATCH','A_RAW') and job['end_epoch']==41:request(out,'drop',key=key)
        write_json(out/'runs'/key/'COMPLETE.json',dict(status='COMPLETE',job_id=key,config_hash=job['config_hash'],start_epoch=job['start_epoch'],end_epoch=job['end_epoch'],
            logical_epochs=job['end_epoch']-job['start_epoch'],logical_adam_steps=(job['end_epoch']-job['start_epoch'])*job['config']['steps_per_epoch']))
    except Exception:
        request(out,'failed',key=key,pid=os.getpid(),error=traceback.format_exc());raise


def smoke(out,ds,device):
    from .trajectory import create_parent,advance_epoch,export_session,restore_session,fork_at_epoch,train_one_step
    from .distillation import create_student
    from experiments.ema_mechanism_20261106.worker import create_session as source_session,train_batch as source_batch
    from experiments.mdmt_history_information_20261103 import protocol as source
    out=Path(out);p=read_json(out/'PREPARED.json');job=next(j for j in p['jobs'] if j['dataset']==ds and j['family']=='PARENT' and j['seed']==2022)
    if torch.device(device).type!='cuda':raise ValueError('GPU smoke requires CUDA')
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
    observed=create_parent(job,p,device);baseline=source_session(job,observed['bundle'],device,observed['panels'])
    batch=job['config']['batch_size'];order=torch.randperm(len(observed['bundle'].train_y),generator=baseline['loader']).numpy()
    for offset in range(0,len(order),batch):
        ids=order[offset:offset+batch];x=torch.as_tensor(np.array(observed['bundle'].train_x[ids]),device=device).T;y=torch.as_tensor(np.array(observed['bundle'].train_y[ids]),device=device)
        source_batch(baseline,x,y,observe=False)
    baseline_rng=source.rng_state(device);record,objects=advance_epoch(observed)
    for name in ('model','optimizer'):
        if state_digest(baseline[name].state_dict())!=state_digest(observed[name].state_dict()):raise AssertionError('No-action changed '+name)
    if not torch.equal(baseline['bank'].get('F'),observed['bank'].get('F')):raise AssertionError('No-action changed F')
    if not torch.equal(baseline['loader'].get_state(),observed['loader'].get_state()) or not source.equal(baseline_rng,observed['rng']):raise AssertionError('Observer changed source streams')
    snapshot=export_session(observed)
    from experiments.ema_causal_20261110.run import fault_roundtrip
    recovered=fault_roundtrip(out,job,snapshot)
    restored=restore_session(job,p,device,recovered,observed['objects']);del recovered
    ids=order[:batch];x=torch.as_tensor(np.array(observed['bundle'].train_x[ids]),device=device).T;y=torch.as_tensor(np.array(observed['bundle'].train_y[ids]),device=device)
    source.restore_rng(observed['rng'],device)
    train_one_step(observed,x,y,ids);train_one_step(restored,x,y,ids)
    if state_digest(observed['model'].state_dict())!=state_digest(restored['model'].state_dict()) or state_digest(observed['optimizer'].state_dict())!=state_digest(restored['optimizer'].state_dict()):raise AssertionError('RAM recovery next update differs')
    ajob=make_job(p,ds,2022,'A_MATCH',start_epoch=1,end_epoch=2,direction_alpha=1.,job_id='SMOKE_A_'+ds)
    a=fork_at_epoch(ajob,p,device,snapshot,observed['objects']);a_result=train_one_step(a,x,y,ids)
    if a_result['direction'] is None:raise AssertionError('A feedback was not active')
    teacher_id=f'B_H{p["manifest"]["datasets"][ds]["history_base"]}_S_logit';teacher=observed['objects'][snapshot['selected'][teacher_id]['packet_id']]
    kdjob=make_job(p,ds,2022,'B_DISTILL',start_epoch=0,end_epoch=5,student_lr=.001,distill_weight=.25,job_id='SMOKE_KD_'+ds)
    kd=create_student(kdjob,p,device,teacher,observed['objects']);kd_result=train_one_step(kd,x,y,ids)
    result=dict(status='PASS',role='ENGINEERING_ONLY',dataset=ds,seed=2022,implementation_hash=implementation_identity()['sha256'],
        no_action_exact=True,ram_recovery_next_step_equal=True,worker_fault_exit_code=37,full_epoch_steps=job['config']['steps_per_epoch'],readout_count=len(record['readouts']),
        epoch_cost=record['cost'],A_step=a_result,KD_step=kd_result,smoke_seconds=time.perf_counter()-start,
        gpu_peak_bytes=torch.cuda.max_memory_allocated(device),live_snapshot_bytes=unique_bytes(snapshot),objects_bytes=unique_bytes(objects))
    write_json(out/f'SMOKE_{ds.upper()}.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['audit','prepare','smoke','execute','status','resume','analyze','freeze','evaluate','report','release','_guardian','_worker'])
    parser.add_argument('--plan-dir',default='research_plans/26-11-11_ema_next_phase');parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--cache-root');parser.add_argument('--dataset',choices=['movielens','kuairand']);parser.add_argument('--devices',nargs='+',type=int,default=[0,1]);parser.add_argument('--device',default='cuda:0');parser.add_argument('--key');parser.add_argument('--through',choices=['report'],default='report')
    args=parser.parse_args(argv);out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.command=='_guardian':return serve(out)
    if args.command=='_worker':return worker(out,args.key,args.device)
    if args.command in ('audit','prepare'):
        study=resolve_plan(args.plan_dir);result=audit_source(study,out) if args.command=='audit' else prepare(study,out,args.cache_root)
        print(json.dumps(dict(status='COMPLETE',command=args.command,identity=result.get('identity'))));return
    if args.command=='smoke':result=smoke(out,args.dataset,args.device)
    elif args.command in ('execute','resume'):
        from .scheduler import execute
        result=execute(out,args.devices)
    elif args.command=='status':
        from .state import alive
        info=read_json(out/'guardian.json') if (out/'guardian.json').exists() else {}
        result=request(out,'status') if alive(info.get('owner')) else dict(status='NO_LIVE_GUARDIAN',release=read_json(out/'RELEASE_RECEIPT.json') if (out/'RELEASE_RECEIPT.json').exists() else None)
    elif args.command=='release':result=request(out,'release')
    else:
        from .report import freeze,evaluate_test,report
        p=read_json(out/'PREPARED.json')
        if args.command=='freeze':result=freeze(out,p,read_json(out/'DECISIONS.json')['datasets'])
        elif args.command=='evaluate':result=evaluate_test(out,p,args.device)
        else:result=report(out,p)
    print(json.dumps(result,default=str))

if __name__=='__main__':main()
