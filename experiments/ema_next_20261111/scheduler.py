"""Single-owner, resource-admitted execution of the finite conditional study."""
import copy
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from .protocol import ROOT,SEEDS,make_job,parent_id,implementation_identity
from .state import request,ensure_guardian,alive,unique_bytes,disk_admit
from experiments.overnight_ab_20261103.state import read_json,write_json,append_json,digest
from . import decision as D


def load_records(out,key):
    rows=[]
    for path in sorted((Path(out)/'runs'/key/'epochs').glob('E*.json')):
        row=read_json(path)
        if row['epoch'] and not (Path(out)/'runs'/key/'commits'/path.name).exists():continue
        rows.append(row)
    return rows


def committed(job,out):
    path=Path(out)/'runs'/job['job_id']/'COMPLETE.json'
    return path.exists() and read_json(path).get('config_hash')==job['config_hash']


def gpu_state(devices):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,power.draw,power.limit','--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(v[0]),total=float(v[1])*2**20,used=float(v[2])*2**20,power=float(v[3]),power_limit=float(v[4]))
            for line in raw.splitlines() if (v:=[x.strip() for x in line.split(',')]) and int(v[0]) in devices]


def admit(prepared,status,job,devices,active_count):
    P=job['config']['parameter_count'];ds=job['dataset'];peak=prepared['resource_estimate']['datasets'][ds]['peak']
    extra=(20*P*4)+(32<<20)
    guardian=max(status['payload_bytes']+3*extra,peak['guardian_predicate'])+active_count*2*extra
    layout=prepared['resource_estimate']['datasets'][ds]['layout_elements']
    closure_bytes=4*(sum(layout['best'].values())+sum(layout['prefix'].values())+sum(layout['anchors'].values())+layout['active'])
    physical=max(status['payload_bytes']+5*extra+closure_bytes,peak['physical_tensor_bytes'])+active_count*(closure_bytes+4*extra)
    memory={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
    host_used=memory['MemTotal']-memory['MemAvailable'];host_reserve=(2<<30)+4*extra
    receipt=dict(job_id=job['job_id'],guardian_forecast=guardian,physical_forecast=physical,host_used=host_used,host_reserve=host_reserve)
    if guardian>4<<30:return None,dict(receipt,reason='guardian_commit_4GiB')
    if physical>7<<30:return None,dict(receipt,reason='physical_tensor_7GiB')
    if host_used+host_reserve>=.9*memory['MemTotal']:return None,dict(receipt,reason='host_90pct')
    calibration=prepared.get('calibration',{}).get(ds,{})
    reserve=calibration.get('gpu_peak_bytes',2<<30)
    gpus=gpu_state(devices)
    for g in sorted(gpus,key=lambda v:v['used']/v['total']):
        if g['used']+reserve<.85*g['total'] and g['power']<.85*g['power_limit']:
            return g['index'],dict(receipt,gpu=g,gpu_reserve=reserve,reason='ADMITTED')
    return None,dict(receipt,gpus=gpus,gpu_reserve=reserve,reason='GPU_memory_or_power_85pct')


def run_jobs(out,prepared,jobs,devices):
    out=Path(out);active={};logs={}
    for job in jobs:
        path=out/'runs'/job['job_id']/'JOB.json'
        if path.exists() and read_json(path)!=job:raise ValueError('Job identity changed '+job['job_id'])
        write_json(path,job)
    last_reason=None
    while True:
        status=request(out,'status');rows=status['jobs']
        for key,p in list(active.items()):
            if p.poll() is not None:
                logs.pop(key).close();active.pop(key)
                if p.returncode and rows.get(key,{}).get('launches',0)>=2:raise RuntimeError('Worker restart exhausted '+key)
        done=[j for j in jobs if rows.get(j['job_id'],{}).get('status') in ('COMPLETE','RELEASED') and committed(j,out)]
        if len(done)==len(jobs):return
        launched=False
        for job in jobs:
            key=job['job_id'];row=rows.get(key,{})
            if job in done or key in active or alive(row.get('worker')):continue
            if row.get('launches',0)>=2:raise RuntimeError('Worker restart exhausted '+key)
            active_count=sum(alive(r.get('worker')) for r in rows.values())
            device,receipt=admit(prepared,status,job,devices,active_count)
            reason=(key,receipt['reason'])
            if reason!=last_reason or device is not None:append_json(out/'RESOURCE_ADMISSION.jsonl',dict(receipt,time=time.time()));last_reason=reason
            if device is None:continue
            directory=out/'runs'/key;logs[key]=(directory/'worker.log').open('ab')
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            active[key]=subprocess.Popen([sys.executable,'-m','experiments.ema_next_20261111.run','_worker','--out',str(out),'--key',key,'--device',f'cuda:{device}'],
                cwd=ROOT,env=env,stdout=logs[key],stderr=logs[key],stdin=subprocess.DEVNULL,start_new_session=True)
            launched=True;time.sleep(3);break
        if not launched:time.sleep(3)


def points_for(out,jobs,rid,prefix=None):
    result={}
    for j in jobs:
        rows=(prefix or {}).get(j['seed'],[])+load_records(out,j['job_id'])
        result[j['seed']]=D.selected(rows,rid)
    return result


def gate_write(out,dataset,name,value):
    path=Path(out)/'decisions'/dataset/(name+'.json')
    if path.exists() and read_json(path)!=value:raise ValueError('Frozen decision changed '+dataset+'/'+name)
    write_json(path,value)


def candidate_registry(jobs,rid,spec):
    return dict(spec=spec,owners={str(j['seed']):dict(job_id=j['job_id'],readout_id=rid) for j in jobs})


def append_registry(registry,key,jobs,rid,spec):registry[key]=candidate_registry(jobs,rid,spec)


def a_short_jobs(prepared,ds,family,alpha=None,frozen_B=None,objective='clean'):
    jobs=[]
    for seed in SEEDS:
        for anchor in (20,35):
            suffix=f'.a{alpha:g}' if alpha is not None else ''
            job=make_job(prepared,ds,seed,family,objective=objective,parent_id=parent_id(ds,seed,objective),start_epoch=anchor,end_epoch=anchor+6,
                direction_alpha=alpha,anchor=anchor,frozen_B=frozen_B,job_id=f'A.{"M" if family=="A_MATCH" else "R"}.{ds}.{seed}.e{anchor}{suffix}')
            jobs.append(job)
    return jobs


def full_jobs(prepared,ds,short_jobs,family,frozen_B):
    return [make_job(prepared,ds,j['seed'],family,parent_id=j['job_id'],start_epoch=26,end_epoch=50,direction_alpha=j.get('direction_alpha'),
        frozen_B=frozen_B,continuation=True,job_id=f'A.FULL.{ds}.{j["seed"]}.{family}') for j in short_jobs if j['start_epoch']==20]


def student_jobs(prepared,ds,parents,teacher_rid,family,lr,weight,objective='clean',tag=''):
    return [make_job(prepared,ds,p['seed'],family,objective=objective,parent_id=p['job_id'],start_epoch=0,end_epoch=5,
        teacher_readout=teacher_rid,student_lr=lr,distill_weight=weight,job_id=f'KD.{tag}{family}.{ds}.{p["seed"]}.l{weight:g}.lr{lr:g}') for p in parents]


def compact_registry(out,registries):
    retained={}
    for registry in registries:
        for entry in registry.values():
            for owner in entry['owners'].values():retained.setdefault(owner['job_id'],set()).add(owner['readout_id'])
    for key,readouts in retained.items():
        request(out,'retain',key=key,readouts=sorted(readouts),anchors=False)
        request(out,'strip_teacher_refs',key=key)


def run_dataset(out,prepared,ds,devices):
    complete_path=Path(out)/'decisions'/ds/'DATASET_COMPLETE.json'
    if complete_path.exists():
        saved=read_json(complete_path);live=request(out,'summaries')['summaries']
        owners=[owner for registry in (saved['registry'],saved['gaussian']['registry']) for entry in registry.values() for owner in entry['owners'].values()]
        if all(o['job_id'] in live and o['readout_id'] in live[o['job_id']]['selected'] for o in owners):
            compact_registry(out,[saved['registry'],saved['gaussian']['registry']]);return saved
    m=prepared['manifest'];menu=prepared['readouts'][ds];specs={r['id']:dict(r,parameter_bytes=4*m['datasets'][ds]['parameter_count']*(2 if r['forwards']==2 else 1),extra_steps=0) for r in menu}
    parents=[j for j in prepared['jobs'] if j['dataset']==ds and j['family']=='PARENT'];run_jobs(out,prepared,parents,devices)
    curves={j['seed']:load_records(out,j['job_id']) for j in parents}
    result=D.parent_decisions(curves,menu,m['datasets'][ds]['history_base'])
    if result['status']!='COMPLETE':raise RuntimeError('Parent barrier incomplete')
    if any(g.get('status')=='BLOCKED_PANEL' for g in result['gates'].values()):raise RuntimeError('C future panel evidence incomplete')
    gate_write(out,ds,'PARENT_BC',result)
    frozen_B=result['B_spec'] if result['gates']['G_B']['passed'] else None
    points=result['points'];accepted=list(result['accepted']);registry={}
    keep={'O','F','B_SHARED',result['B_winner'],'C_GLOBAL_PRED','C_GLOBAL_PARAM'}
    bspec=result['B_spec'];keep.add(f'B_H{bspec["H"]}_{"all" if bspec["scope"]=="S" else "S"}_{bspec["mode"]}')
    cell_options={k:v for k,v in points.items() if k.startswith('C_CELL_')}
    cell_accepted={k:cell_options[k] for k in accepted if k in cell_options}
    cell_winner=D.choose(cell_accepted or cell_options,points['F'],order=sorted(cell_options,key=lambda k:('_s1' in k,k))) or D.choose(cell_options)
    shrink=next(r['shrinkage'] for r in menu if r['id']==cell_winner)
    keep.update(f'C_CELL_{mode}_s{shrink:g}' for mode in ('PRED','PARAM'))
    accepted=[k for k in accepted if not k.startswith('C_CELL_') or k.endswith(f'_s{shrink:g}')]
    keep.update(accepted)
    gate_write(out,ds,'C_CONFIG',dict(winner=cell_winner,shrinkage=shrink,readouts=[f'C_CELL_{mode}_s{shrink:g}' for mode in ('PRED','PARAM')]))
    for rid in sorted(keep):append_registry(registry,rid,parents,rid,specs[rid])
    for p in parents:
        request(out,'retain',key=p['job_id'],readouts=sorted(keep),anchors=True)
        request(out,'prune_anchors',key=p['job_id'],frozen_B=frozen_B)
    shortjobs=[]
    for alpha in (.5,1.):shortjobs+=a_short_jobs(prepared,ds,'A_MATCH',alpha,frozen_B)
    run_jobs(out,prepared,shortjobs,devices)
    shorts={(j['seed'],j['start_epoch'],j['direction_alpha']):load_records(out,j['job_id']) for j in shortjobs}
    ma=D.match_decision(curves,shorts);gate_write(out,ds,'A_MATCH',ma)
    decisions=dict(parent=result['gates'],A_MATCH=ma);fulls={};raw_jobs=[]
    if ma['winner'] is not None:
        for j in shortjobs:
            if j['direction_alpha']!=ma['winner']:request(out,'drop',key=j['job_id'])
        raw_jobs=a_short_jobs(prepared,ds,'A_RAW',None,frozen_B);run_jobs(out,prepared,raw_jobs,devices)
        raw=D.short_gate(curves,{(j['seed'],j['start_epoch'],None):load_records(out,j['job_id']) for j in raw_jobs},'A_RAW')
        gate_write(out,ds,'A_RAW',raw);decisions['A_RAW']=raw
        fulls['MATCH']=full_jobs(prepared,ds,[j for j in shortjobs if j['direction_alpha']==ma['winner']],'A_MATCH',frozen_B)
        if raw['passed']:fulls['RAW']=full_jobs(prepared,ds,raw_jobs,'A_RAW',frozen_B)
    else:decisions['A_RAW']=dict(status='NOT_TRIGGERED',reason='G_A_NOT_PASSED',inputs=ma)
    needed_short={j['parent_id'] for jobs in fulls.values() for j in jobs}
    for j in shortjobs+raw_jobs:
        if j['job_id'] not in needed_short:request(out,'drop',key=j['job_id'])
    for label,jobs in fulls.items():
        run_jobs(out,prepared,jobs,devices)
        prefix={s:[r for r in curves[s] if r['epoch']<=20]+load_records(out,next(j['parent_id'] for j in jobs if j['seed']==s)) for s in SEEDS}
        for rid in ('O','F'):
            key=f'A_FULL_{label}_{rid}';points[key]=points_for(out,jobs,rid,prefix);specs[key]=dict(id=key,family='A_MATCH' if label=='MATCH' else 'A_RAW',forwards=1,parameter_bytes=4*m['datasets'][ds]['parameter_count'],extra_steps=0,direction_alpha=ma['winner'] if label=='MATCH' else None)
            append_registry(registry,key,jobs,rid,specs[key])
        gate=D.value_gate('G_A_FULL',points[f'A_FULL_{label}_F'],points['F']);decisions['A_FULL_'+label]=gate
        if gate['passed']:accepted.append(f'A_FULL_{label}_F')
        if frozen_B:
            key='AB_'+label;points[key]=points_for(out,jobs,frozen_B['id'],prefix);specs[key]=dict(frozen_B,id=key,family='AB_READOUT',A_family='A_MATCH' if label=='MATCH' else 'A_RAW',direction_alpha=ma['winner'] if label=='MATCH' else None,forwards=2,parameter_bytes=8*m['datasets'][ds]['parameter_count'],extra_steps=0)
            append_registry(registry,key,jobs,frozen_B['id'],specs[key]);ab=D.ab_gate(points[key],points[f'A_FULL_{label}_F'],points[result['B_winner']]);decisions[key]=ab
            if ab['passed']:accepted.append(key)
        for j in jobs:request(out,'drop',key=j['parent_id'])
    for p in parents:request(out,'retain',key=p['job_id'],readouts=sorted(keep),anchors=False)
    for label in ('MATCH','RAW'):
        if label not in fulls:decisions['A_FULL_'+label]=dict(status='NOT_TRIGGERED',reason='CORRESPONDING_SHORT_GATE_NOT_PASSED')
        if 'AB_'+label not in decisions:decisions['AB_'+label]=dict(status='NOT_TRIGGERED',reason='B_OR_A_GATE_NOT_PASSED')
    teacher=D.teacher_choice(points,accepted,menu);decisions['KD_ENTER']=teacher;gate_write(out,ds,'KD_ENTER',teacher)
    kd_choice=None;kd_jobs={};teacher_id=teacher['teacher']
    if teacher_id:
        for lr in (.001,.003):
            for weight in (.25,.5):
                key=f'KD_B_l{weight:g}_lr{lr:g}';kd_jobs[key]=student_jobs(prepared,ds,parents,teacher_id,'B_DISTILL',lr,weight)
            key=f'KD_LABEL_lr{lr:g}';kd_jobs[key]=student_jobs(prepared,ds,parents,teacher_id,'KD_LABEL',lr,0.)
        for key,jobs in kd_jobs.items():
            run_jobs(out,prepared,jobs,devices)
            points[key]=points_for(out,jobs,'F');specs[key]=dict(id=key,family=jobs[0]['family'],student_lr=jobs[0]['student_lr'],distill_weight=jobs[0]['distill_weight'],forwards=1,parameter_bytes=4*m['datasets'][ds]['parameter_count'],extra_steps=5*m['datasets'][ds]['steps_per_epoch'])
        kds={k:v for k,v in points.items() if k.startswith('KD_B_')};order=sorted(kds,key=lambda k:(specs[k]['distill_weight'],specs[k]['student_lr']))
        kd_choice=D.choose(kds,points['F'],order=order)
        if kd_choice is None:
            decisions['KD_HISTORY']=dict(status='STOP',reason='NO_CELL_GUARD_FEASIBLE_KD_CONFIGURATION',best_unprotected=D.choose(kds,order=order))
            decisions['KD_FTEACHER']=dict(status='NOT_TRIGGERED',reason='NO_CELL_GUARD_FEASIBLE_KD_CONFIGURATION')
            for jobs in kd_jobs.values():
                for j in jobs:request(out,'drop',key=j['job_id'])
        else:
            conf=specs[kd_choice]
            gate_write(out,ds,'KD_CONFIG',dict(winner=kd_choice,spec=conf))
            f_jobs=student_jobs(prepared,ds,parents,teacher_id,'KD_FTEACHER',conf['student_lr'],conf['distill_weight']);run_jobs(out,prepared,f_jobs,devices)
            kd_jobs['KD_FTEACHER']=f_jobs;points['KD_FTEACHER']=points_for(out,f_jobs,'F');specs['KD_FTEACHER']=dict(conf,id='KD_FTEACHER',family='KD_FTEACHER')
            label='KD_LABEL_lr'+format(conf['student_lr'],'g');history=D.kd_history_gate(points[kd_choice],points['F'],points[label],points['KD_FTEACHER'],points[teacher_id]);decisions['KD_HISTORY']=history
            if history['passed']:accepted.append(kd_choice)
            for key in ('KD_LABEL_lr0.001','KD_LABEL_lr0.003','KD_FTEACHER'):
                simple=D.value_gate('G_SIMPLE_KD',points[key],points['F']);decisions[key]=simple
                if simple['passed']:accepted.append(key)
            retain_students={kd_choice,label,'KD_FTEACHER',*[k for k in accepted if k.startswith('KD_')]}
            for key,jobs in kd_jobs.items():
                if key in retain_students:
                    for rid in ('O','F'):
                        ckey=key if rid=='F' else key+'_O';points[ckey]=points_for(out,jobs,rid);specs[ckey]=dict(specs[key],id=ckey)
                        append_registry(registry,ckey,jobs,rid,specs[ckey])
                else:
                    for j in jobs:request(out,'drop',key=j['job_id'])
    if not teacher_id:
        decisions['KD_HISTORY']=dict(status='NOT_TRIGGERED',reason='G_KD_ENTER_NOT_PASSED',gate=teacher['gate'])
        decisions['KD_FTEACHER']=dict(status='NOT_TRIGGERED',reason='G_KD_ENTER_NOT_PASSED')
    request(out,'teacher_clear')
    if all(k in accepted for k in ('A_FULL_MATCH_F','A_FULL_RAW_F')):
        if abs(sum(D.auc(v) for v in points['A_FULL_MATCH_F'].values())/5-sum(D.auc(v) for v in points['A_FULL_RAW_F'].values())/5)<=1e-6:accepted.remove('A_FULL_MATCH_F')
    primary=D.primary_choice(points,accepted,specs)
    gaussian_gate=D.value_gate('G_G',points[primary],points['F']) if primary else dict(gate='G_G',passed=False,status='NOT_TRIGGERED',reason='NO_ACCEPTED_CLEAN_CANDIDATE')
    decisions['G_G']=gaussian_gate;decisions['primary']=primary or 'F';gate_write(out,ds,'CLEAN_FINAL',decisions)
    if gaussian_gate['passed']:
        gaussian=run_gaussian(out,prepared,ds,devices,primary,specs[primary],teacher_id,kd_choice,specs,menu)
    else:gaussian=dict(status='NOT_TRIGGERED',gate=gaussian_gate,registry={})
    summary=dict(dataset=ds,status='COMPLETE',primary=primary or 'F',registry=registry,gaussian=gaussian,decisions=decisions)
    gate_write(out,ds,'DATASET_COMPLETE',summary)
    compact_registry(out,[registry,gaussian['registry']])
    return summary


def run_gaussian(out,prepared,ds,devices,primary,spec,teacher_id,kd_choice,specs,menu):
    family=spec['family'];requested={'O','F'};selected_specs={x['id']:x for x in menu}
    if family in ('B_SPLIT','B_SHARED_ONLY','C_GLOBAL_PRED','C_GLOBAL_PARAM','C_CELL_PRED','C_CELL_PARAM'):requested.add(primary)
    if family=='B_SPLIT':requested.add(f'B_H{spec["H"]}_{"all" if spec["scope"]=="S" else "S"}_{spec["mode"]}');requested.add('B_SHARED')
    if family.startswith('C_'):
        requested.update(('C_GLOBAL_PRED','C_GLOBAL_PARAM'))
        if family.startswith('C_CELL'):requested.add(primary.replace('_PRED_','_PARAM_') if '_PRED_' in primary else primary.replace('_PARAM_','_PRED_'))
    if family=='AB_READOUT':requested.add(next(r['id'] for r in menu if r['family']=='B_SPLIT' and all(r[k]==spec[k] for k in ('H','scope','mode'))))
    if family in ('B_DISTILL','KD_LABEL','KD_FTEACHER'):requested.add(teacher_id)
    readouts=[selected_specs[k] for k in sorted(requested)]
    parents=[make_job(prepared,ds,s,'PARENT',objective='gaussian_d0',job_id=parent_id(ds,s,'gaussian_d0'),readout_menu=readouts) for s in SEEDS]
    run_jobs(out,prepared,parents,devices);curves={p['seed']:load_records(out,p['job_id']) for p in parents};registry={}
    for rid in requested:append_registry(registry,rid,parents,rid,selected_specs[rid])
    if family in ('A_MATCH','A_RAW','AB_READOUT'):
        afamily=spec.get('A_family',family);b=next((r for r in readouts if r['family']=='B_SPLIT'),None) if family=='AB_READOUT' else None
        jobs=[make_job(prepared,ds,s,afamily,objective='gaussian_d0',parent_id=parent_id(ds,s,'gaussian_d0'),start_epoch=20,end_epoch=50,anchor=20,direction_alpha=spec.get('direction_alpha'),frozen_B=b,job_id=f'G.CAND.{ds}.{s}.{afamily}') for s in SEEDS]
        run_jobs(out,prepared,jobs,devices)
        for rid in ('O','F'):
            append_registry(registry,'G_A_'+rid,jobs,rid,dict(spec,id='G_A_'+rid))
        if b:append_registry(registry,'G_AB',jobs,b['id'],spec)
    elif family in ('B_DISTILL','KD_LABEL','KD_FTEACHER'):
        conf=specs[kd_choice];transfer=dict(teacher_id=teacher_id,student_lr=spec['student_lr'],distill_weight=conf['distill_weight'],condition='gaussian_d0_parent__clean_student')
        gate_write(out,ds,'GAUSSIAN_TRANSFER_FREEZE',transfer)
        for fam in ('B_DISTILL','KD_LABEL','KD_FTEACHER'):
            jobs=student_jobs(prepared,ds,parents,teacher_id,fam,spec['student_lr'],0. if fam=='KD_LABEL' else conf['distill_weight'],objective='gaussian_d0_parent__clean_student',tag='G.')
            run_jobs(out,prepared,jobs,devices)
            for rid in ('O','F'):append_registry(registry,'G_'+fam+'_'+rid,jobs,rid,dict(spec,id='G_'+fam+'_'+rid,family=fam))
    request(out,'teacher_clear')
    for p in parents:request(out,'retain',key=p['job_id'],readouts=sorted(requested),anchors=False)
    return dict(status='COMPLETE',inherited_primary=primary,inherited_spec=spec,registry=registry,primary_candidate='G_AB' if family=='AB_READOUT' else 'G_A_F' if family in ('A_MATCH','A_RAW') else 'G_'+family+'_F' if family in ('B_DISTILL','KD_LABEL','KD_FTEACHER') else primary)


def execute(out,devices):
    out=Path(out).resolve();prepared=read_json(out/'PREPARED.json')
    if all((out/n).exists() for n in ('RELEASE_RECEIPT.json','TEST_COMPLETE.json','FINAL_REPORT.md')):return dict(status='COMPLETE',released=True)
    if prepared['identity']['implementation_hash']!=implementation_identity()['sha256']:raise ValueError('Prepared implementation changed')
    review=read_json(out/'IMPLEMENTATION_REVIEW.json')
    if review.get('verdict') not in ('PASS','PASS WITH FIXES') or review.get('blocking_issues') or review.get('implementation_hash')!=implementation_identity()['sha256']:raise RuntimeError('Independent hash-bound PASS required')
    for ds in prepared['manifest']['dataset_order']:
        receipt=read_json(out/f'SMOKE_{ds.upper()}.json')
        if receipt['status']!='PASS' or receipt['implementation_hash']!=implementation_identity()['sha256']:raise ValueError('Current smoke required '+ds)
        prepared.setdefault('calibration',{})[ds]=receipt
    with (out/'scheduler.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Scheduler owner already active')
        ensure_guardian(out);summaries=[]
        for ds in prepared['manifest']['dataset_order']:summaries.append(run_dataset(out,prepared,ds,devices))
        write_json(out/'DECISIONS.json',dict(status='COMPLETE',datasets=summaries))
        write_json(out/'HPO_SELECTION.json',{s['dataset']:dict(primary=s['primary'],B=read_json(out/'decisions'/s['dataset']/'PARENT_BC.json')['B_spec'],C=read_json(out/'decisions'/s['dataset']/'C_CONFIG.json'),A=s['decisions']['A_MATCH'],KD=s['decisions']['KD_ENTER']) for s in summaries})
        from .report import freeze,evaluate_test,report
        freeze(out,prepared,summaries)
        while True:
            gpus=gpu_state(devices);available=[g for g in gpus if g['used']+(2<<30)<.85*g['total'] and g['power']<.85*g['power_limit']]
            if available:break
            time.sleep(3)
        evaluate_test(out,prepared,f'cuda:{available[0]["index"]}')
        report(out,prepared)
        request(out,'release')
