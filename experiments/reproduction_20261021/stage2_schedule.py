#!/usr/bin/env python
"""Raw run scheduling only; no scientific analysis or automatic hyperparameter changes."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def gpu_stats(gpus):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    all_stats={int(a):(int(b)*1024**2,int(c)*1024**2) for a,b,c in (line.split(',') for line in rows.splitlines())}
    return {g:all_stats[g] for g in gpus}
def process_memory(gpus):
    """Per-GPU/PID bytes; unavailable accounting falls back to full reservation."""
    result={}
    for gpu in gpus:
        try:
            rows=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
            for line in rows.splitlines():
                pid,used=line.split(',')
                result[(gpu,int(pid))]=int(used)*1024**2
        except (subprocess.CalledProcessError,ValueError):
            # Missing/N/A process accounting must never release a reservation.
            for key in list(result):
                if key[0]==gpu: del result[key]
    return result

def projected_memory(used,jobs,observed,new_estimate):
    """Keep external use plus every outstanding reservation below the limit."""
    owned=[observed.get((j['gpu'],j['proc'].pid),0) for j in jobs]
    if sum(owned)>used:
        # Separate nvidia-smi samples can race; do not credit inconsistent data.
        outstanding=sum(j['estimate'] for j in jobs)
    else:
        outstanding=sum(max(j['estimate']-current,0) for j,current in zip(jobs,owned))
    return used+outstanding+new_estimate

def valid_completions(root):
    out={}
    for p in (root/'stage2').glob('*/*/seed*/attempt_*/completion.json'):
        j=json.loads(p.read_text())
        if j.get('valid') and j.get('formal') and j.get('stage')=='stage2':
            key=(j['dataset'],j['arm'],j['seed'])
            if key in out: raise RuntimeError('Multiple valid formal attempts: '+str(key))
            out[key]=j
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-root',default='/home_new/xzf_m3oe_reproduction_20261021');p.add_argument('--gpus',default='0,1');p.add_argument('--stage',choices=['all'],default='all');p.add_argument('--profile',action='store_true');p.add_argument('--dataset',choices=['movielens','kuairand','both'],default='both');p.add_argument('--poll-seconds',type=int,default=20);p.add_argument('--max-per-gpu',type=int,default=5); args=p.parse_args()
    if not 1<=args.max_per_gpu<=5: raise ValueError('max-per-gpu must be 1..5')
    root=Path(args.output_root); root.mkdir(parents=True,exist_ok=True);gpus=[int(x) for x in args.gpus.split(',')]
    datasets=['movielens','kuairand'] if args.dataset=='both' else [args.dataset]
    profiles_path=root/'stage2_profiles.json'; profiles=json.loads(profiles_path.read_text()) if profiles_path.exists() else {}
    if args.profile:
        # Full actual one-epoch pipelines; formal batch size stays unchanged.
        for dataset in datasets:
            for arm in ['Gtau0','Dconst01']:
                key=dataset+'/'+arm
                if key in profiles: continue
                snapshot=gpu_stats(gpus)
                # First profile has no proven bound. Require meaningful headroom,
                # measure one process at a time, and record that limitation.
                minimum_headroom=2*1024**3
                eligible=[g for g,(total,used) in snapshot.items() if used<total*.90 and total*.90-used>=minimum_headroom]
                preflight={'time':time.time(),'dataset':dataset,'arm':arm,'gpu_memory_bytes':snapshot,'eligible_gpus':eligible,'required_headroom_bytes':minimum_headroom,'estimate':'unknown until this one-process profile completes'}
                with (root/'stage2_profile_preflight.jsonl').open('a') as receipt: receipt.write(json.dumps(preflight)+'\n')
                if not eligible: raise RuntimeError('No GPU has 2GiB headroom below 90% for the unknown first-profile footprint; see stage2_profile_preflight.jsonl')
                g=max(eligible,key=lambda g:snapshot[g][0]*.90-snapshot[g][1])
                log=root/f'stage2_profile_{dataset}_{arm}.log' 
                command=[sys.executable,str(HERE/'stage2_run.py'),'--dataset',dataset,'--arm',arm,'--seed','2022','--device',f'cuda:{g}','--epochs','1','--smoke','--output-root',str(root)]
                print('PROFILE',command,flush=True)
                with log.open('w') as f: result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
                if result.returncode: raise RuntimeError('Profile failed: '+str(log))
                matches=sorted((root/'stage2_pipeline'/dataset/arm/'seed2022').glob('attempt_*/completion.json'))
                comp=json.loads(matches[-1].read_text()); assert comp['valid']
                profiles[key]={'peak_memory_bytes':comp['peak_memory_bytes'],'wall_seconds':comp['wall_seconds'],'completion':str(matches[-1])}
                profiles_path.write_text(json.dumps(profiles,indent=2))
        return
    seeds=[2022,2023,2024,2025]
    todo=[(ds,arm,seed) for ds in datasets for seed in seeds for arm in ['Gtau0','Dconst01']]
    done=valid_completions(root);todo=[key for key in todo if key not in done]
    for ds,arm,seed in todo:
        if ds+'/'+arm not in profiles: raise RuntimeError('Missing one-epoch profile '+ds+'/'+arm+'; run --profile first')
    running=[]; queue_log=root/f'stage2_queue_{args.stage}_{int(time.time())}.jsonl'
    def record(event):
        with queue_log.open('a') as f: f.write(json.dumps({'time':time.time(),**event})+'\n')
        print(json.dumps(event),flush=True)
    while todo or running:
        for job in running[:]:
            code=job['proc'].poll()
            if code is not None:
                job['log'].close();running.remove(job);record({'event':'exit','key':job['key'],'gpu':job['gpu'],'returncode':code})
                if code:
                    record({'event':'queue_halted','reason':'failed attempt; existing jobs finish and remain preserved'})
                    for active in running: active['proc'].wait(); active['log'].close()
                    raise RuntimeError('Formal run failed '+str(job['key']))
        observed=process_memory(gpus)
        stats=gpu_stats(gpus)
        for key in todo[:]:
            ds,arm,seed=key; estimate=int(profiles[ds+'/'+arm]['peak_memory_bytes']*1.15+384*1024**2)
            choices=[]
            for g,(total,used) in stats.items():
                jobs=[j for j in running if j['gpu']==g]
                # Account for external use and not-yet-realized owned reservations.
                projected=projected_memory(used,jobs,observed,estimate)
                if len(jobs)<args.max_per_gpu and projected<total*.90: choices.append((projected/total,len(jobs),g))
            if not choices: continue
            _,_,g=min(choices);command=[sys.executable,str(HERE/'stage2_run.py'),'--dataset',ds,'--arm',arm,'--seed',str(seed),'--device',f'cuda:{g}','--output-root',str(root)]
            lp=root/f'stage2_launch_{ds}_{arm}_{seed}_{int(time.time())}.log';log=lp.open('w');proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            running.append(dict(proc=proc,log=log,gpu=g,key=key,estimate=estimate)); todo.remove(key)
            record({'event':'launch','key':key,'gpu':g,'pid':proc.pid,'command':command,'log':str(lp),'estimated_memory_bytes':estimate,'gpu_used_bytes':stats[g][1],'projected_memory_bytes':projected_memory(stats[g][1],[j for j in running if j['gpu']==g],observed,0)})
        if todo or running: time.sleep(args.poll_seconds)
    done=valid_completions(root)
    missing=[key for key in [(ds,a,s) for ds in datasets for s in seeds for a in ['Gtau0','Dconst01']] if key not in done]
    if missing: raise RuntimeError('Missing valid completions '+str(missing))
    record({'event':'stage_complete','stage':'stage2','count':len(done)})
if __name__=='__main__': main()
