"""Executable staged DFD-v4.0 workflow with immutable gates and exposed freeze."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from .state import atomic_json,digest,sha_file,load_checkpoint,restore_module,state_hash
from .data import RoleBundle
from .plans import full_plan,DATASETS,DEV_SEEDS,FORMAL_SEEDS
from .prefix import DEFAULT_ROOT,paths,run_prefix,load_prefix,source_manifest,configure,make_native,predict_native
from .demand import train_eta_and_oof
from .diagnostics import decide_p1,audit_rhd,oof_support_tables


def inspect(root,full=True):
    reports={}
    for d in DATASETS:
        data=RoleBundle(d);reports[d]=data.verify(Path(root)/'manifests',full=full)
    atomic_json(Path(root)/'DATA_ROLE_MANIFEST.json',reports);atomic_json(Path(root)/'SOURCE_MANIFEST.json',source_manifest());audit_rhd(root)
    return dict(status='PASS',data_hashes={d:r['data_hash'] for d,r in reports.items()},source_hash=source_manifest()['source_hash'])


def _gate(root,stage,dataset):
    path=Path(root)/f'DECISION_{stage.upper()}.json'
    if not path.exists():raise RuntimeError(f'Missing completed decision {stage}')
    d=json.loads(path.read_text())['datasets'][dataset]
    return d.get('status')=='PASS'


def run_stage(root,stage,datasets,seeds,device,resume,stop_after=None,arms=None):
    configure();root=Path(root);out=[]
    for dataset in datasets:
        for seed in seeds:
            if stage=='p1':
                if seed not in DEV_SEEDS:raise ValueError('P1 uses fixed development seeds')
                data=RoleBundle(dataset,access_log=root/'prefix'/dataset/f'seed{seed}'/'data_access.jsonl')
                r,h=paths(data,seed,root)
                if not resume and (h/'native_last.pt').exists():raise FileExistsError('Seed identity already used for this method; use exact --resume, never replace seed')
                result=run_prefix(data,seed,root,device,resume,stop_after)
                if result['status']=='RETAINED_BOUNDARY':out.append(result);continue
                cache,pilot=load_prefix(data,seed,root,device)
                result=train_eta_and_oof(data,cache,pilot,seed,r,device,resume,stop_after)
                if result['status']=='COMPLETE':oof_support_tables(data,r)
                out.append(dict(dataset=dataset,seed=seed,**result))
            elif stage in ('p2','p3'):
                previous='p1' if stage=='p2' else 'p2'
                if not _gate(root,previous,dataset):out.append(dict(dataset=dataset,seed=seed,status='NO_START',reason=previous.upper()+'_FAILED'));continue
                from .pipeline import run_pipeline
                data=RoleBundle(dataset);out.append(run_pipeline(data,seed,root,device=device,stage=stage,arms=arms,resume=resume))
            elif stage=='p4':
                if not _gate(root,'p3',dataset):out.append(dict(dataset=dataset,seed=seed,status='NO_START',reason='P3_NOT_PASSED'));continue
                freeze=Path(root)/'FROZEN_EVALUATION_SPEC.json'
                if not freeze.exists():raise RuntimeError('Formal runs require immutable full evaluation freeze')
                raise RuntimeError('P4 is triggered: exact external adaptations require separate implementation/review before any formal package launches')
            else:raise ValueError(stage)
    return out


def decide(root,stage):
    if stage=='p1':return decide_p1(root)
    from .pipeline import decide_stage
    result={}
    for dataset in DATASETS:
        previous='p1' if stage=='p2' else 'p2'
        if not _gate(root,previous,dataset):result[dataset]=dict(status='NO_START',reason=previous.upper()+'_NOT_PASSED');continue
        row=decide_stage(root,stage,dataset);result[dataset]=dict(row,status='PASS' if row['PASS'] else 'FAIL')
    report=dict(stage=stage,datasets=result);atomic_json(Path(root)/f'DECISION_{stage.upper()}.json',report);return report


def freeze_evaluation(root):
    root=Path(root);p1=decide_p1(root);decisions={'p1':p1}
    for stage,previous in [('p2','p1'),('p3','p2')]:
        p=root/f'DECISION_{stage.upper()}.json'
        if p.exists():decisions[stage]=json.loads(p.read_text())
        else:
            ds={}
            for dataset in DATASETS:
                prev=decisions[previous]['datasets'][dataset]
                if prev['status'] not in ('FAIL','NO_START'):raise RuntimeError('Triggered stage still incomplete: '+stage+' '+dataset)
                ds[dataset]=dict(status='NO_START',reason=previous.upper()+'_NOT_PASSED')
            decisions[stage]=dict(stage=stage,datasets=ds);atomic_json(p,decisions[stage])
    actual=[]
    for dataset in DATASETS:
        for seed in DEV_SEEDS:
            r=root/'prefix'/dataset/f'seed{seed}'
            if not (r/'PREFIX_COMPLETE.json').exists():raise RuntimeError('Missing required prefix')
            actual.extend(dict(dataset=dataset,seed=seed,arm=arm,stage='prefix',roles=['test_exposed','Q_legacy_confirm_exposed','Q_explore','validation']) for arm in ('NATIVE','LOCAL','NATIVE_EPOCH24'))
    all_sources={str(p.relative_to(Path(__file__).parent)):sha_file(p) for p in Path(__file__).parent.rglob('*.py')}
    snapshot=dict(all_source_files=all_sources,all_source_hash=digest(all_sources),spec='DFD-v4.0',plan=full_plan(),decisions=decisions,development_prefix_outputs=actual,source=source_manifest(),exposure='all test/legacy roles historically exposed; no unseen-data claim',storage='user revision: temporary full state only, final models and large caches removed after prediction/reconstruction audit',formal_datasets=[d for d in DATASETS if decisions['p3']['datasets'][d]['status']=='PASS'])
    snapshot['freeze_hash']=digest(snapshot);path=root/'FROZEN_EVALUATION_SPEC.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old!=snapshot:raise ValueError('Frozen evaluation spec already exists with different identity')
    else:atomic_json(path,snapshot,exclusive=True)
    return snapshot


def evaluate_prefix(data,seed,root,roles,device,authorization):
    import numpy as np,torch
    from .metrics import save_predictions
    r,h=paths(data,seed,root);cache,pilot=load_prefix(data,seed,root,device);out=[];reconstruction=[]
    selected_model,_=make_native(data.schema,seed,data.dataset,device);selected=load_checkpoint(h/'native_selected.pt');restore_module(selected_model,None,selected['models']['native']);selected_model.eval()
    for role in roles:
        ids=data.ids(role);data._check(ids,role,'evaluate',authorization);local=[];native_max=0.;local_max=0.
        with torch.no_grad():
            for i in range(0,len(ids),4096):
                sub=ids[i:i+4096];packet=cache.get(sub,device);lp=pilot.local_logits(packet['h'],packet['z'],packet['domain']).sigmoid();local.append(lp.cpu().numpy())
                raw=selected_model.encode(torch.as_tensor(np.array(data.x[sub]),device=device));raw_local=pilot.local_logits(raw.h,raw.logits,packet['domain']).sigmoid();native_max=max(native_max,float((raw.logits.sigmoid()-packet['z'].sigmoid()).abs().max()));local_max=max(local_max,float((raw_local-lp).abs().max()))
        if native_max>1e-6 or local_max>1e-6:raise ValueError('Prefix raw-X predictor reconstruction mismatch')
        reconstruction.append(dict(role=role,rows=len(ids),native_max_error=native_max,local_max_error=local_max,tolerance=1e-6))
        for arm,p,hash_ in [('LOCAL',np.concatenate(local),state_hash(pilot.state_dict())),('NATIVE',torch.sigmoid(torch.from_numpy(np.array(cache.z[ids]))).numpy(),cache.native_hash)]:
            m=save_predictions(r/f'{arm}_{role}.npz',ids,data.y[ids],p,data.domain[ids],role,hash_);out.append(dict(arm=arm,role=role,metrics=m))
    model,_=make_native(data.schema,seed,data.dataset,device);saved=load_checkpoint(h/'native_epoch24.pt');restore_module(model,None,saved['models']['native']);model.eval();hh=state_hash(model.state_dict())
    for role in roles:
        ids=data.ids(role);data._check(ids,role,'evaluate',authorization);p=predict_native(model,data,ids,device);m=save_predictions(r/f'NATIVE_EPOCH24_{role}.npz',ids,data.y[ids],p,data.domain[ids],role,hh);out.append(dict(arm='NATIVE_EPOCH24',role=role,metrics=m))
    atomic_json(r/'PREFIX_RECONSTRUCTION_RECEIPT.json',dict(checks=reconstruction,PASS=True))
    return out


def evaluate_exposed(root,device):
    root=Path(root);path=root/'FROZEN_EVALUATION_SPEC.json'
    if not path.exists():raise PermissionError('Exposed labels locked until immutable freeze')
    frozen=json.loads(path.read_text());payload=dict(frozen);key=payload.pop('freeze_hash')
    if digest(payload)!=key:raise ValueError('Evaluation freeze digest mismatch')
    result=[]
    for dataset in DATASETS:
        data=RoleBundle(dataset)
        for seed in DEV_SEEDS:
            result.extend(evaluate_prefix(data,seed,root,['test_exposed','Q_legacy_confirm_exposed','Q_explore','validation'],device,data.data_hash))
            if frozen['decisions']['p1']['datasets'][dataset]['status']=='PASS':
                from .pipeline import evaluate_saved
                arms=['NATIVE','LOCAL','H_DIRECT','ETA_DIRECT','UNIFORM','GEOM','SHUFFLE','CONST_UNIFORM','LOCAL_CAP','DFD']
                if frozen['decisions']['p2']['datasets'][dataset]['status']=='PASS':arms+=['U_BUDGET','POOL']
                result.extend(evaluate_saved(data,seed,root,arms,['test_exposed','Q_legacy_confirm_exposed'],device,data.data_hash))
    atomic_json(root/'EXPOSED_EVALUATION_COMPLETE.json',dict(freeze_hash=key,outputs=len(result)));return dict(status='COMPLETE',outputs=len(result))


def summarize(root):
    root=Path(root);index=[]
    for p in root.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.npz','.md'):
            index.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size))
    atomic_json(root/'ARTIFACT_INDEX.json',index)
    return dict(status='INDEXED',files=len(index),research_analysis='orchestrator owns conclusions; see research_reports/2026-10-26_v4')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['inspect','build-plan','verify-semantics','run-stage','decide','freeze-evaluation','evaluate-exposed','cleanup','summarize']);ap.add_argument('--artifact-root',default=str(DEFAULT_ROOT));ap.add_argument('--spec',default='DFD-v4.0');ap.add_argument('--stage',default='all');ap.add_argument('--dataset',choices=DATASETS);ap.add_argument('--seed',type=int);ap.add_argument('--device',default='cuda:0');ap.add_argument('--resume',action='store_true');ap.add_argument('--dry-run',action='store_true');ap.add_argument('--locked-only',action='store_true');ap.add_argument('--stop-after',choices=['native1','pilot1','eta1']);ap.add_argument('--arms',nargs='*');ap.add_argument('--logical-rows',type=int,default=40960)
    a=ap.parse_args();root=Path(a.artifact_root);root.mkdir(parents=True,exist_ok=True)
    if a.spec!='DFD-v4.0':raise ValueError('Unsupported scientific spec')
    if a.command=='inspect':out=inspect(root)
    elif a.command=='build-plan':
        out=full_plan();p=root/'PLAN.json'
        if p.exists() and json.loads(p.read_text())!=out:raise ValueError('Immutable plan mismatch')
        if not p.exists():atomic_json(p,out,exclusive=True)
    elif a.command=='verify-semantics':
        from .semantics import verify
        out=verify(root,a.device,a.logical_rows)
    elif a.command=='run-stage':out=run_stage(root,a.stage,[a.dataset] if a.dataset else DATASETS,[a.seed] if a.seed else (FORMAL_SEEDS if a.stage=='p4' else DEV_SEEDS),a.device,a.resume,a.stop_after,a.arms)
    elif a.command=='decide':out=decide(root,a.stage)
    elif a.command=='freeze-evaluation':out=freeze_evaluation(root)
    elif a.command=='evaluate-exposed':
        if not a.locked_only:raise ValueError('Explicit --locked-only required')
        out=evaluate_exposed(root,a.device)
    elif a.command=='cleanup':
        if not (root/'EXPOSED_EVALUATION_COMPLETE.json').exists():raise RuntimeError('Cleanup waits for all locked exposed evaluation and dependent stages')
        from .pipeline import cleanup_intermediates
        out=cleanup_intermediates(root)
    else:out=summarize(root)
    from .state import plain
    print(json.dumps(plain(out),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
