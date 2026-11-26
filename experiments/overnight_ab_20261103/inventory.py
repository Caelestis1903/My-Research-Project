"""Signed scalar reuse, legacy separation and external ownership inventory."""
import csv
from pathlib import Path
from .state import sha_file,read_json,digest
from .selection import score

METRICS=tuple(f'd{d}t{t}_{name}' for d in range(3) for t in range(2)
              for name in ('auc','logloss','count','positive','negative'))
def read_csv(path):
    with Path(path).open(newline='') as stream: return list(csv.DictReader(stream))
def scalar_row(row):
    out=dict(row)
    for key,value in out.items():
        if key=='seed' or key=='epoch': out[key]=int(value)
        elif key.endswith(('_auc','_logloss','_count','_positive','_negative')): out[key]=float(value) if value else None
    return out
def scan(history_root,plan):
    root=Path(history_root); source=root/'research_reports/2026-11-03_mdmt_five_seed_completion'
    endpoint_file=source/'endpoints.csv'; curve_file=source/'validation_epochs.csv'
    endpoints=[]; curves=[]; aliases=[]; audits=[]
    if endpoint_file.exists() and curve_file.exists():
        original=[scalar_row(r) for r in read_csv(endpoint_file)]
        curves=[scalar_row(r) for r in read_csv(curve_file)]
        index={(r['dataset'],r['arm'],r['seed'],r['model_kind'],r['checkpoint'],r['split']):r for r in original}
        verified={}
        for row in original:
            if row['arm'] not in ('A0','C_MIX','P01','Gaussian'): continue
            path=root/row['source_result']
            if path not in verified:
                verified[path]=sha_file(path)
            if verified[path]!=row['source_result_sha256']: raise ValueError(f'Historical source checksum mismatch: {path}')
            arm='GAUSSIAN_A1' if row['arm']=='Gaussian' else row['arm']
            estimator=row['model_kind']; identity=(row['dataset'],row['arm'],row['seed'],estimator)
            if row['checkpoint']=='selected':
                history=[c for c in curves if (c['dataset'],c['arm'],c['seed'],c['model_kind'])==identity]
                if {c['epoch'] for c in history}!=set(range(1,51)): raise ValueError(f'Missing historical validation curve {identity}')
                selected=max(sorted(history,key=lambda c:c['epoch']),key=score)
                if selected['epoch']!=row['epoch']: raise ValueError('Historical selected epoch mismatch')
                peer=index[(*identity,'selected','test' if row['split']=='validation' else 'validation')]
                if (peer['epoch'],peer['model_hash'])!=(row['epoch'],row['model_hash']): raise ValueError('Historical V/test state mismatch')
            selection_id=digest(dict(trajectory=row['key'],estimator=estimator,selection=plan['selection']))
            out=dict(row,arm=arm,method='M3OE_FIXED1_ADAPTED' if arm=='A0' and estimator=='online' else arm,
                estimator=estimator,protocol_id=plan['protocol_id'],data_protocol=plan['data_protocol'],
                config_hash=row['configuration'],source_hash=(read_json(path)['job']['source_hash'] if path.suffix=='.json' else
                    read_json(path.parent/'formal_configuration_frozen.json')['code_hash']),
                trajectory_id=row['key'],selection_id=selection_id,
                endpoint_id=digest(dict(selection=selection_id,checkpoint=row['checkpoint'],epoch=row['epoch'],
                    model_hash=row['model_hash'],split=row['split'])),stage='train',
                role='PUBLISHED_BASELINE' if arm=='A0' and estimator=='online' else 'INTERNAL_REFERENCE' if arm=='A0' else 'COMPONENT',
                source_path=str(path),source_sha256=verified[path],reused=True,independent_run_id=row['key'])
            endpoints.append(out)
            for d in range(3):
                for task in range(2):
                    cell=f'd{d}t{task}'
                    if out.get(cell+'_negative') is None:
                        out[cell+'_negative']=out[cell+'_count']-out[cell+'_positive']
        aliases=[dict(dataset=r['dataset'],arm=r['arm'],seed=r['seed'],trajectory_id=r['trajectory_id'],
            source_path=r['source_path'],source_sha256=r['source_sha256'],model_hash=r['model_hash'])
            for r in endpoints if r['checkpoint']=='selected' and r['split']=='test']
    legacy_file=Path(__file__).resolve().parents[2]/'research_reports/2026-11-03_overnight_AB/LEGACY_A_RESELECTION_AUDIT.csv'
    legacy_rows=read_csv(legacy_file) if legacy_file.exists() else []
    for ds in plan['datasets']:
        for method in ('M3OE_FIXED1_ADAPTED','HYPERGATE','AUTOIFS'):
            for seed in plan['seeds']:
                source_method='M3OE_FIXED1' if method=='M3OE_FIXED1_ADAPTED' else method
                legacy=next((r for r in legacy_rows if r['dataset']==ds and r['method']==source_method and int(r['seed'])==seed),{})
                audits.append(dict(legacy,dataset=ds,method=method,seed=seed,eligible_main=False,legacy=True,
                    source_commit=plan['source_refs']['baseline'],action='INCOMPATIBLE',
                    reason='TRAIN_POPULATION_AND_PREPROCESS_MISMATCH',
                    historical_reselection=legacy.get('historical_reselection','AUDIT_PENDING')))
    for ds in plan['datasets']:
        for arm in ('A0','C_MIX','P01','GAUSSIAN_A1'):
            for seed in plan['seeds']:
                found=[r for r in endpoints if (r['dataset'],r['arm'],r['seed'],r['estimator'],r['checkpoint'],r['split'])==
                       (ds,arm,seed,'fixed_ema','selected','test')]
                row=found[0] if found else {}
                audits.append(dict(dataset=ds,method=arm,seed=seed,eligible_main=True,legacy=False,
                    action='REUSE_DONE' if found else 'REPLAY_MISSING',
                    reason='SIGNED_FULL_PROTOCOL_ENDPOINT' if found else 'NO_LOCAL_SIGNED_ENDPOINT_OR_ACTIVE_OWNER',
                    **{k:row.get(k) for k in ('source_path','source_sha256','trajectory_id','selection_id','epoch','model_hash')}))
    return dict(endpoints=endpoints,validation_curves=curves,audits=audits,aliases=aliases,
        source_files={str(p):sha_file(p) for p in (endpoint_file,curve_file) if p.exists()})

