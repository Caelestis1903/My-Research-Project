"""Source-locked final export and campaign-only temporary-state cleanup.

This module is outside the running prefix/P2 dependency hashes. It never trains,
selects, refits, or changes an experiment decision. Exposed labels are read only
by evaluate() after an immutable complete predictor manifest has been frozen.
"""
from __future__ import annotations
import argparse,copy,json,os,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .state import atomic_json,digest,sha_file,state_hash,load_checkpoint,array_hash
from .plans import DATASETS,DEV_SEEDS,FORMAL_SEEDS,full_plan
from .data import RoleBundle
from .prefix import DEFAULT_ROOT,paths as prefix_paths,source_manifest as prefix_source,configure,load_prefix,identity as prefix_identity
from .native_adapter import make_native
from .functions import freeze as freeze_module
from .pipeline import (P2_ARMS,P3_ARMS,CapsulePredictor,source_identity as pipeline_source,
                       _identity as pipeline_identity,_direct_features)
from .metrics import save_predictions
from .costs import CostLedger
ROLES=('validation','Q_explore','test_exposed','Q_legacy_confirm_exposed')
MODEL_SUFFIXES={'.pt','.pth','.ckpt','.safetensors','.pkl','.pickle','.bin'}
PACKAGE=Path(__file__).resolve().parent
FREEZE_FILE='FROZEN_EVALUATION_SPEC.json'


def _inside(path,root):
    p=Path(path).resolve();r=Path(root).resolve()
    if p==r or r not in p.parents:raise ValueError('Path must be a strict descendant of campaign root: '+str(path))
    return p


def _root(root):
    root=Path(root).resolve()
    if root!=Path(DEFAULT_ROOT).resolve() or root.name!='v4_0' or root.parent.name!='mdmt_dfd_v4' or Path('/home_new') in root.parents:
        raise ValueError('Only this explicit DFD-v4 campaign root is authorized')
    return root


def _json(path):return json.loads(Path(path).read_text())


def _decision(root,stage,dataset,required=True):
    aggregate=root/f'DECISION_{stage.upper()}.json';single=root/f'DECISION_{stage.upper()}_{dataset}.json'
    if aggregate.is_file():
        value=_json(aggregate).get('datasets',{}).get(dataset)
        if value is not None:
            status=value.get('status')
            if status is None and 'PASS' in value:status='PASS' if value['PASS'] else 'FAIL'
            if status not in ('PASS','FAIL','NO_START'):raise RuntimeError(f'{stage}/{dataset} decision incomplete: {status}')
            return dict(status=status,record=value,file=str(aggregate.relative_to(root)),sha256=sha_file(aggregate))
    if single.is_file():
        value=_json(single);status=value.get('status') or ('PASS' if value.get('PASS') is True else 'FAIL' if value.get('PASS') is False else None)
        if status not in ('PASS','FAIL','NO_START'):raise RuntimeError('Incomplete stage decision '+str(single))
        return dict(status=status,record=value,file=str(single.relative_to(root)),sha256=sha_file(single))
    if required:raise RuntimeError('Required completed stage decision missing: '+stage+'/'+dataset)
    return None


def _sources():
    files={str(p.relative_to(PACKAGE)):sha_file(p) for p in PACKAGE.rglob('*.py')}
    return dict(files=files,all_source_hash=digest(files),prefix_source_hash=prefix_source()['source_hash'],pipeline_source_hash=digest(pipeline_source()))


def _checkpoint(path,root,expected_identity=None,require_complete=False):
    path=_inside(path,root)
    if not path.is_file():raise RuntimeError('Required temporary complete state is missing: '+str(path))
    st=load_checkpoint(path,expected_identity)
    if require_complete and not st.get('complete'):raise RuntimeError('Incomplete state: '+str(path))
    record=dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,file_sha256=sha_file(path),full_state_hash=state_hash(st),identity=st.get('identity'),stage=st.get('stage'),complete=st.get('complete'))
    return st,record


def _prefix_records(root,data,seed):
    run=root/'prefix'/data.dataset/f'seed{seed}';done_path=run/'PREFIX_COMPLETE.json'
    if not done_path.is_file():raise RuntimeError('Required scratch prefix incomplete: '+str(run))
    done=_json(done_path);ident=done['identity'];config_path=run/'config.json';data_path=run/'data_manifest.json';cfg=_json(config_path)
    if ident!=prefix_identity(data,seed) or any(cfg.get(k)!=v for k,v in ident.items()):raise ValueError('Prefix effective config identity mismatch')
    if cfg.get('native')!=full_plan()['native'] or cfg.get('pilot_passes')!=12 or cfg.get('fusion')!=[1,1,1,1] or cfg.get('automl') is not False or cfg.get('optimizer_count')!=1:raise ValueError('Prefix effective hyperparameters mismatch')
    if _json(data_path)!=data.manifest:raise ValueError('Prefix saved data manifest changed')
    if ident['data_hash']!=data.data_hash or ident['seed']!=seed or ident['dataset']!=data.dataset:raise ValueError('Prefix identity/data mismatch')
    if ident['source_hash']!=prefix_source()['source_hash']:raise ValueError('Prefix source changed since training')
    if done['native_epochs']!=24 or done['pilot_passes']!=12:raise ValueError('Prefix schedule mismatch')
    _,heavy=prefix_paths(data,seed,root)
    selected,sr=_checkpoint(heavy/'native_selected.pt',root,ident)
    endpoint,er=_checkpoint(heavy/'native_epoch24.pt',root,ident)
    pilot,pr=_checkpoint(heavy/'pilot.pt',root,ident,True)
    last,lr=_checkpoint(heavy/'native_last.pt',root,ident)
    if selected['pass']!=done['selected_epoch'] or selected['pass'] not in range(12,25):raise ValueError('Native selected checkpoint outside legal epochs')
    if endpoint['pass']!=24 or last['pass']!=25 or pilot['pass']!=13:raise ValueError('Required fixed prefix endpoint not complete')
    native_hash=state_hash(selected['models']['native']['model']);endpoint_hash=state_hash(endpoint['models']['native']['model']);pilot_hash=state_hash(pilot['models']['pilot']['model'])
    if native_hash!=done['native_hash'] or pilot_hash!=done['pilot_hash']:raise ValueError('Prefix selected component hash mismatch')
    if state_hash(last['models']['native']['model'])!=endpoint_hash:raise ValueError('Native final and epoch24 differ')
    cache_path=_inside(heavy/'cache',root);cache_meta=_json(cache_path/'CACHE.json')
    if cache_meta['native_hash']!=native_hash or cache_meta['cache_hash']!=done['cache_hash'] or cache_meta['data_hash']!=data.data_hash:raise ValueError('Cache dependency identity mismatch')
    if pilot.get('dependencies',{}).get('cache_hash')!=done['cache_hash']:raise ValueError('Pilot/cache dependency mismatch')
    oof_complete=run/'P1_OOF_COMPLETE.json';oof_path=run/'p1_oof.npz';oof_metrics=run/'P1_METRICS.json';oof_receipts=run/'P1_OOF_RECEIPTS.json'
    oc=_json(oof_complete);om=_json(oof_metrics)
    from .demand import source_identity as demand_source
    if oc['npz_sha256']!=sha_file(oof_path) or oc['checkpoint_hash']!=om['checkpoint_hash'] or oc['identity']['source_hash']!=demand_source() or oc['identity']['data_hash']!=data.data_hash or oc['identity']['cache_hash']!=done['cache_hash'] or oc['identity']['pilot_hash']!=pilot_hash:raise ValueError('P1 OOF/decision dependency mismatch')
    with np.load(oof_path) as oa:
        if not np.array_equal(oa['rows'],data.ids('P_form')) or str(oa['checkpoint_hash'])!=oc['checkpoint_hash']:raise ValueError('P1 OOF row/checkpoint identity mismatch')
        oof_fold_hash=array_hash(oa['fold'])
    oof_info=dict(complete_sha256=sha_file(oof_complete),npz_sha256=oc['npz_sha256'],metrics_sha256=sha_file(oof_metrics),receipts_sha256=sha_file(oof_receipts),checkpoint_hash=oc['checkpoint_hash'],identity=oc['identity'],fold_hash=oof_fold_hash)
    info=dict(config_path=str(config_path.relative_to(root)),config_sha256=sha_file(config_path),data_manifest_path=str(data_path.relative_to(root)),data_manifest_sha256=sha_file(data_path),p1_oof=oof_info,dataset=data.dataset,seed=seed,complete_path=str(done_path.relative_to(root)),complete_sha256=sha_file(done_path),identity=ident,components=dict(native_hash=native_hash,pilot_hash=pilot_hash,epoch24_hash=endpoint_hash,cache_hash=done['cache_hash']),checkpoints=dict(native_selected=sr,native_epoch24=er,pilot_pass12=pr,native_last=lr),cache_metadata=cache_meta)
    records=[]
    for arm,variant,cp,predictor_hash in [('NATIVE','selected',sr,native_hash),('NATIVE','epoch24',er,endpoint_hash),('LOCAL','pass12',pr,digest(dict(native=native_hash,pilot=pilot_hash,policy='LOCAL_PASS12')))]:
        records.append(dict(id=f'{data.dataset}_s{seed}_PREFIX_{arm}_{variant}',dataset=data.dataset,seed=seed,stage='prefix',arm=arm,variant=variant,run_dir=str(run.relative_to(root)),predictor_hash=predictor_hash,identity=ident,primary_checkpoint=cp,dependencies=dict(native_selected=sr,pilot_pass12=pr,cache_hash=done['cache_hash']),roles=list(ROLES),probability_precision='FP32 network and original prefix sigmoid; probabilities exported as FP64',endpoint_semantics='native epoch24' if variant=='epoch24' else 'fixed pilot pass12' if arm=='LOCAL' else 'validation LL-selected native'))
    return info,records


def _pipeline_records(root,data,seed,arm,stage,prefix_info):
    run=root/'runs'/data.dataset/arm/f'seed{seed}';cp=run/'COMPLETE.json';config=run/'config.json'
    if not cp.is_file():raise RuntimeError('Triggered predictor missing: '+str(run))
    done=_json(cp);ident=done['identity'];cfg=_json(config)
    prefix_hash=digest(dict(pilot=prefix_info['components']['pilot_hash'],cache_hash=prefix_info['components']['cache_hash'],native_hash=prefix_info['components']['native_hash']))
    expected=pipeline_identity(data,seed,arm,stage,prefix_hash)
    if ident!=expected or cfg!=ident:raise ValueError('Pipeline source/config/data/plan identity changed: '+str(run))
    if done.get('status')!='COMPLETE':raise ValueError('Incomplete triggered predictor: '+str(run))
    capsule,cr=_checkpoint(run/'transient'/'predictor.pt',root,ident,True)
    components=done['component_hashes']
    if capsule['predictor_hash']!=done['predictor_hash'] or capsule['component_hashes']!=components:raise ValueError('Capsule/COMPLETE predictor identity mismatch')
    if state_hash(capsule['native_state'])!=components['native_hash'] or state_hash(capsule['pilot'])!=components['pilot_hash']:raise ValueError('Capsule native/pilot tensors mismatch')
    if components['native_hash']!=prefix_info['components']['native_hash'] or components['pilot_hash']!=prefix_info['components']['pilot_hash']:raise ValueError('Pipeline borrowed another prefix')
    snapshots=dict(predictor=cr)
    if capsule['bank'] is not None:
        formation_path=run/'transient'/'formation.pt';formation_identity=ident
        if arm in ('CONST_UNIFORM','U_BUDGET'):
            formation_path=root/'runs'/data.dataset/'UNIFORM'/f'seed{seed}'/'transient'/'formation.pt'
            formation_identity=pipeline_identity(data,seed,'UNIFORM','p2' if stage=='p3' else stage,prefix_hash)
        formation,fr=_checkpoint(formation_path,root,formation_identity,True)
        if state_hash(capsule['bank']['state'])!=components['bank_hash'] or state_hash(formation['models']['main'])!=components['bank_hash']:raise ValueError('Fixed pass12 library mismatch')
        if formation.get('pass_id') not in (13,None):raise ValueError('Formation did not complete fixed pass12')
        snapshots['formation_pass12']=fr
    selected_state=None
    if capsule['reader'] is not None:
        selected_state,rr=_checkpoint(run/'transient'/'reader_selected.pt',root,ident,True);snapshots['selected_readout']=rr
        if state_hash(selected_state['models']['main'])!=state_hash(capsule['reader']['state']) or state_hash(selected_state['extra'])!=state_hash(capsule['reader']['extra']):raise ValueError('Selected whole reader capsule mismatch')
    elif capsule['direct'] is not None:
        selected_state,rr=_checkpoint(run/'transient'/'direct_selected.pt',root,ident,True);snapshots['selected_readout']=rr
        if state_hash(selected_state['models']['main'])!=state_hash(capsule['direct']['state']):raise ValueError('Selected whole direct capsule mismatch')
    elif capsule['constant'] is not None:
        selected_state,rr=_checkpoint(run/'transient'/'constant.pt',root,ident);snapshots['selected_readout']=rr
    if selected_state is not None and state_hash(selected_state)!=components['readout_state_hash']:raise ValueError('Selected readout complete state hash mismatch')
    for role in ('validation','Q_explore'):
        p=run/f'{role}_selected.npz'
        with np.load(p) as a:
            if str(a['checkpoint_hash'])!=done['predictor_hash'] or not np.array_equal(a['row_id'],data.ids(role)):raise ValueError('Saved selected prediction identity mismatch')
    base=dict(dataset=data.dataset,seed=seed,stage=stage,arm=arm,run_dir=str(run.relative_to(root)),identity=ident,complete_sha256=sha_file(cp),config_sha256=sha_file(config),component_hashes=components,capsule=cr,snapshots=snapshots,roles=list(ROLES),probability_precision='FP32 network; original P2 capsule FP64 sigmoid/mixture',formation_endpoint='fixed pass12 final library' if capsule['bank'] is not None else None)
    records=[dict(base,id=f'{data.dataset}_s{seed}_{stage}_{arm}_selected',variant='selected',predictor_hash=done['predictor_hash'],selection=selected_state.get('selection') if selected_state is not None else 'EXACT_PREFIX_POLICY')]
    if capsule['reader'] is not None or capsule['direct'] is not None:
        name='reader_progress.pt' if capsule['reader'] is not None else 'direct_progress.pt'
        terminal,tr=_checkpoint(run/'transient'/name,root,ident,True)
        if terminal.get('phase')!='P_read' or terminal.get('pass_id')!=11 or terminal.get('next_head')!=0 or terminal.get('next_logical_step')!=0:raise ValueError('Read endpoint is not complete pass10 boundary')
        endpoint_hash=digest(dict(identity=ident,components=dict(components,readout_state_hash=state_hash(terminal)),endpoint='P_read_pass10'))
        for role in ('validation','Q_explore'):
            with np.load(run/f'{role}_read_pass10.npz') as a:
                if str(a['checkpoint_hash'])!=endpoint_hash or not np.array_equal(a['row_id'],data.ids(role)):raise ValueError('Saved pass10 endpoint identity mismatch')
        records.append(dict(base,id=f'{data.dataset}_s{seed}_{stage}_{arm}_read_pass10',variant='read_pass10',predictor_hash=endpoint_hash,endpoint_checkpoint=tr,selection='FIXED_P_READ_PASS10'))
    return records


def build_manifest(root=DEFAULT_ROOT):
    """Refuse until every triggered package and its exact decision is complete."""
    root=_root(root);sources=_sources();datasets={};prefixes={};predictors=[];expected_runs=set();decisions={}
    for dataset in DATASETS:
        data=RoleBundle(dataset)
        if sha_file(data.raw_path)!=data.raw_manifest['raw_sha256']:raise ValueError('Raw source changed before final freeze')
        datasets[dataset]=dict(data_hash=data.data_hash,manifest=data.manifest,raw_manifest=data.raw_manifest,role_row_hashes={r:array_hash(data.ids(r)) for r in ROLES},user_item_support=data.support_reason or 'raw identity available')
        p1=_decision(root,'p1',dataset);decisions[dataset]={'p1':p1}
        if p1['status']=='PASS':p2=_decision(root,'p2',dataset)
        else:
            existing=_decision(root,'p2',dataset,False)
            if existing is not None and existing['status']!='NO_START':raise ValueError('P2 decision contradicts failed P1 trigger')
            p2=existing or dict(status='NO_START',reason='P1_NOT_PASSED')
        decisions[dataset]['p2']=p2
        if p2['status']=='PASS':p3=_decision(root,'p3',dataset)
        else:
            existing=_decision(root,'p3',dataset,False)
            if existing is not None and existing['status']!='NO_START':raise ValueError('P3 decision contradicts failed predecessor trigger')
            p3=existing or dict(status='NO_START',reason='P2_NOT_PASSED' if p1['status']=='PASS' else 'P1_NOT_PASSED')
        decisions[dataset]['p3']=p3
        decisions[dataset]['p4']=dict(status='REQUIRED' if p3['status']=='PASS' else 'NO_START',reason='P3_PASSED' if p3['status']=='PASS' else 'P3_NOT_PASSED')
        seeds=list(DEV_SEEDS)+(list(FORMAL_SEEDS) if p3['status']=='PASS' else [])
        if p3['status']=='PASS':
            # Exact formal author adapters need their own approved finalizer if
            # triggered; this module never silently omits them or weakens them.
            raise RuntimeError('P4 triggered: wait for complete formal internal and exact author raw/INFO manifests and their reviewed finalizer adapter')
        for seed in seeds:
            info,records=_prefix_records(root,data,seed);prefixes[f'{dataset}:{seed}']=info;predictors.extend(records)
            arms=list(P2_ARMS) if p1['status']=='PASS' else []
            if p2['status']=='PASS':arms+=list(P3_ARMS)
            for arm in arms:
                stage='p3' if arm in P3_ARMS else 'p2';expected_runs.add((dataset,arm,seed));predictors.extend(_pipeline_records(root,data,seed,arm,stage,info))
    observed=set()
    for path in (root/'runs').glob('*/*/seed*/COMPLETE.json') if (root/'runs').exists() else []:
        item=_json(path)['identity'];observed.add((item['dataset'],item['method'],int(item['seed'])))
    if observed!=expected_runs:raise RuntimeError('Actual completed predictor list differs from triggered manifest: '+str(dict(missing=sorted(expected_runs-observed),unexpected=sorted(observed-expected_runs))))
    temporary_states=[];seen_state_paths=set()
    for p in root.rglob('*.pt'):
        if p.is_symlink():continue
        absolute=_inside(p,root)
        if absolute in seen_state_paths:continue
        seen_state_paths.add(absolute);_,state_record=_checkpoint(absolute,root);temporary_states.append(state_record)
    # All reference files are immutable by this point. Decisions are recorded,
    # never recomputed using exposed-role labels or replaced by another seed.
    manifest=dict(temporary_state_identities=temporary_states,spec='DFD-v4.0',type='COMPLETE_ACTUAL_PREDICTOR_FREEZE',plan=full_plan(),sources=sources,datasets=datasets,decisions=decisions,prefixes=prefixes,predictors=predictors,roles=list(ROLES),expected_prediction_exports=len(predictors)*len(ROLES),exposure=dict(test_exposed='historically exposed benchmark',Q_legacy_confirm_exposed='historically exposed companion',Q_explore='development',validation='native selection and companion; no new selection'),selection='frozen validation-selected native and whole P_cal policy; no selection during evaluation',storage='only campaign-root temporary state; remove model/optimizer/cache after all exports and raw-X reconstruction receipts pass',finalization_source_sha256=sha_file(__file__))
    manifest['freeze_hash']=digest(manifest)
    return manifest


def freeze(root=DEFAULT_ROOT):
    root=_root(root);manifest=build_manifest(root);predictors=manifest['predictors'];path=root/FREEZE_FILE
    if path.exists():
        if _json(path)!=manifest:raise RuntimeError('Existing frozen evaluation identity differs; do not overwrite an exposed-method lock')
    else:atomic_json(path,manifest,exclusive=True)
    atomic_json(root/'ACTUAL_PREDICTOR_MANIFEST.json',dict(freeze_hash=manifest['freeze_hash'],predictors=predictors,expected_prediction_exports=manifest['expected_prediction_exports']))
    return dict(status='FROZEN',freeze_hash=manifest['freeze_hash'],predictors=len(predictors),exports=manifest['expected_prediction_exports'])


def _locked(root,allow_missing_models=False):
    root=_root(root);m=_json(root/FREEZE_FILE);payload=dict(m);key=payload.pop('freeze_hash')
    if digest(payload)!=key:raise ValueError('Frozen manifest digest mismatch')
    if m['sources']!=_sources():raise ValueError('Source changed after final evaluation freeze')
    for dataset,info in m['datasets'].items():
        checked_data=RoleBundle(dataset)
        if checked_data.data_hash!=info['data_hash'] or sha_file(checked_data.raw_path)!=info['raw_manifest']['raw_sha256']:raise ValueError('Frozen data/raw identity changed')
    for info in m['prefixes'].values():
        p=root/info['complete_path']
        if not p.is_file() or sha_file(p)!=info['complete_sha256']:raise ValueError('Prefix completion metadata changed after freeze')
        if sha_file(root/info['config_path'])!=info['config_sha256'] or sha_file(root/info['data_manifest_path'])!=info['data_manifest_sha256']:raise ValueError('Prefix source/config/data metadata changed after freeze')
    for cp in m.get('temporary_state_identities',[]):
        p=_inside(root/cp['path'],root)
        if not p.is_file():
            if allow_missing_models:continue
            raise ValueError('Temporary complete state is missing after final freeze: '+str(p))
        if sha_file(p)!=cp['file_sha256']:raise ValueError('Temporary complete state changed after final freeze: '+str(p))
    for dataset,stages in m['decisions'].items():
        for item in stages.values():
            if 'file' in item and sha_file(root/item['file'])!=item['sha256']:raise ValueError('Frozen stage decision changed')
    for rec in m['predictors']:
        run=root/rec['run_dir']
        if rec['stage']!='prefix':
            if sha_file(run/'config.json')!=rec['config_sha256'] or sha_file(run/'COMPLETE.json')!=rec['complete_sha256']:raise ValueError('Pipeline config/completion changed after freeze')
    return root,m


class PrefixPredictor(nn.Module):
    def __init__(self,record,prefix_info,data,root,device):
        super().__init__();self.arm=record['arm'];self.device=device
        state=load_checkpoint(root/record['primary_checkpoint']['path'],record['identity'])
        native_state=state['models']['native']['model'] if self.arm=='NATIVE' else load_checkpoint(root/prefix_info['checkpoints']['native_selected']['path'],record['identity'])['models']['native']['model']
        self.native,_=make_native(data.schema,record['seed'],data.dataset,device);self.native.load_state_dict(native_state);freeze_module(self.native);self.pilot=None
        if self.arm=='LOCAL':
            from .functions import PilotBank
            self.pilot=PilotBank().to(device);self.pilot.load_state_dict(state['models']['pilot']['model']);freeze_module(self.pilot)
        self.eval()
    def forward(self,x):
        with torch.no_grad():
            packet=self.native.encode(x)
            if self.arm=='NATIVE':return packet.logits.sigmoid().double()
            return self.pilot.local_logits(packet.h,packet.logits,x[:,-1].long()).sigmoid().double()


def _build(record,prefix_info,data,root,device):
    checks=[record['primary_checkpoint']] if record['stage']=='prefix' else [record['capsule']]+([record['endpoint_checkpoint']] if record['variant']=='read_pass10' else [])
    for cp in checks:
        p=_inside(root/cp['path'],root)
        if sha_file(p)!=cp['file_sha256']:raise ValueError('Frozen checkpoint bytes changed: '+str(p))
    if record['stage']=='prefix':return PrefixPredictor(record,prefix_info,data,root,device)
    cap=load_checkpoint(root/record['capsule']['path'],record['identity'])
    if record['variant']=='read_pass10':
        cap=copy.deepcopy(cap);end=load_checkpoint(root/record['endpoint_checkpoint']['path'],record['identity'])
        if cap['reader'] is not None:cap['reader']['state']=end['models']['main'];cap['reader']['extra']=end['extra']
        else:
            cap['direct']['state']=end['models']['main'];cap['direct']['normalizers']=end['normalizers']
        cap['predictor_hash']=record['predictor_hash']
    return CapsulePredictor(cap,device)


def _cached(model,packet):
    h,z,d=packet['h'],packet['z'],packet['domain']
    if isinstance(model,PrefixPredictor):
        return z.sigmoid().double() if model.arm=='NATIVE' else model.pilot.local_logits(h,z,d).sigmoid().double()
    if model.arm=='NATIVE':return z.double().sigmoid()
    if model.arm=='LOCAL':return model.pilot.local_logits(h,z,d).double().sigmoid()
    if model.direct is not None:return model.direct.predict(_direct_features(packet,model.pilot,model.direct.input_dim),d)
    a=model.bank.candidates(h,z,d,model.pilot)
    if model.reader is not None:return model.reader.predict(h,z,d,a)[0]
    probabilities=a.double().sigmoid();p=probabilities[...,0].clone()
    if model.constant['selection']!='OFF':
        for t in range(3):
            mask=d==t
            for q in range(2):p[mask,q]=probabilities[mask,q]@torch.as_tensor(model.constant['alphas'][t,q],device=h.device)
    return p


def _export_dir(root,rec):return root/'final_evaluation'/rec['dataset']/f"seed{rec['seed']}"/rec['stage']/rec['arm']/rec['variant']


def evaluate(root=DEFAULT_ROOT,device='cuda:0'):
    """Read-only frozen inference; exposed Y is fetched only after prediction."""
    configure();root,manifest=_locked(root);frozen_hash=manifest['freeze_hash'];receipts=[]
    data_cache={};feature_cache={};ledger=CostLedger(root/'final_evaluation'/'costs.jsonl',dict(stage='final_evaluation',freeze_hash=frozen_hash))
    for rec in manifest['predictors']:
        dataset=rec['dataset'];seed=rec['seed']
        if dataset not in data_cache:data_cache[dataset]=RoleBundle(dataset)
        data=data_cache[dataset];pinfo=manifest['prefixes'][f'{dataset}:{seed}']
        key=(dataset,seed)
        if key not in feature_cache:feature_cache[key]=load_prefix(data,seed,root,device)[0]
        cache=feature_cache[key];dest=_export_dir(root,rec);dest.mkdir(parents=True,exist_ok=True)
        model=None
        for role in ROLES:
            pred_path=dest/f'{role}.npz';receipt_path=dest/f'{role}.receipt.json'
            if receipt_path.is_file() and pred_path.is_file():
                old=_json(receipt_path)
                if old['freeze_hash']==frozen_hash and old['predictor_hash']==rec['predictor_hash'] and old['prediction_sha256']==sha_file(pred_path) and old.get('reconstruction_pass'):
                    receipts.append(dict(path=str(receipt_path.relative_to(root)),sha256=sha_file(receipt_path)));continue
            if model is None:model=_build(rec,pinfo,data,root,device)
            before=state_hash(model.state_dict());rows=data.ids(role);ps=[];ys=[];max_error=0.;saved_error=0.;reference=None
            existing_name=f'{role}_{"read_pass10" if rec["variant"]=="read_pass10" else "selected"}.npz'
            reference_path=root/rec['run_dir']/existing_name
            if rec['stage']!='prefix' and role in ('validation','Q_explore'):
                with np.load(reference_path) as a:
                    if not np.array_equal(a['row_id'],rows) or str(a['checkpoint_hash'])!=rec['predictor_hash']:raise ValueError('Frozen development prediction changed')
                    reference=np.asarray(a['p'])
            with ledger.measure('raw_x_prediction_and_reconstruction',predictor_id=rec['id'],role=role):
                with torch.no_grad():
                    for i in range(0,len(rows),4096):
                        ids=rows[i:i+4096];x=torch.as_tensor(np.array(data.x[ids]),device=device);p=model(x)
                        # Epoch24 has no selected-native feature cache. Rebuild
                        # against pinned fixed probability forward, independently of
                        # adapter.encode and without reusing selected h/z.
                        if rec['stage']=='prefix' and rec['variant']=='epoch24':
                            baseline=model.native.native(x.transpose(0,1)).double()
                        else:baseline=_cached(model,cache.get(ids,device))
                        err=float((p-baseline).abs().max());max_error=max(max_error,err)
                        if not torch.isfinite(p).all() or err>1e-6:raise ValueError('Raw-X reconstruction outside locked tolerance: '+rec['id']+'/'+role+' '+str(err))
                        a=p.cpu().numpy();ps.append(a)
                        if reference is not None:saved_error=max(saved_error,float(np.max(np.abs(a-reference[i:i+len(ids)]))))
                        # Labels cannot enter either predictor path above.
                        batch=data.batch(ids,role,'cpu',purpose='evaluate',authorization=data.data_hash);ys.append(batch.y.numpy())
                if state_hash(model.state_dict())!=before:raise ValueError('Evaluation mutated frozen model state')
            if saved_error>1e-6:raise ValueError('Raw-X final predictor disagrees with frozen development export')
            with ledger.measure('prediction_metrics_and_io',predictor_id=rec['id'],role=role):
                metrics=save_predictions(pred_path,rows,np.concatenate(ys),np.concatenate(ps),data.domain[rows],role,rec['predictor_hash'],freeze_hash=np.asarray(frozen_hash),predictor_id=np.asarray(rec['id']),variant=np.asarray(rec['variant']))
            ledger.access('final_evaluation',role,len(rows),tasks=2,predictor_id=rec['id'])
            receipt=dict(freeze_hash=frozen_hash,predictor_id=rec['id'],predictor_hash=rec['predictor_hash'],variant=rec['variant'],role=role,row_hash=array_hash(rows),rows=len(rows),prediction_path=str(pred_path.relative_to(root)),prediction_sha256=sha_file(pred_path),metrics_path=str(pred_path.with_suffix('.metrics.json').relative_to(root)),metrics_sha256=sha_file(pred_path.with_suffix('.metrics.json')),raw_x_reconstruction_max_error=max_error,frozen_development_export_max_error=saved_error,tolerance=1e-6,reconstruction_pass=True,model_state_unchanged=True,cache_used_only_as_independent_reference=True,label_use='evaluation only; both predictions computed before label fetch')
            atomic_json(receipt_path,receipt);receipts.append(dict(path=str(receipt_path.relative_to(root)),sha256=sha_file(receipt_path)))
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    if len(receipts)!=manifest['expected_prediction_exports']:raise RuntimeError('Incomplete final prediction coverage')
    completion=dict(freeze_hash=frozen_hash,status='COMPLETE',exports=len(receipts),required_exports=manifest['expected_prediction_exports'],receipts=receipts,all_raw_x_reconstructions_pass=True,source_hash=manifest['sources']['all_source_hash'],cleanup_authorized_after_receipt_verification=True)
    atomic_json(root/'EXPOSED_EVALUATION_COMPLETE.json',completion);return dict(status='COMPLETE',exports=len(receipts),freeze_hash=frozen_hash)


def cleanup(root=DEFAULT_ROOT):
    """Remove only this campaign's temporary models after all frozen exports."""
    root,manifest=_locked(root,allow_missing_models=True);completion=_json(root/'EXPOSED_EVALUATION_COMPLETE.json')
    if completion.get('freeze_hash')!=manifest['freeze_hash'] or completion.get('exports')!=manifest['expected_prediction_exports'] or not completion.get('all_raw_x_reconstructions_pass'):raise RuntimeError('Cleanup requires every frozen predictor/role export and raw-X check')
    for entry in completion['receipts']:
        path=_inside(root/entry['path'],root)
        if sha_file(path)!=entry['sha256']:raise ValueError('Final reconstruction receipt changed')
        r=_json(path)
        if not r['reconstruction_pass'] or r['raw_x_reconstruction_max_error']>1e-6 or r['freeze_hash']!=manifest['freeze_hash']:raise ValueError('Invalid reconstruction receipt')
        for pkey,hkey in [('prediction_path','prediction_sha256'),('metrics_path','metrics_sha256')]:
            p=_inside(root/r[pkey],root)
            if sha_file(p)!=r[hkey]:raise ValueError('Final information export changed before cleanup')
    targets=[];aliases=[];extra=[]
    if (root/'_temporary').is_symlink():raise ValueError('Campaign temporary root must not be an external alias')
    if (root/'_temporary').is_dir():targets.append(root/'_temporary')
    for branch in ('prefix','runs'):
        base=root/branch
        if not base.exists():continue
        for current,dirs,files in os.walk(base,followlinks=False):
            current=Path(current)
            for name in dirs+files:
                path=current/name
                if path.is_symlink():
                    if name in ('transient','states','cache'):
                        target=_inside(path,root)
                        aliases.append(path)
                        if target.exists() and target.is_dir():targets.append(target)
                    continue
                if path.is_dir() and name=='transient':targets.append(path)
                elif path.is_file() and (path.suffix.lower() in MODEL_SUFFIXES or name in ('h.npy','z.npy','pilot_residuals.npy','responsibilities.npz')):extra.append(path)
    reduced=[]
    for p in sorted(set(targets),key=lambda p:len(p.parts)):
        p=_inside(p,root)
        if p.is_symlink():raise ValueError('Cleanup target is an unexpected alias')
        if not any(p==q or q in p.parents for q in reduced):reduced.append(p)
    metadata=[];removed=[]
    for target in reduced:
        size=0
        for current,dirs,files in os.walk(target,followlinks=False):
            for name in files:
                p=Path(current)/name
                if p.is_symlink():continue
                size+=p.stat().st_size
                if p.suffix=='.json':metadata.append(dict(path=str(p.relative_to(root)),sha256=sha_file(p),content=_json(p)))
        removed.append(dict(path=str(target.relative_to(root)),bytes=size,kind='temporary_directory'))
    for p in extra:
        if not any(p==q or q in p.parents for q in reduced):removed.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha_file(p),kind='temporary_model_or_cache'))
    # Information/identity exports are written before the first destructive step.
    atomic_json(root/'TRANSIENT_IDENTITY_INDEX.json',dict(freeze_hash=manifest['freeze_hash'],metadata=metadata,complete_state_hashes={p['id']:dict(primary=p.get('primary_checkpoint'),capsule=p.get('capsule'),snapshots=p.get('snapshots'),endpoint=p.get('endpoint_checkpoint')) for p in manifest['predictors']}))
    atomic_json(root/'CLEANUP_PLAN.json',dict(freeze_hash=manifest['freeze_hash'],targets=removed,aliases=[str(p.relative_to(root)) for p in aliases],scope='this campaign only; no external path traversal or /home_new access'))
    for p in aliases:p.unlink()
    for p in extra:
        if p.exists() and not any(p==q or q in p.parents for q in reduced):p.unlink()
    for p in reduced:
        if p.exists():shutil.rmtree(p)
    remaining=[]
    for current,dirs,files in os.walk(root,followlinks=False):
        for name in files:
            p=Path(current)/name
            if not p.is_symlink() and p.suffix.lower() in MODEL_SUFFIXES:remaining.append(str(p.relative_to(root)))
    receipt=dict(freeze_hash=manifest['freeze_hash'],status='PASS' if not remaining else 'FAIL',removed=removed,aliases_removed=[str(p.relative_to(root)) for p in aliases],models_remaining=remaining,final_models_retained=bool(remaining),information_exports_preserved=completion['exports'],external_directories_touched=False)
    atomic_json(root/'INTERMEDIATE_CLEANUP.json',receipt)
    if remaining:raise RuntimeError('Unexpected model archives remain in campaign: '+str(remaining))
    return receipt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('freeze','evaluate','cleanup'));ap.add_argument('--artifact-root',default=str(DEFAULT_ROOT));ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    result=freeze(a.artifact_root) if a.command=='freeze' else evaluate(a.artifact_root,a.device) if a.command=='evaluate' else cleanup(a.artifact_root)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
