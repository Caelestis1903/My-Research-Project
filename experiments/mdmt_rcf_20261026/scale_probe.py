"""Isolated historical SCALE bridge and bounded constant/no-augmentation runs."""
from __future__ import annotations
import copy
import importlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .artifacts import _json,_npz,_write_csv,_sha

SNAPSHOT=Path(__file__).resolve().parent/'vendor_snapshot'/'scale_20261021'
SOURCE_COMMIT='c5618e281785a4b7063ff9bf2a61cf38d4faf1be'

def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def _historical_data(dataset,cache):
    directory=SNAPSHOT/'experiments'/'reproduction_20261021'
    # Absolute historical imports are bridged only to byte-frozen snapshot files.
    aliases={name:sys.modules.get(name) for name in ('stage_S1_protocol','audited_data')}
    try:
        _load('stage_S1_protocol',directory/'stage_S1_protocol.py')
        _load('audited_data',directory/'audited_data.py')
        mod=_load('_rcf_scale_data',directory/'data.py')
        return mod.load_data(dataset,cache)
    finally:
        # The returned data bundle classes remain reachable; no old trainer runs.
        for key,old in aliases.items():
            if old is not None:sys.modules[key]=old
            else:sys.modules.pop(key,None)

def _historical_model(bundle,seed,device,arm='Dconst01_NO_AUG'):
    base='experiments.mdmt_rcf_20261026.vendor_snapshot.scale_20261021.experiments.data_generate.MDMTRec'
    feat=importlib.import_module(base+'.basic.features')
    def features():
        return [feat.DenseFeature(i) if c in bundle.features.dense_columns else feat.SparseFeature(i,bundle.features.vocab_sizes[c],embed_dim=16) for i,c in enumerate(bundle.features.columns)]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    kwargs=dict(domain_num=3,task_num=2,fcn_dims=[512,256,256,64],expert_num=4,exp_d=1,exp_t=1,bal_d=1,bal_t=1)
    original=importlib.import_module(base+'.models.multi_domain.original').MDMTRec(features(),**kwargs)
    backbone={k:v for k,v in original.state_dict().items() if not k.startswith('_weight_')}
    state_hash=_hist_protocol().state_hash
    canonical=state_hash(backbone)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+700000)
        mod=importlib.import_module(base+'.models.multi_domain.'+('fixed' if arm=='B' else 'model'))
        model=mod.MDMTRec(features(),**kwargs)
    missing,unexpected=model.load_state_dict(backbone,strict=False)
    if unexpected or not all(k.startswith('dbo_') for k in missing):raise ValueError('Historical backbone copy mismatch')
    actual={k:v for k,v in model.state_dict().items() if not k.startswith(('_weight_','dbo_'))}
    if state_hash(actual)!=canonical:raise ValueError('Historical backbone mismatch')
    return model.to(device),dict(canonical_backbone_sha256=canonical,copied_backbone_sha256=state_hash(actual),complete_initial_sha256=state_hash(model.state_dict()),source_commit=SOURCE_COMMIT)

def run_scale_numeric(config,output_root):
    """Cold and correctly corresponding warm coupled-Adam equivalence."""
    c=.1;wd=1e-5;eps=1e-8;results={}
    for warm in (False,True):
        a=torch.nn.Parameter(torch.tensor([.7,-.9,1.3],dtype=torch.float64));b=torch.nn.Parameter(a.detach().clone())
        oa=torch.optim.Adam([a],lr=.01,betas=(.9,.999),eps=eps,weight_decay=wd)
        ob=torch.optim.Adam([b],lr=.01,betas=(.9,.999),eps=eps/c,weight_decay=wd/c)
        if warm:
            oa.state[a]={'step':torch.tensor(4.),'exp_avg':torch.tensor([.04,-.02,.08],dtype=torch.float64)*c,'exp_avg_sq':torch.tensor([.02,.03,.04],dtype=torch.float64)*c*c}
            ob.state[b]={'step':torch.tensor(4.),'exp_avg':oa.state[a]['exp_avg'].clone()/c,'exp_avg_sq':oa.state[a]['exp_avg_sq'].clone()/(c*c)}
        errors=[]
        for step in range(3):
            target=torch.tensor([.1,.2,-.3],dtype=torch.float64)+step*.01
            oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
            (c*(a-target).square().mean()).backward();((b-target).square().mean()).backward();oa.step();ob.step()
            errors.append(float((a-b).detach().abs().max()))
        results['warm' if warm else 'cold']={'max_abs_parameter_error':max(errors),'passed':max(errors)<1e-12,'steps':3}
    out=dict(kind='actual FP64 numerical bridge, not performance experiment',source_commit=SOURCE_COMMIT,c=c,wd=wd,eps=eps,lr=.01,betas=[.9,.999],identity='Adam(cL,wd,eps) == Adam(L,wd/c,eps/c) for corresponding history',tests=results,status='PASS' if all(r['passed'] for r in results.values()) else 'FAIL')
    _json(Path(output_root)/'scale'/'NUMERIC_BRIDGE.json',out)
    if out['status']!='PASS':raise ArithmeticError('Adam bridge numerical check failed')
    return out

def _hist_protocol():
    return _load('_rcf_scale_protocol',SNAPSHOT/'experiments'/'reproduction_20261021'/'protocol.py')

def _predict(model,x,device,chunk=40960):
    values=[];model.eval()
    with torch.no_grad():
        for i in range(0,len(x),chunk):
            out=model(torch.as_tensor(x[i:i+chunk],device=device).T)
            if isinstance(out,tuple):out=out[0]
            values.append(out.cpu().numpy())
    return np.concatenate(values)

def _train_one(dataset,seed,config,root,device,arm='Dconst01_NO_AUG'):
    from torch.utils.data import TensorDataset,DataLoader
    from .state import state_hash
    out=Path(root)/'scale'/dataset/arm/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    identity={'dataset':dataset,'seed':int(seed),'arm':arm,'source_commit':SOURCE_COMMIT,'model_storage':'RAM_ONLY','batch_size':40960,'objective':'sum_(task,domain) group_mean(constant_weight * probability_BCE)','constant_weight':.1 if arm!='B' else 1.,'augmentation':False,'lr':.01,'weight_decay':1e-5,'eps':1e-8,'betas':[.9,.999],'max_epochs':50,'selection':'validation_macro_auc_strict_earliest','patience':10,'tf32':False,'amp':False}
    done=out/'COMPLETE.json'
    if done.exists():
        old=json.loads(done.read_text())
        if old['identity']!=identity:raise ValueError('SCALE identity mismatch')
        return old
    if (out/'epoch_metrics.csv').exists():raise RuntimeError('Interrupted SCALE trajectory requires explicit same-identity replay with recorded cause')
    bundle,manifest=_historical_data(dataset,Path(root)/'scale'/'data_cache')
    model,initial=_historical_model(bundle,seed,device,arm);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=1e-5,betas=(.9,.999),eps=1e-8)
    protocol=_hist_protocol();_json(out/'config.json',dict(identity=identity,initialization=initial));_json(out/'data_manifest.json',manifest)
    generator=torch.Generator().manual_seed(seed+11000)
    loader=DataLoader(TensorDataset(torch.from_numpy(bundle.train_x),torch.from_numpy(bundle.train_y)),batch_size=40960,shuffle=True,drop_last=False,num_workers=0,generator=generator)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    history=[];best=-np.inf;selected=None;selected_epoch=0;bad=0;steps=0;started=time.monotonic()
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,51):
        model.train();total=0.;epoch_started=time.monotonic()
        for xb,yb in loader:
            x=xb.T.to(device);y=yb.to(device);opt.zero_grad(set_to_none=True)
            outp=model(x)
            p=outp[0] if isinstance(outp,tuple) else outp
            bce=F.binary_cross_entropy(p,y,reduction='none');cells=[]
            for q in range(2):
                for d in range(3):
                    mask=x[-1]==d
                    cells.append((bce[mask,q]*identity['constant_weight']).mean() if bool(mask.any()) else p.sum()*0)
            loss=sum(cells);loss.backward();opt.step();steps+=1;total+=float(loss.detach())
        vp=_predict(model,bundle.val_x,device);vm=protocol.metrics(bundle.val_y,vp,bundle.val_x[:,-1])
        row=dict(epoch=epoch,steps=steps,optimized_group_sum=total/len(loader),seconds=time.monotonic()-epoch_started,**{'validation_'+k:v for k,v in vm.items()})
        history.append(row);_write_csv(out/'epoch_metrics.csv',history)
        print(json.dumps(dict(component='scale',dataset=dataset,seed=seed,epoch=epoch,validation_macro_auc=vm['macro_auc'],validation_macro_logloss=vm['macro_logloss'])),flush=True)
        if vm['macro_auc']>best:
            best=vm['macro_auc'];selected_epoch=epoch;bad=0;selected=copy.deepcopy(model.state_dict())
        else:bad+=1
        if bad>=10:break
    terminal=_predict(model,bundle.test_x,device);model.load_state_dict(selected);p=_predict(model,bundle.test_x,device);validation=_predict(model,bundle.val_x,device)
    tm={'status':'DEFERRED_UNTIL_FROZEN_COMPARISON_MANIFEST'};term=dict(tm)
    files=[_npz(out/'test_predictions.npz',row_id=bundle.test_row_ids,domain=bundle.test_x[:,-1],y=bundle.test_y,p=p,p_terminal=terminal),_npz(out/'validation_predictions.npz',row_id=bundle.val_row_ids,domain=bundle.val_x[:,-1],y=bundle.val_y,p=validation)]
    result=dict(identity=identity,status='COMPLETE',selected_epoch=selected_epoch,stopped_epoch=epoch,steps=steps,wall_seconds=time.monotonic()-started,selected_model_hash=state_hash(selected),selected_test=tm,terminal_test=term,selected_validation=protocol.metrics(bundle.val_y,validation,bundle.val_x[:,-1]),files=files,peak_memory_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,checkpoint_file='NEVER_WRITTEN',parent_comparison='public historical metrics only unless raw paired predictions independently available')
    _json(done,result);return result

def run_scale_training(config,output_root,*,device):
    root=Path(output_root);numeric=root/'scale'/'NUMERIC_BRIDGE.json'
    if not numeric.exists() or json.loads(numeric.read_text())['status']!='PASS':raise PermissionError('SCALE numeric bridge must pass first')
    seeds=config.get('_scale_seeds',config['seeds']['scale_first'])
    if any(s in config['seeds']['scale_conditional_locked'] for s in seeds) and not config.get('_scale_locked_authorization'):
        raise PermissionError('SCALE 2024/2025 requires registered conditional decision')
    rows=[]
    for dataset in config.get('_scale_datasets',list(config['datasets'])):
        for seed in seeds:rows.append(_train_one(dataset,seed,config,root,device))
    result={'status':'COMPLETE','runs':rows,'conditional_locked':'NO_START_UNTIL_REPEATED_AUGMENTATION_INCREMENT_DECISION','new_model_files':0}
    _json(root/'scale'/'SCALE_RUNS.json',result);return result


def evaluate_scale_results(config,output_root):
    """Open saved exposed predictions only after all registered first-stage runs finish."""
    root=Path(output_root)/'scale';expected=[(d,s) for d in config['datasets'] for s in config['seeds']['scale_first']]
    records=[]
    for d,s in expected:
        p=root/d/'Dconst01_NO_AUG'/f'seed{s}'/'COMPLETE.json'
        if not p.exists():raise PermissionError('SCALE comparison inventory is incomplete')
        records.append((p,json.loads(p.read_text())))
    manifest={'status':'FROZEN','comparison':'Dconst01_NO_AUG vs public historical B/Dconst01','source_commit':SOURCE_COMMIT,'config_hash':config['config_hash'],'runs':[r['identity'] for _,r in records],'historical_raw_prediction_pairing':'UNVERIFIED_UNLESS_SEPARATELY_IDENTIFIED'}
    _json(root/'FROZEN_COMPARISON_MANIFEST.json',manifest)
    protocol=_hist_protocol();summary=[]
    for path,record in records:
        a=np.load(path.parent/'test_predictions.npz',allow_pickle=False)
        summary.append(dict(dataset=record['identity']['dataset'],seed=record['identity']['seed'],selected=protocol.metrics(a['y'],a['p'],a['domain']),terminal=protocol.metrics(a['y'],a['p_terminal'],a['domain'])))
    _json(root/'EXPOSED_EVALUATION.json',summary);return summary
