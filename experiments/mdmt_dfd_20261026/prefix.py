"""Scratch native -> frozen cache -> owner-only pilots, with complete resume."""
from __future__ import annotations
from pathlib import Path
import json,os
import numpy as np
import torch
from torch.nn import functional as F
from .data import RoleBundle
from .plans import batches,torch_seed,full_plan
from .state import (atomic_json,sha_file,digest,state_hash,array_hash,save_checkpoint,load_checkpoint,capture_rng,restore_rng,module_state,restore_module)
from .metrics import evaluate_probabilities,save_predictions
from .costs import CostLedger,measured_budget
from .native_adapter import make_native
from .functions import PilotBank
DEFAULT_ROOT=Path('/home/xzf/Code/phase0-3/artifacts/mdmt_dfd_v4/v4_0')
# User storage revision: all temporary state stays inside this experiment root.
HEAVY_ROOT=DEFAULT_ROOT
PACKAGE=Path(__file__).resolve().parent
PREFIX_FILES=('state.py','plans.py','data.py','metrics.py','costs.py','native_adapter.py','prefix.py','functions.py')

def configure():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')

def source_manifest():
    files={}
    for f in PREFIX_FILES:
        p=PACKAGE/f
        if p.exists():files[f]=sha_file(p)
    for p in (PACKAGE/'vendor').rglob('*.py'):files[str(p.relative_to(PACKAGE))]=sha_file(p)
    return dict(files=files,source_hash=digest(files),scope='P0/P1 dependency files; conditional pipeline outside hash')

def paths(data,seed,root=DEFAULT_ROOT):
    ds=data.dataset if hasattr(data,'dataset') else data
    r=Path(root)/'prefix'/ds/f'seed{seed}';h=Path(root)/'_temporary'/'prefix'/ds/f'seed{seed}'
    r.mkdir(parents=True,exist_ok=True);h.mkdir(parents=True,exist_ok=True)
    # No writes to /home_new; bounded last/selected/epoch24 files are temporary.
    link=r/'states'
    if not link.exists():link.symlink_to(h,target_is_directory=True)
    transient=r/'transient'
    if not transient.exists():transient.symlink_to(h,target_is_directory=True)
    return r,h

def identity(data,seed):
    source=source_manifest();p=full_plan()
    return dict(method='DFD-v4.0',run_id=f'v4_{data.dataset}_s{seed}_PREFIX_p1',dataset=data.dataset,seed=int(seed),source_hash=source['source_hash'],data_hash=data.data_hash,config_hash=digest(p['native']),plan_hash=p['plan_hash'])

def _load(path,ident):return load_checkpoint(path,ident)
def _payload(ident,stage,pass_id,step,model,opt,selection=None,**other):
    out=dict(identity=ident,stage=stage,**{'pass':pass_id},next_logical_step=step,models={stage:module_state(model,opt)},optimizers={stage:opt.state_dict()},rng=capture_rng(),selection=selection or {},normalizers={},dependencies={},cost_state={});out.update(other);return out

def predict_native(model,data,ids,device,chunk=4096):
    values=[];model.eval()
    with torch.no_grad():
        for i in range(0,len(ids),chunk):
            x=torch.as_tensor(np.array(data.x[ids[i:i+chunk]]),device=device);values.append(model(x).sigmoid().cpu().numpy())
    return np.concatenate(values)

class FrozenCache:
    def __init__(self,data,directory):
        self.data=data;self.directory=Path(directory);self.manifest=json.loads((self.directory/'CACHE.json').read_text())
        self.h=np.load(self.directory/'h.npy',mmap_mode='r');self.z=np.load(self.directory/'z.npy',mmap_mode='r')
        if self.manifest['data_hash']!=data.data_hash:raise ValueError('Cache data identity mismatch')
        self.hash=self.manifest['cache_hash'];self.native_hash=self.manifest['native_hash']
    def get(self,ids,device='cpu'):
        ids=np.asarray(ids,dtype=np.int64)
        return dict(h=torch.as_tensor(np.array(self.h[ids]),device=device),z=torch.as_tensor(np.array(self.z[ids]),device=device),domain=torch.as_tensor(self.data.domain[ids],device=device),row_id=torch.as_tensor(ids,device=device))

def build_cache(data,model,native_hash,directory,ledger):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    identity=dict(data_hash=data.data_hash,native_hash=native_hash,encoder_hash=data.schema['encoder_hash'],x_sha256=data.raw_manifest['array_hashes']['x'],extractor_sha256=sha_file(PACKAGE/'native_adapter.py'),layout='global_original_row',dtype='float32',rows=len(data.x))
    key=digest(identity)
    if (directory/'CACHE.json').exists():
        saved=json.loads((directory/'CACHE.json').read_text())
        if saved['cache_hash']!=key:raise ValueError('Cache identity mismatch')
        return FrozenCache(data,directory)
    cursor=0
    if (directory/'CACHE_PROGRESS.json').exists():
        progress=json.loads((directory/'CACHE_PROGRESS.json').read_text())
        if progress['cache_hash']!=key:raise ValueError('Partial cache identity mismatch')
        cursor=progress['completed_rows']
    mode='r+' if cursor else 'w+'
    h=np.lib.format.open_memmap(directory/'h.npy',mode=mode,dtype=np.float32,shape=(len(data.x),256));z=np.lib.format.open_memmap(directory/'z.npy',mode=mode,dtype=np.float32,shape=(len(data.x),2))
    model.eval();device=next(model.parameters()).device
    with ledger.measure('feature_cache',rows=len(data.x)-cursor):
        with torch.no_grad():
            for i in range(cursor,len(data.x),4096):
                end=min(len(data.x),i+4096);packet=model.encode(torch.as_tensor(np.array(data.x[i:end]),device=device));h[i:end]=packet.h.cpu().numpy();z[i:end]=packet.logits.cpu().numpy()
                if not np.isfinite(h[i:end]).all() or not np.isfinite(z[i:end]).all():raise FloatingPointError('Nonfinite frozen cache')
                if (i//4096)%64==63 or end==len(data.x):
                    h.flush();z.flush();atomic_json(directory/'CACHE_PROGRESS.json',dict(cache_hash=key,completed_rows=end))
    h.flush();z.flush();atomic_json(directory/'CACHE.json',dict(identity,cache_hash=key,completed_rows=len(data.x)))
    return FrozenCache(data,directory)

def make_pilot(data,seed,device):return PilotBank(seed_fn=lambda s,q,k:torch_seed(seed,data.dataset,30,s,q,k)).to(device)

def run_prefix(data,seed,artifact_root=DEFAULT_ROOT,device='cuda:0',resume=True,stop_after=None):
    configure();r,h=paths(data,seed,artifact_root);ident=identity(data,seed);ledger=CostLedger(r/'costs.jsonl',ident)
    atomic_json(r/'config.json',dict(ident,native=full_plan()['native'],pilot_passes=12,loss='sum(row,task BCE)/(2*logicalB)',fusion=[1,1,1,1],automl=False,optimizer_count=1))
    atomic_json(r/'data_manifest.json',data.manifest);atomic_json(r/'SOURCE_MANIFEST.json',source_manifest())
    model,init=make_native(data.schema,seed,data.dataset,device);atomic_json(r/'native_init.json',init)
    opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
    epoch=1;cursor=0;selection=dict(best_ll=None,best_epoch=None);history=[]
    if (h/'native_last.pt').exists():
        if not resume:raise FileExistsError('Existing run; explicit --resume required')
        saved=_load(h/'native_last.pt',ident);restore_module(model,opt,saved['models']['native']);restore_rng(saved['rng']);epoch=saved['pass'];cursor=saved['next_logical_step'];selection=saved['selection'];history=saved['validation_history']
    while epoch<=24:
        plan=batches(data.ids('fit'),seed,data.dataset,20,epoch);model.train()
        with ledger.measure('native_epoch',epoch=epoch,start_step=cursor,planned_steps=len(plan)-cursor):
            for step in range(cursor,len(plan)):
                ids=plan[step];opt.zero_grad(set_to_none=True)
                for start in range(0,len(ids),40960):
                    batch=data.batch(ids[start:start+40960],'fit',device,purpose='native')
                    loss=F.binary_cross_entropy_with_logits(model(batch.x),batch.y,reduction='sum')/(2*len(ids));loss.backward()
                    if not torch.isfinite(loss):raise FloatingPointError('Native nonfinite loss')
                opt.step();ledger.access('native','fit',len(ids),epoch=epoch,logical_step=step)
                save_checkpoint(h/'native_last.pt',_payload(ident,'native',epoch,step+1,model,opt,selection,validation_history=history,row_plan_hash=array_hash(np.concatenate(plan))))
        with ledger.measure('native_validation',epoch=epoch):
            ids=data.ids('validation');data._check(ids,'validation','select');p=predict_native(model,data,ids,device);m=evaluate_probabilities(data.y[ids],p,data.domain[ids]);history.append(dict(epoch=epoch,**m));ledger.access('native_validation','validation',len(ids),epoch=epoch)
        if epoch>=12 and (selection['best_ll'] is None or m['macro_logloss']<selection['best_ll']):
            selection=dict(best_ll=m['macro_logloss'],best_epoch=epoch)
            save_checkpoint(h/'native_selected.pt',_payload(ident,'native',epoch,len(plan),model,opt,selection,validation_history=history,row_plan_hash=array_hash(np.concatenate(plan))))
        save_checkpoint(h/'native_last.pt',_payload(ident,'native',epoch+1,0,model,opt,selection,validation_history=history,row_plan_hash=None))
        atomic_json(r/'epoch_metrics.json',history)
        if epoch==24:save_checkpoint(h/'native_epoch24.pt',_payload(ident,'native',24,len(plan),model,opt,selection,validation_history=history))
        if stop_after=='native1' and epoch==1:return dict(status='RETAINED_BOUNDARY',component='native_epoch1',run_dir=str(r))
        epoch+=1;cursor=0
    selected=_load(h/'native_selected.pt',ident);restore_module(model,None,selected['models']['native']);model.eval()
    for p in model.parameters():p.requires_grad_(False);p.grad=None
    native_hash=state_hash(model.state_dict());cache=build_cache(data,model,native_hash,h/'cache',ledger)
    if not (r/'cache').exists():(r/'cache').symlink_to(h/'cache',target_is_directory=True)
    pilot=make_pilot(data,seed,device);popt=torch.optim.Adam(pilot.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=1e-5);pass_id=1;cursor=0
    if (h/'pilot.pt').exists():
        saved=_load(h/'pilot.pt',ident);restore_module(pilot,popt,saved['models']['pilot']);restore_rng(saved['rng']);pass_id=saved['pass'];cursor=saved['next_logical_step']
        if saved.get('complete'):pass_id=13
    while pass_id<=12:
        plan=batches(data.ids('fit'),seed,data.dataset,31,pass_id);pilot.train()
        with ledger.measure('pilot_pass',pass_id=pass_id,start_step=cursor):
            for step in range(cursor,len(plan)):
                ids=plan[step];popt.zero_grad(set_to_none=True)
                for start in range(0,len(ids),40960):
                    subset=ids[start:start+40960];packet=cache.get(subset,device);b=data.batch(subset,'fit',device,purpose='pilot')
                    loss=pilot.pilot_loss(packet['h'],packet['z'],b.y,b.domain,denominator=len(ids));loss.backward()
                    if not torch.isfinite(loss):raise FloatingPointError('Pilot nonfinite loss')
                popt.step();ledger.access('pilot','fit',len(ids),pass_id=pass_id,logical_step=step)
                save_checkpoint(h/'pilot.pt',_payload(ident,'pilot',pass_id,step+1,pilot,popt,dependencies=dict(cache_hash=cache.hash),complete=False,row_plan_hash=array_hash(np.concatenate(plan))))
        save_checkpoint(h/'pilot.pt',_payload(ident,'pilot',pass_id+1,0,pilot,popt,dependencies=dict(cache_hash=cache.hash),complete=pass_id==12))
        if stop_after=='pilot1' and pass_id==1:return dict(status='RETAINED_BOUNDARY',component='pilot_pass1',run_dir=str(r))
        pass_id+=1;cursor=0
    pilot.eval()
    for p in pilot.parameters():p.requires_grad_(False);p.grad=None
    # Stable deployment exports are different from the native epoch24 endpoint.
    for role in ('validation','Q_explore'):
        ids=data.ids(role);data._check(ids,role,'evaluate');out=[]
        with ledger.measure('prefix_evaluation',role=role):
            with torch.no_grad():
                for i in range(0,len(ids),4096):
                    packet=cache.get(ids[i:i+4096],device);out.append(pilot.local_logits(packet['h'],packet['z'],packet['domain']).sigmoid().cpu().numpy())
        save_predictions(r/f'LOCAL_{role}.npz',ids,data.y[ids],np.concatenate(out),data.domain[ids],role,state_hash(pilot.state_dict()))
        save_predictions(r/f'NATIVE_{role}.npz',ids,data.y[ids],torch.sigmoid(torch.from_numpy(np.array(cache.z[ids]))).numpy(),data.domain[ids],role,native_hash)
    atomic_json(r/'PREFIX_COMPLETE.json',dict(identity=ident,native_hash=native_hash,pilot_hash=state_hash(pilot.state_dict()),cache_hash=cache.hash,selected_epoch=selected['pass'],native_epochs=24,pilot_passes=12,states=str(h)))
    return dict(status='COMPLETE',run_dir=str(r))

def load_prefix(data,seed,artifact_root=DEFAULT_ROOT,device='cuda:0'):
    if isinstance(data,str):data=RoleBundle(data)
    r,h=paths(data,seed,artifact_root);ident=identity(data,seed);cache=FrozenCache(data,h/'cache');pilot=make_pilot(data,seed,device);saved=_load(h/'pilot.pt',ident)
    if not saved.get('complete'):raise ValueError('Pilot incomplete')
    if saved.get('dependencies',{}).get('cache_hash')!=cache.hash:raise ValueError('Pilot/cache dependency mismatch')
    selected=_load(h/'native_selected.pt',ident)
    selected_hash=state_hash(selected['models']['native']['model'])
    if selected_hash!=cache.native_hash:raise ValueError('Selected native/cache dependency mismatch')
    pilot.load_state_dict(saved['models']['pilot']['model']);pilot.eval()
    for p in pilot.parameters():p.requires_grad_(False);p.grad=None
    return cache,pilot
