"""Native and pilot common prefixes, with all trainable state retained in RAM."""
from __future__ import annotations
from pathlib import Path
import copy,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .protocol import PrefixBundle,RoleView,LabelLedger,RoleBundle,derive_rng,torch_seed,evaluate
from .state import cpu_state,freeze,digest,state_hash,array_hash,write_json,append_jsonl,capture_memory_state,sha_file
from .vendor_snapshot.mdmt_dfd_20261026.native_adapter import NativeAdapter,make_features
from .vendor_snapshot.mdmt_dfd_20261026.functions import PilotBank
from .vendor_snapshot.mdmt_dfd_20261026.vendor.pinned_fixed.models.multi_domain.fixed import MDMTRec
from .vendor_snapshot.mdmt_dfd_20261026.vendor.pinned_fixed.models.multi_domain.original import MDMTRec as Original
from .memory_data import RowDataStore,persist,release_data_pages
_LIVE={}

def source_manifest():
    package=Path(__file__).resolve().parent
    files={name:sha_file(package/name) for name in ('protocol.py','state.py','prefix.py','memory_data.py')}
    for path in sorted((package/'vendor_snapshot'/'mdmt_dfd_20261026').rglob('*.py')):
        files[str(path.relative_to(package))]=sha_file(path)
    return dict(scope='S0 protocol/state/prefix and frozen imported v4/vendor dependency snapshot',files=files,source_hash=digest(files))

def _check_identity_file(path,identity):
    if path.exists():
        import json
        prior=json.loads(path.read_text()).get('identity')
        if not isinstance(prior,dict):raise ValueError('Existing prefix missing run identity: '+str(path))
        mismatch=[k for k,v in identity.items() if prior.get(k)!=v]
        if mismatch:raise ValueError('Existing prefix identity mismatch '+','.join(mismatch)+': '+str(path))


def configure():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')

def make_native(schema,seed,dataset,device):
    kwargs=dict(domain_num=3,task_num=2,fcn_dims=[512,256,256,64],expert_num=4,exp_d=1,exp_t=1,bal_d=1,bal_t=1)
    init_seed=torch_seed(seed,dataset,'native_init')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed);original=Original(make_features(schema),**kwargs)
        backbone={k:v for k,v in original.state_dict().items() if not k.startswith('_weight_')}
        native=MDMTRec(make_features(schema),**kwargs);native.load_state_dict(backbone,strict=True)
    receipt=dict(initial_seed=init_seed,canonical_backbone_hash=state_hash(backbone),copied_backbone_hash=state_hash(native.state_dict()),parameters=sum(p.numel() for p in native.parameters()))
    if receipt['canonical_backbone_hash']!=receipt['copied_backbone_hash']:raise ValueError('canonical initialization mismatch')
    return NativeAdapter(native=native).to(device),receipt

def _plan(ids,batch,seed,dataset,stage,epoch,source=0,task=0,unit=0):
    order=derive_rng(seed,dataset,stage,source,task,unit,epoch).permutation(ids)
    return [order[i:i+batch] for i in range(0,len(order),batch)]

def _native_predictions(model,data,ids,device,chunk=4096,context=False):
    zz=[];hh=[]
    with torch.no_grad():
        for start in range(0,len(ids),chunk):
            x=torch.as_tensor(np.array(data.x[ids[start:start+chunk]]),device=device)
            p=model.encode(x);zz.append(p.logits.cpu().numpy())
            if context:hh.append(p.h.cpu().numpy())
    return (np.concatenate(hh),np.concatenate(zz)) if context else np.concatenate(zz)

class PilotAccess:
    def __init__(self,bundle,source_bank,local_banks,initial_bank):
        self.bundle=bundle;self.source_bank=source_bank;self.local_banks=local_banks;self.initial_bank=initial_bank
        self.phi_normalizers={};self.local_normalizers={};self._cache={}
        for s in range(3):
            for q in range(2):
                a=self.hidden('fit',s,q,standardized=False);a=a[bundle.views['fit'].domain==s]
                self.phi_normalizers[s,q]=dict(mean=a.mean(0,dtype=np.float64),std=np.maximum(a.std(0,dtype=np.float64),1e-3))
                for k in range(2):
                    a=self.local_hidden('fit',s,q,k,standardized=False);a=a[bundle.views['fit'].domain==s]
                    self.local_normalizers[s,q,k]=dict(mean=a.mean(0,dtype=np.float64),std=np.maximum(a.std(0,dtype=np.float64),1e-3))
    def _hidden(self,role,head,key):
        if key in self._cache:return self._cache[key]
        h=self.bundle.views[role].h;device=next(head.parameters()).device;vals=[]
        with torch.no_grad():
            for i in range(0,len(h),40960):vals.append(F.relu(head.hidden(torch.as_tensor(np.array(h[i:i+40960]),device=device))).cpu().numpy())
        out=persist(self.bundle,('pilot_hidden',key),np.concatenate(vals));self._cache[key]=out;return out
    def hidden(self,role,source,task,*,standardized=True,initial=False):
        bank=self.initial_bank if initial else self.source_bank;key=('initial' if initial else 'source',role,source,task)
        a=self._hidden(role,bank.head(source,task),key)
        if standardized:
            # Initial controls use their own source-fit normalization, never refitted on target.
            if initial:
                f=self._hidden('fit',bank.head(source,task),('initial','fit',source,task));f=f[self.bundle.views['fit'].domain==source]
                mean=f.mean(0,dtype=np.float64);std=np.maximum(f.std(0,dtype=np.float64),1e-3)
            else:mean=self.phi_normalizers[source,task]['mean'];std=self.phi_normalizers[source,task]['std']
            return ((a-mean)/std).astype(np.float32)
        return a
    def residual(self,role,source,task):
        head=self.source_bank.head(source,task);a=self.hidden(role,source,task,standardized=False)
        w=head.output.weight.detach().cpu().numpy().reshape(-1);b=float(head.output.bias.detach().cpu().item())
        return a@w+b
    def logits(self,role,source,task):return self.bundle.views[role].z[:,task]+self.residual(role,source,task)
    def clone_hidden(self,source,task):
        h=self.source_bank.head(source,task).hidden;return h.weight.detach().cpu().clone(),h.bias.detach().cpu().clone()
    def local_hidden(self,role,target,task,replica,*,standardized=True):
        a=self._hidden(role,self.local_banks[replica].head(target,task),('local',replica,role,target,task))
        if standardized:
            n=self.local_normalizers[target,task,replica];return ((a-n['mean'])/n['std']).astype(np.float32)
        return a
    def local_clone_hidden(self,target,task,replica):
        h=self.local_banks[replica].head(target,task).hidden;return h.weight.detach().cpu().clone(),h.bias.detach().cpu().clone()

def _fit_pilots(bundle,config,source_bank,stage,method,unit=0,shuffled=False):
    device=bundle.device;v=bundle.views['fit'];optimizer=torch.optim.Adam(source_bank.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
    labels=np.array(v.y)
    if shuffled:
        for s in range(3):
            rows=np.flatnonzero(v.domain==s);per=derive_rng(bundle.seed,bundle.dataset,'source_label_shuffle',s).permutation(rows);labels[rows]=v.y[per]
    for epoch in range(1,13):
        order=_plan(np.arange(len(v.row_id)),40960,bundle.seed,bundle.dataset,stage,epoch,unit=unit)
        total=0.;started=time.monotonic()
        for pos in order:
            optimizer.zero_grad(set_to_none=True);h=torch.as_tensor(np.array(v.h[pos]),device=device);z=torch.as_tensor(np.array(v.z[pos]),device=device)
            y=torch.as_tensor(labels[pos],device=device);d=torch.as_tensor(v.domain[pos],device=device)
            loss=source_bank.pilot_loss(h,z,y,d,denominator=len(pos));loss.backward();optimizer.step();total+=float(loss.detach())
            bundle.ledger.access(method,'fit',v.row_id[pos],purpose='pilot' if method=='PILOT' or shuffled else 'local_pilot',pass_id=epoch)
        append_jsonl(Path(bundle.output_root)/'training.jsonl',dict(component=method,pass_id=epoch,loss=total/len(order),seconds=time.monotonic()-started))
    return freeze(source_bank),optimizer

def build_prefix(dataset,seed,config,output_root,*,device,resume=True):
    configure();root=(Path(output_root)/'prefix'/dataset/f'seed{seed}').resolve()
    data=RoleBundle(dataset);source=source_manifest()
    ident=dict(spec='RCF-v1',dataset=dataset,seed=int(seed),config_hash=config['config_hash'],data_hash=data.data_hash,source_hash=source['source_hash'],output_root=str(root),storage='RAM_ONLY')
    # Validate all existing identities before opening logs or rewriting any receipt.
    for name in ('config.json','PREFIX_COMPLETE.json'):_check_identity_file(root/name,ident)
    key=(dataset,int(seed),config['config_hash'],data.data_hash,source['source_hash'],str(root),str(device))
    if key in _LIVE:
        if not resume:raise FileExistsError('Prefix already resident; resume required')
        return _LIVE[key]
    root.mkdir(parents=True,exist_ok=True);ledger=LabelLedger(data,root/'LABEL_ACCESS_LEDGER.jsonl')
    write_json(root/'SOURCE_MANIFEST.json',source)
    progress=root/'PREFIX_COMPLETE.json'
    if progress.exists():
        if not resume:raise FileExistsError('Completed prefix exists; explicit replay required')
        append_jsonl(root/'replay_events.jsonl',dict(reason='process-local state unavailable; deterministic same-identity rebuild',identity=ident))
    write_json(root/'config.json',dict(identity=ident,native=config['native'],pilot=config['pilot'],model_storage='RAM_ONLY'))
    model,init=make_native(data.schema,seed,dataset,device);write_json(root/'native_init.json',init)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
    history=[];best_ll=float('inf');selected=None;selected_epoch=None;started=time.monotonic()
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,25):
        model.train();batch_plan=_plan(data.ids('fit'),40960,seed,dataset,'native_plan',epoch);loss_sum=0.;ep_start=time.monotonic()
        for step,ids in enumerate(batch_plan):
            ledger.access('NATIVE','fit',ids,purpose='native',epoch=epoch,step=step)
            optimizer.zero_grad(set_to_none=True);x=torch.as_tensor(np.array(data.x[ids]),device=device);y=torch.as_tensor(np.array(data.y[ids]),device=device)
            loss=F.binary_cross_entropy_with_logits(model(x),y,reduction='sum')/(2*len(ids))
            if not torch.isfinite(loss):raise FloatingPointError('Native nonfinite loss')
            loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(ids)
        model.eval();ids=data.ids('validation');ledger.access('NATIVE','validation',ids,purpose='native_select',epoch=epoch)
        z=_native_predictions(model,data,ids,device);p=torch.sigmoid(torch.from_numpy(z)).numpy();metrics=evaluate(data.y[ids],p,data.domain[ids])
        rec=dict(epoch=epoch,train_loss=loss_sum/len(data.ids('fit')),validation=metrics,seconds=time.monotonic()-ep_start,row_plan_hash=array_hash(np.concatenate(batch_plan)))
        history.append(rec);append_jsonl(root/'epoch_metrics.jsonl',rec)
        if epoch>=12 and metrics['macro_logloss']<best_ll:
            best_ll=metrics['macro_logloss'];selected_epoch=epoch;selected=capture_memory_state({'model':model,'optimizer':optimizer},positions={'epoch':epoch,'next_epoch':epoch+1})
        print(f'PREFIX {dataset} seed={seed} epoch={epoch}/24 val_ll={metrics["macro_logloss"]:.9f} seconds={rec["seconds"]:.2f}',flush=True)
    terminal_complete=capture_memory_state({'model':model,'optimizer':optimizer},positions={'epoch':24,'next_epoch':25,'next_step':0});terminal_state=cpu_state(model);terminal_hash=state_hash(terminal_state)
    terminal_logits={role:_native_predictions(model,data,data.ids(role),device) for role in data.roles}
    selected_hash=state_hash(selected['objects']['model']);model.load_state_dict(selected['objects']['model']);freeze(model)
    views={};cut=np.empty((3,2),dtype=np.float64);row_store=RowDataStore(root)
    # Data caches are permitted; no trainable tensors/state are serialized.
    for role in data.roles:
        ids=data.ids(role);h,z=_native_predictions(model,data,ids,device,context=True)
        h=row_store.persist(('native_h',role),h);z=row_store.persist(('native_z',role),z)
        views[role]=RoleView(ids.copy(),data.domain[ids].copy(),np.array(data.y[ids]),h,z,np.zeros((len(ids),2),dtype=np.int64))
    for t in range(3):
        for q in range(2):cut[t,q]=np.median(views['fit'].z[views['fit'].domain==t,q])
    for v in views.values():
        for t in range(3):v.group[v.domain==t]=(v.z[v.domain==t]>cut[t]).astype(np.int64)
    bundle=PrefixBundle(dataset,int(seed),views,{},None,None,model,{},dict(group_cut=cut),ident,ledger,data,str(device),root,terminal_state)
    bundle._row_data_store=row_store
    init_bank=PilotBank(lambda s,q,k:torch_seed(seed,dataset,'pilot_init',s,q,k)).to(device);initial=copy.deepcopy(init_bank).cpu()
    source_bank,source_opt=_fit_pilots(bundle,config,init_bank,'pilot_plan','PILOT')
    local_banks=[];local_opts=[]
    for replica in range(2):
        bank=PilotBank(lambda s,q,k:torch_seed(seed,dataset,'local_init',s,q,replica)).to(device)
        bank,opt=_fit_pilots(bundle,config,bank,'local_plan',f'LOCAL_PILOT_{replica}',unit=replica);local_banks.append(bank);local_opts.append(opt)
    bundle.pilots=PilotAccess(bundle,source_bank,local_banks,initial)
    bundle.identity.update(native_hash=selected_hash,native_epoch24_hash=terminal_hash,pilot_hash=state_hash(source_bank.state_dict()),selected_epoch=selected_epoch)
    bundle.diagnostics=dict(native_history=history,native_init=init,native_terminal_logits=terminal_logits,selected_state=selected,native_terminal_complete=terminal_complete,pilot_optimizer=source_opt,local_optimizers=local_opts,elapsed_prefix_seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0)
    # Raw X to frozen feature path comparison uses an independent invocation.
    errors=[]
    for role,v in views.items():
        n=min(4096,len(v.row_id));h,z=_native_predictions(model,data,v.row_id[:n],device,context=True)
        errors.append(max(float(np.max(np.abs(h-v.h[:n]))),float(np.max(np.abs(z-v.z[:n])))))
    bundle.diagnostics['raw_native_cache_max_error']=max(errors)
    if max(errors)>1e-6:raise ValueError('raw-X/native cache mismatch')
    write_json(progress,dict(identity=bundle.identity,native_epochs=24,pilot_passes=12,local_pilot_passes=12,storage='RAM_ONLY',elapsed_seconds=bundle.diagnostics['elapsed_prefix_seconds'],peak_cuda_bytes=bundle.diagnostics['peak_cuda_bytes']))
    release_data_pages(bundle)
    _LIVE[key]=bundle;return bundle

def rebuild_shuffled_pilots(bundle,config):
    other=copy.copy(bundle);other.normalizers=copy.deepcopy(bundle.normalizers)
    bank=PilotBank(lambda s,q,k:torch_seed(bundle.seed,bundle.dataset,'pilot_init',s,q,k)).to(bundle.device);initial=copy.deepcopy(bank).cpu()
    bank,_=_fit_pilots(bundle,config,bank,'pilot_plan','SOURCE_LABEL_SHUFFLE_PILOT',shuffled=True)
    other.pilots=PilotAccess(other,bank,bundle.pilots.local_banks,initial);other.identity=dict(bundle.identity,pilot_intervention='SOURCE_LABEL_SHUFFLE',pilot_hash=state_hash(bank.state_dict()))
    return other
