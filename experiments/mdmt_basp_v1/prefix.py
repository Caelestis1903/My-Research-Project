"""Fixed-fusion native and owner pilots; model states stay in RAM."""
from pathlib import Path
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.mdmt_relation_basis_v2.vendor_snapshot.mdmt_dfd_20261026.native_adapter import NativeAdapter,make_features
from experiments.mdmt_relation_basis_v2.vendor_snapshot.mdmt_dfd_20261026.vendor.pinned_fixed.models.multi_domain.fixed import MDMTRec
from experiments.mdmt_relation_basis_v2.vendor_snapshot.mdmt_dfd_20261026.vendor.pinned_fixed.models.multi_domain.original import MDMTRec as Original
from experiments.mdmt_relation_basis_v2.vendor_snapshot.mdmt_dfd_20261026.functions import PilotBank
from .config import rng,torch_seed
from .roles import tensor
from .identity import freeze,write_json,append_jsonl,cpu_copy,capture_rng
from .metrics import evaluate


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def make_native(schema,seed,dataset,device):
    canonical_seed=torch_seed(seed,dataset,10)
    kwargs=dict(domain_num=3,task_num=2,fcn_dims=[512,256,256,64],expert_num=4,
                exp_d=1,exp_t=1,bal_d=1,bal_t=1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(canonical_seed)
        original=Original(make_features(schema),**kwargs)
        backbone={k:v for k,v in original.state_dict().items() if not k.startswith('_weight_')}
        native=MDMTRec(make_features(schema),**kwargs)
        native.load_state_dict(backbone,strict=True)
        if not all(torch.equal(v,native.state_dict()[k]) for k,v in backbone.items()):
            raise AssertionError('Canonical backbone copy mismatch')
    return NativeAdapter(native=native).to(device),dict(canonical_seed=canonical_seed,
               canonical_copy='PASS_DIRECT_TENSOR_EQUALITY',fixed_fusion=[1,1,1,1],
               learned_fusion=False,automl=False,optimizers=1)


@torch.no_grad()
def encode_native(native,x,device,chunk=4096):
    native.eval()
    hs,zs=[],[]
    for i in range(0,len(x),chunk):
        packet=native.encode(tensor(x[i:i+chunk],device))
        hs.append(packet.h.cpu().numpy());zs.append(packet.logits.cpu().numpy())
    return np.concatenate(hs),np.concatenate(zs)


def _state(model,optimizer,epoch,step):
    return dict(model=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
                rng=capture_rng(),epoch=epoch,step=step,storage='RAM_ONLY')


def train_native(data,seed,cfg,root,device):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    native,init=make_native(data.schema,seed,data.dataset,device)
    write_json(root/'native_initialization.json',init)
    optimizer=torch.optim.Adam(native.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
    started=time.monotonic();best=float('inf');selected=None;step=0
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    ids=data.roles['T0']
    for epoch in range(1,25):
        native.train();total=0.;ep=time.monotonic()
        order=rng(seed,data.dataset,20,epoch).permutation(ids)
        for begin in range(0,len(order),40960):
            rows=order[begin:begin+40960]
            y,_=data.labels('T0','native',rows,epoch=epoch,step=step)
            optimizer.zero_grad(set_to_none=True)
            logits=native(tensor(data.x[rows],device))
            loss=F.binary_cross_entropy_with_logits(logits,tensor(y,device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite native risk')
            loss.backward();optimizer.step();step+=1
            total+=float(loss.detach())*len(rows)
        yv,mv=data.labels('V','select',component='native',epoch=epoch)
        _,zv=encode_native(native,data.inputs('V'),device)
        metrics=evaluate(yv,zv,data.domain[data.roles['V']],mv)
        synchronize(device)
        append_jsonl(root/'native_epoch_metrics.jsonl',dict(epoch=epoch,step=step,
                     train_loss=total/len(ids),validation=metrics,seconds=time.monotonic()-ep))
        if epoch>=12 and metrics['macro_logloss']<best-1e-12:
            best=metrics['macro_logloss'];selected=_state(native,optimizer,epoch,step)
        print(f'NATIVE {data.dataset} seed={seed} epoch={epoch}/24 V_LL={metrics["macro_logloss"]:.9f}',flush=True)
    final=_state(native,optimizer,24,step)
    native.load_state_dict(selected['model']);freeze(native)
    receipt=dict(selected_epoch=selected['epoch'],final_epoch=24,selected_macro_logloss=best,
                 seconds=time.monotonic()-started,parameter_count=sum(p.numel() for p in native.parameters()),
                 peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
                 model_storage='RAM_ONLY',checkpoint_identity=f'{data.dataset}/seed{seed}/native/selected/epoch{selected["epoch"]}')
    write_json(root/'native_receipt.json',receipt)
    return native,selected,final,receipt


@torch.no_grad()
def cache_native(native,data,role,root,device):
    ids=data.roles[role]
    dest=Path(root)/'row_data'/role;dest.mkdir(parents=True,exist_ok=True)
    h=np.lib.format.open_memmap(dest/'h.npy',mode='w+',dtype=np.float32,shape=(len(ids),256))
    z=np.lib.format.open_memmap(dest/'z.npy',mode='w+',dtype=np.float32,shape=(len(ids),2))
    native.eval()
    for begin in range(0,len(ids),4096):
        packet=native.encode(tensor(data.x[ids[begin:begin+4096]],device))
        h[begin:begin+len(packet.h)]=packet.h.cpu().numpy()
        z[begin:begin+len(packet.h)]=packet.logits.cpu().numpy()
    h.flush();z.flush();del h,z
    return dict(h=np.load(dest/'h.npy',mmap_mode='r'),z=np.load(dest/'z.npy',mmap_mode='r'),
                ids=ids,domain=data.domain[ids])


class SourceFunctions(nn.Module):
    def __init__(self,bank,mean,std):
        super().__init__();self.bank=bank
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        ws,bs=[],[]
        for s in range(3):
            sw,sb=[],[]
            for t in range(2):
                head=bank.head(s,t)
                expanded=head.output.weight.new_zeros(64)
                expanded[32*t:32*(t+1)]=head.output.weight.detach().reshape(32)
                sw.append(expanded*self.std[s].to(expanded.device))
                sb.append(head.output.bias.detach().reshape(())+(expanded*self.mean[s].to(expanded.device)).sum())
            ws.append(torch.stack(sw));bs.append(torch.stack(sb))
        self.register_buffer('teacher_w',torch.stack(ws))
        self.register_buffer('teacher_b',torch.stack(bs));freeze(self)

    def one(self,h,source):
        hidden=torch.cat([F.relu(self.bank.head(source,t).hidden(h)) for t in range(2)],-1)
        return (hidden-self.mean[source])/self.std[source]

    def forward(self,h):
        return torch.stack([self.one(h,s) for s in range(3)],1)

    def teacher(self,phi,z,source):
        return z+phi@self.teacher_w[source].T+self.teacher_b[source]


def train_source(data,view,seed,root,device):
    def pilot_seed(s,t,k=0):
        return int(rng(seed,data.dataset,30,source=s,task=t).integers(0,2**32,dtype=np.uint32))
    bank=PilotBank(pilot_seed).to(device)
    optimizer=torch.optim.Adam(bank.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
    started=time.monotonic();step=0
    for epoch in range(1,13):
        ep=time.monotonic();total=0.
        order=rng(seed,data.dataset,31,epoch).permutation(len(view['ids']))
        for begin in range(0,len(order),40960):
            pos=order[begin:begin+40960]
            y,_=data.labels('T0','pilot',view['ids'][pos],epoch=epoch,step=step)
            optimizer.zero_grad(set_to_none=True)
            loss=bank.pilot_loss(tensor(view['h'][pos],device),tensor(view['z'][pos],device),
                                 tensor(y,device),tensor(view['domain'][pos],device),denominator=len(pos))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite source pilot risk')
            loss.backward();optimizer.step();step+=1;total+=float(loss.detach())*len(pos)
        synchronize(device)
        append_jsonl(Path(root)/'source_epoch_metrics.jsonl',dict(epoch=epoch,step=step,
                     train_loss=total/len(order),seconds=time.monotonic()-ep))
        print(f'SOURCE {data.dataset} seed={seed} pass={epoch}/12 loss={total/len(order):.9f}',flush=True)
    snapshot=_state(bank,optimizer,12,step);freeze(bank)
    means,stds=[],[]
    with torch.no_grad():
        for s in range(3):
            total=np.zeros(64,np.float64);square=np.zeros(64,np.float64);count=0
            own=np.flatnonzero(view['domain']==s)
            for begin in range(0,len(own),40960):
                h=tensor(view['h'][own[begin:begin+40960]],device)
                hidden=torch.cat([F.relu(bank.head(s,t).hidden(h)) for t in range(2)],-1).cpu().numpy().astype(np.float64)
                total+=hidden.sum(0);square+=(hidden*hidden).sum(0);count+=len(hidden)
            mean=total/count;std=np.maximum(np.sqrt(np.maximum(square/count-mean*mean,0)),.001)
            means.append(mean.astype(np.float32));stds.append(std.astype(np.float32))
    source=SourceFunctions(bank,np.stack(means),np.stack(stds)).to(device)
    max_diff=0.
    with torch.no_grad():
        for s in range(3):
            pos=np.flatnonzero(view['domain']==s)[:256]
            h,z=tensor(view['h'][pos],device),tensor(view['z'][pos],device)
            original=z+torch.stack([bank.head(s,t)(h) for t in range(2)],-1)
            converted=source.teacher(source.one(h,s),z,s)
            max_diff=max(max_diff,float((converted-original).abs().max()))
            torch.testing.assert_close(converted,original,atol=1e-6,rtol=0)
    synchronize(device)
    receipt=dict(seconds=time.monotonic()-started,pilot_parameters=sum(p.numel() for p in bank.parameters()),
                 teacher_max_abs_diff=max_diff,passes=12,normalization='T0_source_population_std_floor_0.001',
                 model_storage='RAM_ONLY',checkpoint_identity=f'{data.dataset}/seed{seed}/pilot/pass12')
    write_json(Path(root)/'source_receipt.json',receipt)
    return source,snapshot,receipt


@torch.no_grad()
def cache_phi(source,view,role,root,device):
    path=Path(root)/'row_data'/role/'phi.npy'
    out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(view['ids']),3,64))
    for begin in range(0,len(out),40960):
        out[begin:begin+40960]=source(tensor(view['h'][begin:begin+40960],device)).cpu().numpy()
    out.flush();del out;view['phi']=np.load(path,mmap_mode='r')
    return view


def packet(view,positions,device):
    return {k:tensor(view[k][positions],device) for k in ('h','z','phi')}


def build_prefix(data,seed,cfg,root,device):
    native,selected,final,nreceipt=train_native(data,seed,cfg,root,device)
    views={r:cache_native(native,data,r,root,device) for r in ('T0','F','S','Q','V')}
    source,pilot,sreceipt=train_source(data,views['T0'],seed,root,device)
    for role,view in views.items():
        cache_phi(source,view,role,root,device)
    return dict(native=native,source=source,native_selected=selected,native_final=final,
                pilot_state=pilot,views=views,native_receipt=nreceipt,source_receipt=sreceipt)
