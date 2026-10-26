"""Reviewed experiment semantics. No alternative backbone is defined here."""
import contextlib, hashlib, importlib, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def make_model(bundle,arm,seed,device):
    prefix='experiments.grad.MDMTRec' if arm=='G' else 'experiments.data_generate.MDMTRec'
    feature_mod=importlib.import_module(prefix+'.basic.features')
    def features():
        return [feature_mod.DenseFeature(i) if c in bundle.features.dense_columns else feature_mod.SparseFeature(i,bundle.features.vocab_sizes[c],embed_dim=16) for i,c in enumerate(bundle.features.columns)]
    kwargs=dict(domain_num=3,task_num=2,fcn_dims=[512,256,256,64],expert_num=4,exp_d=1,exp_t=1,bal_d=1,bal_t=1)
    seed_all(seed)
    original=importlib.import_module(prefix+'.models.multi_domain.original').MDMTRec(features(),**kwargs)
    backbone={k:v for k,v in original.state_dict().items() if not k.startswith('_weight_')}
    canonical_hash=state_hash(backbone)
    if arm=='O': model=original
    else:
        # D/G share auxiliary draws; canonical tensors are then copied exactly.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+700000)
            mod=importlib.import_module(prefix+'.models.multi_domain.'+('fixed' if arm=='B' else 'model'))
            model=mod.MDMTRec(features(),**kwargs)
        missing,unexpected=model.load_state_dict(backbone,strict=False)
        assert not unexpected and all(k.startswith('dbo_') for k in missing),(missing,unexpected)
    actual={k:v for k,v in model.state_dict().items() if not k.startswith(('_weight_','dbo_'))}
    assert state_hash(actual)==canonical_hash
    receipt={'canonical_backbone_sha256':canonical_hash,'copied_backbone_sha256':state_hash(actual),'complete_initial_sha256':state_hash(model.state_dict()),'model_module':model.__class__.__module__}
    return model.to(device),receipt

def optimizers(model,arm):
    # Historical ~bool predicate includes every parameter in O's main Adam.
    main=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=1e-5)
    aux=torch.optim.Adam([p for n,p in model.named_parameters() if n.startswith('_weight_')],lr=.001,weight_decay=1e-5) if arm=='O' else None
    return main,aux

def batch_loss(model,x,y,arm,scarce,noise_seed,noise_std=.1):
    n=y.shape[0]; mask=x[-1]==scarce
    aug=None
    if arm in ('D','G'):
        x=torch.cat([x,x[:,mask]],dim=1); y_input=torch.cat([y,y[mask]],dim=0)
        aug=torch.arange(x.shape[1],device=x.device)>=n
        devs=[x.device.index] if x.is_cuda else []
        # Noise reproducibility is independent of loader, model, and diagnostics.
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(noise_seed)
            pred,complexity,specificity=model(x,aug_mask=aug,noise_std=noise_std)
        realw=.1+.5*specificity[:n]
        w=torch.cat([realw,realw.mean().detach().expand(x.shape[1]-n)])
    else:
        pred=model(x); y_input=y; w=torch.ones(n,device=x.device); complexity=specificity=None
    bce=F.binary_cross_entropy(pred,y_input,reduction='none')
    cells=[]
    for t in range(2):
        for d in range(3):
            m=x[-1]==d
            cells.append((bce[m,t]*w[m]).mean() if m.any() else pred.sum()*0)
    return cells,dict(pred=pred,bce=bce,w=w,complexity=complexity,specificity=specificity,n=n,domain=x[-1],aug=aug)

def gradients(cells,params):
    return [torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True) for loss in cells]
def dot(a,b,indices=None):
    pairs=zip(a,b) if indices is None else ((a[i],b[i]) for i in indices)
    vals=[(x*y).sum() for x,y in pairs if x is not None and y is not None]
    return torch.stack(vals).sum() if vals else torch.tensor(0.,device=next(x.device for x in a if x is not None))
def norm(a,indices=None): return dot(a,a,indices).sqrt()
def cosine(a,b,indices=None):
    return float((dot(a,b,indices)/(norm(a,indices)*norm(b,indices)).clamp_min(1e-30)).detach().cpu())
def apply_backward(cells,model,arm,tau=.5,gs=None):
    model.zero_grad(set_to_none=True)
    if arm=='G':
        params=list(model.parameters()); gs=gs if gs is not None else gradients(cells,params)
        norms=torch.stack([norm(g) for g in gs]); mx=norms.max()
        scales=torch.where(norms>0,1-tau+tau*mx/norms.clamp_min(1e-30),torch.ones_like(norms)).detach()
        for j,p in enumerate(params):
            p.grad=sum((g[j]*s for g,s in zip(gs,scales) if g[j] is not None),torch.zeros_like(p))
        return norms.detach().cpu().tolist(),scales.cpu().tolist()
    sum(cells).backward(); return None,None

def stats(v):
    a=v.detach().float(); return {'mean':float(a.mean()),'std':float(a.std(unbiased=False)),'min':float(a.min()),'max':float(a.max())}
def probe(model,cells,info,tau):
    named=list(model.named_parameters()); ps=[p for _,p in named]
    gs=gradients(cells,ps)
    shared=[i for i,(n,p) in enumerate(named) if n.startswith(('embedding.','shared_weight','shared_bias','star_mlp.','skip_conn.','expert.'))]
    norms=torch.stack([norm(g) for g in gs]); mx=norms.max()
    cos=[[cosine(a,b) for b in gs] for a in gs]
    result={'cell_order':'t0d0,t0d1,t0d2,t1d0,t1d1,t1d2','norm_all':norms.detach().cpu().tolist(),'norm_shared':[float(norm(g,shared)) for g in gs],'cosine_all':cos,'cosine_shared':[[cosine(a,b,shared) for b in gs] for a in gs],'conflict_rate':sum(cos[i][j]<0 for i in range(6) for j in range(i))/15,'scales':torch.where(norms>0,1-tau+tau*mx/norms.clamp_min(1e-30),torch.ones_like(norms)).detach().cpu().tolist(),'real_count':info['n'],'copy_count':int(info['bce'].shape[0]-info['n'])}
    for key in ['w','specificity','complexity']:
        if info[key] is not None: result[key]=stats(info[key][:info['n']])
    w=info['w'][:info['n']].detach(); e=info['bce'][:info['n']].mean(1).detach()
    result['weight_bce_correlation']=float(torch.corrcoef(torch.stack([w,e]))[0,1]) if float(w.std())>0 else None
    total=sum(cells)
    aux_idx=[i for i,(n,p) in enumerate(named) if n.startswith('dbo_')]
    gtot=torch.autograd.grad(total,ps,retain_graph=True,allow_unused=True)
    result['auxiliary_gradient_norm']=float(norm(gtot,aux_idx))
    if result['copy_count']:
        real=[]; generated=[]; n=info['n']; bce=info['bce']; domain=info['domain']; weights=info['w']
        for t in range(2):
            for d in range(3):
                m=domain==d; den=m.sum().clamp_min(1)
                real.append((bce[:n,t]*weights[:n]*m[:n]).sum()/den)
                generated.append((bce[n:,t]*weights[n:]*m[n:]).sum()/den)
        gr=torch.autograd.grad(sum(real),ps,retain_graph=True,allow_unused=True)
        ga=torch.autograd.grad(sum(generated),ps,retain_graph=True,allow_unused=True)
        result['real_generated_cosine_all']=cosine(gr,ga)
        result['real_generated_cosine_shared']=cosine(gr,ga,shared)
    return result,gs

def evaluate(model,loader,device):
    model.eval(); yy=[]; pp=[]; dd=[]
    with torch.no_grad():
        for x,y in loader:
            p=model(x.T.to(device)); yy.append(y.numpy()); pp.append(p.cpu().numpy()); dd.append(x[:,-1].numpy())
    return metrics(np.concatenate(yy),np.concatenate(pp),np.concatenate(dd))
def metrics(y,p,domains):
    p=np.clip(p.astype(np.float64),1e-7,1-1e-7)
    bce=-(y*np.log(p)+(1-y)*np.log1p(-p)); out={}; aucs=[]; losses=[]; counts=[]
    for t in range(2):
        for d in range(3):
            m=domains==d; a=roc_auc_score(y[m,t],p[m,t]) if len(np.unique(y[m,t]))==2 else float('nan'); l=bce[m,t].mean() if m.any() else float('nan')
            out[f'd{d}t{t}_auc']=float(a); out[f'd{d}t{t}_logloss']=float(l); out[f'd{d}t{t}_count']=int(m.sum()); out[f'd{d}t{t}_positive']=int(y[m,t].sum()); aucs.append(a); losses.append(l); counts.append(m.sum())
    out.update(macro_auc=float(np.mean(aucs)),macro_logloss=float(np.mean(losses)),pooled_logloss=float(bce.mean()),sample_weighted_cell_auc=float(np.average(aucs,weights=counts)),worst_cell_auc=float(np.min(aucs)))
    for t in range(2):
        out[f'task{t}_macro_auc']=float(np.mean(aucs[t*3:t*3+3])); out[f'task{t}_macro_logloss']=float(np.mean(losses[t*3:t*3+3])); out[f'task{t}_pooled_auc']=float(roc_auc_score(y[:,t],p[:,t])); out[f'task{t}_pooled_logloss']=float(bce[:,t].mean())
    for d in range(3):
        out[f'domain{d}_macro_auc']=float(np.mean([aucs[d],aucs[d+3]])); out[f'domain{d}_macro_logloss']=float(np.mean([losses[d],losses[d+3]]))
    out['pooled_task_mean_auc']=float(np.mean([out[f'task{t}_pooled_auc'] for t in range(2)]))
    return out

def fusion(model,arm):
    return {n:float(torch.sigmoid(getattr(model,'_weight_'+n).deep_weights).detach()) if arm=='O' else 1. for n in ['exp_d','exp_t','bal_d','bal_t']}
def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def json_write(path,value): Path(path).write_text(json.dumps(value,indent=2,default=str,allow_nan=True))
