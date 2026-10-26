"""Bounded canonical-forward fixtures, without fitting or resident RPC access."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,json,hashlib,marshal
import numpy as np
import torch
from experiments.mdmt_rcf_20261026 import baselines
from experiments.mdmt_rcf_20261026.raw_reconstruction_observer import raw_features,raw_source,_formed_values,install_on_reader_reload,callable_fingerprint
from experiments.mdmt_rcf_20261026.prefix import PilotAccess,configure
from experiments.mdmt_rcf_20261026.protocol import PrefixBundle,RoleView,FeatureBank
from experiments.mdmt_rcf_20261026.formation import DeltaMLP,DeltaFunction
from experiments.mdmt_rcf_20261026.vendor_snapshot.mdmt_dfd_20261026.functions import PilotBank
from experiments.mdmt_rcf_20261026.state import state_hash,capture_rng
from experiments.mdmt_rcf_20261026.memory_data import data_hash

class ForbiddenCache(dict):
    def __getitem__(self,key):raise AssertionError('Training content cache read')
    def get(self,*args,**kwargs):raise AssertionError('Training content cache read')

class IndependentInput(torch.nn.Module):
    def __init__(self,h):
        super().__init__();self.anchor=torch.nn.Parameter(torch.tensor(0.),requires_grad=False);self.raw_h=h
    def encode(self,x):
        ids=x[:,0].cpu().numpy().astype(np.int64)
        return SimpleNamespace(h=torch.as_tensor(self.raw_h[ids],device=x.device),logits=torch.zeros((len(ids),2),device=x.device))

configure();torch.manual_seed(22811);rng=np.random.default_rng(93301)
n=41063;h=rng.normal(size=(n,256)).astype(np.float32);z=np.zeros((n,2),np.float32);domain=np.arange(n)%3;ids=np.arange(n,dtype=np.int64)
root=Path('artifacts/mdmt_rcf_20261026/v2_four_seed/verification_raw_reconstruction').resolve()
root.mkdir(parents=True,exist_ok=True)
data=SimpleNamespace(x=ids.astype(np.float32)[:,None],user_ids=None)
view=RoleView(ids,domain,np.zeros((n,2),np.float32),h,z,np.zeros((n,2),np.int64))
native=IndependentInput(h)
b=PrefixBundle('movielens',5201,{'fit':view},{},None,None,native,{}, {},{'native_hash':'read_only_fixture_input_identity'},None,data,'cuda:0' if torch.cuda.is_available() else 'cpu',root)
base=PilotBank(lambda s,q,k:2400+30*s+q)
with torch.no_grad():
    for head in base.heads.values():head.output.weight.fill_(.023);head.output.bias.fill_(.017)
initial=copy.deepcopy(base).cpu();source=copy.deepcopy(base).to(b.device);locals_=[copy.deepcopy(base).to(b.device),copy.deepcopy(base).to(b.device)]
b.pilots=PilotAccess(b,source,locals_,initial)
oracles={};legacy_errors={}
for method in ('FULL_SOURCE','FULL_SOURCE_ALL','FULL_TARGET','RANDOM_FEATURE','PILOT_LOGIT'):
    oracles[method]={}
    for t in range(3):
        mask=domain==t
        for q in range(2):
            sources=[s for s in range(3) if s!=t]
            if method=='FULL_TARGET':expected=np.concatenate([b.pilots.local_hidden('fit',t,q,k)[mask] for k in range(2)],1)
            elif method=='PILOT_LOGIT':expected=np.stack([b.pilots.residual('fit',s,q)[mask]-b.pilots.residual('fit',t,q)[mask] for s in sources],1)
            else:expected=np.concatenate([b.pilots.hidden('fit',s,q,initial=method=='RANDOM_FEATURE')[mask] for s in sources],1)
            oracles[method][t,q]=expected
            if method=='FULL_SOURCE':
                old=np.concatenate([((baselines._raw_hidden(source.head(s,q),h[mask])-b.pilots.phi_normalizers[s,q]['mean'])/b.pilots.phi_normalizers[s,q]['std']).astype(np.float32) for s in sources],1)
                legacy_errors[f'{t}:{q}']=float(np.max(np.abs(old-expected)))
residual_oracle=np.stack([np.stack([b.pilots.residual('fit',s,q) for q in range(2)],1) for s in range(3)],1)
b.pilots._cache=ForbiddenCache()
weight=base.head(0,0).hidden.weight.detach().clone();bias=base.head(0,0).hidden.bias.detach().clone()
module=DeltaMLP(weight,bias)
with torch.no_grad():module.output.weight.fill_(.032);module.output.bias.fill_(.02)
module.eval().requires_grad_(False)
function=DeltaFunction(module,.713,0,0,0,'CANONICAL_DELTA',outputs=ForbiddenCache())
canonical=copy.copy(function);canonical.module=copy.deepcopy(module).to(b.device)
delta_oracle=canonical.values(h)
columns={(t,q):([dict(function_id=function.function_id)] if t!=0 and q==0 else []) for t in range(3) for q in range(2)}
bank=FeatureBank(ForbiddenCache(),columns,{},dict(method='FIXTURE_FORMED'));bank.functions={function.function_id:function}
models=[native,source,initial,*locals_,module]
def fingerprint():
    return dict(state=[state_hash(m.state_dict()) for m in models],modes=[[a.training for a in m.modules()] for m in models],devices=[[str(p.device) for p in m.parameters()] for m in models],requires=[[p.requires_grad for p in m.parameters()] for m in models],gradients=[state_hash({k:p.grad for k,p in m.named_parameters()}) for m in models],rng=state_hash(capture_rng()))
before=fingerprint();records=[]
for method in oracles:
    fixed=SimpleNamespace(fixed_method=method,identity={'method':method},values=ForbiddenCache())
    got=raw_features(b,fixed,'fit',h,z,domain)
    error=max(float(np.max(np.abs(got[k]-oracles[method][k]))) for k in got)
    assert error==0,(method,error)
    records.append(dict(method=method,max_content_error=error,dtypes=sorted({str(v.dtype) for v in got.values()}),exact=True))
assert np.array_equal(raw_source(b,h),residual_oracle)
got=raw_features(b,bank,'fit',h,z,domain)
assert all(np.array_equal(got[t,q],delta_oracle[domain==t,None] if columns[t,q] else np.empty((int((domain==t).sum()),0))) for t in range(3) for q in range(2))
old_formed=max(float(np.max(np.abs(function.values(h[domain==t])-delta_oracle[domain==t]))) for t in (1,2))
assert before==fingerprint()
assert install_on_reader_reload()['status']=='NO_OP'
# Frozen training/readout definitions: only the two named observer functions differ.
previous=Path('research_reports/2026-10-26_v2/evidence/pre_raw_reconstruction_observer')
asts={};old_code={}
for name in ('baselines.py','readers.py'):
    old=ast.parse((previous/name).read_text());new=ast.parse((Path('experiments/mdmt_rcf_20261026')/name).read_text())
    olddefs={a.name:ast.dump(a,include_attributes=False) for a in old.body if isinstance(a,(ast.FunctionDef,ast.ClassDef))}
    newdefs={a.name:ast.dump(a,include_attributes=False) for a in new.body if isinstance(a,(ast.FunctionDef,ast.ClassDef))}
    changed=[key for key in olddefs if olddefs[key]!=newdefs.get(key)]
    assert changed==(['_raw_source','raw_features'] if name=='baselines.py' else []),changed
    asts[name]=dict(changed_definitions=changed,other_original_definitions_unchanged=True)
    if name=='baselines.py':
        code=compile((previous/name).read_text(),str((Path('experiments/mdmt_rcf_20261026')/name).resolve()),'exec')
        old_code={c.co_name:hashlib.sha256(marshal.dumps(c)).hexdigest() for c in code.co_consts if hasattr(c,'co_name') and c.co_name in ('raw_features','_raw_source')}
result=dict(status='BASIC_CHECKS_COMPLETE',formal_training=False,resident_service_contacted=False,model_files_written=0,
    rows=n,pilot_chunk=40960,delta_chunk=16384,initial_device=str(next(initial.parameters()).device),formation_device=b.device,
    fixed_content=records,pilot_residual_exact=True,formed_new_max_error=0.,old_masked_cpu_delta_max_error=old_formed,
    old_masked_source_content_errors=legacy_errors,training_cache_access_forbidden=True,frozen_fingerprint_unchanged=True,
    non_worker_install_no_op=True,definition_ast_audit=asts,old_callable_fingerprints=old_code,
    new_callable_fingerprints={'raw_features':callable_fingerprint(raw_features),'_raw_source':callable_fingerprint(raw_source)},
    source_sha256={name:hashlib.sha256((Path('experiments/mdmt_rcf_20261026')/name).read_bytes()).hexdigest() for name in ('raw_reconstruction_observer.py','baselines.py','readers.py')},
    actual_trained_case_after='PENDING_INDEPENDENT_PASS_AND_NORMAL_EVALUATE',original_tolerance=1e-6)
Path('research_reports/2026-10-26_v2/RAW_RECONSTRUCTION_OBSERVER_BASIC_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
