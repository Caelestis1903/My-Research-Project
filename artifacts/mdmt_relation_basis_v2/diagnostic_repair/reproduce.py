from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from experiments.mdmt_relation_basis_v2.config import load_config
from experiments.mdmt_relation_basis_v2.data import Data,tensor
from experiments.mdmt_relation_basis_v2.native import build_native,synchronize
from experiments.mdmt_relation_basis_v2.source_functions import fit_source
from experiments.mdmt_relation_basis_v2.trainer import train_method,predict
from experiments.mdmt_relation_basis_v2.diagnostics import raw_forward,metric_difference
from experiments.mdmt_relation_basis_v2.metrics import evaluate
from experiments.mdmt_relation_basis_v2.state import configure,write_json,state_hash
configure()
started=time.monotonic()
base=Path('artifacts/mdmt_relation_basis_v2')
root=base/'diagnostic_repair'/'seed2022_v2'
root.mkdir(exist_ok=False)
(root/'prefix').mkdir()
data=Data('movielens',base,root/'label_access.jsonl')
cfg=load_config();device='cuda:0'
old=json.loads((base/'workers/movielens/seed2022/PREFIX_COMPLETE.json').read_text())
identity=old['identity']
views={}
for role in ('T0','F','S','Q','V'):
 p=base/'workers/movielens/seed2022/prefix/row_data'/role
 views[role]=dict(h=np.load(p/'h.npy',mmap_mode='r'),z=np.load(p/'z0.npy',mmap_mode='r'),ids=data.roles[role],domain=data.domain[data.roles[role]])
 if role!='T0':views[role]['phi']=np.load(p/'phi.npy',mmap_mode='r')
native,ns,nf,nr=build_native(data,2022,cfg,root/'prefix',device,identity)
assert nr['selected_model_hash']==old['native']['selected_model_hash'],(nr,old['native'])
source,ss,sr=fit_source(data,views['T0'],2022,root/'prefix',device,identity)
assert sr['source_function_hash']==old['source']['source_function_hash'],(sr,old['source'])
resident=train_method('READ',data,views,2022,cfg,root,device,identity)
prior=json.loads((base/'runs/movielens/READ/seed2022/run_receipt.json').read_text())
assert resident['receipt']['selected_model_hash']==prior['selected_model_hash']
model=resident['model'].to(device).eval()

def summary(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);diff=np.abs(a-b)
 idx=np.unravel_index(np.argmax(diff),diff.shape)
 return dict(max_abs=float(diff[idx]),mean_abs=float(diff.mean()),rms=float(np.sqrt(np.mean(diff*diff))),max_index=[int(i) for i in idx],cached_value=float(a[idx]),raw_value=float(b[idx]))

records=[]
with torch.no_grad():
 for batch in (1,256):
  ids=data.roles['Q'][:batch];x=tensor(data.x[ids],device);domain=x[:,-1].long()
  packet=native.encode(x);phi=source(packet.h,domain)
  hc,zc,pc=(tensor(views['Q'][key][:batch],device) for key in ('h','z','phi'))
  cached=model(hc,zc,domain,pc)
  raw=raw_forward(native,source,model,x)
  replay_phi=source(hc)
  rec=dict(batch=batch,logits=summary(cached.cpu(),raw.cpu()),h=summary(hc.cpu(),packet.h.cpu()),z=summary(zc.cpu(),packet.logits.cpu()),phi_foreign=summary(pc.cpu().numpy()[np.arange(batch)[:,None],np.array([[1,2],[0,2],[0,1]])[domain.cpu().numpy()]],phi.cpu().numpy()[np.arange(batch)[:,None],np.array([[1,2],[0,2],[0,1]])[domain.cpu().numpy()]]),phi_same_h=summary(pc.cpu(),replay_phi.cpu()))
  context_n=min(4096,len(views['Q']['ids']))
  context_packet=native.encode(tensor(data.x[data.roles['Q'][:context_n]],device))
  rec['native_same_4096_context_h']=summary(hc.cpu(),context_packet.h[:batch].cpu())
  rec['native_same_4096_context_z']=summary(zc.cpu(),context_packet.logits[:batch].cpu())
  source_context_n=min(40960,len(views['Q']['ids']))
  h_context=tensor(views['Q']['h'][:source_context_n],device)
  rec['source_details']=[]
  for source_id in range(3):
   foreign=domain!=source_id
   if not bool(foreign.any()):continue
   context_hidden=source.hidden(h_context,source_id)
   context_phi=source.one(h_context,source_id)
   raw_hidden=source.hidden(packet.h[foreign],source_id)
   same_h_hidden=source.hidden(hc[foreign],source_id)
   local_context_phi=context_phi[:batch][foreign]
   stored_phi=pc[foreign,source_id]
   norm_std=source.std[source_id]
   rec['source_details'].append(dict(source=source_id,source_same_40960_context_phi=summary(stored_phi.cpu(),local_context_phi.cpu()),hidden_different_geometry=summary(context_hidden[:batch][foreign].cpu(),raw_hidden.cpu()),hidden_same_h_different_geometry=summary(context_hidden[:batch][foreign].cpu(),same_h_hidden.cpu()),normalized_same_h_different_geometry=summary(stored_phi.cpu(),source.one(hc[foreign],source_id).cpu()),std_min=float(norm_std.min()),std_max=float(norm_std.max()),std_at_floor=int((norm_std<=.001000001).sum())))
  rec['cached_hz_raw_phi_logits']=summary(cached.cpu(),model(hc,zc,domain,phi).cpu())
  rec['raw_hz_cached_phi_logits']=summary(cached.cpu(),model(packet.h,packet.logits,domain,pc).cpu())
  records.append(rec);print('FIRST_RAW_DIFF',json.dumps(rec),flush=True)
  write_json(root/'point_differences.json',records)
 q=views['Q'];y,mask=data.labels('Q','diagnostic',method='READ_replay')
 cached_z=predict(model,q,device)
 raw_z=[];same_shape_reencoded=[]
 for begin in range(0,len(q['ids']),4096):
  rows=q['ids'][begin:begin+4096]
  raw_z.append(raw_forward(native,source,model,tensor(data.x[rows],device)).cpu().numpy())
 raw_z=np.concatenate(raw_z)
 cached_m=evaluate(y,cached_z,q['domain'],mask);raw_m=evaluate(y,raw_z,q['domain'],mask)
 result=dict(status='REPRODUCED',native_hash_match=True,source_hash_match=True,READ_selected_hash_match=True,points=records,Q_cached=cached_m,Q_raw=raw_m,Q_raw_minus_cached=metric_difference(raw_m,cached_m),Q_logit_difference=summary(cached_z,raw_z),seconds=time.monotonic()-started,source_native_seconds=nr['seconds']+sr['seconds'],READ_seconds=resident['receipt']['training_seconds'],model_files_written=0)
 np.savez_compressed(root/'Q_logit_comparison.npz',row_id=q['ids'],cached_logits=cached_z,raw_logits=raw_z,labels=y,domain=q['domain'])
 write_json(root/'REPRODUCTION.json',result)
 print('DIAGNOSTIC_COMPLETE',json.dumps(result),flush=True)
