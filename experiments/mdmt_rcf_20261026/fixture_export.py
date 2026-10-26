"""Read-only execution data and cost delivery; never training-state transport."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import expit
from .artifacts import _json,_npz,_sha


def live_model_hashes(bundle):
    from .state import state_hash
    result={'native':state_hash(bundle.native.state_dict())}
    for name in ('source_bank','initial_bank'):
        module=getattr(bundle.pilots,name,None)
        if module is not None:result[name]=state_hash(module.state_dict())
    for i,module in enumerate(bundle.pilots.local_banks):result[f'local_bank_{i}']=state_hash(module.state_dict())
    for key,value in bundle.base_models.items():
        module=value[0] if isinstance(value,tuple) else value
        result['base_'+str(key)]=state_hash(module.state_dict())
    return result


def export_reader_fixture(bundle,config):
    """Exact failed FULL_SOURCE group, freshly replayed under the same identity."""
    before=live_model_hashes(bundle);view=bundle.views['P_read'];q=0;t=0;g=0
    mask=(view.domain==t)&(view.group[:,q]==g);rows=view.row_id[mask]
    x=np.concatenate([bundle.pilots.hidden('P_read',s,q,standardized=True)[mask] for s in (1,2)],1).astype(np.float64)
    b=np.asarray(bundle.base_logits['P_read'][mask,q],np.float64);y=np.asarray(view.y[mask,q],np.float64)
    if x.shape!=(len(rows),64) or not np.isfinite(x).all() or not np.isfinite(b).all():raise ValueError('Invalid actual FULL_SOURCE fixture')
    lam=.001;ridge=config['reader']['coefficient_ridge'];a=np.column_stack((b,np.ones(len(b)),x));v=expit(b)*(1-expit(b))
    h=(a.T*v)@a/len(b);h[2:,2:]+=lam*(x.T@x)/len(b)+ridge*np.eye(64)
    eig=np.linalg.eigvalsh((h+h.T)/2);sv=np.linalg.svd(x,compute_uv=False)
    root=Path(config['paths']['artifact'])/'reader_fixture'/bundle.dataset/f'seed{bundle.seed}';root.mkdir(parents=True,exist_ok=True)
    path=root/'FULL_SOURCE_P_read_t0q0g0.npz'
    arrays=dict(b=b,x=x,y=y,row_id=rows,domain=view.domain[mask],group=view.group[mask])
    from .state import array_hash
    hashes={k:array_hash(np.asarray(a)) for k,a in arrays.items()}
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(arrays) or any(array_hash(old[k])!=hv for k,hv in hashes.items()):raise FileExistsError('Existing fixture differs from replayed identity')
        file=dict(path=str(path.resolve()),sha256=_sha(path),bytes=path.stat().st_size)
    else:file=_npz(path,**arrays)
    bundle.ledger.access('READER_NUMERICAL_FIXTURE','P_read',rows=rows,task=q,purpose='audit')
    after=live_model_hashes(bundle)
    if before!=after:raise RuntimeError('Read-only fixture export changed a trained model')
    result=dict(status='EXPORTED',method='FULL_SOURCE',original_start=[1.,0.]+[0.]*64,extraction_source_sha256={'fixture_export.py':_sha(__file__),'baselines.py':_sha(Path(__file__).with_name('baselines.py'))},current_disk_readers_sha256=_sha(Path(__file__).with_name('readers.py')),solver_loaded_by_export=False,file=file,shape=list(x.shape),historical_expected_rows=2359,actual_rows=len(rows),same_as_historical_row_count=len(rows)==2359,array_hashes=hashes,role='P_read',dataset=bundle.dataset,seed=bundle.seed,target=t,task=q,group=g,source_columns=[dict(source=s,task=q,coordinate=j) for s in (1,2) for j in range(32)],lambda_value=lam,coefficient_ridge=ridge,bounds=[[.25,4.],[-4.,4.]]+[[None,None]]*64,positive=int(y.sum()),negative=int(len(y)-y.sum()),condition=dict(feature_rank=int((sv>max(x.shape)*np.finfo(float).eps*sv[0]).sum()),feature_condition=None if sv[-1]==0 else float(sv[0]/sv[-1]),initial_hessian_eigen_min=float(eig[0]),initial_hessian_eigen_max=float(eig[-1]),initial_hessian_condition=None if eig[0]<=0 else float(eig[-1]/eig[0])),prefix_identity=bundle.identity,models_before=before,models_after=after,models_unchanged=True,solver_executed=False,model_files_written=0,fixture_source='same-identity deterministic replay after documented process loss')
    _json(root/'FIXTURE_RECEIPT.json',result);return result
