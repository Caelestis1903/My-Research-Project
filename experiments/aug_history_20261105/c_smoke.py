"""Original-batch C correctness/peak check; no scientific selection."""
import copy
import time
from pathlib import Path
import numpy as np
import torch
from experiments.overnight_ab_20261103 import queue
from experiments.overnight_ab_20261103.data_bridge import cached_bundle
from experiments.overnight_ab_20261103.state import state_digest,write_json
from . import worker
from .objectives import source_losses
from .c_search import make_job,scope_check


def gpu_smoke_C(output,prepared,devices):
    from .run import CostBlocked
    scope_check(prepared,prepared['scope_datasets'])
    if not torch.cuda.is_available(): raise CostBlocked('CUDA is unavailable')
    resources=[queue.gpu_resources(device) for device in devices]
    eligible=[r for r in resources if r['memory_used_mib']/r['memory_total_mib']<.85 and r['power_draw_w']/r['power_limit_w']<.85]
    free,total=queue.available_ram()
    if not eligible or free/total<=.1: raise CostBlocked('C smoke awaits an eligible GPU/RAM admission')
    resource=min(eligible,key=lambda r:r['memory_used_mib']); device=f'cuda:{resource["device"]}'
    bundle=cached_bundle(prepared['cache_root'],'movielens',include_test=False)
    generator=torch.Generator().manual_seed(2022+11000)
    ids=torch.randperm(len(bundle.train_y),generator=generator)[:40960].numpy()
    x=torch.as_tensor(np.array(bundle.train_x[ids]),device=device).T
    y=torch.as_tensor(np.array(bundle.train_y[ids]),device=device); row_ids=np.asarray(bundle.train_row_ids[ids])
    gamma0=make_job(prepared,dict(id='C0_ENGINEERING_ONLY',gamma=0.,refresh_steps=20,control_mode='task'),2022,'ENGINEERING_ONLY')
    session=worker.create_session(gamma0,bundle,device); reference=copy.deepcopy(session['model'])
    teacher=copy.deepcopy(reference); c=gamma0['config']
    optimizer=torch.optim.Adam(reference.parameters(),lr=.01,betas=(.9,.999),eps=1e-8,
        weight_decay=c['weight_decay'],foreach=False,fused=False)
    torch.cuda.reset_peak_memory_stats(device); started=time.perf_counter()
    for step in (1,2):
        value=worker.train_batch(session,x,y,row_ids=row_ids)
        optimizer.zero_grad(set_to_none=True)
        loss,_=source_losses(reference,x,y,0,dict(num_views=1,view_family='gaussian',pairing='iid'),
            'movielens',2022,step,mix_lambda=.5,kappa=.05)
        loss.backward(); optimizer.step()
        with torch.no_grad():
            for phi,theta in zip(teacher.parameters(),reference.parameters()): phi.mul_(c['rho']).add_(theta,alpha=1-c['rho'])
        if value['loss']!=float(loss.detach()) or any(not torch.equal(a,b) for a,b in zip(reference.parameters(),session['model'].parameters())):
            raise RuntimeError('Gamma-zero original-batch source update differs')
        if any(not torch.equal(a,b) for a,b in zip(teacher.parameters(),session['bank'].teacher.parameters())):
            raise RuntimeError('Gamma-zero source EMA differs')
    identity_hash=state_digest(session['model'].state_dict())
    del session,reference,teacher,optimizer
    conditional_job=next(job for job in prepared['jobs'] if job['candidates'][0]['id']=='C1_NEG')
    session=worker.create_session(conditional_job,bundle,device)
    for step in range(1,22): value=worker.train_batch(session,x,y,row_ids=row_ids)
    if session['bank'].last_valid_step!=21 or not bool(session['bank'].A.ne(0).any()):
        raise RuntimeError('C original-batch smoke did not exercise nonzero old-teacher drift')
    torch.cuda.synchronize(device)
    result=dict(route='C',scope_datasets=['movielens'],source_hash=prepared['identity']['source_hash'],datasets=dict(movielens=dict(
        status='PASS',purpose='ENGINEERING_ONLY',device=device,batch_size=40960,admission=resource,
        peak_gpu_bytes=torch.cuda.max_memory_reserved(device),wall_seconds=time.perf_counter()-started,
        gamma_zero_steps=2,gamma_zero_model_hash=identity_hash,conditional_steps=21,
        last_valid_statistic_step=session['bank'].last_valid_step,root_hash=state_digest(session['bank'].root),
        nonzero_statistics=True,old_teacher_step=value['response']['teacher_preupdate_step'])))
    write_json(Path(output)/'GPU_SMOKE.json',result)
    return result
