"""S0 source/data audit and first/tail, selection, RAM, and admission contract checks."""
import copy
import gc
import os
import shutil
import sys
import time
import uuid

import numpy as np
import torch
from experiments.conditional_information_momentum_20261102_v3.objectives import train_step
from .protocol import (DATASETS, PLAN, PLAN_COMMIT, ROOT, M, V3_MANIFEST, digest, git,
                       pinned, read, sha, source, v3, write)
from .reuse import inventory
from .runner import apply_step, batch_inputs, select
from .scheduler import admit, sample, residency
from .state import capture, commit, generation, request, restore

FIELDS = ('raw_sha256','array_hashes','split_row_fingerprints','split_row_id_hashes','encoder_hash',
          'scaler_hash','train_domain_counts','split_rows')


def audit(art,stage):
    art.mkdir(parents=True,exist_ok=True)
    try:
        authority = source(stage)
        v3_files = [pinned(v3.HERE/r['path'],M) for r in read(V3_MANIFEST)['files']]
        paths = git('ls-tree','-r','--name-only',PLAN_COMMIT,str(PLAN.relative_to(ROOT))).decode().splitlines()
        assert len(paths)==15
        plans = [pinned(ROOT/p,PLAN_COMMIT) for p in paths]
        write(art/f'{stage}_SOURCE_MANIFEST.json',dict(**authority,historical_files=v3_files,plan_files=plans))
        reuse = inventory()
        write(art/'REUSE_MANIFEST.json',reuse)
        bridge = read(art/'endpoint_bridge.json')
        assert bridge['match_count']==96 and bridge['model_hash_epoch_macro_and_cell_auc_match']
        assert bridge['max_macro_logloss_difference']<=6e-17
        data = {}
        for name,conf in v3.load_protocol()['datasets'].items():
            manifest = read(art/f'{name}_data_identity.json')
            assert sha(conf['csv']) == manifest['raw_sha256']
            assert all(manifest[k]==conf['source'][k] for k in FIELDS)
            assert manifest['dataset']==name and manifest['split_id']=='original'
            assert manifest['preprocessing_fit']=='train_only' and not manifest['test_metrics_accessed']
            data[name] = digest(manifest)
        write(art/'DATA_IDENTITY.json',data)
        assert sys.version_info[:3]==(3,13,5) and torch.__version__=='2.9.0+cu126'
        assert torch.cuda.is_available() and torch.version.cuda=='12.6'
        usable = {str(i):torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())}
        owner = generation(art/'attempt1')
        value = torch.arange(16)
        assert torch.equal(request(art/'attempt1','roundtrip',value=value)['value'],value)
        report = dict(status='SOURCE_READY',stage=stage,**authority,datasets=data,
            bridge_hash=sha(art/'endpoint_bridge.json'),reuse_sha256=sha(art/'REUSE_MANIFEST.json'),
            cuda_usable_total_bytes=usable,owner_generation=owner,environment=dict(python=sys.version,
            executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
            requirements_sha256=sha(ROOT/'requirements-phase0-3.lock')),telemetry=sample(),
            disk_free_bytes=shutil.disk_usage(art).free,time=time.time())
        write(art/f'{stage}_AUDIT.json',report)
        write(art/f'{stage}_BLOCKERS.json',dict(status='SOURCE_READY',blockers=[]))
        return report
    except Exception as exc:
        write(art/f'{stage}_BLOCKERS.json',dict(status='TECHNICAL_BLOCKED',blockers=[repr(exc)]))
        raise


def check(art,stage,gpu):
    if stage != 'A_CORE':
        raise RuntimeError('B_CORE needs its independently reviewed observer and B-specific checks')
    attempt = art/'attempt1'
    audit_report = read(art/f'{stage}_AUDIT.json')
    assert audit_report['source_hash']==source(stage)['source_hash']
    owner = generation(attempt)
    assert owner==audit_report['owner_generation']
    v3.setup(2022)
    model = torch.nn.Linear(2,1)
    selected,epochs,values = {},{},{}
    for e,(auc,ll) in enumerate(((.6,2.),(.6,1.),(.7,3.),(.65,.5)),1):
        select(e,'fixed_ema',model,dict(macro_auc=auc,macro_logloss=ll),selected,epochs,values)
    assert epochs['fixed_ema']==3 and values['fixed_ema']['macro_logloss']==3.
    row = dict(used=80,total=100,watts=84,limit=100)
    assert admit(row,.89,99,19) and not admit(row,.89,99,20) and not admit(row,.90,99,1)
    protocol = v3.load_protocol()
    profile, records = {}, []
    device = torch.device(f'cuda:{gpu}')
    physical = read(ROOT.parent/'phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/PHYSICAL_CAPACITY_MEASUREMENTS.json')
    for dataset in DATASETS:
        worker_bytes = physical['profiles'][dataset]['observed_full_worker_nvml_bytes']
        telem = sample()
        card = next(r for r in telem['devices'] if r['index']==gpu)
        if not admit(card,telem['host_ram_used'],audit_report['cuda_usable_total_bytes'][str(gpu)],worker_bytes):
            raise RuntimeError(f'GPU {gpu} current capacity insufficient for {dataset} contract fixture')
        bundle, identity = v3.load_dataset(dataset,p=protocol)
        data_hash = digest(identity)
        assert data_hash==audit_report['datasets'][dataset]
        profile[dataset] = worker_bytes
        for arm,lam in [('A0',0.),('P00',.05),('P01',.05),('P10',.05)]:
            cfg = v3.config(arm,dataset,lam)
            loader = torch.Generator().manual_seed(13022)
            order = torch.randperm(len(bundle.train_y),generator=loader).numpy()
            first,tail = order[:40960],order[(len(order)-1)//40960*40960:]
            assert len(tail)==(21927 if dataset=='movielens' else 40052)
            reference = []
            for variant in ('source','adapter'):
                v3.setup(2022,device)
                model = v3.make_model(bundle,protocol['datasets'][dataset]['source']).to(device)
                ema = copy.deepcopy(model).requires_grad_(False)
                opt = v3.make_optimizer(model,cfg)
                for step,ids in enumerate((first,tail),1):
                    x,y,rows = batch_inputs(bundle,ids,device)
                    ctx = dict(study='EMA02V2',dataset=dataset,split_id='original',seed=2022,epoch=1,global_step=step)
                    val = (train_step(model,opt,ema,x,y,rows,cfg,ctx) if variant=='source' else
                           apply_step(None,model,opt,ema,x,y,rows,cfg,ctx,loader))
                    hashes = dict(model=v3.state_digest(model.state_dict()),ema=v3.state_digest(ema.state_dict()),
                                  optimizer=v3.state_digest(opt.state_dict()),pair=val['diagnostics'].get('pair_index_hash'),
                                  loss=val['loss'])
                    if variant=='source':
                        reference.append(hashes)
                    else:
                        assert hashes==reference[step-1], (dataset,arm,step)
                        if step==1 and arm=='P10':
                            job = dict(arm=arm,dataset=dataset,purpose='S0_ENGINEERING_FIXTURE')
                            key = 'S0-'+uuid.uuid4().hex
                            request(attempt,'register',key=key,job=job,pid=os.getpid())
                            gh = digest(opt.parameter_manifest)
                            packet = capture(model,opt,ema,loader,first,None,dict(epoch=1,job=job,counts={},
                                owner_generation=owner,data_hash=data_hash,group_hash=gh),device)
                            ack = commit(attempt,key,packet,'S0_ENGINEERING_FIXTURE')
                            saved = request(attempt,'get',key=key)['state']
                            restore(saved,job,model,opt,ema,loader,None,data_hash,gh,owner,device)
                            request(attempt,'paused',key=key,reason='S0_CHECK_COMPLETE')
                            request(attempt,'release',key=key)
                            records.append(dict(dataset=dataset,ram_get_hash=ack['state_hash']))
                    torch.cuda.synchronize(device)
                    peak = residency(sample(),os.getpid())
                    if arm in ('P00','P10'):
                        profile[dataset]=max(profile[dataset],peak)
                    del x,y,val
                del model,ema,opt
                gc.collect()
                torch.cuda.empty_cache()
            records.append(dict(dataset=dataset,arm=arm,first_tail_match=True,rows=[len(first),len(tail)],
                                source_and_adapter_hashes=reference))
        del bundle
    result = dict(status='PASS',stage=stage,source_hash=audit_report['source_hash'],owner_generation=owner,
                  full_process_bytes=profile,cuda_usable_total_bytes=audit_report['cuda_usable_total_bytes'],
                  records=records,new_scientific_runs=0,test_evaluations=0)
    write(art/f'{stage}_CHECK.json',result)
    return result
