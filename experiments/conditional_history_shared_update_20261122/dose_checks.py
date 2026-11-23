"""Source-equivalence, paired-draw and RAM anchor checks on natural first/tail batches."""
import copy
import gc
import os
import time
import uuid

import torch

from experiments.conditional_information_momentum_20261102_v3.objectives import train_step
from .dose_runner import dose_config, make_optimizer
from .dose_stage import STAGE, dose_source, stopping
from .history_checks import live_id
from .history_runner import step_anchor, store_anchor
from .protocol import DATASETS, digest, read, v3, write
from .results import DOSE
from .runner import batch_inputs
from .scheduler import admit, residency, sample
from .state import generation, request


def check(art, gpu):
    attempt = art/'attempt1'
    audit = read(art/f'{STAGE}_AUDIT.json')
    assert audit['source_hash'] == dose_source()['source_hash']
    assert audit['owner_generation'] == generation(attempt)
    history_check = read(art/'A_HISTORY_CHECK.json')
    recent = read(art/'A_HISTORY_RUN.json')['full_process_bytes']
    profile = {d:max(recent[d],history_check['full_process_bytes'][d]) for d in DATASETS}
    protocol = v3.load_protocol()
    device = torch.device(f'cuda:{gpu}')
    records = []
    for dataset in DATASETS:
        bundle,identity = v3.load_dataset(dataset,p=protocol)
        assert digest(identity) == audit['datasets'][dataset]
        high = next(r for r in history_check['records'] if r['dataset']==dataset and r['arm']=='A21')
        paired = None
        for arm in DOSE:
            cfg = dose_config(arm,dataset)
            source = v3.config('PCOLL' if arm==DOSE[0] else 'P00',dataset,.05)
            source.update(lambda_pair=.025,dynamic=False)
            assert all(cfg[k]==v for k,v in source.items() if k!='arm')
            assert dose_config(arm,dataset,0.) == v3.config('A0',dataset,0.)
            expected = []
            anchor_receipt = None
            for variant in ('reference','live'):
                while True:
                    if stopping(art):
                        raise RuntimeError('Dose fixture stopped')
                    obs = sample()
                    card = next(c for c in obs['devices'] if c['index']==gpu)
                    if admit(card,obs['host_ram_used'],audit['cuda_usable_total_bytes'][str(gpu)],profile[dataset]):
                        break
                    time.sleep(2)
                v3.setup(2022,device)
                model = v3.make_model(bundle,protocol['datasets'][dataset]['source']).to(device)
                ema = copy.deepcopy(model).requires_grad_(False)
                active_cfg = source if variant=='reference' else cfg
                optimizer = (v3.make_optimizer(model,source) if variant=='reference' else make_optimizer(model,cfg))
                loader = torch.Generator().manual_seed(13022)
                order = torch.randperm(len(bundle.train_y),generator=loader).numpy()
                draws = []
                for step,offset in enumerate((0,(len(order)-1)//40960*40960),1):
                    ids = order[offset:offset+40960]
                    x,y,rows = batch_inputs(bundle,ids,device)
                    ctx = dict(study='EMA02V2',dataset=dataset,split_id='original',seed=2022,epoch=1,global_step=step)
                    if arm==DOSE[0] and variant=='live' and step==2:
                        job = dict(dataset=dataset,arm=arm,purpose='DOSE_ENGINEERING',key=uuid.uuid4().hex)
                        meta = dict(job=job,owner_generation=audit['owner_generation'],data_hash=digest(identity),
                                    group_hash=digest(optimizer.parameter_manifest),global_step_before=step-1)
                        value,anchor = step_anchor(model,optimizer,ema,x,y,rows,cfg,ctx,loader,order,offset,meta)
                        anchor_receipt = store_anchor(attempt,job,anchor)
                        try:
                            saved = request(attempt,'get',key=anchor_receipt['key'])['state']['anchor']
                            clone = v3.make_model(bundle,protocol['datasets'][dataset]['source']).to(device)
                            clone.load_state_dict(saved['model'])
                            fclone = copy.deepcopy(clone).requires_grad_(False)
                            fclone.load_state_dict(saved['ema'])
                            replay = make_optimizer(clone,cfg)
                            replay.load_state_dict(saved['optimizer'])
                            for name,param in clone.named_parameters():
                                grad = saved['mixed_grad'][name]
                                param.grad = grad.to(device) if grad is not None else None
                            replay.step(saved['pair'],[g.to(device) for g in saved['raw_pair']])
                            v3.update_ema(fclone,clone,cfg['rho'])
                            assert live_id(clone,fclone,replay) == live_id(model,ema,optimizer)
                            assert v3.state_digest(replay.state_dict()) == anchor['actual_post']['optimizer']
                            del clone,fclone,replay,saved,anchor
                        finally:
                            request(attempt,'release',key=anchor_receipt['key'])
                    else:
                        value = train_step(model,optimizer,ema,x,y,rows,active_cfg,ctx)
                    pair = value['diagnostics']['pair_index_hash']
                    assert pair == high['first_tail'][step-1]['pair']
                    draws.append([v3.state_digest(rows),pair])
                    measured = dict(**live_id(model,ema,optimizer),loss=value['loss'],pair=pair)
                    if variant=='reference':
                        expected.append(measured)
                    else:
                        assert measured == expected[step-1], (dataset, arm, step)
                    torch.cuda.synchronize(device)
                    profile[dataset] = max(profile[dataset],residency(sample(),os.getpid()))
                    del x,y,value
                if paired is None:
                    paired = draws
                assert paired == draws
                del optimizer,model,ema
                gc.collect()
                torch.cuda.empty_cache()
            records.append(dict(dataset=dataset,arm=arm,first_tail=expected,rows_and_pairs=paired,
                                reference_source_arm=source['arm'],restored_anchor=anchor_receipt))
        del bundle
    result = dict(stage=STAGE,status='PASS',source_hash=audit['source_hash'],owner_generation=audit['owner_generation'],
                  full_process_bytes=profile,cuda_usable_total_bytes=audit['cuda_usable_total_bytes'],records=records,
                  new_scientific_runs=0,test_evaluations=0)
    write(art/f'{STAGE}_CHECK.json',result)
    return result
