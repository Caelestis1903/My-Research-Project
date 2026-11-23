"""Engineering checks of live steps, shadows, paired draws, anchor readback and residency."""
import copy
import gc
import os
import time
import uuid

import numpy as np
import torch

from experiments.conditional_information_momentum_20261102_v3.objectives import train_step
from experiments.conditional_information_momentum_20261102_v3.optimizers import shared_name
from .history_controls import make_history_optimizer, history_config
from .history_runner import step_anchor, store_anchor
from .history_stage import STAGE, history_source, stopping
from .protocol import DATASETS, digest, read, v3, write
from .runner import batch_inputs
from .scheduler import admit, residency, sample
from .state import generation, request


def reference(model, cfg):
    if cfg['arm']=='A21':
        return v3.make_optimizer(model,cfg)
    # Independent native reference. No diagnostic autograd or shadow state.
    ref = v3.make_optimizer(model,dict(cfg,mode='adam'))
    groups = [dict(params=[p for n,p in model.named_parameters() if shared_name(n)==dense])
              for dense in (False, True)]
    groups[1]['betas'] = (.99, .999)
    ref.adam = torch.optim.Adam(groups,lr=cfg['lr'],weight_decay=cfg['weight_decay'],eps=1e-8,
                                amsgrad=False,foreach=False,fused=False)
    return ref


def live_id(model, ema, opt):
    return {k:v3.state_digest(v) for k,v in dict(model=model.state_dict(),ema=ema.state_dict(),
               adam=opt.adam.state_dict(),history=opt.history,moments=opt.moments,selected=opt.selected,
               pair_clock=opt.pair_clock,demand=opt.demand).items()}


def basic():
    """Independent .9 Adam first-moment reconstruction, empty pair and None/zero semantics."""
    v3.setup(2022)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_weight=torch.nn.Parameter(torch.tensor([.2, -.1]))
            self.zero=torch.nn.Parameter(torch.tensor([1.]))
            self.unused=torch.nn.Parameter(torch.tensor([1.]))
    for arm in ('A21','A22'):
        cfg = history_config(arm,'movielens')
        model = Model()
        opt = make_history_optimizer(model,cfg)
        ordinary = torch.zeros_like(model.shared_weight)
        for i,cond in enumerate((0,7,None,10,None)):
            opt.zero_grad()
            p = model.shared_weight
            mixed = torch.tensor([.05*(i+1),-.4/(i+1)])
            p.grad = mixed.clone()
            model.zero.grad = torch.zeros_like(model.zero)
            ordinary.lerp_(mixed.add(p.detach(),alpha=cfg['weight_decay']), .1)
            pair = dict(selected=cond,index=0,w=np.array([.4]),q=np.array([.5])) if cond is not None else None
            r = [torch.tensor([1./(i+1),-.2])] if pair else None
            old = opt.shadow.get('shared_weight',{}).get('Q')
            old = old.clone() if old is not None else None
            opt.step(pair,r)
            s = opt.shadow['shared_weight']
            m = opt.moments['shared_weight']['m'] if arm=='A21' else s['R9']
            torch.testing.assert_close(m+s['M9'].sum(0),ordinary,atol=2e-7,rtol=1e-5)
            if arm=='A21' and pair is None:
                assert torch.equal(old,s['Q'])
            opt.reconstruction_error()
        assert opt.pair_clock==(3 if arm=='A21' else 0)
        assert model.unused not in opt.adam.state and opt.adam.state[model.zero]['step'].item()==5
        assert not opt.demand
        cfg0 = history_config(arm,'movielens',0.)
        m0,ref = Model(),Model()
        o0 = make_history_optimizer(m0,cfg0)
        oref = v3.make_optimizer(ref,v3.config('A0','movielens',0))
        for model_ in (m0,ref):
            model_.shared_weight.grad=torch.ones_like(model_.shared_weight)
        o0.step()
        oref.step()
        assert cfg0==oref.config and live_id(m0,m0,o0)==live_id(ref,ref,oref)


def check(art,gpu):
    basic()
    attempt = art/'attempt1'
    audit = read(art/f'{STAGE}_AUDIT.json')
    assert audit['source_hash']==history_source()['source_hash'] and audit['owner_generation']==generation(attempt)
    old_profile = read(art/'A_CORE_CHECK.json')['full_process_bytes']
    profile = dict(old_profile)
    protocol = v3.load_protocol()
    device = torch.device(f'cuda:{gpu}')
    records = []
    for dataset in DATASETS:
        bundle,identity = v3.load_dataset(dataset,p=protocol)
        assert digest(identity)==audit['datasets'][dataset]
        paired = None
        for arm in ('A21','A22'):
            cfg = history_config(arm,dataset)
            signatures = []
            for variant in ('reference','adapter'):
                while True:
                    if stopping(art):
                        raise RuntimeError('Fixture paused by stop control')
                    obs = sample()
                    card = next(c for c in obs['devices'] if c['index']==gpu)
                    if admit(card,obs['host_ram_used'],audit['cuda_usable_total_bytes'][str(gpu)],profile[dataset]):
                        break
                    time.sleep(2)
                v3.setup(2022,device)
                model = v3.make_model(bundle,protocol['datasets'][dataset]['source']).to(device)
                ema = copy.deepcopy(model).requires_grad_(False)
                loader = torch.Generator().manual_seed(13022)
                order = torch.randperm(len(bundle.train_y),generator=loader).numpy()
                opt = reference(model,cfg) if variant=='reference' else make_history_optimizer(model,cfg)
                ids_hashes = []
                for step,offset in enumerate((0,(len(order)-1)//40960*40960),1):
                    ids = order[offset:offset+40960]
                    x,y,rows = batch_inputs(bundle,ids,device)
                    ctx = dict(study='EMA02V2',dataset=dataset,split_id='original',seed=2022,epoch=1,global_step=step)
                    if variant=='adapter' and step==2:
                        job = dict(dataset=dataset,arm=arm,purpose='HISTORY_ENGINEERING',key=uuid.uuid4().hex)
                        meta = dict(job=job,owner_generation=audit['owner_generation'],data_hash=digest(identity),
                                    group_hash=digest(opt.parameter_manifest),global_step_before=1)
                        value,anchor = step_anchor(model,opt,ema,x,y,rows,cfg,ctx,loader,order,offset,meta)
                        receipt = store_anchor(attempt,job,anchor)
                        saved = request(attempt,'get',key=receipt['key'])['state']['anchor']
                        # Reapply the captured real gradients and raw r, including the pre-update shadows.
                        clone = v3.make_model(bundle,protocol['datasets'][dataset]['source']).to(device)
                        clone.load_state_dict(saved['model'])
                        fclone = copy.deepcopy(clone).requires_grad_(False)
                        fclone.load_state_dict(saved['ema'])
                        test_opt = make_history_optimizer(clone,cfg)
                        test_opt.load_state_dict(saved['optimizer'])
                        for n,p in clone.named_parameters():
                            g = saved['mixed_grad'][n]
                            p.grad = g.to(device) if g is not None else None
                        test_opt.step(saved['pair'],[t.to(device) for t in saved['raw_pair']])
                        v3.update_ema(fclone,clone,cfg['rho'])
                        assert live_id(model,ema,opt)==live_id(clone,fclone,test_opt)
                        assert v3.state_digest(test_opt.state_dict())==saved['actual_post']['optimizer']
                        assert np.array_equal(saved['row_ids'],rows)
                        request(attempt,'release',key=receipt['key'])
                        del clone,fclone,test_opt,saved,anchor
                    else:
                        value = train_step(model,opt,ema,x,y,rows,cfg,ctx)
                    ids_hashes.append((v3.state_digest(rows),value['diagnostics']['pair_index_hash']))
                    signature = dict(**live_id(model,ema,opt),loss=value['loss'],pair=ids_hashes[-1][1])
                    if variant=='reference':
                        signatures.append(signature)
                    else:
                        assert signature==signatures[step-1], (dataset,arm,step)
                    torch.cuda.synchronize(device)
                    profile[dataset]=max(profile[dataset],residency(sample(),os.getpid()))
                    del x,y,value
                if paired is None:
                    paired = ids_hashes
                assert ids_hashes==paired
                del model,ema,opt
                gc.collect()
                torch.cuda.empty_cache()
            records.append(dict(dataset=dataset,arm=arm,first_tail=signatures,rows=[40960,len(ids)],
                                restored_anchor=receipt, paired_rows_and_draws=True))
        del bundle
    result = dict(stage=STAGE,status='PASS',source_hash=audit['source_hash'],owner_generation=audit['owner_generation'],
                  full_process_bytes=profile,cuda_usable_total_bytes=audit['cuda_usable_total_bytes'],records=records,
                  new_scientific_runs=0,test_evaluations=0)
    write(art/f'{STAGE}_CHECK.json',result)
    return result
