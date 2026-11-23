"""B source/observer equivalence and residency fixtures, outside the science queue."""
import copy
import gc
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.conditional_information_momentum_20261102_v3 import protocol as v3
from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.objectives import train_step
from .b_observer import BObserver, identity
from .b_pairs import exact_transition_table, transition_metrics
from .b_panels import Panel, compare, selection_score
from .b_gate import _choice
from .b_pulse import run_pulse
from .b_virtual import partition, registry, virtual_step
from .protocol import DATASETS, ROOT, digest, read, source, write
from .scheduler import admit, residency, sample, stopping
from .state import generation


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Linear(2,3)
        self.expert = torch.nn.Linear(3,3)
        self.shared_weight = torch.nn.Parameter(torch.tensor(1.))
        self.none = torch.nn.Parameter(torch.tensor(1.))
        self.zero = torch.nn.Parameter(torch.tensor(1.))
        self.tower = torch.nn.ModuleList(torch.nn.Linear(3,1) for _ in range(6))

    def forward(self,x):
        h = self.expert(self.embedding(x[:2].T))*self.shared_weight + self.zero*0
        z = torch.stack([net(h).flatten() for net in self.tower],dim=1)
        d = x[-1].long()
        idx = torch.stack([d,d+3],dim=1)
        return z.gather(1,idx).sigmoid()


def basic():
    v3.setup(2022)
    y = torch.tensor([[0.,1.],[1.,0.]]*6)
    d = torch.tensor([0,0,1,1,2,2]*2)
    x = torch.stack([torch.arange(12.)/4,torch.arange(12.).sin(),d.float()])
    ids = np.arange(len(y))
    bundle = SimpleNamespace(dataset='movielens',train_x=x.T.numpy(),train_y=y.numpy(),train_row_ids=ids)
    model = _Toy()
    ema = copy.deepcopy(model).requires_grad_(False)
    c = v3.config('A0','movielens',0)
    opt = v3.make_optimizer(model,c)
    ctx = dict(dataset='movielens',seed=2022,epoch=20,global_step=40)
    masks,part = partition(y[:2],d[:2],ctx)
    assert part['capacities'][2:]==[[0,0]]*4 and all(len(v)==4 for v in masks.values())
    rows = [np.flatnonzero(d.numpy()==q//2) for q in range(6)]
    pp = []
    for q,r in enumerate(rows):
        yq = y.numpy()[:,q%2]
        pp.append(np.array([(i,j) for i in r[yq[r]==1] for j in r[yq[r]==0]]))
    panel = Panel(bundle.train_x,bundle.train_y,rows,pp,rows[::2])
    loader = torch.Generator().manual_seed(13022)
    # Seed non-uniform private Adam clocks. Unused skips every step, native zero steps with WD.
    opt.zero_grad()
    model.shared_weight.grad=torch.tensor(0.)
    opt.step()
    ref,ropt,rf = copy.deepcopy((model,opt,ema))
    expected = train_step(ref,ropt,rf,x,y,ids,c,ctx)
    rstate = v3.state_digest(v3.rng_state())
    with tempfile.TemporaryDirectory() as tmp:
        obs = BObserver.__new__(BObserver)
        obs.bundle,obs.out,obs.K,obs.seed,obs.dataset = bundle,Path(tmp),2,2022,'movielens'
        obs.ram = dict(anchors=[],windows={},panels={},active=None)
        obs.scopes = {n:'EMBED' if n.startswith('embedding') else 'S_DENSE' if n=="shared_weight" or n.startswith("expert.") else "N"
                      for n,_ in model.named_parameters()}
        obs.group_hash=v3.state_digest(obs.scopes)
        obs.panels = {k:panel for k in ('P_A','P_B','V_A','V_B')}
        prev = {n:p.detach().clone() for n,p in ema.named_parameters()}
        result = obs._anchor(model,opt,ema,x,y,ids,c,ctx,loader)
        assert result['loss']==expected['loss'] and identity(ref,ropt,rf)==identity(model,opt,ema)
        assert rstate==v3.state_digest(v3.rng_state())
        raw = read(Path(tmp)/'B_ANCHOR_E20.json')
        assert raw['counts']['virtual_adam']==52 and raw['identity']['native_none']==['none']
        assert 'zero' in raw['identity']['native_tensor_zero']
        assert raw['identity']['post_steps']['shared_weight']==2 and raw['identity']['post_steps']['zero']==1
        obs._window_step(model,ema,ids,c,ctx,prev,{})
        # Force the low-level suffix check even if the toy panels select NO_ACTION.
        # Its stored first post is JOINT, so it must recover the parent's entire suffix exactly.
        win = obs.ram['windows']['20']
        win['branches']['true'] = dict(q=0,model=v3.cpu_copy(model.state_dict()),adam=v3.cpu_copy(opt.adam.state_dict()),
            ema=v3.cpu_copy(ema.state_dict()),mode=True,scores=win['parents']['1']['online'],
            ema_scores=win['parents']['1']['fixed_ema'],first_ema_residual=win['residuals'][0])
        order = torch.randperm(len(y),generator=loader).numpy()
        obs.step(model,opt,ema,x[:,order].contiguous().T.contiguous().T,y[order],ids[order],c,
                 dict(ctx,epoch=21,global_step=41),loader)
        gate = dict(dataset='movielens',B_PULSE='TRIGGERED')
        pulse = run_pulse(obs,model,c,20,'true',gate=gate)
        assert pulse['counts']['counterfactual_suffix_adam']==1
        for pop in ('P_B','V_B'):
            assert pulse['milestones']['2']['fixed_ema'][pop]['macro_auc_delta']==0
            assert pulse['milestones']['2']['online'][pop]['macro_ll_delta']==0
        assert pulse['terminal_common_radius']['radius']==0
        win['branches']['pseudo0'] = dict(q=None)
        zero = run_pulse(obs,model,c,20,'pseudo0',gate=gate)
        assert zero['counts']['counterfactual_suffix_adam']==0 and zero['counts']['logical_steps']==2
    # A fixed sampled pair worsens while the full panel population AUC improves.
    lab = np.repeat([[1.,1.],[0.,0.]],2,axis=0)
    ix = np.arange(4)
    mini = Panel(np.empty((4,0)),lab,[ix]*6,[np.array([[0,2]])]*6,[ix]*3)
    before = {'P_A': {'p':np.full((4,2),.5)}}
    changed = {'P_A': {'p':np.repeat(np.array([[.1],[.9],[.2],[.05]]),2,axis=1)}}
    outcome = compare({'P_A':mini},before,changed)
    assert selection_score(outcome)==.25 and outcome['P_A']['macro_pair_delta']==-.5
    assert _choice([0.,selection_score(outcome),0.,0.,0.,0.])==1
    for ds in DATASETS:
        manifest = read(ROOT/f'artifacts/conditional_history_shared_update_20261122/{ds}_data_identity.json')
        bundle = SimpleNamespace(dataset=ds,features=SimpleNamespace(**manifest['features']))
        model = v3.make_model(bundle,v3.load_protocol()['datasets'][ds]['source'])
        registry(model,bundle)
    t = exact_transition_table([1,1,0,0],[0,1,0,1],[1,1,0,1])
    assert abs(transition_metrics(t)['closure_error'])<1e-14
    # A permanent displacement and one replaced snapshot have different F responses.
    rho=.8
    for permanent in (True,False):
        f=0.
        for k in range(1,5):
            f = rho*f+(1-rho)*(1. if permanent or k==1 else 0.)
            target=1-rho**k if permanent else (1-rho)*rho**(k-1)
            assert abs(f-target)<1e-14
    return dict(virtual_joint_exact=True,synthetic_sweeps=18,synthetic_virtual_steps=52,
                equal_suffix=True,no_action_suffix=True,none_zero_empty=True,groups_both_datasets=True,
                population_selection_separate_from_sampled_pairs=True)


def check(art,stage,gpu):
    if stage!='B_CORE':
        raise ValueError(stage)
    from .runner import batch_inputs
    art = Path(art)
    audit = read(art/f'{stage}_AUDIT.json')
    assert audit['source_hash']==source(stage)['source_hash'] and audit['status']=='SOURCE_READY'
    owner = generation(art/'attempt1')
    assert owner==audit['owner_generation']
    fixtures = basic()
    protocol = v3.load_protocol()
    physical = read(ROOT.parent/'phase0-3_tvg_rm_20261116/artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/PHYSICAL_CAPACITY_MEASUREMENTS.json')
    device = torch.device(f'cuda:{gpu}')
    profiles,rows = {},[]
    for ds in DATASETS:
        if stopping(art,stage):
            raise RuntimeError('Stage stopped during B fixture')
        prior = physical['profiles'][ds]['observed_full_worker_nvml_bytes']
        telem = sample()
        card = next(r for r in telem['devices'] if r['index']==gpu)
        usable = audit['cuda_usable_total_bytes']
        if not admit(card,telem['host_ram_used'],usable[str(gpu)],prior):
            raise RuntimeError('B engineering fixture needs current capacity')
        bundle,data = v3.load_dataset(ds,p=protocol)
        assert digest(data)==audit['datasets'][ds]
        c = v3.config('A0',ds,0)
        K = 20 if ds=='movielens' else 49
        reference=[]
        probes = None
        peak = 0
        with tempfile.TemporaryDirectory() as tmp:
            for variant in ('source','observer'):
                v3.setup(2022,device)
                model = v3.make_model(bundle,protocol['datasets'][ds]['source']).to(device)
                ema = copy.deepcopy(model).requires_grad_(False)
                opt = v3.make_optimizer(model,c)
                loader = torch.Generator().manual_seed(13022)
                order = torch.randperm(len(bundle.train_y),generator=loader).numpy()
                first,tail = order[:40960],order[(K-1)*40960:]
                obs = BObserver(bundle,ds,2022,tmp) if variant=='observer' else None
                for idx,indices in enumerate((first,tail)):
                    x,y,ids = batch_inputs(bundle,indices,device)
                    # Physical fixture uses real source population and a two-step optimizer
                    # prestate; synthetic epoch/step labels only select the production hook.
                    ctx = dict(study='EMA02V2',dataset=ds,split_id='original',seed=2022,
                        epoch=20,global_step=20*K-1+idx,batch=K-1+idx,last_batch=bool(idx))
                    current = (obs.step(model,opt,ema,x,y,ids,c,ctx,loader) if obs else
                               train_step(model,opt,ema,x,y,ids,c,ctx))
                    record = dict(**identity(model,opt,ema),loss=current['loss'],row_hash=v3.state_digest(ids),
                                  rng=v3.state_digest(v3.rng_state(device)),loader=v3.state_digest(loader.get_state()))
                    if obs:
                        assert record==reference[idx],(ds,idx)
                    else:
                        reference.append(record)
                    torch.cuda.synchronize(device)
                    peak = max(peak,residency(sample(),os.getpid()))
                    del x,y,current
                if obs:
                    probes=read(Path(tmp)/'B_ANCHOR_E20.json')
                    assert probes['counts']['virtual_adam']==52
                del model,ema,opt,obs
                gc.collect()
                torch.cuda.empty_cache()
        profiles[ds]=peak
        rows.append(dict(dataset=ds,real_batch_rows=[len(first),len(tail)],first_tail_reference=reference,
            full_process_bytes=peak,observer_identity=probes['identity'],partitions=probes['partitions'],
            counts=probes['counts'],source_noninterference=True,role='S0_ENGINEERING_FIXTURE'))
        del bundle,probes
    result = dict(status='PASS',stage=stage,source_hash=audit['source_hash'],owner_generation=owner,
        full_process_bytes=profiles,cuda_usable_total_bytes=audit['cuda_usable_total_bytes'],records=rows,
        fixtures=fixtures,test_evaluations=0,new_scientific_runs=0)
    write(art/f'{stage}_CHECK.json',result)
    return result
