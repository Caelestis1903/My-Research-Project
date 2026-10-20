"""Same-batch original-producer oracle for every required S5/S6 control arm."""
import copy
from pathlib import Path
import tempfile
import numpy as np
import torch
from .bootstrap import ROOT,dump,source_hashes
from .manifest import configurations
from .controlled import Runner, S5MLRunner, S6MLRunner, RecordedLoader

def main():
    root=ROOT/'basicchecks';root.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='controlled_same_batch_',dir=root));reports=[]
    for stage,arms in [('AC-S5',('F','T','HT','DH')),('AC-S6',('F','DH'))]:
        for arm in arms:
            cfg=next(c.copy() for c in configurations() if c['stage']==stage and c['arm']==arm and c['budget']=='full50')
            cfg['run_dir']=str(directory/(stage+'_'+arm));r=Runner(cfg,'cpu');b=r.bundle
            rows=np.concatenate([np.flatnonzero(b.train_x[:,-1]==d)[:7] for d in range(3)])
            x=torch.tensor(b.train_x[rows]);y=torch.tensor(b.train_y[rows]);r.train_loader.last_indices=rows
            fp=r._record_batch(0,0,'train',x,y)
            losses=r._objective_components(r.model(x.T),y,x[:,-1].long())
            r._controlled_step(losses,fp,x,0,0,'main');r.optimizer_steps+=1
            # Nonzero persistent m/v baseline. Compare SAME batch and SAME
            # pre-step state to the actual unmodified archived producer method.
            reference=copy.deepcopy(r)
            fp=r._record_batch(0,1,'train',x,y);fr=reference._record_batch(0,1,'train',x,y)
            losses=r._objective_components(r.model(x.T),y,x[:,-1].long())
            lr=reference._objective_components(reference.model(x.T),y,x[:,-1].long())
            r.optimizer.zero_grad(set_to_none=True);r.arch_optimizer.zero_grad(set_to_none=True)
            reference.optimizer.zero_grad(set_to_none=True);reference.arch_optimizer.zero_grad(set_to_none=True)
            original=S5MLRunner if stage=='AC-S5' else S6MLRunner
            original._controlled_step(reference,lr,fr,x,0,1,'main')
            r._controlled_step(losses,fp,x,0,1,'main')
            parameter_error=max(float((r.named[n].detach()-reference.named[n].detach()).abs().max()) for n in r.named)
            moment_error=0.;counter_match=True
            for n in r.main_names:
                a=r.optimizer.state.get(r.named[n],{});z=reference.optimizer.state.get(reference.named[n],{})
                assert set(a)==set(z)
                for key in a:
                    if torch.is_tensor(a[key]):moment_error=max(moment_error,float((a[key]-z[key]).abs().max()))
                    else:counter_match &= a[key]==z[key]
            assert parameter_error==0. and moment_error==0. and counter_match
            # Architecture controlled step gets its own non-overlapping Adam.
            r.train_loader.last_indices=rows;reference.train_loader.last_indices=rows
            fp=r._record_batch(0,-1,'architecture',x,y);fr=reference._record_batch(0,-1,'architecture',x,y)
            losses=r._objective_components(r.model(x.T),y,x[:,-1].long());lr=reference._objective_components(reference.model(x.T),y,x[:,-1].long())
            r.model.zero_grad(set_to_none=True);reference.model.zero_grad(set_to_none=True)
            original._controlled_step(reference,lr,fr,x,0,-1,'architecture');r._controlled_step(losses,fp,x,0,-1,'architecture')
            architecture_error=max(float((r.named[n].detach()-reference.named[n].detach()).abs().max()) for n in r.arch_names)
            assert architecture_error==0.
            reports.append(dict(stage=stage,arm=arm,history=r.history_source,current=r.current_source,
                same_batch_rows=len(rows),nonzero_baseline_parameter_error=parameter_error,persistent_m_v_step_error=moment_error,
                architecture_step_parameter_error=architecture_error,
                original_controlled_step_identity=original.__module__+'.'+original.__name__+'._controlled_step',
                partition=r.partition,main_lr=r.optimizer.param_groups[0]['lr'],arch_lr=r.arch_optimizer.param_groups[0]['lr']))
    # Stateful arch iterator must reproduce original DataLoader's next batch
    # after round-trip without advancing ambient RNG.
    loader=r.train_loader;iterator=iter(loader)
    for _ in range(2):next(iterator)
    state=iterator.state();ambient=torch.get_rng_state().clone();restored=loader.restore_iterator(state)
    assert torch.equal(ambient,torch.get_rng_state())
    one=next(iterator);two=next(restored)
    assert all(torch.equal(a,b) for a,b in zip(one,two))
    loader.limit=187500;sizes=[];all_rows=[]
    for x,y in loader:sizes.append(len(x));all_rows.append(loader.last_indices.copy())
    loader.limit=None
    assert sizes==[40960,40960,40960,40960,23660]
    assert len(np.unique(np.concatenate(all_rows)))==187500
    initial=torch.load(r.dir/'last_resume.pt',map_location='cpu',weights_only=False)
    recovered=Runner(cfg,'cpu')
    assert recovered.optimizer_steps==0 and recovered.architecture_optimizer_steps==0
    assert all(torch.equal(recovered.model.state_dict()[n],value) for n,value in initial['model'].items())
    dump(root/'controlled_same_batch.json',dict(status='PASS',formal_training_epochs_run=0,checks=reports,
         architecture_iterator_state_roundtrip='PASS',matched_budget_batch_sizes=sizes,
         initial_full_state_resume='PASS',source_hashes=source_hashes(('controlled.py',))))
    print(str(root/'controlled_same_batch.json'))

if __name__=='__main__':main()
