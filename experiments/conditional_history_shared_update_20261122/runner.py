"""Single mechanical trajectory. The scheduler owns dispatch; model state stays in RAM."""
import copy
import os
import resource
import time
import traceback

import torch
from experiments.conditional_information_momentum_20261102_v3.objectives import train_step
from .protocol import append, digest, read, sha, v3, write
from .state import capture, commit, generation, request, restore


def batch_inputs(bundle, ids, device):
    return (torch.as_tensor(bundle.train_x[ids],device=device).T,
            torch.as_tensor(bundle.train_y[ids],device=device),bundle.train_row_ids[ids])


def apply_step(observer, model, optimizer, ema, x, y, rows, cfg, context, loader):
    if observer:
        return observer.step(model,optimizer,ema,x,y,rows,cfg,context,loader)
    return train_step(model,optimizer,ema,x,y,rows,cfg,context)


def select(epoch, kind, estimator, metrics, selected, epochs, values):
    score = metrics['macro_auc']
    if score is None:
        raise RuntimeError('Undefined full Validation Macro AUC')
    if kind not in values or score > values[kind]['macro_auc']:
        selected[kind] = v3.cpu_copy(estimator.state_dict())
        epochs[kind] = epoch
        values[kind] = metrics


def add_counts(total, extra):
    for name, value in extra.items():
        total[name] = total.get(name,0) + value


def endpoint(job, attempt, kind, checkpoint, epoch, model_hash, scores):
    return dict(dataset=job['dataset'], arm=job['arm'], seed=job['seed'], split='validation',
                readout=kind, checkpoint_rule=checkpoint, epoch=epoch, model_hash=model_hash,
                source_ref=str(attempt/'runs'/job['key']/'result.json'), source_hash=job['source_hash'],
                config_hash=digest(job['config']), trajectory_id=job['key'],
                reuse_status='NEW_TRAIN' if job['stage']=='A_CORE' else 'STATE_RECONSTRUCTION', **scores)


def observer_store(attempt,job,state):
    """The owner's normal completion compacts the parent. Retain deferred pulses separately."""
    key = 'B-branches-'+job['key']
    existing = request(attempt,'status')['jobs'].get(key)
    if not existing or existing['epoch'] == 0:
        if not existing:
            request(attempt,'register',key=key,job=job,pid=os.getpid())
        request(attempt,'commit',key=key,state=dict(epoch=1,counts={},observer=state),
                epoch_file='RAM_ONLY_B_OBSERVER')
    saved = request(attempt,'get',key=key)['state']
    if v3.state_digest(saved['observer']) != v3.state_digest(state):
        raise RuntimeError('B deferred branch state readback failed')
    request(attempt,'paused',key=key,reason='RETAINED_FOR_B_GATE')
    return key


def worker(attempt, job, device, manifest, stop):
    import signal
    from .scheduler import sample, residency
    key, cfg, dataset = job['key'], job['config'], job['dataset']
    owner = generation(attempt)
    reg = request(attempt,'register',key=key,job=job,pid=os.getpid())
    out = attempt/'runs'/key/f"{owner[:10]}-attempt{reg['attempt']}"
    out.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    device = torch.device(device)
    completed, steps, prior_seconds = 0, 0, 0.0
    physical_steps = physical_epochs = 0
    interrupted = False
    def on_stop(_signal, _frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM,on_stop)
    signal.signal(signal.SIGINT,on_stop)
    def stopping():
        return interrupted or stop()
    try:
        protocol = v3.load_protocol()
        bundle, identity = v3.load_dataset(dataset,p=protocol)
        data_hash = digest(identity)
        assert data_hash == manifest['datasets'][dataset]
        v3.setup(job['seed'],device)
        model = v3.make_model(bundle,protocol['datasets'][dataset]['source']).to(device)
        ema = copy.deepcopy(model).requires_grad_(False)
        optimizer = v3.make_optimizer(model,cfg)
        group_hash = digest(optimizer.parameter_manifest)
        loader = torch.Generator().manual_seed(job['seed']+11000)
        obs = None
        if job['stage'] == 'B_CORE':
            from .b_observer import BObserver
            obs = BObserver(bundle,dataset,job['seed'],out)
        selected, epochs, values, counts, checkpoints = {}, {}, {}, {}, []
        b_outputs = {}
        state = reg['state']
        if state is not None:
            verified = request(attempt,'get',key=key)['state']
            assert v3.state_digest(verified) == v3.state_digest(state)
            del verified
            receipt_files = sorted((attempt/'runs'/key).glob(f'{owner[:10]}-attempt*/RAM_{state["epoch"]:02d}.json'))
            if receipt_files:
                assert read(receipt_files[-1])['state_hash'] == v3.state_digest(state)
            restore(state,job,model,optimizer,ema,loader,obs,data_hash,group_hash,owner,device)
            completed, steps = state['epoch'],state['global_step']
            prior_seconds = state['physical_worker_seconds']
            selected, epochs, values = (state[k] for k in ('selected_models','selected_epochs','selected_validation'))
            counts, checkpoints = state['counts'], state['checkpoint_rows']
            b_outputs = state['b_outputs']
            ack_dir = attempt/'runs'/key/state['attempt_id']
            epoch_path = ack_dir/f'epoch_{completed:02d}.json'
            if not epoch_path.exists():
                packet_hash = v3.state_digest(state)
                write(ack_dir/f'RAM_{completed:02d}.json',dict(epoch=completed,state_hash=packet_hash,
                      owner_generation=owner,status='RAM_VERIFIED'))
                write(epoch_path,dict(**state['epoch_metrics'],ram_state_hash=packet_hash))
        del reg, state
        peak_nvml = 0
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(completed+1,51):
            if stopping():
                request(attempt,'paused',key=key,reason='SUPERVISOR_STOP')
                return
            t0 = time.monotonic()
            model.train()
            order = torch.randperm(len(bundle.train_y),generator=loader).numpy()
            losses, real_losses, epoch_counts = 0., 0., {}
            for batch,offset in enumerate(range(0,len(order),cfg['batch_size']),1):
                ids = order[offset:offset+cfg['batch_size']]
                x,y,rows = batch_inputs(bundle,ids,device)
                steps += 1
                context = dict(study='EMA02V2',dataset=dataset,split_id='original',seed=job['seed'],
                               epoch=epoch,global_step=steps,batch=batch,last_batch=offset+len(ids)==len(order),
                               role='STATE_RECONSTRUCTION' if obs else 'NEW_SCIENTIFIC_TRAIN')
                value = apply_step(obs,model,optimizer,ema,x,y,rows,cfg,context,loader)
                physical_steps += 1
                add_counts(counts,value['counts'])
                add_counts(epoch_counts,value['counts'])
                losses += len(ids)*value['loss']
                real_losses += len(ids)*value['real_loss']
                append(out/'batches.jsonl',dict(epoch=epoch,batch=batch,step=steps,rows=len(ids),loss=value['loss'],
                                                real_loss=value['real_loss'],diagnostics=value['diagnostics'],counts=value['counts']))
                if physical_steps == 1 or (obs and context['last_batch'] and epoch in (5,20,35,50)):
                    torch.cuda.synchronize(device)
                    telemetry = sample()
                    peak_nvml = max(peak_nvml,residency(telemetry,os.getpid()))
                    write(out/'ALLOCATION.json',dict(epoch=epoch,batch=batch,device=str(device),
                          peak_process_gpu_bytes=peak_nvml,telemetry=telemetry))
                    request(attempt,'allocated',key=key,peak_memory_bytes=peak_nvml)
                del x,y,value
            validation = {}
            for kind, estimator in [('online',model),('fixed_ema',ema)]:
                scores,cost = v3.evaluate(estimator,bundle.val_x,bundle.val_y,device)
                select(epoch,kind,estimator,scores,selected,epochs,values)
                validation[kind] = scores
                add_counts(counts,{'validation_'+n:v for n,v in cost.items()})
                if epoch in (10,50):
                    checkpoints.append(endpoint(job,attempt,kind,f'E{epoch}',epoch,
                                        v3.state_digest(estimator.state_dict()),scores))
            physical_epochs += 1
            file = out/f'epoch_{epoch:02d}.json'
            record = dict(epoch=epoch,global_step=steps,rows=len(order),wall_seconds=time.monotonic()-t0,
                          loss=losses/len(order),real_loss=real_losses/len(order),epoch_counts=epoch_counts,
                          validation=validation,selected_epochs=epochs,counts=counts)
            if obs:
                for prefix in ('B_ANCHOR_E','B_WINDOW_E'):
                    for path in out.glob(prefix+'*.json'):
                        b_outputs[path.name] = dict(path=str(path),sha256=sha(path))
            meta = dict(job=job, data_hash=data_hash,group_hash=group_hash,owner_generation=owner,
                        b_outputs=b_outputs,epoch_metrics=record,attempt_id=out.name, transaction=epoch,epoch=epoch,global_step=steps,
                        counts=counts,selected_models=selected,selected_epochs=epochs,
                        selected_validation=values,checkpoint_rows=checkpoints,
                        physical_worker_seconds=prior_seconds+time.monotonic()-started)
            packet = capture(model,optimizer,ema,loader,order,obs,meta,device)
            ack = commit(attempt,key,packet,file)
            write(out/f'RAM_{epoch:02d}.json',ack)
            write(file,dict(**record,**{'ram_state_hash':ack['state_hash']}))
            completed = epoch
            del packet
        rows = list(checkpoints)
        selected_hashes = {kind:v3.state_digest(value) for kind,value in selected.items()}
        for kind in ('online','fixed_ema'):
            rows.append(endpoint(job,attempt,kind,'selected',epochs[kind],selected_hashes[kind],values[kind]))
        branch_key = observer_store(attempt,job,obs.state_dict()) if obs else None
        result = dict(job=job,owner_generation=owner,attempt_id=out.name,rows=rows,selected_epochs=epochs,
                      selected_hashes=selected_hashes,data_hash=data_hash,group_hash=group_hash,
                      counts=counts,epochs=completed,global_step=steps,physical_worker_seconds=prior_seconds+time.monotonic()-started,
                      cost_class='STATE_RECONSTRUCTION' if obs else 'NEW_SCIENTIFIC_TRAIN',branch_key=branch_key,b_outputs=b_outputs,
                      peak_torch_reserved=torch.cuda.max_memory_reserved(device),peak_process_gpu_bytes=peak_nvml,
                      max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,test_evaluations=0)
        request(attempt,'complete',key=key,result=result)
        retrieved = request(attempt,'get',key=key)
        assert retrieved['result'] == result
        for kind,h in selected_hashes.items():
            assert v3.state_digest(retrieved['state']['selected_models'][kind])==h
        write(attempt/'runs'/key/'COMPLETE_VERIFIED.json',dict(owner_generation=owner,result_hash=digest(result)))
    except BaseException:
        failure = dict(physical_steps=physical_steps,physical_epochs=physical_epochs,last_verified_epoch=completed,
                       worker_seconds=time.monotonic()-started,traceback=traceback.format_exc(),
                       peak_torch_reserved=torch.cuda.max_memory_reserved(device) if torch.cuda.is_initialized() else 0)
        write(out/'failure.json',failure)
        # PAUSED accepts a fresh attempt directly from the last acknowledged epoch.
        status = request(attempt,'status')['jobs'][key]
        if status['status']!='COMPLETE':
            request(attempt,'paused',key=key,reason=failure['traceback'][-1500:])
        raise
