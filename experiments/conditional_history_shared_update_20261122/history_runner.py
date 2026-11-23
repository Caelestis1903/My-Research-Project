"""A_HISTORY trajectories and RAM-only pre-step states for possible localization."""
import copy
import os
import resource
import signal
import time
import traceback

import torch

from experiments.conditional_information_momentum_20261102_v3.objectives import train_step
from .history_controls import make_history_optimizer
from .protocol import append, digest, read, v3, write
from .runner import add_counts, batch_inputs, endpoint, select
from .scheduler import sample, residency
from .state import capture, commit, generation, request, restore

ANCHORS = (5, 20, 35, 50)


def step_anchor(model, optimizer, ema, x, y, rows, cfg, context, loader, order, offset, meta):
    """Capture the live post-forward, pre-update state and exact inputs to optimizer.step."""
    packet = {}
    device = x.device
    pre_rng = v3.rng_state(device)
    pre_buffers = v3.cpu_copy(dict(model.named_buffers()))

    def capture_before_update(pair, raw_pair):
        packet.update(capture(model, optimizer, ema, loader, order, None, meta, device))
        packet.update(cursor=offset, pre_forward_rng=pre_rng, pre_forward_buffers=pre_buffers,
                      context=dict(context), row_ids=rows.copy(), batch_indices=order[offset:offset+len(rows)].copy(),
                      pair=v3.cpu_copy(pair), raw_pair=v3.cpu_copy(raw_pair),
                      mixed_grad=v3.cpu_copy({n:p.grad for n,p in model.named_parameters()}))

    optimizer.on_pre_step = capture_before_update
    try:
        value = train_step(model, optimizer, ema, x, y, rows, cfg, context)
    finally:
        optimizer.on_pre_step = None
    packet['actual_post'] = dict(model=v3.state_digest(model.state_dict()), ema=v3.state_digest(ema.state_dict()),
                                 optimizer=v3.state_digest(optimizer.state_dict()))
    packet['reconstruction_max_abs'] = optimizer.reconstruction_error()
    return value, packet


def store_anchor(attempt, job, packet):
    epoch = packet['context']['epoch']
    key = f"A-history-anchor-{job['key']}-E{epoch}"
    owner = packet['owner_generation']
    signature = v3.state_digest(packet)
    # Replaying an interrupted unacknowledged epoch must reproduce its anchor.
    existing = request(attempt, 'status')['jobs'].get(key)
    if not existing:
        request(attempt, 'register', key=key, job=dict(parent=job['key'], epoch=epoch), pid=os.getpid())
    if not existing or existing['epoch'] == 0:
        request(attempt, 'commit', key=key, state=dict(epoch=1, counts={}, anchor=packet),
                epoch_file='RAM_ONLY_HISTORY_ANCHOR')
    saved = request(attempt, 'get', key=key)['state']['anchor']
    if v3.state_digest(saved) != signature or saved['owner_generation'] != owner:
        raise RuntimeError(f'History anchor readback differs: {key}')
    request(attempt, 'paused', key=key, reason='RETAINED_FOR_A_TARGET_LOCALIZATION')
    return dict(key=key, state_hash=signature, owner_generation=owner, epoch=epoch,
                actual_post=packet['actual_post'], reconstruction_max_abs=packet['reconstruction_max_abs'])


def science_endpoint(job, attempt, kind, checkpoint, epoch, model_hash, scores):
    return dict(endpoint(job, attempt, kind, checkpoint, epoch, model_hash, scores), reuse_status='NEW_TRAIN')


def worker(attempt, job, device, manifest, stop):
    key, cfg, dataset = job['key'], job['config'], job['dataset']
    owner = generation(attempt)
    reg = request(attempt, 'register', key=key, job=job, pid=os.getpid())
    out = attempt/'runs'/key/f"{owner[:10]}-attempt{reg['attempt']}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = torch.device(device)
    completed = steps = physical_steps = physical_epochs = peak_nvml = 0
    prior_seconds = 0.
    interrupted = False
    def on_stop(_signal, _frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM,on_stop)
    signal.signal(signal.SIGINT,on_stop)
    try:
        protocol = v3.load_protocol()
        bundle, identity = v3.load_dataset(dataset, p=protocol)
        data_hash = digest(identity)
        assert data_hash == manifest['datasets'][dataset]
        v3.setup(job['seed'],device)
        model = v3.make_model(bundle,protocol['datasets'][dataset]['source']).to(device)
        ema = copy.deepcopy(model).requires_grad_(False)
        optimizer = make_history_optimizer(model,cfg)
        group_hash = digest(optimizer.parameter_manifest)
        loader = torch.Generator().manual_seed(job['seed']+11000)
        selected, epochs, values, counts, checkpoints, anchors = {}, {}, {}, {}, [], {}
        state = reg['state']
        if state is not None:
            verified = request(attempt,'get',key=key)['state']
            signature = v3.state_digest(verified)
            assert signature == v3.state_digest(state)
            del verified
            restore(state,job,model,optimizer,ema,loader,None,data_hash,group_hash,owner,device)
            completed, steps = state['epoch'], state['global_step']
            prior_seconds = state['physical_worker_seconds']
            selected, epochs, values = (state[k] for k in ('selected_models','selected_epochs','selected_validation'))
            counts, checkpoints, anchors = state['counts'], state['checkpoint_rows'], state['anchors']
            ack_dir = attempt/'runs'/key/state['attempt_id']
            ack_path = ack_dir/f'RAM_{completed:02d}.json'
            if ack_path.exists():
                assert read(ack_path)['state_hash'] == signature
            else:
                write(ack_path, dict(epoch=completed,state_hash=signature,owner_generation=owner,status='RAM_VERIFIED'))
            if not (ack_dir/f'epoch_{completed:02d}.json').exists():
                write(ack_dir/f'epoch_{completed:02d}.json',dict(**state['epoch_metrics'],ram_state_hash=signature))
        del reg,state
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(completed+1,51):
            if interrupted or stop():
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
                context = dict(study='EMA02V2',dataset=dataset,split_id='original',seed=job['seed'],epoch=epoch,
                               global_step=steps,batch=batch,last_batch=offset+len(ids)==len(order),role='NEW_SCIENTIFIC_TRAIN')
                at_anchor = context['last_batch'] and epoch in ANCHORS
                if at_anchor:
                    meta = dict(job=job, owner_generation=owner, data_hash=data_hash, group_hash=group_hash,
                                selected_epochs=dict(epochs), selected_validation=v3.cpu_copy(values),
                                selected_models=v3.cpu_copy(selected), global_step_before=steps-1)
                    value, packet = step_anchor(model,optimizer,ema,x,y,rows,cfg,context,loader,order,offset,meta)
                    anchors[str(epoch)] = store_anchor(attempt,job,packet)
                    del packet,meta
                else:
                    value = train_step(model,optimizer,ema,x,y,rows,cfg,context)
                physical_steps += 1
                value['counts']['diagnostic_shared_autograd_calls'] = (
                    value['counts']['shared_autograd_calls'] if cfg['arm']=='A22' else 0)
                add_counts(counts,value['counts'])
                add_counts(epoch_counts,value['counts'])
                losses += len(ids)*value['loss']
                real_losses += len(ids)*value['real_loss']
                append(out/'batches.jsonl',dict(epoch=epoch,batch=batch,step=steps,rows=len(ids),
                       row_id_hash=v3.state_digest(rows),loss=value['loss'],real_loss=value['real_loss'],
                       diagnostics=value['diagnostics'],counts=value['counts']))
                if physical_steps == 1 or at_anchor:
                    torch.cuda.synchronize(device)
                    telemetry = sample()
                    peak_nvml = max(peak_nvml,residency(telemetry,os.getpid()))
                    write(out/'ALLOCATION.json',dict(epoch=epoch,batch=batch,device=str(device),
                          peak_process_gpu_bytes=peak_nvml,telemetry=telemetry))
                    request(attempt,'allocated',key=key,peak_memory_bytes=peak_nvml)
                del x,y,value
            validation = {}
            for kind,estimator in [('online',model),('fixed_ema',ema)]:
                scores,cost = v3.evaluate(estimator,bundle.val_x,bundle.val_y,device)
                select(epoch,kind,estimator,scores,selected,epochs,values)
                validation[kind] = scores
                add_counts(counts,{'validation_'+n:v for n,v in cost.items()})
                if epoch in (10,50):
                    checkpoints.append(science_endpoint(job,attempt,kind,f'E{epoch}',epoch,
                                       v3.state_digest(estimator.state_dict()),scores))
            physical_epochs += 1
            file = out/f'epoch_{epoch:02d}.json'
            record = dict(epoch=epoch,global_step=steps,rows=len(order),wall_seconds=time.monotonic()-t0,
                          loss=losses/len(order),real_loss=real_losses/len(order),epoch_counts=epoch_counts,
                          validation=validation,selected_epochs=epochs,counts=counts)
            meta = dict(job=job,data_hash=data_hash,group_hash=group_hash,owner_generation=owner,anchors=anchors,
                        epoch_metrics=record,attempt_id=out.name,transaction=epoch,epoch=epoch,global_step=steps,
                        counts=counts,selected_models=selected,selected_epochs=epochs,selected_validation=values,
                        checkpoint_rows=checkpoints,physical_worker_seconds=prior_seconds+time.monotonic()-started)
            packet = capture(model,optimizer,ema,loader,order,None,meta,device)
            ack = commit(attempt,key,packet,file)
            write(out/f'RAM_{epoch:02d}.json',ack)
            write(file,dict(**record,ram_state_hash=ack['state_hash']))
            completed = epoch
            del packet
        assert set(anchors)=={str(e) for e in ANCHORS}
        selected_hashes = {kind:v3.state_digest(value) for kind,value in selected.items()}
        rows = list(checkpoints)
        for kind in ('online','fixed_ema'):
            rows.append(science_endpoint(job,attempt,kind,'selected',epochs[kind],selected_hashes[kind],values[kind]))
        result = dict(job=job,owner_generation=owner,attempt_id=out.name,rows=rows,selected_epochs=epochs,
                      selected_hashes=selected_hashes,data_hash=data_hash,group_hash=group_hash,
                      counts=counts,epochs=completed,global_step=steps,physical_worker_seconds=prior_seconds+time.monotonic()-started,
                      cost_class='NEW_SCIENTIFIC_TRAIN',anchors=anchors,peak_torch_reserved=torch.cuda.max_memory_reserved(device),
                      peak_process_gpu_bytes=peak_nvml,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      test_evaluations=0)
        request(attempt,'complete',key=key,result=result)
        retrieved = request(attempt,'get',key=key)
        assert retrieved['result']==result
        for kind,h in selected_hashes.items():
            assert v3.state_digest(retrieved['state']['selected_models'][kind])==h
        write(attempt/'runs'/key/'COMPLETE_VERIFIED.json',dict(owner_generation=owner,result_hash=digest(result)))
    except BaseException:
        failure = dict(physical_steps=physical_steps,physical_epochs=physical_epochs,last_verified_epoch=completed,
                       worker_seconds=time.monotonic()-started,traceback=traceback.format_exc(),
                       peak_torch_reserved=torch.cuda.max_memory_reserved(device) if torch.cuda.is_initialized() else 0)
        write(out/'failure.json',failure)
        status = request(attempt,'status')['jobs'][key]
        if status['status']!='COMPLETE':
            request(attempt,'paused',key=key,reason=failure['traceback'][-1500:])
        raise
