"""Full epoch-boundary states in the existing RAM store, verified by readback."""
from experiments.conditional_information_momentum_20261102_v3.guardian import request
from .protocol import digest, read, v3


def generation(attempt):
    info = read(attempt/'guardian.json')
    status = request(attempt,'status')
    if info['pid'] != status['pid']:
        raise RuntimeError('RAM owner identity changed')
    return digest([info['address'],info['pid']])


def capture(model, optimizer, ema, loader, order, observer, meta, device):
    return dict(**meta, model=v3.cpu_copy(model.state_dict()), optimizer=v3.cpu_copy(optimizer.state_dict()),
                ema=v3.cpu_copy(ema.state_dict()), model_mode=model.training, ema_mode=ema.training,
                loader=loader.get_state().clone(), permutation=order.copy(), cursor=len(order),
                rng=v3.rng_state(device), observer=v3.cpu_copy(observer.state_dict()) if observer else None)


def restore(state, job, model, optimizer, ema, loader, observer, data_hash, group_hash, owner, device):
    assert state['job'] == job and state['owner_generation'] == owner
    assert (state['data_hash'],state['group_hash']) == (data_hash,group_hash)
    assert state['cursor'] == len(state['permutation'])
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    ema.load_state_dict(state['ema'])
    model.train(state['model_mode'])
    ema.train(state['ema_mode'])
    loader.set_state(state['loader'])
    if observer:
        observer.load_state_dict(state['observer'])
    v3.restore_rng(state['rng'],device)


def commit(attempt,key,packet,file):
    expected = v3.state_digest(packet)
    ack = request(attempt,'commit',key=key,state=packet,epoch_file=str(file))
    actual = request(attempt,'get',key=key)['state']
    if ack['epoch'] != packet['epoch'] or v3.state_digest(actual) != expected:
        raise RuntimeError('RAM readback differs from committed packet')
    return dict(epoch=ack['epoch'], state_hash=expected, owner_generation=packet['owner_generation'],
                status='RAM_VERIFIED')
