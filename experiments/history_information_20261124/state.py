"""Verified live packet custody within the foreground supervisor."""
import copy
import hashlib
import io
import time
import uuid

import torch

from experiments.conditional_information_momentum_20261102_v3.vendor.ema02.protocol import (
    cpu_copy, rng_state, restore_rng, state_digest,
)

EPOCH = 'StatePacketV1'
ANCHOR = 'AnchorPacketV1'
SELECTED = 'SelectedPacketV1'


def encode(packet):
    stream = io.BytesIO()
    torch.save(packet, stream)
    return stream.getvalue()


def decode(payload):
    return torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)


def validate_packet(packet, identity):
    if packet['identity'] != identity:
        raise ValueError('RAM packet identity differs from the active owner and job')
    kind = packet['schema']
    if kind == SELECTED:
        if state_digest(packet['model_state']) != identity['model_hash']:
            raise ValueError('Selected model hash differs from its packet identity')
        return
    cursor, order = packet['cursor'], packet['permutation']
    if kind == EPOCH:
        if packet['phase'] != 'EPOCH_BOUNDARY' or cursor != len(order):
            raise ValueError('Epoch packet requires the end-of-permutation cursor')
    elif kind == ANCHOR:
        length = len(packet['row_ids'])
        if (packet['phase'] != 'POST_BACKWARD_PRE_STEP' or cursor < 0 or length <= 0
                or cursor + length != len(order)
                or not (order[cursor:cursor+length] == packet['batch_indices']).all()
                or set(packet['actual_post']) != {'model', 'optimizer', 'ema'}):
            raise ValueError('Anchor packet requires the final batch start and three live hashes')
    else:
        raise ValueError(f'Unknown RAM packet schema {kind}')


class RamStore:
    def __init__(self):
        self.generation = uuid.uuid4().hex
        self._records = {}

    def commit(self, key, packet, identity):
        self._validate(packet, identity)
        payload = encode(packet)
        put_hash = hashlib.sha256(payload).hexdigest()
        old = self._records.get(key)
        self._records[key] = (payload, put_hash, copy.deepcopy(identity))
        try:
            actual, get_hash = self.get(key, identity)
            if get_hash != put_hash or state_digest(actual) != state_digest(packet):
                raise RuntimeError('Actual RAM readback differs from committed values')
        except BaseException:
            if old is None:
                self.release(key)
            else:
                self._records[key] = old
            raise
        return dict(run_id=identity['job_key'], state_kind=packet['schema'], guardian_key=key,
                    schema_version=1, epoch=packet['epoch'], step=packet['global_step'],
                    put_hash=put_hash, get_hash=get_hash, bytes=len(payload), owner=self.generation,
                    available=True, verified_at=time.time())

    def _validate(self, packet, identity):
        if identity['owner_generation'] != self.generation:
            raise ValueError('RAM owner generation differs')
        validate_packet(packet, identity)

    def get(self, key, identity=None):
        payload, expected, stored_identity = self._records[key]
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected or (identity is not None and stored_identity != identity):
            raise RuntimeError('RAM payload or expected identity differs from its receipt')
        packet = decode(payload)
        self._validate(packet, stored_identity)
        return packet, actual

    def release(self, key):
        self._records.pop(key, None)

    def promote(self, pending, key):
        self._records[key] = self._records.pop(pending)

    def close(self):
        self._records.clear()


def capture(schema, identity, model, optimizer, ema, loader, observer, device,
            epoch, step, order, selected, *, cursor, model_meta=None, **progress):
    return dict(schema=schema, identity=copy.deepcopy(identity), epoch=epoch, global_step=step,
                phase='EPOCH_BOUNDARY' if schema == EPOCH else 'POST_BACKWARD_PRE_STEP',
                permutation=order.copy(), cursor=cursor, selected=cpu_copy(selected),
                online=cpu_copy(model.state_dict()), model_mode=model.training,
                optimizer=cpu_copy(optimizer.state_dict()), ema=cpu_copy(ema.state_dict()) if ema is not None else None,
                ema_mode=ema.training if ema is not None else None, model_meta=cpu_copy(model_meta), rng=rng_state(device),
                loader=loader.get_state().clone(),
                observer=cpu_copy(observer.state_dict()) if observer else None, **progress)


def restore(packet, identity, model, optimizer, ema, loader, observer, device, restore_model_meta):
    validate_packet(packet, identity)
    if packet['schema'] != EPOCH:
        raise ValueError('Training continuation requires an epoch packet')
    restore_model_meta(model, copy.deepcopy(packet['model_meta']))
    model.load_state_dict(packet['online'])
    optimizer.load_state_dict(copy.deepcopy(packet['optimizer']))
    model.train(packet['model_mode'])
    if ema is not None:
        restore_model_meta(ema, copy.deepcopy(packet['model_meta']))
        ema.load_state_dict(packet['ema'])
        ema.train(packet['ema_mode'])
    loader.set_state(packet['loader'])
    if observer:
        observer.load_state_dict(copy.deepcopy(packet['observer']))
    restore_rng(packet['rng'], device)


def live_hashes(model, optimizer, ema):
    return dict(model=state_digest(model.state_dict()), optimizer=state_digest(optimizer.state_dict()),
                ema=state_digest(ema.state_dict()))
