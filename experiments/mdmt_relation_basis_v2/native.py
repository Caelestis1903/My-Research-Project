"""Reviewed fixed1 prefix, with selection snapshots only in RAM."""
from pathlib import Path
import time
import numpy as np
import torch
from .config import rng
from .data import tensor
from .losses import masked_bce
from .metrics import evaluate
from .state import capture, freeze, state_hash, array_hash, write_json, append_jsonl
from .vendor_snapshot.mdmt_dfd_20261026.native_adapter import make_native


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


@torch.no_grad()
def encode(model, x, device, chunk=4096):
    hs, zs = [], []
    model.eval()
    for i in range(0, len(x), chunk):
        packet = model.encode(tensor(x[i:i+chunk], device))
        hs.append(packet.h.cpu().numpy())
        zs.append(packet.logits.cpu().numpy())
    return np.concatenate(hs), np.concatenate(zs)


def build_native(data, seed, cfg, root, device, identity):
    root = Path(root)
    model, init = make_native(data.schema, seed, data.dataset, device)
    write_json(root/'native_initialization.json', init)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=1e-5)
    best, selected, global_step = float('inf'), None, 0
    ids = data.roles['T0']
    synchronize(device)
    started = time.monotonic()
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, 25):
        ep_start = time.monotonic()
        model.train()
        order = rng(seed, data.dataset, 20, epoch).permutation(ids)
        loss_sum = 0.
        for begin in range(0, len(order), 40960):
            rows = order[begin:begin+40960]
            y, mask = data.labels('T0', 'native', rows, epoch=epoch, step=global_step)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tensor(data.x[rows], device))
            loss = masked_bce(logits, tensor(y, device), tensor(mask, device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite native loss')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())*len(rows)
            global_step += 1
        model.eval()
        yv, maskv = data.labels('V', 'select', component='native', epoch=epoch)
        _, zv = encode(model, data.inputs('V'), device)
        metrics = evaluate(yv, zv, data.domain[data.roles['V']], maskv)
        synchronize(device)
        record = dict(component='native', epoch=epoch, train_loss=loss_sum/len(ids),
                      validation=metrics, seconds=time.monotonic()-ep_start,
                      row_plan_hash=array_hash(order), global_step=global_step)
        append_jsonl(root/'native_epoch_metrics.jsonl', record)
        if epoch >= 12 and metrics['macro_logloss'] < best-1e-12:
            best = metrics['macro_logloss']
            selected = capture(model, optimizer, identity, epoch=epoch, global_step=global_step,
                               next_epoch=epoch+1, row_cursor=0, row_plan_hash=array_hash(order))
        print(f'NATIVE {data.dataset} seed={seed} epoch={epoch}/24 V_LL={metrics["macro_logloss"]:.9f} seconds={record["seconds"]:.2f}', flush=True)
    final = capture(model, optimizer, identity, epoch=24, global_step=global_step, next_epoch=25,
                    row_cursor=0, row_plan_hash=array_hash(order))
    model.load_state_dict(selected['model'])
    freeze(model)
    synchronize(device)
    receipt = dict(selected_epoch=selected['position']['epoch'], selected_model_hash=state_hash(selected['model']),
                   final_model_hash=state_hash(final['model']), selected_macro_logloss=best,
                   seconds=time.monotonic()-started,
                   parameter_count=sum(p.numel() for p in model.parameters()),
                   peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0,
                   model_storage='RAM_ONLY')
    write_json(root/'native_receipt.json', receipt)
    return model, selected, final, receipt


@torch.no_grad()
def cache_role(native, data, role, root, device):
    ids = data.roles[role]
    root = Path(root)/'row_data'/role
    root.mkdir(parents=True, exist_ok=True)
    paths = [root/'h.npy', root/'z0.npy']
    hh = np.lib.format.open_memmap(paths[0], mode='w+', dtype=np.float32, shape=(len(ids), 256))
    zz = np.lib.format.open_memmap(paths[1], mode='w+', dtype=np.float32, shape=(len(ids), 2))
    native.eval()
    for begin in range(0, len(ids), 4096):
        rows = ids[begin:begin+4096]
        packet = native.encode(tensor(data.x[rows], device))
        hh[begin:begin+len(rows)] = packet.h.cpu().numpy()
        zz[begin:begin+len(rows)] = packet.logits.cpu().numpy()
    hh.flush()
    zz.flush()
    del hh, zz
    return dict(h=np.load(paths[0], mmap_mode='r'), z=np.load(paths[1], mmap_mode='r'),
                ids=ids, domain=data.domain[ids])

