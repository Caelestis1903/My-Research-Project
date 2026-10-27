"""Equal-budget adaptation, V-only selection and resident RAM states."""
from pathlib import Path
import csv
import time
import numpy as np
import torch
from .config import rng
from .data import tensor
from .losses import masked_bce
from .metrics import evaluate
from .model import Model
from .native import synchronize
from .state import capture, state_hash, array_hash, write_json, append_jsonl, freeze


@torch.no_grad()
def predict(model, view, device, **kwargs):
    model.eval()
    result = []
    for begin in range(0, len(view['ids']), 4096):
        sl = slice(begin, begin+4096)
        phi = tensor(view['phi'][sl], device) if model.reads_source else None
        result.append(model(tensor(view['h'][sl], device), tensor(view['z'][sl], device),
                            tensor(view['domain'][sl], device), phi, **kwargs).cpu().numpy())
    return np.concatenate(result)


def flatten_metrics(metrics):
    out = {key:metrics[key] for key in ('macro_logloss', 'macro_auc', 'natural_logloss', 'valid_requests')}
    for task, auc in enumerate(metrics['pooled_auc_by_task']):
        out[f'pooled_auc_task{task}'] = auc
    for cell in metrics['cells']:
        for key in ('n', 'positive', 'negative', 'auc', 'logloss'):
            out[f'd{cell["domain"]}_t{cell["task"]}_{key}'] = cell[key]
    return out


def append_csv(path, record):
    path = Path(path)
    exists = path.exists()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def export_prediction(root, role, state, view, labels, mask, logits, epoch, model_hash):
    root = Path(root)
    metrics = evaluate(labels, logits, view['domain'], mask)
    identity = dict(role=role, state=state, epoch=epoch, model_hash=model_hash,
                    row_id_hash=array_hash(view['ids']), logit_hash=array_hash(logits))
    write_json(root/f'{role}_{state}_metrics.json', dict(identity=identity, metrics=metrics))
    np.savez_compressed(root/f'{role}_{state}_predictions.npz', row_id=view['ids'],
                        domain=view['domain'], labels=labels, mask=mask, logits=logits)
    append_csv(root/'evaluation_metrics.csv', dict(**identity, **flatten_metrics(metrics)))
    return metrics


def train_method(method, data, views, seed, cfg, artifact_root, device, prefix_identity):
    root = Path(artifact_root)/'runs'/data.dataset/method/f'seed{seed}'
    root.mkdir(parents=True, exist_ok=False)
    identity = dict(prefix_identity, method=method)
    write_json(root/'config.json', dict(effective=cfg, identity=identity,
                                      loss='sum_valid_logit_bce / valid_label_count'))
    write_json(root/'data_manifest.json', data.manifest)
    model = Model(method, seed, data.dataset).to(device)
    counts = model.parameter_counts()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    view = views['F']
    best, selected, global_step = float('inf'), None, 0
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.monotonic()
    for epoch in range(1, 21):
        ep_start = time.monotonic()
        model.train()
        model.warmup(epoch == 1)
        order = rng(seed, data.dataset, 50, epoch).permutation(len(view['ids']))
        total, requests = 0., 0
        for begin in range(0, len(order), 2048):
            pos = order[begin:begin+2048]
            y, mask = data.labels('F', 'adapt', view['ids'][pos], method=method, epoch=epoch, step=global_step)
            optimizer.zero_grad(set_to_none=True)
            phi = tensor(view['phi'][pos], device) if model.reads_source else None
            logits = model(tensor(view['h'][pos], device), tensor(view['z'][pos], device),
                           tensor(view['domain'][pos], device), phi)
            loss = masked_bce(logits, tensor(y, device), tensor(mask, device))
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite adaptation loss')
            loss.backward()
            optimizer.step()
            model.project()
            total += float(loss.detach())*int(mask.sum())
            requests += int(mask.sum())
            global_step += 1
        yv, mv = data.labels('V', 'select', method=method, epoch=epoch)
        logits = predict(model, views['V'], device)
        metrics = evaluate(yv, logits, views['V']['domain'], mv)
        synchronize(device)
        record = dict(epoch=epoch, train_loss=total/requests, seconds=time.monotonic()-ep_start,
                      global_step=global_step, warmup=epoch == 1, row_plan_hash=array_hash(view['ids'][order]),
                      **flatten_metrics(metrics))
        append_csv(root/'epoch_metrics.csv', record)
        append_jsonl(root/'epoch_metrics.jsonl', dict(**record, validation=metrics))
        if epoch >= 5 and metrics['macro_logloss'] < best-1e-12:
            best = metrics['macro_logloss']
            selected = capture(model, optimizer, identity, epoch=epoch, global_step=global_step,
                               next_epoch=epoch+1, row_cursor=0, row_plan_hash=record['row_plan_hash'])
        line = f'ADAPT {data.dataset} seed={seed} method={method} epoch={epoch}/20 V_LL={metrics["macro_logloss"]:.9f} seconds={record["seconds"]:.2f}'
        print(line, flush=True)
        with (root/'stdout.log').open('a') as handle:
            handle.write(line+'\n')
    final = capture(model, optimizer, identity, epoch=20, global_step=global_step,
                    next_epoch=21, row_cursor=0, row_plan_hash=record['row_plan_hash'])
    synchronize(device)
    training_seconds = time.monotonic()-started
    peak = torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0
    scores = {}
    for name, snapshot in [('selected', selected), ('final', final)]:
        model.load_state_dict(snapshot['model'])
        for role, purpose in [('V', 'select'), ('Q', 'development')]:
            y, mask = data.labels(role, purpose, method=method, state=name)
            logits = predict(model, views[role], device)
            scores[f'{role}_{name}'] = export_prediction(root, role, name, views[role], y, mask,
                                                       logits, snapshot['position']['epoch'], state_hash(snapshot['model']))
    receipt = dict(identity=identity, counts=counts, selected_epoch=selected['position']['epoch'], final_epoch=20,
                   selected_model_hash=state_hash(selected['model']), final_model_hash=state_hash(final['model']),
                   selected_complete_state_hash=state_hash(selected), final_complete_state_hash=state_hash(final),
                   training_seconds=training_seconds, peak_cuda_bytes=peak,
                   global_steps=global_step, storage='RAM_ONLY', scores=scores)
    write_json(root/'run_receipt.json', receipt)
    model.load_state_dict(selected['model'])
    freeze(model).cpu()
    del optimizer
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return dict(model=model, selected=selected, final=final, root=root, receipt=receipt)

