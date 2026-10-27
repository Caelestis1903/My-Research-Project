"""Owner-only source pilots and T0-normalized concatenated 64d functions."""
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .config import rng, torch_seed
from .data import tensor
from .native import synchronize
from .state import freeze, capture, state_hash, array_hash, write_json, append_jsonl
from .vendor_snapshot.mdmt_dfd_20261026.functions import PilotBank


class SourceFunctions(nn.Module):
    def __init__(self, bank, mean, std):
        super().__init__()
        self.bank = bank
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
        teacher_w, teacher_b = [], []
        for source in range(3):
            ws, bs = [], []
            for task in range(2):
                head = bank.head(source, task)
                weight = head.output.weight.detach().reshape(32)
                expanded = weight.new_zeros(64)
                expanded[32*task:32*(task+1)] = weight
                ws.append(expanded*self.std[source].to(weight.device))
                bs.append(head.output.bias.detach().reshape(()) + (expanded*self.mean[source].to(weight.device)).sum())
            teacher_w.append(torch.stack(ws))
            teacher_b.append(torch.stack(bs))
        self.register_buffer('teacher_w', torch.stack(teacher_w))
        self.register_buffer('teacher_b', torch.stack(teacher_b))
        freeze(self)

    def hidden(self, h, source):
        return torch.cat([F.relu(self.bank.head(source, task).hidden(h)) for task in range(2)], dim=-1)

    def one(self, h, source):
        return (self.hidden(h, source)-self.mean[source])/self.std[source]

    def forward(self, h, domain=None):
        values = h.new_zeros(len(h), 3, 64)
        for source in range(3):
            rows = torch.ones(len(h), dtype=torch.bool, device=h.device) if domain is None else domain != source
            if bool(rows.any()):
                values[rows, source] = self.one(h[rows], source)
        return values

    def teacher(self, phi, z, source):
        return z + phi @ self.teacher_w[source].T + self.teacher_b[source]


def fit_source(data, view, seed, root, device, identity):
    bank = PilotBank(lambda s, q, k: torch_seed(seed, data.dataset, 30, source=s, task=q)).to(device)
    optimizer = torch.optim.Adam(bank.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=1e-5)
    synchronize(device)
    started, global_step = time.monotonic(), 0
    for epoch in range(1, 13):
        ep_start, loss_sum = time.monotonic(), 0.
        order = rng(seed, data.dataset, 31, epoch).permutation(len(view['ids']))
        for begin in range(0, len(order), 40960):
            pos = order[begin:begin+40960]
            y, _ = data.labels('T0', 'pilot', view['ids'][pos], epoch=epoch, step=global_step)
            optimizer.zero_grad(set_to_none=True)
            loss = bank.pilot_loss(tensor(view['h'][pos], device), tensor(view['z'][pos], device),
                                   tensor(y, device), tensor(view['domain'][pos], device), denominator=len(pos))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite pilot loss')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())*len(pos)
            global_step += 1
        synchronize(device)
        record = dict(component='source_pilot', epoch=epoch, train_loss=loss_sum/len(order),
                      seconds=time.monotonic()-ep_start, row_plan_hash=array_hash(view['ids'][order]))
        append_jsonl(Path(root)/'source_epoch_metrics.jsonl', record)
        print(f'SOURCE {data.dataset} seed={seed} pass={epoch}/12 loss={record["train_loss"]:.9f} seconds={record["seconds"]:.2f}', flush=True)
    snapshot = capture(bank, optimizer, identity, epoch=12, global_step=global_step, next_epoch=13,
                       row_cursor=0, row_plan_hash=array_hash(view['ids'][order]))
    freeze(bank)
    means, stds = [], []
    with torch.no_grad():
        for source in range(3):
            total = np.zeros(64, np.float64)
            square = np.zeros(64, np.float64)
            count = 0
            own = np.flatnonzero(view['domain'] == source)
            data.labels('T0', 'normalizer', view['ids'][own], source=source)
            for begin in range(0, len(own), 40960):
                h = tensor(view['h'][own[begin:begin+40960]], device)
                hidden = torch.cat([F.relu(bank.head(source, t).hidden(h)) for t in range(2)], -1).cpu().numpy().astype(np.float64)
                total += hidden.sum(0)
                square += (hidden*hidden).sum(0)
                count += len(hidden)
            mean = total/count
            std = np.maximum(np.sqrt(np.maximum(square/count-mean*mean, 0)), .001)
            means.append(mean.astype(np.float32))
            stds.append(std.astype(np.float32))
    source_model = SourceFunctions(bank, np.stack(means), np.stack(stds)).to(device)
    max_diff = 0.
    with torch.no_grad():
        for source in range(3):
            pos = np.flatnonzero(view['domain'] == source)[:256]
            h, z = tensor(view['h'][pos], device), tensor(view['z'][pos], device)
            original = z + torch.stack([bank.head(source, t)(h) for t in range(2)], -1)
            converted = source_model.teacher(source_model.one(h, source), z, source)
            max_diff = max(max_diff, float((converted-original).abs().max()))
    if max_diff > 1e-6:
        raise ValueError(f'Source teacher conversion differs: {max_diff}')
    synchronize(device)
    receipt = dict(seconds=time.monotonic()-started, pilot_hash=state_hash(snapshot['model']),
                   source_function_hash=state_hash(source_model.state_dict()),
                   teacher_max_abs_diff=max_diff, hidden_parameters=49344,
                   teacher_parameters=198, selected_pass=12,
                   normalization='T0_source_population_std_floor_0.001', storage='RAM_ONLY')
    write_json(Path(root)/'source_receipt.json', receipt)
    return source_model, snapshot, receipt


@torch.no_grad()
def cache_phi(source, view, role, root, device):
    path = Path(root)/'row_data'/role/'phi.npy'
    out = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(view['ids']), 3, 64))
    for begin in range(0, len(out), 40960):
        out[begin:begin+40960] = source(tensor(view['h'][begin:begin+40960], device)).cpu().numpy()
    out.flush()
    del out
    view['phi'] = np.load(path, mmap_mode='r')

