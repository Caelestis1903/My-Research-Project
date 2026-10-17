"""Engineering-only diagnosis of the exact first N2 draw that failed identity.

Design: preserve checkpoint/rows; compare the two scalar aggregations on one
graph, then separate-domain/full-batch forwards in float32 and float64. No
training, threshold modification, selection, or test prediction is performed.
CPU output is an arithmetic oracle, not a reproduction of CUDA roundoff.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import n17_common as c


def gradient(loss, model, retain=False):
    parameters = list(model.parameters())
    values = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for p, g in zip(parameters, values)]).double()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(); started = time.time()
    config = json.loads(args.config.read_text()); assert not args.out.exists()
    torch.set_num_threads(2); device = torch.device(args.device)
    if device.type == 'cuda': torch.cuda.set_device(device)
    b = c.bundle(seed=config['seed']); p = np.asarray(config['p'])
    sampler = c.base.Sampler(b, p, config['seed'] + config['epoch'] * 100 + 500000, 'iid')
    rows = sampler.sample(config['batch_size']); domains = b.train_x[rows, -1].astype(int)
    counts = np.bincount(domains, minlength=3); coefficients = counts / len(rows)
    checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    model = c.base.fixed_model(b, config['seed']).to(device); model.load_state_dict(checkpoint['model'])
    results = []
    for dtype in (torch.float32, torch.float64):
        model.to(dtype=dtype)
        x = torch.tensor(b.train_x[rows], dtype=dtype, device=device).T
        y = torch.tensor(b.train_y[rows], dtype=dtype, device=device)
        prediction = model(x); terms = F.binary_cross_entropy(prediction, y, reduction='none').mean(1)
        full = gradient(terms.mean(), model, retain=True)
        masks = [torch.tensor(domains == d, device=device) for d in range(3)]
        composed_loss = sum(float(coefficients[d]) * terms[masks[d]].mean() for d in range(3))
        same_graph = gradient(composed_loss, model)
        part_gradients = []; prediction_error = 0.
        for d, mask in enumerate(masks):
            subset = model(x[:, mask]); loss = F.binary_cross_entropy(subset, y[mask], reduction='none').mean()
            part_gradients.append(gradient(loss, model))
            prediction_error = max(prediction_error, float((subset.detach() - prediction.detach()[mask]).abs().max()))
        partitioned = torch.tensor(coefficients, dtype=torch.float64, device=device) @ torch.stack(part_gradients)
        norm = float(partitioned.norm()); tolerance = 1e-6 + 1e-4 * norm
        result = dict(dtype=str(dtype), gradient_norm=norm, original_tolerance=tolerance,
                      same_graph_residual_norm=float((full - same_graph).norm()),
                      separate_forward_residual_norm=float((full - partitioned).norm()),
                      max_prediction_difference=prediction_error,
                      loss=float(terms.mean().detach()), elapsed_seconds=time.time() - started)
        results.append(result); print(json.dumps(result), flush=True)
    c.dump(args.out, dict(status='COMPLETE_ENGINEERING', config_sha=c.sha(args.config),
           checkpoint_sha=c.sha(config['checkpoint']), source_sha=c.sha(__file__), device=str(device),
           batch_size=len(rows), domain_counts=counts.tolist(), rows_sha256=hashlib.sha256(b.train_row_ids[rows].astype(np.int64).tobytes()).hexdigest(),
           results=results, test_read=False, scope='one fixed failed draw, same graph and precision diagnosis; no new research replicate'))


if __name__ == '__main__':
    main()
