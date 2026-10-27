"""Foreign-only directed relations and a fixed zero-logit NULL option."""
import torch
from torch import nn
from torch.nn import functional as F


class Relation(nn.Module):
    def __init__(self, kind, basis_count=2, seed=0):
        super().__init__()
        self.kind, self.basis_count = kind, basis_count
        self.register_buffer('foreign', torch.tensor([[1, 2], [0, 2], [0, 1]]))
        self.bias = nn.Parameter(torch.zeros(3, 2, 2, basis_count))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if kind != 'READ':
                self.request = nn.Linear(258, 32)
            if kind == 'LOWRANK_RELATION':
                self.target_task = nn.Embedding(6, 2)
                self.source_basis = nn.Embedding(3*basis_count, 2)
                self.C = nn.Linear(32, 2, bias=False)
            elif kind == 'FULL_RELATION':
                self.full = nn.Linear(32, 3*2*2*basis_count, bias=False)
            elif kind != 'READ':
                raise ValueError('Unknown relation kind')

    def forward(self, hz, domain):
        n, j = len(hz), self.basis_count
        alpha = hz.new_empty(n, 2, 2, j)
        null = hz.new_empty(n, 2)
        request = F.gelu(self.request(hz)) if self.kind != 'READ' else None
        full = self.full(request).reshape(n, 3, 2, 2, j) if self.kind == 'FULL_RELATION' else None
        projected = self.C(request) if self.kind == 'LOWRANK_RELATION' else None
        for d in range(3):
            rows = domain == d
            if not bool(rows.any()):
                continue
            value = self.bias[d].unsqueeze(0).expand(int(rows.sum()), -1, -1, -1)
            if self.kind == 'LOWRANK_RELATION':
                a = self.target_task.weight.reshape(3, 2, 2)[d]
                k = self.source_basis.weight.reshape(3, j, 2)[self.foreign[d]]
                value = value + torch.einsum('tr,sjr,br->btsj', a, k, projected[rows])
            elif self.kind == 'FULL_RELATION':
                value = value + full[rows, d]
            probs = torch.softmax(torch.cat((value.flatten(2), value.new_zeros(len(value), 2, 1)), dim=-1), dim=-1)
            alpha[rows] = probs[:, :, :-1].reshape(-1, 2, 2, j)
            null[rows] = probs[:, :, -1]
        return alpha, null

    def relation_parameters(self):
        return sum(p.numel() for name, p in self.named_parameters() if not name.startswith('request.'))

