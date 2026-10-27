"""Six reviewed adaptation arms; the native prefix is always frozen."""
import torch
from torch import nn
from torch.nn import functional as F
from .config import METHODS, torch_seed
from .relation import Relation


class ResidualHead(nn.Module):
    def __init__(self, width=32, relu=False, seed=0):
        super().__init__()
        self.relu = relu
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hidden = nn.Linear(258, width)
            self.output = nn.Linear(width, 1)
            nn.init.xavier_uniform_(self.hidden.weight)
            nn.init.zeros_(self.hidden.bias)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, hz):
        value = self.hidden(hz)
        return self.output(F.relu(value) if self.relu else F.gelu(value)).squeeze(-1)


class Model(nn.Module):
    def __init__(self, method, seed=2022, dataset='movielens', basis_count=2):
        super().__init__()
        if method not in METHODS:
            raise ValueError('Unknown method')
        self.method, self.basis_count = method, basis_count
        self.reads_source = method in ('READ', 'LOWRANK_RELATION', 'FULL_RELATION')
        self.forms_basis = method in ('LOWRANK_RELATION', 'FULL_RELATION')
        self.heads = nn.ModuleList([ResidualHead(
            width=40 if method == 'TARGET_ONLY' else 32, relu=method == 'B_REFERENCE',
            seed=torch_seed(seed, dataset, 100, source=d, task=t))
            for d in range(3) for t in range(2)])
        if self.reads_source:
            self.relation = Relation(method, basis_count, torch_seed(seed, dataset, 200))
            self.w = nn.Parameter(torch.zeros(3, 2, 2, 64))
        if self.forms_basis:
            self.U = nn.Parameter(torch.zeros(3, basis_count, 64, 4))
            self.V = nn.Parameter(torch.empty(3, basis_count, 4, 64))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch_seed(seed, dataset, 300))
                nn.init.normal_(self.V, std=.02)

    def function(self, phi, source, root=False):
        roots = phi[:, None, :].expand(-1, self.basis_count, -1)
        if root or not self.forms_basis:
            return roots
        hidden = torch.einsum('bp,jrp->bjr', phi, self.V[source])
        return roots + torch.einsum('bjr,jpr->bjp', F.gelu(hidden), self.U[source])

    def target_residual(self, hz, domain):
        result = hz.new_empty(len(hz), 2)
        for d in range(3):
            rows = domain == d
            if bool(rows.any()):
                result[rows] = torch.stack([self.heads[2*d+t](hz[rows]) for t in range(2)], -1)
        return result

    def forward(self, h, z, domain, phi=None, *, root=False, constant_alpha=None,
                force_null=False, return_routes=False):
        if h.shape != (len(z), 256) or z.shape != (len(h), 2):
            raise ValueError('Invalid frozen prefix shapes')
        if domain.shape != (len(h),) or not bool(((domain >= 0) & (domain < 3)).all()):
            raise ValueError('Invalid domain')
        hz = torch.cat((h.detach(), z.detach()), -1)
        logits = z.detach() + self.target_residual(hz, domain)
        alpha = null = None
        if self.reads_source:
            if phi is None or phi.shape != (len(h), 3, 64):
                raise ValueError('Expected [N,3,64] source features')
            alpha, null = self.relation(hz, domain)
            if constant_alpha is not None:
                constant_alpha = constant_alpha.to(h.device)
                alpha = constant_alpha[domain, :, :-1].reshape(-1, 2, 2, self.basis_count)
                null = constant_alpha[domain, :, -1]
            if force_null:
                alpha, null = torch.zeros_like(alpha), torch.ones_like(null)
            addition = logits.new_zeros(len(h), 2)
            for d in range(3):
                rows = domain == d
                if not bool(rows.any()):
                    continue
                for slot, s in enumerate(self.relation.foreign[d].tolist()):
                    value = self.function(phi[rows, s].detach(), s, root)
                    contribution = torch.einsum('bjp,tp->btj', value, self.w[d, :, slot])
                    addition[rows] = addition[rows] + (alpha[rows, :, slot] * contribution).sum(-1)
            logits = logits + addition
        return (logits, alpha, null) if return_routes else logits

    def warmup(self, enabled):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(not enabled or name.startswith('heads.') or name == 'w')
            if not parameter.requires_grad:
                parameter.grad = None

    @torch.no_grad()
    def project(self):
        if self.reads_source:
            self.w.mul_((2. / self.w.norm(dim=-1, keepdim=True).clamp_min(2.)).clamp_max(1.))

    def parameter_counts(self):
        return dict(adaptation=sum(p.numel() for p in self.parameters()),
                    relation=self.relation.relation_parameters() if self.reads_source else 0,
                    source_functions=3*self.basis_count*512 if self.forms_basis else 0)

