"""Stable, complete source parameter coordinates."""
from dataclasses import dataclass
import torch
from .common import digest


@dataclass(frozen=True)
class ParameterLayout:
    names: tuple
    shapes: tuple
    offsets: tuple
    dtypes: tuple
    aliases: tuple
    coordinate_groups: dict
    hash: str

    @classmethod
    def from_model(cls, model):
        named = list(model.named_parameters())
        names = tuple(n for n, _ in named)
        shapes = tuple(tuple(p.shape) for _, p in named)
        offsets, end = [], 0
        for _, p in named:
            offsets.append((end, end+p.numel()))
            end += p.numel()
        canonical = {id(p): n for n, p in named}
        aliases = tuple((n, canonical[id(p)]) for n, p in model.named_parameters(remove_duplicate=False))
        dtypes = tuple(str(p.dtype) for _, p in named)
        identity = dict(names=names, shapes=shapes, offsets=offsets, dtypes=dtypes, aliases=aliases)
        # B uses every coordinate; A's path partition is implemented only if A is triggered.
        return cls(names, shapes, tuple(offsets), dtypes, aliases, {}, digest(identity))

    @property
    def size(self):
        return self.offsets[-1][1]

    def flatten(self, values):
        return torch.cat([v.detach().reshape(-1) for v in values])

    @torch.no_grad()
    def copy_to(self, model, vector, buffers=None):
        if vector.numel() != self.size:
            raise ValueError('Parameter vector size changed')
        for p, shape, (start, end) in zip(model.parameters(), self.shapes, self.offsets):
            p.copy_(vector[start:end].view(shape))
        if buffers is not None:
            for name, value in model.named_buffers():
                value.copy_(buffers[name])

