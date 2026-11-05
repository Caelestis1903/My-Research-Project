"""The fixed1 graph's twelve disjoint parameter-coordinate paths."""
from dataclasses import dataclass, field
import torch
from experiments.overnight_ab_20261103.state import digest
from .registry import ParameterLayout

GROUPS=('S','D0','D1','D2','T0','T1','DT00','DT01','DT10','DT11','DT20','DT21')


@dataclass
class GroupRegistry:
    layout: ParameterLayout
    records: list
    counts: dict
    hash: str
    coordinates: torch.Tensor
    rho0: float | None = None
    device_coordinates: dict = field(default_factory=dict)

    def indices(self,device):
        key=str(device)
        if key not in self.device_coordinates:
            self.device_coordinates[key]=self.coordinates.to(device)
        return self.device_coordinates[key]

    def as_dict(self):
        return dict(group_order=GROUPS,layout_hash=self.layout.hash,group_hash=self.hash,
            counts=self.counts,records=self.records)


def build_path_groups(model,bundle):
    layout=ParameterLayout.from_model(model)
    column=bundle.features.columns.index('domain_indicator')
    domain_name=f'embedding.embed_dict.{column}.weight'
    parameters=dict(model.named_parameters())
    if domain_name not in parameters or parameters[domain_name].shape[1:] != (16,):
        raise ValueError('Domain embedding does not match the actual sparse lookup')
    if set(bundle.train_x[:,-1].astype('int64')) != {0,1,2}:
        raise ValueError('Legal domain lookup rows must be 0/1/2')
    records=[]; coordinates=torch.full((layout.size,),-1,dtype=torch.long)
    cells={g:[[d,t] for d in range(3) for t in range(2) if
        g=='S' or g==f'D{d}' or g==f'T{t}' or g==f'DT{d}{t}'] for g in GROUPS}
    for name,shape,(start,end) in zip(layout.names,layout.shapes,layout.offsets):
        if name==domain_name:
            parts=[(f'D{row}' if row<3 else 'INACTIVE',start+16*row,start+16*(row+1),[row]) for row in range(shape[0])]
        else:
            root=name.split('.')[0]
            if name in ('shared_weight','shared_bias') or root in ('skip_conn','star_mlp','expert','embedding'):
                group='S'
            elif root in ('slot_weight','slot_bias','domain_expert'):
                group='D'+name.split('.')[1]
            elif root=='task_expert':
                group='T'+name.split('.')[1]
            elif root in ('gate','tower'):
                index=int(name.split('.')[1]); group=f'DT{index%3}{index//3}'
            else:
                raise ValueError('Unmapped source parameter: '+name)
            if group not in GROUPS:
                raise ValueError('Unknown graph path: '+group)
            parts=[(group,start,end,None)]
        for group,left,right,rows in parts:
            if bool((coordinates[left:right]!=-1).any()):
                raise ValueError('Parameter coordinates overlap')
            coordinates[left:right]=GROUPS.index(group) if group!='INACTIVE' else 12
            records.append(dict(parameter_name=name,shape=shape,flat_ranges=[[left,right]],
                row_indices=rows,group=group,numel=right-left,source_cells=cells.get(group,[])))
    if bool((coordinates<0).any()):
        raise ValueError('Parameter coordinates are not fully mapped')
    counts={g:int((coordinates==i).sum()) for i,g in enumerate((*GROUPS,'INACTIVE'))}
    expected=dict(S=483152 if bundle.dataset=='movielens' else 661920,
        **{f'D{d}':74960 if bundle.dataset=='movielens' else 295632 for d in range(3)},
        **{f'T{t}':16576 for t in range(2)},**{f'DT{d}{t}':5381 for d in range(3) for t in range(2)},INACTIVE=0)
    if counts!=expected:
        raise ValueError('Frozen path coordinate counts changed')
    identity=dict(layout_hash=layout.hash,records=records,counts=counts,group_order=GROUPS)
    return GroupRegistry(layout,records,counts,digest(identity),coordinates)
