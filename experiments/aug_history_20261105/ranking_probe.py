"""Independent training-row sampling and cell-balanced raw-logit ranking probes."""
import hashlib
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
from experiments.overnight_ab_20261103.state import digest


@dataclass
class ProbeBatch:
    x: np.ndarray
    row_ids: np.ndarray
    cells: list
    metadata: dict


class RankProbeSampler:
    def __init__(self,bundle):
        self.bundle=bundle
        self.pools={(d,t,label):np.flatnonzero((bundle.train_x[:,-1]==d)&(bundle.train_y[:,t]==label))
            for d in range(3) for t in range(2) for label in (0,1)}

    def sample(self,excluded_row_ids,seed,step):
        b=self.bundle; positions=[]; cells=[]; keys=[]; skipped=[]
        excluded=np.isin(b.train_row_ids,np.asarray(excluded_row_ids))
        for d in range(3):
            for t in range(2):
                key=f'AH1105|{b.dataset}|{seed}|{step}|A_RANK|{d}|{t}'
                value=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63-1)
                generator=np.random.default_rng(value)
                chosen={}
                for label in (1,0):
                    pool=self.pools[d,t,label]
                    pool=pool[~excluded[pool]]
                    chosen[label]=generator.choice(pool,size=min(16,len(pool)),replace=False)
                if not len(chosen[1]) or not len(chosen[0]):
                    skipped.append([d,t]); continue
                positive=list(range(len(positions),len(positions)+len(chosen[1])))
                positions.extend(chosen[1].tolist())
                negative=list(range(len(positions),len(positions)+len(chosen[0])))
                positions.extend(chosen[0].tolist())
                cells.append(dict(domain=d,task=t,positive=positive,negative=negative)); keys.append(key)
        ids=np.asarray(positions,dtype=np.int64)
        raw_ids=np.asarray(b.train_row_ids[ids])
        metadata=dict(step=step,keys=keys,valid_cells=len(cells),cell_rows=len(ids),skipped_cells=skipped,
            row_ids_hash=hashlib.sha256(raw_ids.tobytes()).hexdigest(),
            reason='OK' if cells else 'NO_VALID_TRAIN_CELL')
        metadata['probe_hash']=digest(metadata)
        return ProbeBatch(np.asarray(b.train_x[ids]),raw_ids,cells,metadata)


def sample_rank_probe(train_bundle,excluded_row_ids,seed,step):
    return RankProbeSampler(train_bundle).sample(excluded_row_ids,seed,step)


def ranking_objective(params,probe,model_adapter):
    if not probe.cells:
        return None
    device=next(model_adapter.parameters()).device
    x=torch.as_tensor(probe.x,device=device).T
    logits=model_adapter.forward_logits(x,params=params,probe_input_full_graph=False)
    terms=[]
    for cell in probe.cells:
        z=logits[:,cell['task']]
        terms.append(F.softplus(z[cell['negative']][:,None]-z[cell['positive']][None,:]).mean())
    return torch.stack(terms).mean()
