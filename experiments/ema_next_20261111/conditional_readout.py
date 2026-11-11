"""Fixed training rank pairs and one-epoch-lagged finite C policies."""
import hashlib
import json
import numpy as np
from experiments.overnight_ab_20261103.state import digest


def build_unique_pairs(panel, dataset, seed, limit=4096):
    tables=[];identities=[]
    for cell in range(6):
        ids=np.asarray(panel['rank_row_ids'][cell]);labels=np.asarray(panel['rank_y'][cell])
        positive=sorted({int(row):int(i) for i,row in enumerate(ids) if labels[i]==1}.items())
        negative=sorted({int(row):int(i) for i,row in enumerate(ids) if labels[i]==0}.items())
        rows=[]
        for pos,pi in positive:
            for neg,ni in negative:
                key=dict(study='EMA_NEXT_20261111_C_PAIR_V1',dataset=dataset,seed=int(seed),
                         panel=panel['panel_id'],domain=cell//2,task=cell%2,
                         positive_row_id=pos,negative_row_id=neg)
                h=hashlib.blake2b(json.dumps(key,separators=(',',':'),ensure_ascii=False).encode('utf-8'),digest_size=16).digest()
                rows.append((h,pos,neg,pi,ni))
        rows.sort(key=lambda r:(r[0],r[1],r[2]));rows=rows[:limit]
        table=np.asarray([(r[3],r[4]) for r in rows],dtype=np.int64).reshape(-1,2)
        tables.append(table)
        identities.append(dict(cell=cell,count=len(table),unique=True,pair_hash=digest([[r[1],r[2]] for r in rows]),
                               row_hash=hashlib.sha256(ids.tobytes()).hexdigest(),missing_reason=None if len(table) else 'NO_VALID_PAIR'))
    return dict(tables=tables,identity=identities,panel_hash=panel['panel_hash'],pair_hash=digest(identities))


def grid_for(a):
    result=[]
    for v in sorted([0.,.25,.5,.75,1.,float(a)]):
        if not result or v-result[-1]>1e-12: result.append(v)
    return np.asarray(result,dtype=np.float64)


def choose_risk(grid,risks,a):
    minimum=float(np.min(risks));ties=[float(v) for v,r in zip(grid,risks) if r<=minimum+1e-12]
    return min(ties,key=lambda v:(abs(v-a),abs(v-.5),v))


def pair_margins(logits,pairs):
    z=np.asarray(logits,dtype=np.float64);p=np.asarray(pairs)
    return z[p[:,0]]-z[p[:,1]]


def pair_risk(logits,pairs):
    if not len(pairs): return None
    return float(np.logaddexp(0.,-pair_margins(logits,pairs)).mean())


def fit_from_PA(recent,old,pairs,a,epoch):
    grid=grid_for(a);risks=[]
    for c,table in enumerate(pairs['tables']):
        if not len(table): risks.append(None);continue
        r=pair_margins(recent[c],table);o=pair_margins(old[c],table)
        risks.append(np.logaddexp(0.,-((1-grid[:,None])*r+grid[:,None]*o)).mean(1))
    valid=[r for r in risks if r is not None]
    global_alpha=choose_risk(grid,np.mean(valid,axis=0),a) if valid else float(a)
    cell=[choose_risk(grid,r,a) if r is not None else global_alpha for r in risks]
    return dict(fit_epoch=int(epoch),a_fit=float(a),global_alpha=global_alpha,cell_alpha=cell,
                grid=grid.tolist(),risks=[r.tolist() if r is not None else None for r in risks],
                pair_hash=pairs['pair_hash'],missing_cells=[i for i,r in enumerate(risks) if r is None])


def apply_lagged_policy(pending,a,epoch,s=0.):
    if epoch==1:
        return dict(weights=[float(a)]*6,global_alpha=float(a),fit_epoch=0,application_epoch=1,a_fit=float(a),a_apply=float(a),shrinkage=s)
    if pending is None or pending['fit_epoch']!=epoch-1: raise ValueError('C policy must come from previous epoch P_A')
    g=pending['global_alpha'];weights=(1-s)*g+s*np.asarray(pending['cell_alpha'])
    return dict(weights=weights.tolist(),global_alpha=g,fit_epoch=pending['fit_epoch'],application_epoch=epoch,
                a_fit=pending['a_fit'],a_apply=float(a),shrinkage=s)


def mix_logits(recent,old,domains,weights):
    r=np.asarray(recent,dtype=np.float32);o=np.asarray(old,dtype=np.float32)
    w=np.asarray(weights,dtype=np.float32)
    if w.ndim:
        w=w.reshape(3,2)[np.asarray(domains,dtype=np.int64)]
    return ((1-w)*r+w*o).astype(np.float32)


def sign_change_fraction(candidate,reference,pairs):
    if not len(pairs): return None
    return float(np.mean(np.sign(pair_margins(candidate,pairs))!=np.sign(pair_margins(reference,pairs))))


def evaluate_parameter_match(packet,x,context):
    from .evaluation import predict_batch
    if packet.get('mode')!='parameter': raise ValueError('Expected parameter policy packet')
    return predict_batch(packet,x,context)
