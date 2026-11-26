"""Validation-only strict maximum selection shared by A and B."""
import math
from .state import cpu_copy, state_digest, digest

CELLS=tuple(f'd{d}t{t}' for d in range(3) for t in range(2))

def score(validation):
    values=[validation.get(c+'_auc') for c in CELLS]
    if any(v is None or not math.isfinite(v) for v in values):
        raise ValueError('All six validation AUC cells are required')
    return math.fsum(values)/6

def consider(tracker,stage,epoch,validation,model_state,model_meta=None):
    if stage!=tracker['final_stage'] or not 1<=epoch<=50: return False
    value=score(validation)
    if value<=tracker.get('best_score',-math.inf): return False
    state=cpu_copy(model_state)
    tracker.update(best_score=value,selected=dict(model_state=state,stage=stage,epoch=epoch,
        model_hash=state_digest(state),validation=dict(validation),model_meta=model_meta or {}))
    return True

def freeze(tracker):
    selected=tracker.get('selected')
    if selected is None: raise ValueError('No legal selected state')
    return {k:v for k,v in selected.items() if k!='model_state'}

