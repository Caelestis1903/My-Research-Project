"""Disjoint leaf execution costs and explicit label-access events."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
import os,time
import torch
from .state import append_jsonl,atomic_json

class CostLedger:
    def __init__(self,path,identity=None):self.path=Path(path);self.identity=identity or {}
    @contextmanager
    def measure(self,component,**fields):
        if torch.cuda.is_initialized():torch.cuda.synchronize()
        start=time.perf_counter();utc=datetime.now(timezone.utc).isoformat()
        try:yield
        finally:
            if torch.cuda.is_initialized():torch.cuda.synchronize()
            append_jsonl(self.path,dict(self.identity,component=component,started_utc=utc,seconds=time.perf_counter()-start,pid=os.getpid(),**fields))
    def access(self,component,role,rows,tasks=2,passes=1,**fields):
        append_jsonl(self.path.with_name('label_access.jsonl'),dict(self.identity,component=component,role=role,rows=int(rows),tasks=int(tasks),passes=passes,**fields))


def measured_budget(root,stage,remaining_counts):
    import json,shutil
    rows=[]
    for p in Path(root).rglob('costs.jsonl'):
        rows.extend(json.loads(l) for l in p.read_text().splitlines() if l.strip())
    maxima={}
    for row in rows:maxima[row['component']]=max(maxima.get(row['component'],0),row['seconds'])
    known={k:maxima[k]*v for k,v in remaining_counts.items() if k in maxima};missing=[k for k in remaining_counts if k not in maxima]
    r=dict(stage=stage,measured_max_seconds=maxima,remaining_counts=remaining_counts,predicted_remaining_seconds=1.35*sum(known.values()),unmeasured_components=missing,reserve=1.35,disk_free_bytes=shutil.disk_usage(root).free,definition='prediction based on retained real leaf components; missing leaves remain unknown; process wall is not GPU hours')
    atomic_json(Path(root)/f'BUDGET_{stage.upper()}.json',r);return r
