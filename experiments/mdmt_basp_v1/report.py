"""Collect raw measurements across trajectories without making research decisions."""
from pathlib import Path
import json
from .export import write_csv
from .identity import write_json


def collect(artifact_root,output):
    root,out=Path(artifact_root),Path(output)
    seeds,cells,calibrations,costs=[],[],[],[]
    for path in sorted((root/'runs').glob('*/*/seed*/*_metrics.json')):
        dataset,method,seed_text=path.parts[-4:-1]
        role,state=path.stem.removesuffix('_metrics').split('_',1)
        metric=json.loads(path.read_text())
        common=dict(dataset=dataset,method=method,seed=int(seed_text[4:]),role=role,state=state)
        seeds.append(dict(**common,**{k:v for k,v in metric.items() if k not in ('cells','identity')}))
        cells.extend(dict(**common,**c) for c in metric['cells'])
    for path in sorted((root/'runs').glob('*/*/seed*/*_GAIN_CALIBRATION.json')):
        dataset,method,seed_text=path.parts[-4:-1]
        role,state=path.stem.removesuffix('_GAIN_CALIBRATION').split('_',1)
        values=json.loads(path.read_text())
        for kind,metric in values.items():
            if not isinstance(metric,dict) or not set(metric).intersection({'mse','status','skill','constant_mse'}):
                continue
            calibrations.append(dict(dataset=dataset,method=method,seed=int(seed_text[4:]),
                                     role=role,state=state,kind=kind,**metric))
    for path in sorted((root/'runs').glob('*/*/seed*/COST_LEDGER.json')):
        dataset,method,seed_text=path.parts[-4:-1]
        costs.append(dict(dataset=dataset,method=method,seed=int(seed_text[4:]),**json.loads(path.read_text())))
    write_csv(out/'SEED_METRICS.csv',seeds);write_csv(out/'CELL_METRICS.csv',cells)
    write_csv(out/'GAIN_CALIBRATION.csv',calibrations);write_csv(out/'COST_LEDGER.csv',costs)
    statuses=[]
    for path in sorted((root/'workers').glob('*/seed*/STATUS.json')):
        statuses.append(json.loads(path.read_text()))
    result=dict(status='RAW_COLLECTION',metric_records=len(seeds),cell_records=len(cells),
                trajectories=len(costs),workers=statuses)
    write_json(out/'COLLECTION_STATUS.json',result)
    return result
