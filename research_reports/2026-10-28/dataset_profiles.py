"""Describe the existing frozen populations and T0 vocabulary coverage."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PREPARED = {
    'movielens': ROOT/'artifacts/mdmt_relation_basis_v2/data/movielens',
    'kuairand': ROOT/'artifacts/mdmt_relation_basis_v2/revision1/data/kuairand',
}


def profile(output):
    records = []
    for dataset, root in PREPARED.items():
        manifest = json.loads((root/'DATA_MANIFEST.json').read_text())
        x, y = [np.load(root/f'{key}.npy', mmap_mode='r') for key in ('x','y')]
        columns = manifest['features']['columns']
        sparse = [columns.index(key) for key in manifest['features']['sparse_columns'] if key != 'domain_indicator']
        for role, names in {'T0':['fit'],'F':['P_label'],'S':['P_cal','P_update'],
                            'Q':['Q_explore'],'V':['validation'],'E':['test']}.items():
            ids = np.sort(np.concatenate([np.load(root/f'rows_{name}.npy') for name in names]))
            total = 0
            for domain in range(3):
                rows = ids[x[ids,-1] == domain]
                total += len(rows)
                xx, yy = x[rows], y[rows]
                user_oov = float((xx[:,columns.index('user_id')] == 0).mean()) if 'user_id' in columns else None
                for task in range(2):
                    records.append(dict(dataset=dataset,role=role,domain=domain,task=task,
                        n=len(rows),domain_fraction=len(rows)/len(ids),positive=int(yy[:,task].sum()),
                        positive_rate=float(yy[:,task].mean()),user_id_oov_fraction=user_oov,
                        sparse_field_oov_fraction=float((xx[:,sparse] == 0).mean())))
            assert total == len(ids)
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(output,index=False)
    return records


if __name__ == '__main__':
    rows = profile(Path(__file__).resolve().parent/'analysis/DATASET_PROFILES.csv')
    print(json.dumps([r for r in rows if r['dataset']=='movielens' and r['task']==0 and r['role'] in ('Q','E')]))
