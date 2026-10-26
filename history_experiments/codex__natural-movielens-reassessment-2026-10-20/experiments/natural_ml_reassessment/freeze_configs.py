"""Materialize every scientific configuration and its producer identity."""
import json
from .bootstrap import ROOT,dump,source_hashes
from .manifest import configurations

def main():
    data=json.loads((ROOT/'data/manifest.json').read_text());rows=[]
    for cfg in configurations():
        cfg=dict(cfg);manifest=data[cfg['population']];counts=[manifest['data']['train_domain_counts'][str(d)] for d in range(3)]
        p=[n/sum(counts) for n in counts]
        equal=(cfg['arm'].startswith('E') if cfg['stage']=='AC-2D' else cfg['stage']=='AC-S5')
        alpha=[1/3]*3 if equal else p
        q=[1/3]*3 if cfg['stage'] in ('AC-2D','AC-V9') and cfg['arm'].endswith('B') else p
        extra=('controlled.py',) if cfg['stage'] in ('AC-S5','AC-S6') else ('v9.py',) if cfg['stage']=='AC-V9' else ()
        cfg.update(train_counts=counts,train_population_size=sum(counts),p=p,alpha=alpha,q=q,
            importance_weight=[a/b for a,b in zip(alpha,q)],objective='equal' if equal else 'sample',
            raw_csv_sha256=manifest['data']['preprocessing_metadata']['raw_file_sha256'],
            schema=manifest['schema'],selected_train_row_ids_hash=manifest['data']['preprocessing_metadata']['selected_train_row_ids_hash'],
            base_train_row_ids_hash=manifest['data']['preprocessing_metadata']['base_train_row_ids_hash'],
            split_array_hashes=manifest['array_hashes'],source_hashes=source_hashes(extra),
            optimizer='SGD' if cfg['stage']=='AC-V9' else 'Adam',main_lr=.3 if cfg['stage']=='AC-V9' else .01,
            arch_lr=None if cfg['stage']=='AC-V9' else .001,momentum=0. if cfg['stage']=='AC-V9' else None,
            betas=None if cfg['stage']=='AC-V9' else [.9,.999],eps=None if cfg['stage']=='AC-V9' else 1e-8,
            weight_decay=1e-5,learned_fusion=cfg['stage']!='AC-V9',darts=cfg['stage']!='AC-V9',
            validation_rule='original V3 fixed SHA stratified half-validation earliest strict macro AUC' if cfg['stage']=='AC-V9' else 'six-cell full-validation earliest strict macro AUC')
        rows.append(cfg)
    dump(ROOT/'frozen_core_configurations.json',dict(status='FROZEN_FOR_INDEPENDENT_REVIEW',formal_runs=120,runs=rows))

if __name__=='__main__':main()
