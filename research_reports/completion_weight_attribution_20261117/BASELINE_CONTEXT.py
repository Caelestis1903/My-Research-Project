"""Baseline difficulty and natural train/validation populations from numeric receipts."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

R=Path(__file__).resolve().parent
A4=R.parents[1]/'artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_04'
read=lambda p:json.loads(p.read_text())


def main():
    verified=read(A4/'attempt1/CONTROLS_VERIFIED.json');index=Path(verified['index_path'])
    assert hashlib.sha256(index.read_bytes()).hexdigest()==verified['sha256']
    entries=read(index)['entries'];cells=[];domains=[]
    for dataset in ('movielens','kuairand'):
        manifest=read(A4/'attempt1'/f'{dataset}_data_manifest.json')
        controls=[]
        for entry in entries:
            if entry['dataset']!=dataset or entry['arm']!='CLEAN':continue
            path=Path(entry['result_path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['result_sha256']
            controls.append(read(path)['selected_validation']['fixed_ema'])
        assert len(controls)==5
        for d in range(3):
            joint=manifest['train_joint_labels'][str(d)];count=manifest['train_domain_counts'][d]
            assert sum(joint.values())==count
            row=dict(dataset=dataset,domain=d,train_rows=count)
            for metric in ('auc','logloss'):
                values=[v[f'domain{d}_macro_{metric}'] for v in controls]
                row[f'clean_selected_ema_{metric}_mean']=statistics.mean(values)
                row[f'clean_selected_ema_{metric}_sd']=statistics.stdev(values)
            domains.append(row)
            for t in range(2):
                c=f'd{d}t{t}';n0=sum(n for labels,n in joint.items() if labels[t]=='0');n1=count-n0
                assert len({(v[c+'_count'],v[c+'_positive']) for v in controls})==1
                row=dict(dataset=dataset,domain=d,task=t,train_rows=count,train_negative=n0,train_positive=n1,
                    train_positive_fraction=n1/count,validation_rows=controls[0][c+'_count'],
                    validation_positive=controls[0][c+'_positive'],validation_negative=controls[0][c+'_negative'])
                row['validation_positive_fraction']=row['validation_positive']/row['validation_rows']
                for metric in ('auc','logloss'):
                    values=[v[c+'_'+metric] for v in controls]
                    row[f'clean_selected_ema_{metric}_mean']=statistics.mean(values)
                    row[f'clean_selected_ema_{metric}_sd']=statistics.stdev(values)
                cells.append(row)
    for name,rows in [('BASELINE_CELL_CONTEXT.csv',cells),('BASELINE_DOMAIN_CONTEXT.csv',domains)]:
        with (R/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(domains=domains,cells=len(cells),test_accessed=False)))


if __name__=='__main__':main()
