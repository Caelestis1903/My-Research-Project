"""P2/P4 full exposure identities, label counts, partial matched coverage."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from v2_20261016 import OUT,sha,dump
from stage_S1_data import load_stage_s1_bundle


def main():
    target=OUT/'analysis';coverage=[];identities=[];labels=[]
    manifests=sorted((OUT/'runs').glob('*/*/seed*/*/attempt*/run_manifest.json'))
    for path in manifests:
        c=json.loads(path.read_text())['config'];r=path.parent;e=pd.read_csv(r/'exposure.csv')
        for d in range(3):
            size=c['data']['train_domain_counts'][str(d)]
            for threshold in (.5,.9,.99):
                hits=e[e[f'unique_cumulative_d{d}']/size>=threshold]
                epoch=int(hits.iloc[0].epoch) if len(hits)>0 else None
                metrics=pd.read_csv(r/'metrics_epoch.csv');val=metrics.query('epoch==@epoch') if epoch is not None else None
                coverage.append(dict(dataset=c['dataset'],optimizer=c['optimizer'],variant=c['sampler'],arm=c['arm'],seed=c['seed'],
                    domain=d,train_count=size,coverage_threshold=threshold,first_observed_epoch=epoch,
                    steps=int(hits.iloc[0].steps) if epoch is not None else None,
                    validation_macro_auc=float(val.auc.mean()) if val is not None else None,
                    validation_domain_auc=float(val[val.domain==d].auc.mean()) if val is not None else None,
                    observed_coverage=float(hits.iloc[0][f'unique_cumulative_d{d}']/size) if epoch is not None else None,
                    resolution='INSUFFICIENT_EPOCH1_CENSORING' if epoch==1 else 'EPOCH_GRID_UNEQUAL_BUDGET_PARTIAL_READOUT'))
        if c['sampler']=='cyclic':
            iid=OUT/'runs'/c['dataset']/c['optimizer']/f"seed{c['seed']}"/c['arm']/'attempt1'
            original=pd.read_csv(iid/'batch_trace.csv');control=pd.read_csv(r/'batch_trace.csv')
            match=np.array_equal(original[['epoch','batch','n','d0','d1','d2']].to_numpy(),control[['epoch','batch','n','d0','d1','d2']].to_numpy())
            assert match,'domain sequence changed in cyclic control'
            identities.append(dict(dataset=c['dataset'],optimizer=c['optimizer'],arm=c['arm'],seed=c['seed'],
                exact_domain_exposure_sequence_match=True,rows_changed=not original.row_ids_sha256.equals(control.row_ids_sha256),
                inference='same realized domain counts; within-domain coverage/repetition/temporal dependence jointly changed'))
    pd.DataFrame(coverage).to_csv(target/'coverage_threshold_crossings.csv',index=False)
    pd.DataFrame(identities).to_csv(target/'cyclic_exposure_identity.csv',index=False)
    # Data are loaded once per dataset/seed; label exposure comes from saved row counts,
    # not a replay guessed from q or an expected positive rate.
    for dataset in ('kuairand','movielens'):
        for seed in (2022,2023,2024,2025):
            chosen=[]
            for path in manifests:
                c=json.loads(path.read_text())['config']
                if c['dataset']==dataset and c['seed']==seed:chosen.append((path.parent,c))
            if not chosen:continue
            b=load_stage_s1_bundle(dataset,None,'natural' if dataset=='kuairand' else 'd2_large',seed)
            for r,c in chosen:
                for key,file in [('selected','selected_state.pt'),('final','epoch50.pt')]:
                    state=torch.load(r/file,map_location='cpu',weights_only=False);counts=state['exposure_counts'];assert len(counts)==len(b.train_y)
                    assert int(counts.sum())==state['epoch']*len(b.train_y)
                    state_hash=sha(r/file)
                    for d in range(3):
                        indices=np.flatnonzero(b.train_x[:,-1]==d);exposure=counts[indices].astype(np.int64)
                        w=c['alpha'][d]/c['q'][d]
                        for task in range(2):
                            y=b.train_y[indices,task];pos=int(np.dot(exposure,y));total=int(exposure.sum())
                            labels.append(dict(dataset=dataset,optimizer=c['optimizer'],variant=c['sampler'],arm=c['arm'],seed=seed,
                                checkpoint=key,epoch=state['epoch'],domain=d,task=task,positive_exposures=pos,negative_exposures=total-pos,
                                total_exposures=total,actual_positive_exposure_rate=pos/total,base_train_positive_rate=float(y.mean()),
                                weighted_positive_exposures=pos*w,weighted_total_exposures=total*w,
                                unique_rows=int(np.count_nonzero(exposure)),repeat_median=float(np.median(exposure)),
                                repeat_p95=float(np.quantile(exposure,.95)),repeat_max=int(exposure.max()),
                                state_sha256=state_hash))
    pd.DataFrame(labels).to_csv(target/'actual_label_exposure.csv',index=False)
    frame=pd.DataFrame(coverage);matched=[]
    for keys,z in frame[frame.arm.isin(['SN','SB'])].groupby(['dataset','optimizer','variant','seed','domain','coverage_threshold']):
        n=z[z.arm=='SN'];b=z[z.arm=='SB']
        if n.empty or b.empty:continue
        n=n.iloc[0];b=b.iloc[0]
        matched.append(dict(zip(['dataset','optimizer','variant','seed','domain','threshold'],keys),natural_epoch=n.first_observed_epoch,
            balanced_epoch=b.first_observed_epoch,natural_steps=n.steps,balanced_steps=b.steps,
            domain_auc_difference=float(b.validation_domain_auc-n.validation_domain_auc),
            inference='PARTIAL coverage-threshold readout; unequal budgets and coarse epochs; not causal matched-exposure control'))
    pd.DataFrame(matched).to_csv(target/'partial_matched_coverage.csv',index=False)
    dump(target/'exposure_receipt.json',dict(status='COMPLETE',source_sha256=sha(__file__),label_exposure_rows=len(labels),
        actual_saved_row_counts=True,cyclic_identity_controls=len(identities),matched_coverage_is_partial=True))
    print('Exposure analysis complete',len(labels),len(identities))

if __name__=='__main__':main()
