"""Cross-check raw V1 artifacts and build the gate's numerical evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import v3_audit_20261016 as audit
base=audit.base;OUT=audit.OUT

def main():
    manifest=json.loads((OUT/'audit_manifest.json').read_text());tables=[];checks=[];cosines=[];hashes=[]
    for task in manifest['tasks']:
        args=task['command'];directory=Path(args[args.index('--run-dir')+1])
        receipt=json.loads((directory/'receipt.json').read_text());config=json.loads((directory/'config.json').read_text())
        assert receipt['status']=='PASS' and receipt['k']==512 and config['k']==512
        source=Path(config['source_checkpoint']);old=json.loads((source.parent/'config.json').read_text())
        assert old['data']==config['data'],'Historical checkpoint/current diagnostic population mismatch'
        assert base.sha(source)==config['source_checkpoint_sha']
        for name,expected in receipt['artifact_hashes'].items():
            if name=='progress.json':
                hashes.append(dict(run=directory.name,file=name,sha256=base.sha(directory/name),
                    verification='MUTABLE_STATUS_FINAL_HASH; receipt predates final status write'))
                continue
            # Huge raw matrices keep original producer hashes; all compact numerical
            # inputs are independently rehashed rather than rereading >100GB twice.
            if name.endswith('_gradients.npy'):
                array=np.load(directory/name,mmap_mode='r');assert array.shape==(512,1614254) and array.dtype==np.float32
                kind='PRODUCER_HASH_RETAINED_SHAPE_VERIFIED'
            else:
                assert base.sha(directory/name)==expected;kind='INDEPENDENTLY_REHASHED'
            hashes.append(dict(run=directory.name,file=name,sha256=expected,verification=kind))
        data=np.load(directory/'exact.npz');gd=data['domain_gradients'];ref=data['reference'];p=np.asarray(config['p'])
        assert np.allclose(p@gd,ref,rtol=1e-12,atol=1e-14)
        rows=pd.read_csv(directory/'batches.csv');assert len(rows)==1024
        geometry=json.loads((directory/'geometry.json').read_text())
        groups={key:[] for key in 'CPRE'}
        for parameter in config['parameter_order']:groups[parameter['block']].extend(range(parameter['start'],parameter['end']))
        for arm in ('SN','SB'):
            batch=rows[rows.arm==arm];assert np.array_equal(batch.batch,np.arange(512)) and (batch.n==40960).all()
            assert np.array_equal(batch[['d0','d1','d2']].sum(axis=1),batch.n)
            q=p if arm=='SN' else np.ones(3)/3;total=512*40960
            observed=batch[['d0','d1','d2']].sum().to_numpy()
            assert np.all(abs(observed-total*q)<=6*np.sqrt(total*q*(1-q)))
            summary=np.load(directory/f'{arm}_summary.npz');mean=summary['mean'];gram=summary['gram']
            allrow=next(r for r in geometry if r['arm']==arm and r['block']=='ALL')
            independent_trace=float((np.square(batch['norm']).sum()-512*np.dot(mean,mean))/511)
            assert np.isclose(independent_trace,allrow['covariance_trace'],rtol=1e-6,atol=1e-12)
            assert np.isclose(np.trace(gram)/511,allrow['covariance_trace'],rtol=1e-10,atol=1e-14)
            assert np.isclose(sum(r['covariance_trace'] for r in geometry if r['arm']==arm and r['block']!='ALL'),allrow['covariance_trace'],rtol=1e-10,atol=1e-14)
            assert np.isclose(np.linalg.norm(mean-ref),allrow['bias_norm'],rtol=1e-10,atol=1e-14)
            for row in geometry:
                if row['arm']!=arm:continue
                assert row['bias_within_uncertainty'] and row['coordinates']>0 and row['rank_upper_bound']==511
                assert np.isclose(sum(row['eigenvalues']),row['covariance_trace'],rtol=1e-8,atol=1e-12)
                compact={k:v for k,v in row.items() if k!='eigenvalues'}
                compact['run']=directory.name;compact['bias_over_bootstrap95']=row['bias_norm']/max(row['mc_norm_bootstrap95'],1e-300)
                tables.append(compact)
            checks.append(dict(run=directory.name,arm=arm,k=512,full_coordinates=1614254,exact_identity_residual=receipt['exact_identity_residual'],
                independent_trace=independent_trace,producer_trace=allrow['covariance_trace'],batch_count_and_exposure_pass=True,data_pairing_pass=True))
        for block,ix in {**groups,'ALL':range(gd.shape[1])}.items():
            g=gd[:,ix]
            for i in range(3):
                for j in range(i+1,3):
                    cosine=float(np.dot(g[i],g[j])/max(np.linalg.norm(g[i])*np.linalg.norm(g[j]),1e-300))
                    cosines.append(dict(run=directory.name,seed=config['seed'],optimizer=config['optimizer'],epoch=config['epoch'],block=block,domain_a=i,domain_b=j,cosine=cosine))
    out=OUT/'analysis';out.mkdir(exist_ok=True)
    pd.DataFrame(tables).to_csv(out/'V1_full_coordinate_geometry.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'V1_numerical_cross_checks.csv',index=False)
    pd.DataFrame(cosines).to_csv(out/'V1_exact_domain_cosines.csv',index=False)
    pd.DataFrame(hashes).to_csv(out/'V1_artifact_hashes.csv',index=False)
    deterministic=[]
    for task in json.loads((OUT/'deterministic_manifest.json').read_text())['tasks']:
        args=task['command'];directory=Path(args[args.index('--run-dir')+1]);receipt=json.loads((directory/'receipt.json').read_text())
        assert receipt['status']=='PASS' and receipt['max_exact_pair_absolute_difference']<=1e-6
        deterministic.append(dict(run=directory.name,**receipt))
    base.dump(out/'V1_deterministic_summary.json',deterministic)
    base.dump(OUT/'V1_gate_numerical.json',dict(status='PASS',audits=16,k_per_arm=512,coordinates=1614254,
        deterministic_seeds=4,compact_artifact_hashes_verified=True,raw_gradient_matrices_hash_scope='producer hashes retained, shapes independently verified',
        numerical_checks=len(checks),geometry_rows=len(tables),code_sha=base.sha(__file__),
        mc_scope='Conservative plug-in bound is not a proof of equality; algebraic identity and full-population graph-gradient identity are separate evidence.'))

if __name__=='__main__':main()
