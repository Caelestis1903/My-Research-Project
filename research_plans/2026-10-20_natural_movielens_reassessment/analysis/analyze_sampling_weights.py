"""Exact scalar weight context; not an estimator of gradient or update covariance."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ART=Path('/home_new/xzf_natural_movielens_reassessment/data')
OUT=Path(__file__).resolve().parent/'sampling_weights';OUT.mkdir(exist_ok=True)
rows=[];summary=[]
for population in ['movielens_natural','movielens_artificial_d2_fixed2022','movielens_natural_matched_n187500']:
    path=ART/(population+'.json');data=json.loads(path.read_text())['data']
    n=np.array([data['train_domain_counts'][str(d)] for d in range(3)]);p=n/n.sum()
    for arm in ['SN','SB','EN','EB']:
        alpha=p if arm[0]=='S' else np.ones(3)/3
        q=p if arm[1]=='N' else np.ones(3)/3;w=alpha/q
        expected_mean=float(q@w);second_moment=float(q@np.square(w))
        assert abs(expected_mean-1)<1e-12
        summary.append(dict(population=population,arm=arm,train_N=int(n.sum()),
            nominal_batch_size=40960,weight_expectation=expected_mean,weight_second_moment=second_moment,
            IID_scalar_weight_ESS_fraction=expected_mean**2/second_moment,
            ESS_definition='B*(E_q w)^2/E_q(w^2); scalar weight diagnostic only, not actual gradient covariance or unique coverage',
            applicability='base F0 population, IID domain proposal; not N19/CM19 main-after-meta or shuffled-loader covariance',
            data_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for d in range(3):
            rows.append(dict(population=population,arm=arm,domain=d,train_N=int(n[d]),p=p[d],alpha=alpha[d],q=q[d],w=w[d],
                expected_domain_count_in_40960=40960*q[d],count_SD_IID=np.sqrt(40960*q[d]*(1-q[d])),
                effective_target_domain_coefficient=q[d]*w[d],effective_target_cell_coefficient=q[d]*w[d]/2))
pd.DataFrame(rows).to_csv(OUT/'exact_domain_coefficients.csv',index=False)
table=pd.DataFrame(summary);table.to_csv(OUT/'scalar_weight_summary.csv',index=False)
print(table[['population','arm','weight_second_moment','IID_scalar_weight_ESS_fraction']].to_string(index=False))
