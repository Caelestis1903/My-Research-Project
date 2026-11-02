"""Frozen v3 configuration and the unmodified EMA02 numerical utilities."""
import json
from pathlib import Path
from .vendor.ema02.protocol import (setup, cpu_copy, equal, rng_state, restore_rng,
    send_memory, receive_memory, write_json, append_csv, update_ema, state_digest)
from .vendor.ema02.vendor.gradient_aug_20261031_v2.views import make_model
from .vendor.ema02.data import load_dataset as source_load
from .vendor.ema02.evaluate import evaluate, metrics

HERE = Path(__file__).resolve().parent
METHODS = {'A0':'adam','A1':'adam','P00':'adam','P10':'adam','P01':'eccm',
           'P11':'eccm','PCV':'cv','PCOLL':'collapsed'}

def load_protocol():
    source=json.loads((HERE/'vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31/resolved_protocol.json').read_text())
    datasets={}
    for d,path,rho,wd,target,kappa in [
        ('movielens','/home_new/xzf/datasets/ml-1m/ml-1m.csv',.9828205985452511,1e-4,0,.05),
        ('kuairand','/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv',.995,1e-5,2,.1)]:
        datasets[d]=dict(csv=path,source=source['datasets'][d],rho=rho,weight_decay=wd,target_domain=target,kappa=kappa)
    return dict(study_id='MDMT_INFORMATION_MOMENTUM_V3',datasets=datasets)

def config(arm,dataset,lambda_pair=0.):
    assert arm in METHODS and dataset in ('movielens','kuairand')
    assert lambda_pair in (0.,.05,.2)
    d=load_protocol()['datasets'][dataset]
    return dict(arm=arm,dataset=dataset,mode=METHODS[arm],dynamic=arm in ('P10','P11','PCV','PCOLL'),
        lambda_pair=lambda_pair,lambda_aug=.5,kappa=d['kappa'],target_domain=d['target_domain'],
        rho=d['rho'],weight_decay=d['weight_decay'],lr=.01,epochs=50,batch_size=40960)

def load_dataset(dataset,split_id='original',p=None):
    assert split_id=='original'
    return source_load(dataset,split_id,p or load_protocol())

def make_optimizer(model,c):
    from .optimizers import PairOptimizer
    return PairOptimizer(model,c)
