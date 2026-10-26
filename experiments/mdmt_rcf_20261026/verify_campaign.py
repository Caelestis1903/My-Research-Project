"""Minimal campaign checks: complete seed gate and weighted paired AUC."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from .artifacts import _weighted_auc_plan,_weighted_auc,paired_bootstrap,_json
from .campaign import decide_stage,build_run_manifest,gpu_snapshot

def run_required(config,output_root=None):
    rng=np.random.default_rng(7391)
    y=np.tile(np.array([[0,1],[1,0],[1,1],[0,0]],float),(15,1));p=np.round(rng.uniform(.1,.9,y.shape),1);weights=rng.integers(0,4,len(y))
    errors=[]
    for q in range(2):errors.append(abs(_weighted_auc(_weighted_auc_plan(y[:,q],p[:,q]),weights)-roc_auc_score(y[:,q],p[:,q],sample_weight=weights)))
    assert max(errors)<1e-14
    partial=decide_stage('s2',{},config)
    assert not partial['complete'] and all(x['status']=='BLOCKED_INCOMPLETE_STAGE' for x in partial['datasets'].values())
    manifest=build_run_manifest(config,{'status':'NOT_RUN'})
    assert len([r for r in manifest['runs'] if r['stage']=='s0'])==len(config['datasets'])*len(config['seeds']['development'])
    assert len([r for r in manifest['runs'] if r['stage']=='s2'])==3*len(config['datasets'])*len(config['seeds']['development'])
    from .scale_probe import run_scale_numeric
    # Numeric bridge uses only synthetic FP64 vectors, no experimental training.
    numeric=run_scale_numeric(config,output_root or Path(config['paths']['artifact']))
    simple=verify_simple_raw_callbacks(config)
    result=dict(status='PASS',simple_raw_callbacks=simple,scope='partial seed gate, registered matrix, tie-correct weighted AUC, Adam numeric bridge',auc_max_error=max(errors),s0_runs=len(config['datasets'])*len(config['seeds']['development']),s2_runs=3*len(config['datasets'])*len(config['seeds']['development']),scale_numeric=numeric['status'],formal_training=False,models_written=0)
    if output_root is not None:_json(Path(output_root)/'VERIFY_CAMPAIGN.json',result)
    return result

verify=run_required

def verify_simple_raw_callbacks(config):
    """Actual raw MovieLens minibatch through native and three downstream paths."""
    import copy
    import torch
    from .protocol import RoleBundle,RoleView,PrefixBundle
    from .prefix import make_native,PilotAccess,configure
    from .baselines import DirectHead,predict_direct_models
    from .vendor_snapshot.mdmt_dfd_20261026.functions import PilotBank
    from .campaign import _simple_prediction
    configure();data=RoleBundle('movielens')
    ids=np.concatenate([data.ids('fit',d)[:24] for d in range(3)])
    native,_=make_native(data.schema,5201,'movielens','cpu');native.eval()
    with torch.no_grad():packet=native.encode(torch.as_tensor(np.array(data.x[ids])))
    view=RoleView(ids,data.domain[ids],np.asarray(data.y[ids]),packet.h.numpy(),packet.logits.numpy(),np.zeros((len(ids),2),dtype=np.int64))
    b=PrefixBundle('movielens',5201,{'fit':view},{},None,None,native,{}, {},{},None,data,'cpu',Path(config['paths']['artifact'])/'verification_simple_raw')
    bank=PilotBank(lambda s,q,k:4100+31*s+q)
    with torch.no_grad():
        for head in bank.heads.values():head.output.weight.fill_(.03)
    b.pilots=PilotAccess(b,bank,[copy.deepcopy(bank),copy.deepcopy(bank)],copy.deepcopy(bank))
    for t in range(3):
        for q in range(2):
            head=DirectHead(np.zeros(258),np.ones(258),710+3*q+t,.5)
            with torch.no_grad():head.output.weight.fill_(.02)
            b.base_models[t,q]=head
    b.base_logits={'fit':predict_direct_models(b,b.base_models,'fit')}
    results={}
    for method in ('NATIVE','LOCAL','B_REFIT'):
        pred=_simple_prediction(b,method);receipt=pred.diagnostics['_raw_x_validate'](b)
        assert receipt['status']=='PASS' and receipt['rows']==len(ids)
        results[method]=receipt
    return dict(status='PASS',actual_rows=len(ids),training_performed=False,methods=results)

def verify_capture_fixture(config):
    """Read-only fixture export on a real 72-row minibatch; no fitting."""
    import copy
    import torch
    from .protocol import RoleBundle,RoleView,PrefixBundle,LabelLedger
    from .prefix import make_native,PilotAccess,configure
    from .vendor_snapshot.mdmt_dfd_20261026.functions import PilotBank
    from .fixture_export import export_reader_fixture
    configure();data=RoleBundle('movielens');native,_=make_native(data.schema,5201,'movielens','cpu');native.eval();views={}
    for role in ('fit','P_read'):
        ids=np.concatenate([data.ids(role,d)[:24] for d in range(3)])
        with torch.no_grad():packet=native.encode(torch.as_tensor(np.array(data.x[ids])))
        views[role]=RoleView(ids,data.domain[ids],np.asarray(data.y[ids]),packet.h.numpy(),packet.logits.numpy(),np.zeros((len(ids),2),dtype=np.int64))
    cfg=dict(config,paths=dict(config['paths'],artifact=str(Path(config['paths']['artifact'])/'verification_fixture_rpc')))
    b=PrefixBundle('movielens',5201,views,{'P_read':views['P_read'].z.copy()},None,None,native,{}, {},{'config_hash':config['config_hash']},LabelLedger(data),data,'cpu')
    bank=PilotBank(lambda s,q,k:5700+31*s+q);b.pilots=PilotAccess(b,bank,[copy.deepcopy(bank),copy.deepcopy(bank)],copy.deepcopy(bank))
    receipt=export_reader_fixture(b,cfg)
    assert receipt['shape']==[24,64] and receipt['models_unchanged'] and not receipt['solver_executed']
    with np.load(receipt['file']['path'],allow_pickle=False) as arrays:
        mask=views['P_read'].domain==0
        explicit=np.concatenate([b.pilots.hidden('P_read',s,0,standardized=True)[mask] for s in (1,2)],1).astype(np.float64)
        assert np.array_equal(arrays['x'],explicit) and np.array_equal(arrays['row_id'],views['P_read'].row_id[mask])
    return {'status':'PASS','real_minibatch_rows':72,'exported_rows':24,'columns':64,'exact_column_and_row_order':True,'models_unchanged':True,'training_performed':False}


def verify_four_seed_recovery(config):
    """Only the newly required gate edges, using synthetic metric records."""
    import copy
    from .campaign import _compare,S1_METHODS,S2_METHODS,MANDATORY_S3,MINIMUM_TRAINING_SEEDS
    seeds=config['seeds']['development']
    assert len(set(seeds))>=MINIMUM_TRAINING_SEEDS
    def metric(ll):
        return dict(macro_logloss=ll,macro_auc=.7,cells=[dict(domain=d,task=q,logloss=ll,auc=.7) for q in range(2) for d in range(3)])
    reference={s:metric(.5) for s in seeds};candidate={s:metric(.499) for s in seeds}
    assert _compare(candidate,reference,seeds,config)['passed']
    bad=copy.deepcopy(candidate);bad[seeds[-1]]=metric(.501)
    assert not _compare(bad,reference,seeds,config)['tests']['all_seeds_negative']
    assert not _compare(candidate,reference,seeds[:2],config)['passed']
    entries={d:{s:dict(metrics={m:metric(.5) for m in S1_METHODS+S2_METHODS}) for s in seeds} for d in config['datasets']}
    s2=decide_stage('s2',entries,config)
    assert s2['complete']
    assert set(MANDATORY_S3['kuairand'])<=set(s2['datasets']['kuairand']['triggered_methods'])
    missing=copy.deepcopy(entries);missing['kuairand'].pop(seeds[-1])
    assert not decide_stage('s2',missing,config)['complete']
    old=copy.deepcopy(config);old['seeds']['development']=seeds[:2]
    assert not decide_stage('s2',entries,old)['complete']
    s3=decide_stage('s3',entries,config)
    assert not s3['complete'] and s3['datasets']['kuairand']['status']=='BLOCKED_INCOMPLETE_S3'
    for seed in seeds:
        entries['kuairand'][seed]['s2_decision']=s2['datasets']['kuairand']
        entries['kuairand'][seed]['metrics'].update({m:metric(.5) for m in MANDATORY_S3['kuairand']})
    assert decide_stage('s3',entries,config)['complete']
    del entries['kuairand'][seeds[-1]]['metrics']['K1_WIDE']
    assert not decide_stage('s3',entries,config)['complete']
    return dict(status='PASS',scope='four distinct seeds; fourth-seed adverse direction; 3/4 incomplete; old two-seed config blocked; KR mandatory controls union and missing-control blocking',synthetic_metrics_only=True,formal_training=False,model_files_written=0)
