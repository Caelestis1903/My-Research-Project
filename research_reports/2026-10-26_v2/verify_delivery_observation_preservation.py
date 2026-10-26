"""Mock-only storage/authorization checks; no training or real timings/RPC."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast,copy,hashlib,json
import numpy as np
import torch
from experiments.mdmt_rcf_20261026 import delivery as d
from experiments.mdmt_rcf_20261026 import delivery_observation_preservation as p
from experiments.mdmt_rcf_20261026.protocol import PrefixBundle,PredictionBundle
from experiments.mdmt_rcf_20261026.state import state_hash,capture_rng

root=Path('artifacts/mdmt_rcf_20261026/v2_four_seed/verification_delivery_preservation').resolve()
# Distinct verification instance; real experiment roots and receipts are untouched.
root=root/('mock_'+str(__import__('time').time_ns()));root.mkdir(parents=True)
config={'config_hash':'MOCK_VERIFICATION_ONLY','paths':{'artifact':str(root)}}
methods=list(d.REQUIRED)+['RANDOM_FEATURE'];calls=[]
def bundle(seed):
    native=torch.nn.Linear(1,1);view=SimpleNamespace(row_id=np.arange(4096,dtype=np.int64))
    identity={'dataset':'movielens','seed':seed,'config_hash':config['config_hash'],'native_hash':'mock_native','pilot_hash':'mock_pilot','base_hash':'mock_base'}
    b=PrefixBundle('movielens',seed,{'Q_explore':view},{},None,None,native,{}, {},identity,None,SimpleNamespace(schema={'mock_feature_schema':1}),'cpu',root/'prefix'/f'seed{seed}')
    preds={m:PredictionBundle(m,{'Q_explore':np.full((4096,2),.5)}, {'Q_explore':np.full((4096,2),.5)}, {'selected':'MOCK_FROZEN_POLICY'}, {},dict(identity,method=m,reader_policy_hash='mock_'+m)) for m in methods}
    return b,preds
class Compiler:
    def __init__(self,*args,**kwargs):pass
    def parameter_counts(self):return {'mock_deployed_parameters':7}
def timing(b,pred,method):
    calls.append((b.seed,method))
    identity=p.measurement_identity(b,pred,method,{},config)
    if method=='RANDOM_FEATURE':return dict(status='NOT_MEASURED',reason='Fused/raw reference parity exceeds unchanged1e-6 acceptance',max_abs_probability_error=5.709476791126988e-6,retained_prediction_max_abs_error=5.709476791126988e-6,rows_hash=identity['rows_hash'])
    return dict(status='MEASURED',**identity['recipe'],**identity['environment'],row_ids=identity['row_ids'],rows_hash=identity['rows_hash'],schema_hash=identity['schema_hash'],repetitions=[.01+i*.001 for i in range(10)],max_abs_probability_error=0.,retained_prediction_max_abs_error=0.,native_dtype='float32',linear_readout_dtype='float64',labels_used=False,parameters={'mock_deployed_parameters':7})
def fingerprint(b,preds,formations):return dict(model=state_hash(b.native.state_dict()),rng=state_hash(capture_rng()))
def completions(b,preds):
    for m,pred in preds.items():
        path=root/'runs'/b.dataset/f'seed{b.seed}'/m/'TRAINING_COMPLETE.json'
        p.atomic_json(path,dict(identity=pred.identity,selected_policy=pred.selection,prefix_identity=b.identity))
def reject(fn):
    try:fn()
    except (ValueError,PermissionError,TypeError,FileExistsError):return True
    raise AssertionError('Expected rejection')

patches=[patch.object(d,'FrozenDeployment',Compiler),patch.object(d,'benchmark_predictor',timing),
         patch.object(d,'align_native_terminal_reporting',lambda *a:{'status':'MOCK_NO_CHANGE'}),
         patch.object(d,'_training_costs',lambda *a:[{'component':'MOCK_ONLY','seconds':.1,'scope':'synthetic verification'}]),
         patch.object(d,'_training_parameters',lambda *a:{}),patch.object(d,'_observation_fingerprint',fingerprint)]
for item in patches:item.start()
try:
    b,preds=bundle(5201);completions(b,preds)
    first=d.collect_delivery(b,preds,{}, {},config,benchmark=False)
    again=d.collect_delivery(b,preds,{}, {},config,benchmark=False)
    assert not calls and first['required_measurement_status']=='INCOMPLETE' and all(x['status']=='NOT_MEASURED' for x in again['inference'].values())
    measured=d.collect_delivery(b,preds,{}, {},config,benchmark=True)
    assert len(calls)==10 and measured['required_measurement_status']=='COMPLETE' and measured['status']=='PARTIAL_OBSERVATIONS'
    files=list((root/'delivery'/'measurement_records').rglob('*.json'));before={str(f):(_hash:=hashlib.sha256(f.read_bytes()).hexdigest(),f.stat().st_mtime_ns) for f in files}
    false=d.collect_delivery(b,preds,{}, {},config,benchmark=False)
    false2=d.collect_delivery(b,preds,{}, {},config,benchmark=False)
    repeated=d.collect_delivery(b,preds,{}, {},config,benchmark=True,allowed_benchmark_methods=list(d.REQUIRED))
    assert len(calls)==10 and not false['current_call_benchmark_requested']
    for m in d.REQUIRED:
        assert false['inference'][m]['repetitions']==measured['inference'][m]['repetitions']
        assert false['inference'][m]['measurement_source']==measured['inference'][m]['measurement_source']
    assert false['inference']['RANDOM_FEATURE']['reason']==measured['inference']['RANDOM_FEATURE']['reason']
    assert false['inference']['RANDOM_FEATURE']['max_abs_probability_error']==5.709476791126988e-6
    assert before=={str(f):(hashlib.sha256(f.read_bytes()).hexdigest(),f.stat().st_mtime_ns) for f in (root/'delivery'/'measurement_records').rglob('*.json')}
    assert reject(lambda:p.atomic_json(files[0],{'changed':'forbidden'},immutable=True))
    # Only the one missing, explicitly allowed required method is benchmarked.
    b2,p2=bundle(5202);completions(b2,p2);n=len(calls)
    r=d.collect_delivery(b2,p2,{}, {},config,benchmark=True,allowed_benchmark_methods=['NATIVE'])
    d.collect_delivery(b2,p2,{}, {},config,benchmark=True,allowed_benchmark_methods=['NATIVE'])
    assert len(calls)==n+1 and calls[-1]==(5202,'NATIVE') and len(r['missing_required_methods'])==8
    # Legacy snapshot identity and file SHA are checked against real mock completion receipts.
    b3,p3=bundle(5203);completions(b3,p3)
    snapshot=root/'attempts'/'legacy_mock.json'
    legacy=dict(dataset=b3.dataset,seed=b3.seed,source={'prefix_identity':b3.identity,'module_sha256':'ORIGINAL_MOCK_SOURCE'},benchmark_requested=True,immutability={'parameters_modes_gradients_normalizers_rng_unchanged':True},inference={'NATIVE':timing(b3,p3['NATIVE'],'NATIVE')})
    p.atomic_json(snapshot,legacy);meta={'dataset':b3.dataset,'seed':b3.seed,'path':str(snapshot),'sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest()};n=len(calls)
    imported=d.collect_delivery(b3,p3,{}, {},config,benchmark=False,preserved_measurements=[meta])
    assert len(calls)==n and imported['inference']['NATIVE']['status']=='MEASURED'
    assert imported['inference']['NATIVE']['measurement_source']['origin']['reported_measurement_source']==legacy['source']
    assert reject(lambda:d.collect_delivery(b3,p3,{}, {},config,benchmark=False,preserved_measurements=[dict(meta,sha256='BAD')]))
    identity=p.measurement_identity(b3,p3['NATIVE'],'NATIVE',{},config)
    broken=copy.deepcopy(legacy['inference']['NATIVE']);broken['row_ids'][0]=-1
    assert not p.validate_measurement(broken,identity)[0]
    broken=copy.deepcopy(legacy['inference']['NATIVE']);broken['torch_version']='wrong'
    assert not p.validate_measurement(broken,identity)[0]
    # A policy mismatch is not allowed to reuse the archived method timing.
    p3['NATIVE'].selection={'selected':'DIFFERENT_POLICY'}
    archive=p.MeasurementArchive(b3,p3,{},config,root/'delivery'/b3.dataset/f'seed{b3.seed}',snapshots=[meta])
    assert archive.lookup('NATIVE') is None and archive.rejections
    p3['NATIVE'].selection={'selected':'MOCK_FROZEN_POLICY'}
    bad=copy.deepcopy(legacy);bad['source']['prefix_identity']['config_hash']='WRONG_CONFIG';badpath=root/'attempts'/'wrong_config.json';p.atomic_json(badpath,bad)
    badmeta=dict(meta,path=str(badpath),sha256=hashlib.sha256(badpath.read_bytes()).hexdigest())
    b4,p4=bundle(5204);completions(b4,p4)
    assert p.MeasurementArchive(b4,p4,{},config,root/'delivery'/b4.dataset/f'seed{b4.seed}',snapshots=[dict(badmeta,seed=b4.seed)]).lookup('NATIVE') is None
    # Formation identities missing from legacy JSON require a matching actual receipt.
    formed={'RCF':SimpleNamespace(identity={'functions_hash':'MOCK_FROZEN_FORMATION'})}
    p.atomic_json(root/'runs'/b4.dataset/f'seed{b4.seed}'/'RCF'/'formation.json',{'identity':formed['RCF'].identity})
    formed_snapshot=root/'attempts'/'formed_mock.json'
    formed_legacy=dict(dataset=b4.dataset,seed=b4.seed,source={'prefix_identity':b4.identity},benchmark_requested=True,immutability={'parameters_modes_gradients_normalizers_rng_unchanged':True},inference={'RCF':timing(b4,p4['RCF'],'RCF')})
    p.atomic_json(formed_snapshot,formed_legacy)
    formed_meta=dict(dataset=b4.dataset,seed=b4.seed,path=str(formed_snapshot),sha256=p._sha(formed_snapshot))
    assert p.MeasurementArchive(b4,p4,formed,config,root/'delivery'/b4.dataset/f'seed{b4.seed}',snapshots=[formed_meta]).lookup('RCF')['status']=='MEASURED'
    formed['RCF'].identity={'functions_hash':'WRONG_FORMATION'}
    assert p.MeasurementArchive(b4,p4,formed,config,root/'delivery'/b4.dataset/f'seed{b4.seed}',snapshots=[formed_meta]).lookup('RCF') is None
    assert not list(root.rglob('*.data_tmp'))
finally:
    for item in reversed(patches):item.stop()
# Typed authorization guards are pure checks; no service context or RPC is created.
b0,p0=bundle(5201)
ctx=dict(bundle=b0,predictions=p0,formations={},diagnostics={},config={'config_hash':p.CONFIG_HASH},root=str(p.ARTIFACT_ROOT),frozen=True)
key=p.validate_runtime_state(ctx,pid=p.WORKER_PIDS['movielens:5201'],parent_pid=p.SERVICE_PID,frozen_sha=p.FROZEN_SHA)
assert reject(lambda:p.validate_runtime_state(ctx,pid=1,parent_pid=p.SERVICE_PID,frozen_sha=p.FROZEN_SHA))
assert reject(lambda:p.validate_runtime_state(ctx,pid=p.WORKER_PIDS[key],parent_pid=1,frozen_sha=p.FROZEN_SHA))
assert reject(lambda:p.validate_runtime_state(ctx,pid=p.WORKER_PIDS[key],parent_pid=p.SERVICE_PID,frozen_sha='BAD'))
badctx=dict(ctx,config={'config_hash':'BAD'});assert reject(lambda:p.validate_runtime_state(badctx,pid=p.WORKER_PIDS[key],parent_pid=p.SERVICE_PID,frozen_sha=p.FROZEN_SHA))
token=dict(verdict='PASS',blocking_issues=[],service_pid=p.SERVICE_PID,worker_pids=p.WORKER_PIDS,config_hash=p.CONFIG_HASH,frozen_manifest_sha256=p.FROZEN_SHA,operation='fill_missing_required',source_sha256={n:p._sha(Path(p.__file__).with_name(n)) for n in ('delivery.py','delivery_observation_preservation.py','readers.py')},design_sha256=p._sha(p.WORKSPACE/'research_reports/2026-10-26_v2/DESIGN_DELIVERY_OBSERVATION_PRESERVATION_ADDENDUM.md'),allowed_methods_by_worker={key:['NATIVE']})
assert p.validate_fill_authorization(token,key,p0)==['NATIVE']
wrong=copy.deepcopy(token);wrong['allowed_methods_by_worker'][key]=['RANDOM_FEATURE'];assert reject(lambda:p.validate_fill_authorization(wrong,key,p0))
wrong=copy.deepcopy(token);wrong['verdict']='FAIL';assert reject(lambda:p.validate_fill_authorization(wrong,key,p0))
wrong=copy.deepcopy(token);wrong['source_sha256']['delivery.py']='BAD';assert reject(lambda:p.validate_fill_authorization(wrong,key,p0))
assert p.run_on_reader_reload()['status']=='NO_OP'
oldroot=Path('research_reports/2026-10-26_v2/evidence/pre_delivery_preservation');ast_audit={}
for name in ('delivery.py','readers.py'):
    parse=lambda x:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(x).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old,new=parse((oldroot/name).read_text()),parse((Path('experiments/mdmt_rcf_20261026')/name).read_text())
    changed=[k for k in old if old[k]!=new.get(k)]
    assert changed==(['collect_delivery'] if name=='delivery.py' else []),changed
    ast_audit[name]=dict(changed_original_definitions=changed,all_other_original_definitions_unchanged=True)
result=dict(status='BASIC_CHECKS_COMPLETE',scope='Mock observations/authorization/storage only; all timing values synthetic',formal_training=False,actual_benchmarks=0,resident_service_contacted=False,model_files_written=0,
    true_then_false_preserves_10_values_and_sources=True,false_never_benchmarks=True,repeated_fill_idempotent=True,only_missing_authorized_required_benchmarked=True,
    random_parity_failure_preserved=True,required_and_all_method_status_separate=True,immutable_records_unchanged=True,atomic_json_no_temporary_leftovers=True,
    legacy_snapshot_sha_prefix_policy_rows_environment_checked=True,wrong_pid_parent_config_frozen_authorization_whitelist_rejected=True,client_no_op=True,
    ast_audit=ast_audit,fixture_root=str(root),source_sha256={n:p._sha(Path(p.__file__).with_name(n)) for n in ('delivery.py','delivery_observation_preservation.py','readers.py')})
Path('research_reports/2026-10-26_v2/DELIVERY_PRESERVATION_BASIC_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
