"""Bounded ML5201 allowlist extension; mock observations only, no RPC/timing."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast,copy,json,time
import numpy as np
import torch
from experiments.mdmt_rcf_20261026 import delivery as d
from experiments.mdmt_rcf_20261026 import delivery_observation_preservation as h
from experiments.mdmt_rcf_20261026.protocol import PrefixBundle,PredictionBundle
from experiments.mdmt_rcf_20261026.state import state_hash,capture_rng
root=Path('artifacts/mdmt_rcf_20261026/v2_four_seed/verification_ml5201_completion')/str(time.time_ns());root=root.resolve();root.mkdir(parents=True)
config={'config_hash':'MOCK_COMPLETION_ONLY','paths':{'artifact':str(root)}}
methods=list(d.REQUIRED)+list(h.ML5201_COMPLETION_METHODS)
b=PrefixBundle('movielens',5201,{'Q_explore':SimpleNamespace(row_id=np.arange(4096,dtype=np.int64))},{},None,None,torch.nn.Linear(1,1),{}, {},{'native_hash':'MOCK_NATIVE','pilot_hash':'MOCK_PILOT','config_hash':config['config_hash']},None,SimpleNamespace(schema={'mock':1}),'cpu',root/'prefix')
preds={m:PredictionBundle(m,{}, {},{'selected':'MOCK_POLICY'}, {},dict(b.identity,method=m,reader_policy_hash=m)) for m in methods}
sources={n:h._sha(Path(h.__file__).with_name(n)) for n in ('delivery.py','delivery_observation_preservation.py','readers.py')}
original_design=h.WORKSPACE/'research_reports/2026-10-26_v2/DESIGN_DELIVERY_OBSERVATION_PRESERVATION_ADDENDUM.md'
addendum=h.WORKSPACE/'research_reports/2026-10-26_v2/DESIGN_ML5201_MISSING_OBSERVATION_COMPLETION_ADDENDUM.md'
token=dict(verdict='PASS',blocking_issues=[],service_pid=h.SERVICE_PID,worker_pids=h.WORKER_PIDS,config_hash=h.CONFIG_HASH,frozen_manifest_sha256=h.FROZEN_SHA,operation='fill_missing_required',source_sha256=sources,design_sha256=h._sha(original_design),context_completion_extension=h.ML5201_COMPLETION_EXTENSION,context_completion_addendum_sha256=h._sha(addendum),allowed_methods_by_worker={key:list(d.REQUIRED) for key in h.WORKER_PIDS})
token['allowed_methods_by_worker']['movielens:5201']=methods
assert h.validate_fill_authorization(token,'movielens:5201',preds)==methods
for key in h.WORKER_PIDS:
    if key!='movielens:5201':assert h.validate_fill_authorization(token,key,preds)==list(d.REQUIRED)
def rejected(fn):
    try:fn()
    except (PermissionError,ValueError,TypeError):return True
    raise AssertionError('Expected rejection')
for worker in h.WORKER_PIDS:
    if worker=='movielens:5201':continue
    wrong=copy.deepcopy(token);wrong['allowed_methods_by_worker'][worker].append('PILOT_PROB')
    assert rejected(lambda:h.validate_fill_authorization(wrong,'movielens:5201',preds))
wrong=copy.deepcopy(token);wrong.pop('context_completion_extension');assert rejected(lambda:h.validate_fill_authorization(wrong,'movielens:5201',preds))
wrong=copy.deepcopy(token);wrong['context_completion_addendum_sha256']='BAD';assert rejected(lambda:h.validate_fill_authorization(wrong,'movielens:5201',preds))
wrong=copy.deepcopy(token);wrong['allowed_methods_by_worker']['movielens:5201'].append('SOURCE_INIT_RANDOM');assert rejected(lambda:h.validate_fill_authorization(wrong,'movielens:5201',dict(preds,SOURCE_INIT_RANDOM=preds['NATIVE'])))
base=copy.deepcopy(token);base.pop('context_completion_extension');base.pop('context_completion_addendum_sha256');base['allowed_methods_by_worker']['movielens:5201']=list(d.REQUIRED);assert h.validate_fill_authorization(base,'movielens:5201',preds)==list(d.REQUIRED)
class Compiler:
    def __init__(self,*args):pass
    def parameter_counts(self):return {'mock_parameters':3}
calls=[]
def benchmark(bundle,pred,method):
    calls.append(method);identity=h.measurement_identity(bundle,pred,method,{},config)
    if method=='RANDOM_FEATURE':return dict(status='NOT_MEASURED',reason='Fused/raw reference parity exceeds unchanged1e-6 acceptance',max_abs_probability_error=3e-6,retained_prediction_max_abs_error=3e-6,rows_hash=identity['rows_hash'])
    return dict(status='MEASURED',**identity['recipe'],**identity['environment'],rows_hash=identity['rows_hash'],row_ids=identity['row_ids'],schema_hash=identity['schema_hash'],repetitions=[.02+i*.001 for i in range(10)],max_abs_probability_error=0.,retained_prediction_max_abs_error=0.,native_dtype='float32',linear_readout_dtype='float64',labels_used=False,parameters={'mock_parameters':3})
def fingerprint(*args):return {'model':state_hash(b.native.state_dict()),'rng':state_hash(capture_rng())}
with patch.object(d,'FrozenDeployment',Compiler),patch.object(d,'benchmark_predictor',benchmark),patch.object(d,'align_native_terminal_reporting',lambda *a:{'status':'MOCK'}),patch.object(d,'_training_costs',lambda *a:[]),patch.object(d,'_training_parameters',lambda *a:{}),patch.object(d,'_observation_fingerprint',fingerprint):
    initial=d.collect_delivery(b,preds,{}, {},config,benchmark=True,allowed_benchmark_methods=list(d.REQUIRED))
    assert calls==list(d.REQUIRED)
    before=len(calls);filled=d.collect_delivery(b,preds,{}, {},config,benchmark=True,allowed_benchmark_methods=methods)
    assert calls[before:]==list(h.ML5201_COMPLETION_METHODS)
    assert filled['required_measurement_status']=='COMPLETE' and filled['status']=='PARTIAL_OBSERVATIONS'
    repeated=d.collect_delivery(b,preds,{}, {},config,benchmark=True,allowed_benchmark_methods=methods)
    assert len(calls)==16 and repeated['measurement_actions']['RANDOM_FEATURE']=='PRESERVED_EXISTING_PARITY_FAILURE'
    assert repeated['inference']['RANDOM_FEATURE']['max_abs_probability_error']==3e-6
    for method in d.REQUIRED:assert repeated['inference'][method]['repetitions']==initial['inference'][method]['repetitions']
    # The newly explicit skip also covers a previously observed REQUIRED parity failure.
    second=copy.copy(b);second.identity=dict(b.identity,native_hash='SECOND_MOCK_IDENTITY');second.native=b.native
    second_preds={m:copy.copy(pred) for m,pred in preds.items()}
    for m,pred in second_preds.items():pred.identity=dict(second.identity,method=m,reader_policy_hash=m)
    archive=h.MeasurementArchive(second,second_preds,{},config,root/'delivery'/b.dataset/f'seed{b.seed}')
    failure=dict(status='NOT_MEASURED',reason='Fused/raw reference parity exceeds unchanged1e-6 acceptance',max_abs_probability_error=2e-6,retained_prediction_max_abs_error=2e-6)
    archive.save('NATIVE',failure,{'kind':'MOCK_EXISTING_FAILURE'})
    before=len(calls);skipped=d.collect_delivery(second,second_preds,{}, {},config,benchmark=True,allowed_benchmark_methods=['NATIVE'])
    assert len(calls)==before and skipped['measurement_actions']['NATIVE']=='PRESERVED_EXISTING_PARITY_FAILURE'
    assert skipped['required_measurement_status']=='INCOMPLETE'
oldroot=Path('research_reports/2026-10-26_v2/evidence/pre_ml5201_observation_completion');audits={}
for name in ('delivery.py','readers.py'):
    defs=lambda path:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    aa,bb=defs(oldroot/name),defs(Path('experiments/mdmt_rcf_20261026')/name);changed=[k for k in aa if aa[k]!=bb.get(k)]
    assert changed==(['collect_delivery'] if name=='delivery.py' else [])
    audits[name]={'changed_original_definitions':changed,'all_other_original_definitions_unchanged':True}
result=dict(status='BASIC_CHECKS_COMPLETE',scope='Synthetic timing records and pure token validation only',formal_training=False,actual_benchmarks=0,resident_service_contacted=False,model_files_written=0,token_extension=h.ML5201_COMPLETION_EXTENSION,original_design_sha256=h._sha(original_design),addendum_sha256=h._sha(addendum),unique_ml5201_extra_methods=list(h.ML5201_COMPLETION_METHODS),all_other_worker_whitelists_remain_required9=True,missing7_called_once=True,required9_reused=True,repeat_idempotent=True,known_parity_failures_skipped_including_required=True,extra_failure_retained_not_measured=True,ast_audit=audits,source_sha256=sources,reader_source_unchanged=True,fixture_root=str(root))
Path('research_reports/2026-10-26_v2/ML5201_OBSERVATION_COMPLETION_BASIC_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
