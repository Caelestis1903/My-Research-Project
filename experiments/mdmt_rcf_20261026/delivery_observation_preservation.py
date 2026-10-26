"""Immutable measurement data, identity-checked reuse, and a fixed fill hook.

This module never trains, exports predictions, bootstraps, changes a policy or
changes a compiler. It stores existing observations or invokes the unchanged
benchmark only for missing, explicitly authorized methods.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import numpy as np
import torch
from .artifacts import _plain,_sha
from .state import array_hash,digest

REVISION='DELIVERY_MEASUREMENT_PRESERVATION_V1'
ML5201_COMPLETION_EXTENSION='ML5201_MISSING_OBSERVATION_COMPLETION_V1'
ML5201_COMPLETION_METHODS=('LOCAL','ETA_DIRECT_MATCHED','PILOT_PROB','PILOT_LOGIT',
                         'FULL_SOURCE','FULL_TARGET','RANDOM_FEATURE')
WORKSPACE=Path('/home/xzf/Code/phase0-3')
ARTIFACT_ROOT=WORKSPACE/'artifacts/mdmt_rcf_20261026/v2_four_seed'
SERVICE_PID=2088990
CONFIG_HASH='cefb6d934c774bd78a438e4fba5c560e934f4c56ab09fbb3dfe389f4f4842fd6'
FROZEN_SHA='09def5e2f43786f9c2875e5ae466845a10ec7db47a3c6edd28379827b72e927a'
WORKER_PIDS={'movielens:5201':2089374,'kuairand:5201':2090829,
             'movielens:5202':2091226,'kuairand:5202':2093268,
             'movielens:5203':2093543,'kuairand:5203':2093912,
             'movielens:5204':2094033,'kuairand:5204':2094138}


def _bytes(value):
    return (json.dumps(_plain(value),sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()


def atomic_json(path,value,*,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);raw=_bytes(value)
    if immutable and path.exists():
        if path.read_bytes()!=raw:raise FileExistsError('Immutable observation differs: '+str(path))
        return
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.data_tmp')
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        if immutable:
            try:os.link(tmp,path)
            except FileExistsError:
                if path.read_bytes()!=raw:raise
        else:os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def measurement_identity(bundle,pred,method,formations,config):
    rows=np.asarray(bundle.views['Q_explore'].row_id[:4096],np.int64)
    source=ast.parse(Path(__file__).with_name('delivery.py').read_text())
    numerical=[ast.dump(n,include_attributes=False) for n in source.body
               if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('FrozenDeployment','benchmark_predictor')]
    return _plain(dict(dataset=bundle.dataset,seed=bundle.seed,config_hash=config['config_hash'],
        prefix_identity=bundle.identity,method=method,prediction_identity=pred.identity,selection=pred.selection,
        formation_identity=None if method not in formations else formations[method].identity,
        row_ids=rows.tolist(),rows_hash=array_hash(rows),schema_hash=digest(bundle.data.schema),
        recipe=dict(batch_size=4096,warmups=3,measured_repetitions=10,
            row_rule='first4096 original-order natural Q_explore row_ids',
            parity_tolerance=1e-6,transfer_policy='preloaded CPU FP32 raw X to native device and final p back to CPU, both inside timing'),
        environment=dict(device=str(next(bundle.native.parameters()).device),threads=torch.get_num_threads(),
            torch_version=str(torch.__version__),python_version=platform.python_version(),cuda_runtime=torch.version.cuda),
        numerical_execution_AST_hash=digest(numerical)))


def validate_measurement(value,identity):
    if value.get('status')!='MEASURED':return False,'Not a successful measurement'
    recipe=identity['recipe'];env=identity['environment']
    for key in ('batch_size','warmups','measured_repetitions','row_rule','transfer_policy','parity_tolerance'):
        if value.get(key)!=recipe[key]:return False,'Benchmark recipe differs: '+key
    if value.get('rows_hash')!=identity['rows_hash'] or value.get('row_ids')!=identity['row_ids']:
        return False,'Natural row identities/order differ'
    if len(identity['row_ids'])!=4096 or len(set(identity['row_ids']))!=4096:return False,'Not4096 natural distinct rows'
    if value.get('schema_hash')!=identity['schema_hash']:return False,'Feature schema differs'
    for key,current in env.items():
        if value.get(key)!=current:return False,'Execution environment differs: '+key
    times=value.get('repetitions',[])
    if len(times)!=10 or not all(isinstance(x,(int,float)) and np.isfinite(x) and x>0 for x in times):
        return False,'Ten positive finite actual times are required'
    for key in ('max_abs_probability_error','retained_prediction_max_abs_error'):
        v=value.get(key)
        if not isinstance(v,(int,float)) or not np.isfinite(v) or v>1e-6 or v<0:return False,'Parity not verified: '+key
    if value.get('labels_used') is not False:return False,'Input-only measurement not established'
    if value.get('native_dtype')!='float32' or value.get('linear_readout_dtype')!='float64':return False,'Execution precision differs'
    return True,'Verified actual3+10 raw-X measurement'


class MeasurementArchive:
    def __init__(self,bundle,predictions,formations,config,delivery_root,*,snapshots=()):
        self.bundle=bundle;self.predictions=predictions;self.formations=formations;self.config=config
        self.root=Path(delivery_root);self.artifact=Path(config['paths']['artifact']).resolve()
        self.records=self.artifact/'delivery'/'measurement_records'/bundle.dataset/f'seed{bundle.seed}'
        self.records.mkdir(parents=True,exist_ok=True)
        self.identities={m:measurement_identity(bundle,p,m,formations,config) for m,p in predictions.items()}
        self.available={};self.rejections=[]
        for path in sorted(self.records.glob('*.json')):
            try:
                record=json.loads(path.read_text());method=record.get('identity',{}).get('method')
                if method in self.identities and record['identity']==self.identities[method]:self._accept_record(path,record)
            except (ValueError,KeyError,TypeError) as error:self.rejections.append(dict(path=str(path),reason=str(error)))
        current=self.root/'COSTS_PARAMETERS_AND_INFERENCE.json'
        if current.exists():self._import_legacy(current,dict(kind='previous_canonical',sha256=_sha(current),source_mtime_ns=current.stat().st_mtime_ns))
        for metadata in snapshots:
            if metadata.get('dataset')!=bundle.dataset or int(metadata.get('seed',-1))!=bundle.seed:continue
            path=Path(metadata.get('path',metadata.get('snapshot_path',''))).resolve()
            if not path.is_relative_to(self.artifact/'attempts'):raise PermissionError('Snapshot outside owned attempts')
            if not path.exists() or _sha(path)!=metadata.get('sha256'):raise ValueError('Authorized measurement snapshot SHA differs: '+str(path))
            self._import_legacy(path,dict(metadata,kind='verified_legacy_snapshot'))

    def _accept_record(self,path,record):
        method=record['identity']['method'];value=record['inference']
        expected_name=method+'_'+hashlib.sha256(_bytes(record)).hexdigest()+'.json'
        if path.name!=expected_name:
            self.rejections.append(dict(path=str(path),method=method,reason='Immutable record content hash/name differs'));return
        if value.get('status') not in ('MEASURED','NOT_MEASURED'):
            self.rejections.append(dict(path=str(path),method=method,reason='Unknown measurement status'));return
        valid,reason=validate_measurement(value,self.identities[method])
        if value.get('status')=='MEASURED' and not valid:
            self.rejections.append(dict(path=str(path),method=method,reason=reason));return
        previous=self.available.get(method)
        if previous is not None and previous[0].get('status')=='MEASURED':return
        self.available[method]=(value,dict(path=str(path.resolve()),sha256=_sha(path),
            identity_hash=digest(record['identity']),origin=record['origin'],revision=REVISION))

    def save(self,method,value,origin):
        clean={k:v for k,v in value.items() if k!='measurement_source'}
        if clean.get('status')=='MEASURED':
            valid,reason=validate_measurement(clean,self.identities[method])
            if not valid:raise ValueError('Cannot preserve invalid MEASURED entry: '+reason)
        elif clean.get('status')!='NOT_MEASURED':raise ValueError('Unknown measurement status')
        record=dict(schema_version=1,revision=REVISION,identity=self.identities[method],inference=clean,origin=_plain(origin))
        path=self.records/(method+'_'+hashlib.sha256(_bytes(record)).hexdigest()+'.json')
        atomic_json(path,record,immutable=True);self._accept_record(path,record)
        return dict(clean,measurement_source=dict(path=str(path.resolve()),sha256=_sha(path),
            identity_hash=digest(record['identity']),origin=record['origin'],revision=REVISION))

    def _import_legacy(self,path,origin):
        try:
            data=json.loads(path.read_text())
            if data.get('dataset')!=self.bundle.dataset or int(data.get('seed',-1))!=self.bundle.seed:raise ValueError('Dataset/seed differ')
            if data.get('source',{}).get('prefix_identity')!=_plain(self.bundle.identity):raise ValueError('Prefix/config/native/pilot/base identities differ')
            # New merged canonical summaries refer to separately verified records.
            if data.get('measurement_preservation_revision')==REVISION:return
            if data.get('benchmark_requested') is not True:return
            if data.get('immutability',{}).get('parameters_modes_gradients_normalizers_rng_unchanged') is not True:
                raise ValueError('Legacy measurement lacks original immutable-model proof')
            for method,value in data.get('inference',{}).items():
                if method not in self.identities:continue
                existing=self.available.get(method)
                if existing is not None and (existing[0].get('status')=='MEASURED' or existing[1]['origin'].get('source_sha256')==_sha(path)):
                    continue
                completion=self.artifact/'runs'/self.bundle.dataset/f'seed{self.bundle.seed}'/method/'TRAINING_COMPLETE.json'
                receipt=json.loads(completion.read_text())
                pred=self.predictions[method]
                if receipt.get('identity')!=_plain(pred.identity) or receipt.get('selected_policy')!=_plain(pred.selection):
                    self.rejections.append(dict(path=str(path),method=method,reason='Frozen method identity or selection differs'));continue
                if receipt.get('prefix_identity')!=_plain(self.bundle.identity):
                    self.rejections.append(dict(path=str(path),method=method,reason='Method completion prefix differs'));continue
                observed_mtime=origin.get('source_mtime_ns')
                if observed_mtime is not None and observed_mtime<completion.stat().st_mtime_ns:
                    self.rejections.append(dict(path=str(path),method=method,reason='Measurement predates frozen method completion'));continue
                formation_check=None
                if method in self.formations:
                    formation_path=completion.parent/'formation.json'
                    if not formation_path.exists() or json.loads(formation_path.read_text()).get('identity')!=_plain(self.formations[method].identity):
                        self.rejections.append(dict(path=str(path),method=method,reason='Frozen formation identity receipt missing or differs'));continue
                    formation_check=dict(path=str(formation_path.resolve()),sha256=_sha(formation_path))
                if value.get('status')=='MEASURED':
                    valid,reason=validate_measurement(value,self.identities[method])
                    if not valid:self.rejections.append(dict(path=str(path),method=method,reason=reason));continue
                elif value.get('status')!='NOT_MEASURED':continue
                elif 'rows_hash' in value and value['rows_hash']!=self.identities[method]['rows_hash']:
                    self.rejections.append(dict(path=str(path),method=method,reason='Failed observation row identity differs'));continue
                basis=dict(kind=origin.get('kind'),source_path=str(path.resolve()),source_sha256=_sha(path),
                    reported_measurement_source=data.get('source'),original_source_mtime_ns=origin.get('source_mtime_ns'),
                    original_measurement_timestamp='NOT_RECORDED_IN_LEGACY_JSON',
                    crosscheck_completion_path=str(completion.resolve()),crosscheck_completion_sha256=_sha(completion),
                    crosscheck_formation_receipt=formation_check,
                    crosscheck_basis='Exact frozen prefix and runtime prediction identity/selection matched TRAINING_COMPLETE; original raw-X retained-p parity and environment verified for successes',
                    original_formation_identity_field='NOT_RECORDED_IN_LEGACY_JSON; matching actual formation receipt checked where applicable',
                    imported_at_unix=time.time())
                self.save(method,value,basis)
        except (ValueError,KeyError,TypeError,FileNotFoundError) as error:
            self.rejections.append(dict(path=str(path),reason=str(error)))

    def lookup(self,method):
        found=self.available.get(method)
        return None if found is None else dict(found[0],measurement_source=found[1])


def validate_runtime_state(context,*,pid,parent_pid,frozen_sha):
    from .protocol import PrefixBundle
    if not isinstance(context.get('bundle'),PrefixBundle) or any(not isinstance(context.get(k),dict) for k in ('predictions','formations','diagnostics','config')):
        raise TypeError('Delivery observation requires typed normal worker state')
    bundle=context['bundle'];key=f'{bundle.dataset}:{bundle.seed}'
    if WORKER_PIDS.get(key)!=pid or parent_pid!=SERVICE_PID or not context['frozen']:
        raise PermissionError('Delivery observation worker/frozen identity differs')
    if context['config']['config_hash']!=CONFIG_HASH or Path(context['root']).resolve()!=ARTIFACT_ROOT:
        raise PermissionError('Delivery observation config/root differs')
    if frozen_sha!=FROZEN_SHA:raise PermissionError('Frozen manifest changed')
    return key


def validate_fill_authorization(token,key,predictions):
    from .delivery import REQUIRED
    expected=dict(service_pid=SERVICE_PID,worker_pids=WORKER_PIDS,config_hash=CONFIG_HASH,frozen_manifest_sha256=FROZEN_SHA,
        operation='fill_missing_required')
    if token.get('verdict') not in ('PASS','PASS WITH FIXES') or token.get('blocking_issues',[]) or any(token.get(k)!=v for k,v in expected.items()):
        raise PermissionError('Dedicated delivery preservation review/operation identity invalid')
    for name in ('delivery.py','delivery_observation_preservation.py','readers.py'):
        if token.get('source_sha256',{}).get(name)!=_sha(Path(__file__).with_name(name)):raise PermissionError('Reviewed delivery source changed: '+name)
    design=WORKSPACE/'research_reports/2026-10-26_v2/DESIGN_DELIVERY_OBSERVATION_PRESERVATION_ADDENDUM.md'
    if token.get('design_sha256')!=_sha(design):raise PermissionError('Delivery preservation design differs')
    extension=token.get('context_completion_extension')
    if extension is not None:
        addendum=WORKSPACE/'research_reports/2026-10-26_v2/DESIGN_ML5201_MISSING_OBSERVATION_COMPLETION_ADDENDUM.md'
        if extension!=ML5201_COMPLETION_EXTENSION or token.get('context_completion_addendum_sha256')!=_sha(addendum):
            raise PermissionError('Specific ML5201 observation-completion addendum differs')
    whitelist=token.get('allowed_methods_by_worker',{})
    if not isinstance(whitelist,dict) or not set(whitelist)<=set(WORKER_PIDS):
        raise PermissionError('Unknown delivery observation worker whitelist')
    for worker,methods in whitelist.items():
        ceiling=set(REQUIRED)
        if extension==ML5201_COMPLETION_EXTENSION and worker=='movielens:5201':ceiling.update(ML5201_COMPLETION_METHODS)
        if not isinstance(methods,list) or len(set(methods))!=len(methods) or not set(methods)<=ceiling:
            raise PermissionError('Method exceeds the fixed per-context delivery whitelist')
    allowed=whitelist.get(key)
    if not isinstance(allowed,list) or not set(allowed)<=set(predictions):
        raise PermissionError('Fill authorization names a missing method')
    return allowed


def run_on_reader_reload():
    if os.getpid() not in WORKER_PIDS.values():return dict(status='NO_OP',reason='Not a registered frozen worker')
    campaign=sys.modules.get(__package__+'.campaign')
    if campaign is None:return dict(status='NO_OP',reason='Campaign module not loaded')
    frame=sys._getframe(1);context=None
    try:
        for _ in range(40):
            if frame is None:break
            if frame.f_code is campaign._worker.__code__:
                if frame.f_locals.get('op')=='reload-readers':context=dict(frame.f_locals)
                break
            frame=frame.f_back
    finally:del frame
    if context is None:return dict(status='NO_OP',reason='Outside exact normal worker reload-readers frame')
    from .delivery import collect_delivery
    from .raw_reconstruction_observer import _frozen_fingerprint
    key=validate_runtime_state(context,pid=os.getpid(),parent_pid=os.getppid(),frozen_sha=_sha(ARTIFACT_ROOT/'FROZEN_MANIFEST.json'))
    bundle=context['bundle']
    token_path=ARTIFACT_ROOT/'DELIVERY_OBSERVATION_PRESERVATION_AUTHORIZATION.json'
    if not token_path.exists():return dict(status='DEFINITIONS_ONLY',reason='No dedicated fill authorization')
    token=json.loads(token_path.read_text())
    allowed=validate_fill_authorization(token,key,context['predictions'])
    before=_frozen_fingerprint(bundle,context['predictions'],context['formations'])
    result=collect_delivery(bundle,context['predictions'],context['formations'],context['diagnostics'],context['config'],
        benchmark=True,allowed_benchmark_methods=allowed,preserved_measurements=token.get('verified_snapshots',[]))
    after=_frozen_fingerprint(bundle,context['predictions'],context['formations'])
    if before!=after:raise RuntimeError('Delivery-only fill changed frozen scientific state')
    receipt=dict(status='COMPLETE' if not result['missing_required_methods'] else 'INCOMPLETE_REQUIRED_MEASUREMENTS',
        dataset=bundle.dataset,seed=bundle.seed,worker_pid=os.getpid(),parent_pid=os.getppid(),
        authorization_sha256=_sha(token_path),source_sha256=token['source_sha256'],frozen_manifest_sha256=FROZEN_SHA,
        context_completion_extension=token.get('context_completion_extension'),
        context_completion_addendum_sha256=token.get('context_completion_addendum_sha256'),
        permitted_methods=allowed,measurement_actions=result['measurement_actions'],missing_required_methods=result['missing_required_methods'],
        all_methods_status=result['status'],before=before,after=after,all_frozen_state_unchanged=True,
        summary_path=result['summary_path'],summary_sha256=_sha(result['summary_path']),
        no_export_bootstrap_training=True,model_files_written=0,completed_unix=time.time())
    atomic_json(ARTIFACT_ROOT/'delivery'/'preservation_receipts'/f'{bundle.dataset}_{bundle.seed}_{time.time_ns()}.json',receipt,immutable=True)
    return {k:v for k,v in receipt.items() if k not in ('before','after')}
