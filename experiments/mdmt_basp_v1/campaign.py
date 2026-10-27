"""Resident common prefixes and explicit orchestrator-controlled stage commands."""
from pathlib import Path
import gc
import json
import os
import time
import traceback
import numpy as np
import torch
from .config import DATASETS,DEV_SEEDS,LOCKED_SEEDS,P1,arms,validate_arm,plan
from .roles import Data
from .prefix import build_prefix,cache_native,cache_phi,synchronize
from .trainer import Trainer
from .identity import configure,write_json,freeze
from .export import export_prediction


def _worker_identity(dataset,seed,residents):
    methods={}
    for name,trainer in residents.items():
        methods[name]=dict(identity=trainer.identity,
                           selected=dict(epoch=trainer.selected['state']['epoch'],main_step=trainer.selected['state']['step']),
                           final=dict(epoch=trainer.final['state']['epoch'],main_step=trainer.final['state']['step']))
    return dict(dataset=dataset,seed=seed,methods=methods,storage='RAM_ONLY')


def _ready(root,dataset,seed,residents):
    identity=_worker_identity(dataset,seed,residents)
    ready=dict(status='READY_FOR_STAGE_DECISION',pid=os.getpid(),identity=identity,
               methods=list(residents),model_storage='RAM_ONLY')
    write_json(root/'READY.json',ready);write_json(root/'STATUS.json',ready)
    return identity


def worker(cfg,dataset,seed,artifact_root,device='cuda:0',locked_plan=None):
    if dataset not in DATASETS or seed not in DEV_SEEDS+LOCKED_SEEDS:
        raise ValueError('Dataset/seed outside execution contract')
    if seed in LOCKED_SEEDS and locked_plan is None:
        raise PermissionError('Locked seed requires a pre-existing frozen P5 selection')
    configure()
    artifact_root=Path(artifact_root).resolve()
    root=artifact_root/'workers'/dataset/f'seed{seed}'
    root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    write_json(root/'STATUS.json',dict(status='BUILDING_PREFIX',pid=os.getpid(),dataset=dataset,seed=seed))
    try:
        return _resident(cfg,dataset,seed,artifact_root,root,device,locked_plan,started)
    except BaseException as error:
        write_json(root/'FAILED.json',dict(status='FAILED',error_type=type(error).__name__,error=str(error),
                                           elapsed_seconds=time.monotonic()-started,traceback=traceback.format_exc()))
        raise


def _resident(cfg,dataset,seed,artifact_root,root,device,locked_plan,started):
    review=artifact_root/'REVIEW_RECEIPT.json'
    if not review.is_file():
        raise PermissionError('Independent review receipt required before training')
    receipt=json.loads(review.read_text())
    if receipt.get('verdict') not in ('PASS','PASS WITH FIXES') or receipt.get('blocking_open',True):
        raise PermissionError('Independent review has open blocking issues')
    data=Data(dataset,artifact_root/'data'/dataset/'DIRECT_VALIDATION.json',root/'LABEL_ACCESS.jsonl')
    write_json(root/'DATA_ROLE_MANIFEST.json',data.manifest)
    write_json(root/'EXECUTION_CONFIG.json',cfg)
    prefix_root=root/'prefix';prefix_root.mkdir()
    prefix=build_prefix(data,seed,cfg,prefix_root,device)
    views=prefix['views'];residents={};pool=arms()
    write_json(root/'PREFIX_COMPLETE.json',dict(native=prefix['native_receipt'],source=prefix['source_receipt'],
                                               storage='RAM_ONLY'))
    for role,purpose in (('Q','development'),('V','select')):
        y,m=data.labels(role,purpose,component='NATIVE_RAW')
        export_prediction(root,role,'NATIVE_RAW',views[role],y,m,np.asarray(views[role]['z']),{},
                          dict(dataset=dataset,seed=seed,method='NATIVE_RAW',
                               native_selected_epoch=prefix['native_receipt']['selected_epoch']))
    if seed in LOCKED_SEEDS:
        if locked_plan is None:
            raise PermissionError('Locked seed requires frozen P5 selection')
        frozen=json.loads(Path(locked_plan).read_text())
        if frozen.get('status')!='FROZEN_P5' or dataset not in frozen.get('datasets',{}):
            raise PermissionError('Invalid P5 selection')
        initial=frozen['datasets'][dataset]['arms']
        if len(initial)>4:
            raise ValueError('P5 allows at most four arms per dataset')
    else:
        initial=list(P1)

    def train(registered):
        for item in registered:
            arm=pool[item] if isinstance(item,str) else item
            validate_arm(arm)
            name=arm['id']
            if name in residents:
                raise ValueError('Existing trajectory cannot be overwritten: '+name)
            write_json(root/'STATUS.json',dict(status='TRAINING',pid=os.getpid(),method=name,dataset=dataset,seed=seed))
            trainer=Trainer(arm,data,views,prefix,seed,cfg,artifact_root/'runs'/dataset/name/f'seed{seed}',device)
            trainer.run()
            for which in ('selected','final'):
                for role in ('V','Q'):
                    trainer.export_state(role,which)
            trainer.latency()
            trainer.model.cpu()
            if trainer.cache is not None:
                trainer.cache.to('cpu')
            residents[name]=trainer
            gc.collect()
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()

    train(initial)
    diagnosed=set()
    def run_interventions():
        pending=[name for name,tr in residents.items() if name not in diagnosed and bool(getattr(tr.model,'foreign',False))]
        if not pending:
            return
        review_path=artifact_root/'INTERVENTION_REVIEW_RECEIPT.json'
        write_json(root/'STATUS.json',dict(status='CORE_MATRIX_COMPLETE_WAITING_INTERVENTION_REVIEW',
                                          pid=os.getpid(),methods=list(residents)))
        while not review_path.exists():
            time.sleep(2)
        reviewed=json.loads(review_path.read_text())
        if reviewed.get('verdict') not in ('PASS','PASS WITH FIXES') or reviewed.get('blocking_open',True):
            raise PermissionError('Read-intervention review has open blocking issues')
        from .interventions import run_read_interventions
        for name in pending:
            trainer=residents[name]
            for which in ('selected','final'):
                run_read_interventions(trainer,which)
            trainer.model.cpu()
            if trainer.cache is not None:
                trainer.cache.to('cpu')
            diagnosed.add(name)
    run_interventions()
    identity=_ready(root,dataset,seed,residents)
    version=0;stage_used=dict(P2=0,P3=0,P4=0)
    while True:
        freeze_file=artifact_root/'GLOBAL_FREEZE.json'
        if freeze_file.exists():
            manifest=json.loads(freeze_file.read_text())
            data.authorize_final(manifest,identity)
            break
        command_file=root/'COMMAND.json'
        if command_file.exists():
            command=json.loads(command_file.read_text())
            if int(command.get('version',0))>version:
                if command.get('action')!='train' or not command.get('arms') or not command.get('evidence'):
                    raise PermissionError('A stage command needs registered arms and explicit evidence')
                stage=command.get('stage')
                count=len(command['arms'])
                if stage not in stage_used:
                    raise ValueError('Unknown development stage')
                stage_used[stage]+=count
                if stage_used[stage]>dict(P2=6,P3=2,P4=2)[stage]:
                    raise ValueError('Stage variant limit exceeded')
                write_json(root/f'COMMAND_{command["version"]}_ACCEPTED.json',command)
                train(command['arms']);run_interventions();version=int(command['version'])
                identity=_ready(root,dataset,seed,residents)
        time.sleep(2)
    write_json(root/'STATUS.json',dict(status='FROZEN_WAITING_E_RELEASE',pid=os.getpid(),identity=identity))
    while not (root/'EVALUATE.json').exists():
        time.sleep(2)
    release=json.loads((root/'EVALUATE.json').read_text())
    if release.get('identity')!=identity:
        raise PermissionError('E release differs from resident identity')
    prefix['native'].to(device);prefix['source'].to(device)
    views['E']=cache_native(prefix['native'],data,'E',prefix_root,device)
    cache_phi(prefix['source'],views['E'],'E',prefix_root,device)
    y,m=data.labels('E','final',component='NATIVE_RAW')
    export_prediction(root,'E','NATIVE_RAW',views['E'],y,m,np.asarray(views['E']['z']),{},
                      dict(dataset=dataset,seed=seed,method='NATIVE_RAW',freeze_version=manifest['version']))
    for name,trainer in residents.items():
        trainer.views=views
        for which in ('selected','final'):
            trainer.export_state('E',which)
        trainer.model.cpu()
        if trainer.cache is not None:
            trainer.cache.to('cpu')
    write_json(root/'HISTORICAL_E_RECEIPT.json',dict(status='COMPLETE',identity=identity,
              frozen_version=manifest['version'],models_written=0,
              elapsed_seconds=time.monotonic()-started))
    write_json(root/'STATUS.json',dict(status='COMPLETE',pid=os.getpid(),identity=identity))
    return identity


def freeze_campaign(artifact_root,decision_path,worker_keys):
    artifact_root=Path(artifact_root)
    decision=json.loads(Path(decision_path).read_text())
    if not worker_keys:
        raise ValueError('Explicit complete worker registry required')
    actual={str(p.parent.relative_to(artifact_root/'workers')) for p in (artifact_root/'workers').glob('*/seed*/STATUS.json')}
    if set(worker_keys)!=actual:
        raise ValueError('Frozen registry must include every started resident worker')
    workers={}
    for key in worker_keys:
        dataset,seed_text=key.split('/')
        ready=json.loads((artifact_root/'workers'/dataset/seed_text/'READY.json').read_text())
        if ready['status']!='READY_FOR_STAGE_DECISION':
            raise PermissionError('Worker is not ready: '+key)
        workers[key]=ready['identity']
    manifest=dict(status='FROZEN',version='BASP-A-final-selection-v1',workers=workers,
                  decisions=decision,model_storage='RAM_ONLY',historical_E_exposed=True)
    dest=artifact_root/'GLOBAL_FREEZE.json'
    if dest.exists():
        raise FileExistsError('Final selection is already frozen')
    write_json(dest,manifest)
    return manifest


def release_evaluation(artifact_root,dataset,seed):
    root=Path(artifact_root);key=f'{dataset}/seed{seed}'
    manifest=json.loads((root/'GLOBAL_FREEZE.json').read_text())
    identity=manifest['workers'][key]
    write_json(root/'workers'/dataset/f'seed{seed}'/'EVALUATE.json',dict(identity=identity))
    return dict(status='RELEASED',worker=key)
