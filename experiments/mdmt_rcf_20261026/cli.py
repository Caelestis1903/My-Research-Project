"""CLI contract backed by a resident RAM-only campaign service.

Use ``serve`` once, then the registered commands; ``run-all`` executes the same
stages in one controller. A missing service is never represented as resume.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from multiprocessing.connection import Listener,Client
from .artifacts import _json,_plain

DEFAULT_CONFIG='experiments/mdmt_rcf_20261026/config_four_seed.json'

def _parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=DEFAULT_CONFIG)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('audit','build-plan','verify','serve','prefix','read-probe','form','decide','diagnose','freeze','run-locked','evaluate','finalize','scale-probe','run-all','status','capture-fixture','reload-readers'):
        q=sub.add_parser(name);q.add_argument('--config',default=argparse.SUPPRESS)
        if name in ('serve','run-all'):q.add_argument('--devices',nargs='+',default=None)
        if name=='capture-fixture':
            q.add_argument('--dataset',choices=('movielens','kuairand'),default='movielens');q.add_argument('--seed',type=int,default=5201)
        if name=='prefix':
            q.add_argument('--datasets',nargs='+',choices=('movielens','kuairand'));q.add_argument('--seeds',nargs='+',type=int);q.add_argument('--resume',action='store_true')
        if name=='verify':q.add_argument('--scope',default='required',choices=('required','prefix','formation','campaign'))
        if name=='form':q.add_argument('--stage',default='s2',choices=('s2',));q.add_argument('--locked-plan',action='store_true')
        if name=='decide':q.add_argument('--stage',required=True,choices=('s2','s3'))
        if name in ('read-probe',):q.add_argument('--locked-plan',action='store_true')
        if name in ('diagnose','freeze','run-locked'):q.add_argument('--triggered-only',action='store_true')
        if name=='run-locked':q.add_argument('--resume',action='store_true')
        if name=='evaluate':q.add_argument('--locked-only',action='store_true')
        if name=='finalize':q.add_argument('--verify-exports',action='store_true');q.add_argument('--cleanup-owned-temporary',action='store_true')
        if name=='scale-probe':q.add_argument('--stage',choices=('numeric','train','locked'),default='numeric');q.add_argument('--device',default='cuda:0')
    return p

def _endpoint(config):
    root=Path(config['paths']['artifact']).resolve();root.mkdir(parents=True,exist_ok=True)
    address=str(root/'campaign.sock')
    if len(address.encode())>=104:raise ValueError('Artifact path too long for resident Unix socket')
    auth=hashlib.sha256(('RCF-local-service:'+config['config_hash']).encode()).digest()
    return address,auth

def _dispatch(campaign,request):
    name=request['command'];args=request.get('args',{})
    if name=='prefix':return campaign.prefix(args.get('datasets'),args.get('seeds'))
    if name=='capture-fixture':return campaign.capture_fixture(args.get('dataset','movielens'),args.get('seed',5201))
    if name=='reload-readers':return campaign.reload_readers()
    if name=='read-probe':return campaign.phase('s1')
    if name=='form':return campaign.phase('s2')
    if name=='decide':return campaign.decide(args['stage'])
    if name=='diagnose':return campaign.diagnose()
    if name=='freeze':return campaign.freeze()
    if name=='run-locked':return campaign.run_locked()
    if name=='evaluate':return campaign.evaluate()
    if name=='finalize':return campaign.finalize()
    if name=='run-all':return campaign.run_all()
    if name=='status':return dict(resident_prefixes=[list(k) for k in campaign.workers],decisions=campaign.decisions,frozen=campaign.frozen is not None)
    raise ValueError(name)

def serve(config,devices=None):
    from .campaign import Campaign
    import traceback
    address,auth=_endpoint(config)
    if Path(address).exists():raise FileExistsError('Resident endpoint exists; inspect service, do not overwrite its RAM owner')
    c=Campaign(config,devices=devices)
    with Listener(address=address,family='AF_UNIX',authkey=auth) as listener:
        os.chmod(address,0o600)
        _json(c.root/'SERVICE.json',dict(pid=os.getpid(),socket=address,config_hash=config['config_hash'],model_storage='RAM_ONLY'))
        print(json.dumps(dict(status='SERVICE_READY',pid=os.getpid(),socket=address)),flush=True)
        while True:
            conn=listener.accept()
            try:
                req=conn.recv()
                if req.get('config_hash')!=config['config_hash']:raise PermissionError('Resident config hash mismatch')
                result=_dispatch(c,req);conn.send(dict(ok=True,result=_plain(result)))
            except BaseException as e:conn.send(dict(ok=False,error=repr(e),traceback=traceback.format_exc()))
            finally:conn.close()
            if req.get('command')=='finalize' and not c.workers:break
    Path(address).unlink(missing_ok=True)

def main(argv=None):
    args=vars(_parser().parse_args(argv))
    from .protocol import load_config,audit_dataset
    from .campaign import build_run_manifest,Campaign
    config=load_config(args['config']);root=Path(config['paths']['artifact']).resolve();root.mkdir(parents=True,exist_ok=True)
    name=args['command']
    if name=='audit':
        result={d:audit_dataset(d,config,root/'audit') for d in config['datasets']};_json(root/'DATA_ROLE_MANIFEST.json',result)
    elif name=='build-plan':
        audit=json.loads((root/'DATA_ROLE_MANIFEST.json').read_text()) if (root/'DATA_ROLE_MANIFEST.json').exists() else {'status':'NOT_RUN'}
        result=build_run_manifest(config,audit);_json(root/'RUN_MANIFEST.json',result);_json(root/'IMPLEMENTATION_IDENTITY.json',result['identity']);_json(root/'effective_config.json',config)
    elif name=='verify':
        result={}
        scopes=('prefix','formation','campaign') if args['scope']=='required' else (args['scope'],)
        for scope in scopes:
            mod=__import__('experiments.mdmt_rcf_20261026.verify_'+scope,fromlist=['run_required'])
            fn=getattr(mod,'run_required',None) or getattr(mod,'verify',None) or getattr(mod,'verify_native_required',None)
            if fn is None:raise AttributeError('Verification module lacks run_required/verify: '+scope)
            import inspect
            kwargs={'config':config}
            if 'output_root' in inspect.signature(fn).parameters:kwargs['output_root']=root
            result[scope]=fn(**kwargs)
        _json(root/'VERIFICATION.json',result)
    elif name=='scale-probe':
        from .scale_probe import run_scale_numeric,run_scale_training
        if args['stage']=='numeric':result=run_scale_numeric(config,root)
        else:
            c=Campaign(config);c.require_review()
            if args['stage']=='locked':
                path=root/'scale'/'LOCKED_AUTHORIZATION.json'
                if not path.exists():raise PermissionError('SCALE locked stage was not triggered')
                authorization=json.loads(path.read_text())
                if authorization.get('status')!='TRIGGERED' or authorization.get('config_hash')!=config['config_hash']:raise PermissionError('SCALE locked authorization invalid')
                config=dict(config,_scale_seeds=config['seeds']['scale_conditional_locked'],_scale_locked_authorization=authorization)
            result=run_scale_training(config,root,device=args['device'])
    elif name=='serve':serve(config,args.get('devices'));return
    elif name=='run-all':result=Campaign(config,devices=args.get('devices')).run_all()
    else:
        address,auth=_endpoint(config)
        if not Path(address).exists():raise RuntimeError('No resident campaign service. Start `cli serve`; disk data do not contain trained model/optimizer state. Interrupted work needs same-identity deterministic replay.')
        with Client(address=address,family='AF_UNIX',authkey=auth) as conn:
            conn.send(dict(command=name,args=args,config_hash=config['config_hash']));response=conn.recv()
        if not response['ok']:raise RuntimeError(response.get('traceback',response.get('error')))
        result=response['result']
    print(json.dumps(_plain(result),ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':main()
