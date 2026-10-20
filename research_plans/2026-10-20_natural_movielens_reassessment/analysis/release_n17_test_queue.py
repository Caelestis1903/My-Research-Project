"""Primary release of the already designed N17 zero-update test, after closure only."""
from pathlib import Path
import hashlib
import json

ART=Path('/home_new/xzf_natural_movielens_reassessment')
ANA=Path(__file__).resolve().parent
PYTHON='/home/xzf/miniconda3/envs/phase0-3-env/bin/python'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def release():
    mechanism=json.loads((ANA/'n17_mechanisms/analysis_status.json').read_text())
    assert mechanism['complete'] and mechanism['test_read'] is False
    freeze_path=ART/'N17/test_frozen.json';frozen=json.loads(freeze_path.read_text())
    assert frozen['status']=='FROZEN' and frozen['test_read'] is False
    freeze_sha=sha(freeze_path)
    prepared=json.loads((ANA/'n17_freeze/config_list.json').read_text())
    configs={}
    for path in prepared:
        c=json.loads(Path(path).read_text());assert c['group']!='pilot'
        configs[str(Path(c['run_dir']).resolve())]=(Path(path),c)
    assert len(configs)==len(frozen['runs'])==72 and set(configs)==set(frozen['runs'])
    tasks=[]
    for run_dir,entry in sorted(frozen['runs'].items()):
        config_path,c=configs[run_dir];assert c==entry['config']
        run=Path(run_dir);assert sha(run/'config.json')==entry['config_sha256']
        assert sha(run/'selection_frozen.json')==entry['selection_sha256']
        artifacts=[run/name for name in ('metrics_test.csv','test_prediction_index.json','raw_AULC.json')]
        artifacts += [run/f'test_epoch{e:03d}.npz' for e in range(1,c['epochs']+1)]
        tasks.append(dict(run_id='N17_readout_'+c['id'],study='N17',scope=c['group'],
            config_path=str(config_path),run_dir=run_dir,
            command=[PYTHON,'-m','experiments.natural_ml_extensions.n17_readout','test',
                '--run-dir',run_dir,'--device','cuda:0'],
            receipt=str(run/'test_receipt.json'),
            receipt_checks=dict(status='COMPLETE',zero_updates=True,optimizer_instantiated=False,
                freeze_sha256=freeze_sha,epochs=c['epochs'],selected_epoch=entry['selected_epoch']),
            required_files=[str(config_path),str(freeze_path),str(run/'selection_frozen.json'),str(run/'all_model_index.json')],
            artifacts=[str(p) for p in artifacts],profile_group='TEST_N17'))
    release_id='N17_formal_freeze_'+freeze_sha[:12]
    payload=dict(schema='natural_movielens_test_release_v1',family='N17',release_id=release_id,
        status='RELEASED',test_authorized=True,parent_freeze_path=str(freeze_path),
        parent_freeze_sha256=freeze_sha,tasks=tasks)
    content=(json.dumps(payload,indent=2)+'\n').encode()
    path=ART/'runner/released_test_queues/N17'/(release_id+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_bytes()==content,'never change a published release'
    else:
        temporary=path.with_suffix('.json.tmp');temporary.write_bytes(content);temporary.replace(path)
    archive=ANA/'n17_freeze'/path.name
    if archive.exists():assert archive.read_bytes()==content
    else:archive.write_bytes(content)
    info=dict(release_path=str(path),sha256=sha(path),formal_configurations=len(tasks),
        parent_freeze_sha256=freeze_sha,zero_update_evaluation_only=True)
    (ANA/'n17_freeze/release_receipt.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__':release()
