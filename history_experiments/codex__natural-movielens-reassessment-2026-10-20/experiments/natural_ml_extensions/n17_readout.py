"""Freeze exact checkpoint identities, then mechanically evaluate test with zero updates."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from .n17_bootstrap import OUT,prepare,sources,sha,dump

def freeze(config_paths,closed_receipt):
    closure=json.loads(Path(closed_receipt).read_text())
    assert closure['all_required_or_triggered_stages_closed'] and closure['all_training_and_selection_definitions_frozen']
    entries={}
    for path in config_paths:
        config=json.loads(Path(path).read_text())
        if config['group']=='pilot':continue
        run=Path(config['run_dir']);receipt=json.loads((run/'receipt.json').read_text())
        assert receipt['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
        selected=json.loads((run/'selection_frozen.json').read_text());assert selected['budget']==config['epochs']
        models=json.loads((run/'all_model_index.json').read_text())
        assert len(models)==config['epochs']
        for model in models:assert sha(model['path'])==model['sha256']
        entries[str(run.resolve())]=dict(config=config,config_sha256=sha(run/'config.json'),
            selection_sha256=sha(run/'selection_frozen.json'),selected_epoch=selected['selected_epoch'],models=models)
    dump(OUT/'test_frozen.json',dict(status='FROZEN',runs=entries,source_hashes=sources(),
        closure_receipt_sha256=sha(closed_receipt),test_read=False,exploratory_replication=True))

def execute(run_dir,device='cuda:0'):
    freeze_path=OUT/'test_frozen.json';freeze=json.loads(freeze_path.read_text());assert freeze['status']=='FROZEN'
    assert freeze['source_hashes']==sources()
    run=Path(run_dir);entry=freeze['runs'][str(run.resolve())];config=entry['config']
    assert sha(run/'config.json')==entry['config_sha256']
    assert sha(run/'selection_frozen.json')==entry['selection_sha256']
    b,sp,c,root=prepare(config['population'])
    from v3_train_20261016 import predict,metrics
    model=c.base.fixed_model(b,config['seed']).to(device);rows=[];index=[];start=time.time()
    for info in entry['models']:
        assert sha(info['path'])==info['sha256'];state=torch.load(info['path'],map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);before=c.base.initial_state_hash(model)
        pred=predict(model,b,'test',torch.device(device),batch_size=40960)
        assert c.base.initial_state_hash(model)==before
        epoch=info['epoch'];target=run/f'test_epoch{epoch:03d}.npz'
        np.savez_compressed(target,predictions=pred)
        index.append(dict(epoch=epoch,path=str(target),sha256=sha(target)))
        for cell in metrics(b,pred,'test'):
            rows.append(dict(epoch=epoch,seed=config['seed'],arm=config['arm'],lr=config['lr'],
                selected=epoch==entry['selected_epoch'],final=epoch==config['epochs'],**cell))
    pd.DataFrame(rows).to_csv(run/'metrics_test.csv',index=False)
    dump(run/'test_prediction_index.json',dict(predictions=index,row_ids_sha256=__import__('hashlib').sha256(b.test_row_ids.astype(np.int64).tobytes()).hexdigest()))
    # Raw arithmetic trajectory summaries; primary agent interprets scientific evidence.
    cells=pd.DataFrame(rows);macro=cells.groupby('epoch')[['auc','logloss']].mean();summaries=[]
    for end in (20,100):
        curve=macro.loc[1:end]
        summaries.append(dict(start_epoch=1,end_epoch=end,n=len(curve),
            AULC_definition='original N17 arithmetic mean of completed-epoch metrics',
            AULC_auc=float(curve.auc.mean()),
            AULC_logloss=float(curve.logloss.mean())))
    dump(run/'raw_AULC.json',summaries)
    dump(run/'test_receipt.json',dict(status='COMPLETE',zero_updates=True,optimizer_instantiated=False,
        freeze_sha256=sha(freeze_path),epochs=config['epochs'],selected_epoch=entry['selected_epoch'],
        wall_seconds=time.time()-start,metrics_sha256=sha(run/'metrics_test.csv'),
        exploratory_replication=True,source_hashes=sources()))

def validation(run_dir):
    run=Path(run_dir);config=json.loads((run/'config.json').read_text())['extension_config']
    b,sp,c,root=prepare(config['population'])
    from v3_train_20261016 import metrics
    rows=[]
    for epoch in range(1,config['epochs']+1):
        with np.load(run/f'validation_epoch{epoch:03d}.npz') as saved:pred=saved['predictions']
        for split,subset in [('selection',sp['selection']),('diagnostic',sp['diagnostic']),('validation_all',None)]:
            for cell in metrics(b,pred,'val',subset):
                rows.append(dict(epoch=epoch,split=split,seed=config['seed'],arm=config['arm'],lr=config['lr'],**cell))
    pd.DataFrame(rows).to_csv(run/'metrics_validation_readout.csv',index=False)
    dump(run/'validation_readout_receipt.json',dict(status='COMPLETE',test_access=False,
        epochs=config['epochs'],metrics_sha256=sha(run/'metrics_validation_readout.csv'),sources=sources()))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','validation','test'])
    p.add_argument('--run-dir',type=Path);p.add_argument('--config-list',type=Path)
    p.add_argument('--closed-receipt',type=Path);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.action=='freeze':freeze(json.loads(a.config_list.read_text()),a.closed_receipt)
    elif a.action=='validation':validation(a.run_dir)
    else:execute(a.run_dir,a.device)
if __name__=='__main__':main()
