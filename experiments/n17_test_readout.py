"""Raw test predictions after a primary-signed global model/protocol freeze."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import n17_common as c
from v3_train_20261016 import predict,metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--device',required=True);a=p.parse_args()
    freeze=json.loads(a.freeze.read_text());assert freeze['status']=='FROZEN' and freeze['all_training_and_selection_definitions_frozen'] and freeze['all_required_or_triggered_stages_closed']
    assert freeze['test_readout_source_sha256']==c.sha(__file__)
    key=str(a.run_dir.resolve());entry=freeze['runs'][key];assert c.sha(a.run_dir/'config.json')==entry['config_sha256']
    conf=json.loads((a.run_dir/'config.json').read_text());assert not conf['phase'].endswith('_pilot'),'pilot test is excluded from independent confirmation'
    device=torch.device(a.device);torch.set_num_threads(2);torch.cuda.set_device(device)
    dataset=conf.get('dataset',conf['data']['dataset_key']);population_seed=int(conf.get('subset_seed',conf.get('population_seed',2022)))
    b=c.bundle(dataset,population_seed);b.seed=conf['seed']
    if conf.get('factory_module'):
        module=importlib.import_module(conf['factory_module']);b,model=module.readout_factory(conf,b,device)
    else:model=c.base.fixed_model(b,conf['seed']).to(device)
    started=time.time();rows=[];prediction_index=[]
    for epoch_text,modelinfo in entry['models'].items():
        epoch=int(epoch_text);path=Path(modelinfo['path']);assert c.sha(path)==modelinfo['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False);assert state['epoch']==epoch;model.load_state_dict(state['model'],strict=True)
        pred=predict(model,b,'test',device);prediction_path=a.run_dir/f'test_epoch{epoch:03d}.npz';np.savez_compressed(prediction_path,predictions=pred)
        prediction_index.append(dict(epoch=epoch,model_sha256=modelinfo['sha256'],path=str(prediction_path.resolve()),sha256=c.sha(prediction_path),shape=list(pred.shape),dtype=str(pred.dtype)))
        cells=metrics(b,pred,'test')
        for cell in cells:rows.append(dict(epoch=epoch,seed=conf['seed'],arm=conf['arm'],optimizer=conf['optimizer'],lr=conf['lr'],dataset=dataset,**cell))
        c.dump(a.run_dir/'test_progress.json',dict(status='RUNNING',epoch_completed=epoch,freeze_sha=c.sha(a.freeze),elapsed_seconds=time.time()-started))
    import pandas as pd
    pd.DataFrame(rows).to_csv(a.run_dir/'metrics_test.csv',index=False)
    c.dump(a.run_dir/'test_prediction_index.json',dict(freeze_sha256=c.sha(a.freeze),config_sha256=entry['config_sha256'],row_id_order_sha256=hashlib.sha256(np.asarray(b.test_row_ids,dtype=np.int64).tobytes()).hexdigest(),row_hash_definition='ordered int64 row IDs, native little-endian host bytes',predictions=prediction_index))
    c.dump(a.run_dir/'test_receipt.json',dict(status='COMPLETE',freeze_sha256=c.sha(a.freeze),source_sha256=c.sha(__file__),epochs=sorted(map(int,entry['models'])),config_sha256=entry['config_sha256'],historical_test_exposure=True,wall_seconds=time.time()-started,metrics_sha256=c.sha(a.run_dir/'metrics_test.csv'),prediction_index_sha256=c.sha(a.run_dir/'test_prediction_index.json')))

if __name__=='__main__':main()
