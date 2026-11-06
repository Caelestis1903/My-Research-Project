"""Frozen selected/final test readout. Raw predictions/metrics only."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from cm19_train import ROOT,OUT,ART,dump,cells,append
from cm19_legacy_data import population
from cm19_models import make_model

def evaluate(run_id,device):
    freeze=json.loads((OUT/'TEST_FREEZE.json').read_text());assert run_id in freeze['runs']
    out=OUT/'runs'/run_id;config=json.loads((out/'config.json').read_text());expected=freeze['runs'][run_id]
    for name,digest in freeze['training_source_hashes'].items():assert hashlib.sha256((ROOT/'experiments'/name).read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256((out/'selection_frozen.json').read_bytes()).hexdigest()==expected['selection_sha256']
    receipt=json.loads((out/'receipt.json').read_text());assert receipt['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
    result=out/'test_receipt.json';assert not result.exists(),f'Already evaluated: {run_id}'
    torch.set_num_threads(2);dev=torch.device(device);torch.cuda.set_device(dev);data=population(config['dataset'],'natural');model=make_model(data.features,config['seed'],config['model'],config['variant']).to(dev).eval()
    art=Path(config['state_directory']);records=[];start=time.time();hashes={}
    for label,file in [('selected','selected.pt'),('final','last.pt')]:
        h=hashlib.sha256()
        with (art/file).open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        hashes[label]=h.hexdigest();assert hashes[label]==expected['checkpoint_sha256'][label]
        # Full research states include optimizer history; evaluation only copies
        # model tensors to the GPU, keeping unused optimizer buffers on CPU.
        state=torch.load(art/file,map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
        allp=[];neutral={t:[] for t in model.interventions()};inference_seconds=0.
        with torch.no_grad():
            for o in range(0,len(data.test_x),config['batch_size']):
                xx=torch.tensor(data.test_x[o:o+config['batch_size']],device=dev).T;torch.cuda.synchronize(dev);t=time.perf_counter();pp=model(xx);torch.cuda.synchronize(dev);inference_seconds+=time.perf_counter()-t
                allp.append(pp.cpu().numpy())
                for kind in neutral:neutral[kind].append(model(xx,neutral=kind).cpu().numpy())
        pred=np.concatenate(allp);payload=dict(predictions=pred,labels=data.test_y,domains=data.test_x[:,-1],row_ids=data.test_ids)
        cc=cells(pred,data.test_y,data.test_x[:,-1],model.task_ids)
        for r in cc:r.update(checkpoint=label,epoch=state['epoch'],steps=state['steps'],examples=state['examples'],intervention='none')
        records.extend(cc)
        for kind,arrays in neutral.items():
            npred=np.concatenate(arrays);payload['neutral_'+kind]=npred;cc=cells(npred,data.test_y,data.test_x[:,-1],model.task_ids)
            for r in cc:
                ti=model.task_ids.index(r['task']);ix=data.test_x[:,-1]==r['domain'];delta=np.abs(pred[ix,ti]-npred[ix,ti])
                r.update(checkpoint=label,epoch=state['epoch'],steps=state['steps'],examples=state['examples'],intervention=kind,delta_p=float(delta.mean()),p50=float(np.quantile(delta,.5)),p95=float(np.quantile(delta,.95)),max_delta=float(delta.max()),nonzero=int((delta>0).sum()))
            records.extend(cc)
        np.savez_compressed(art/f'test_{label}.npz',**payload)
        dump(out/f'inference_{label}.json',dict(checkpoint=label,rows=len(pred),batch_size=config['batch_size'],seconds=inference_seconds,seconds_per_million=inference_seconds/len(pred)*1e6,measurement='instrumented Python forward, co-scheduled GPU processes; not isolated deployment latency'))
    append(out/'metrics_test.csv',records)
    dump(result,dict(status='COMPLETE_FROZEN_TEST',run_id=run_id,optimizer_steps_during_evaluation=0,checkpoints=hashes,wall_seconds=time.time()-start,test_rows=len(data.test_x),records=len(records),selected_and_final=True))
    print(json.dumps(dict(run_id=run_id,status='COMPLETE_FROZEN_TEST',records=len(records))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--device',required=True);a=p.parse_args();evaluate(a.run_id,a.device)
