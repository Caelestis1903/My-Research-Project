"""Raw operation counts and model-only latency on frozen checkpoints."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from cm19_train import OUT,dump
from cm19_legacy_data import population
from cm19_models import make_model

def measure(run_id,device):
    out=OUT/'runs'/run_id;cfg=json.loads((out/'config.json').read_text());b=population(cfg['dataset'],'natural');dev=torch.device(device);torch.set_num_threads(2)
    m=make_model(b.features,cfg['seed'],cfg['model'],cfg['variant']).to(dev)
    state=torch.load(Path(cfg['state_directory'])/'selected.pt',map_location=dev,weights_only=False);m.load_state_dict(state['model'])
    rng=np.random.default_rng(191019);ix=np.concatenate([rng.choice(b.meta[b.train_x[b.meta,-1]==d],min(256,int((b.train_x[b.meta,-1]==d).sum())),False) for d in range(3)])
    x=torch.tensor(b.train_x[ix],device=dev).T;y=torch.tensor(b.train_y[ix][:,m.task_ids],device=dev)
    def sync():
        if dev.type=='cuda':torch.cuda.synchronize(dev)
    activities=[torch.profiler.ProfilerActivity.CPU]+([torch.profiler.ProfilerActivity.CUDA] if dev.type=='cuda' else [])
    m.eval()
    with torch.no_grad():
        for _ in range(3):m(x)
        sync();times=[]
        for _ in range(10):
            start=time.perf_counter();m(x);sync();times.append(time.perf_counter()-start)
        with torch.profiler.profile(activities=activities,record_shapes=True,with_flops=True) as prof:m(x)
        sync()
    forward_flops=sum(e.flops for e in prof.key_averages())
    m.train();m.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=activities,record_shapes=True,with_flops=True) as train_prof:
        F.binary_cross_entropy(m(x),y).backward()
    sync();train_flops=sum(e.flops for e in train_prof.key_averages())
    dump(out/'cost_profile.json',dict(run_id=run_id,device=device,rows=len(ix),parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
         forward_dense_flops=int(forward_flops),forward_backward_dense_flops=int(train_flops),forward_dense_flops_per_example=forward_flops/len(ix),forward_backward_dense_flops_per_example=train_flops/len(ix),
         inference_seconds_median=float(np.median(times)),inference_seconds_p95=float(np.quantile(times,.95)),repetitions=10,
         scope='PyTorch supported dense/matmul FLOPs, measured shapes; excludes embedding/normalization/optimizer kernels and data pipeline, hence lower bound. Small stratified train-meta batch, not full physical training batch. Selected checkpoint, temporary training gradients only; checkpoint unchanged; no test labels.',
         source_state=str(Path(cfg['state_directory'])/'selected.pt'),measurement_context='CPU measurements alongside active training/diagnostic processes, not an isolated hardware benchmark',test_read=False))
    print(json.dumps(dict(run_id=run_id,status='COMPLETE_COST_PROFILE')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();measure(a.run_id,a.device)
