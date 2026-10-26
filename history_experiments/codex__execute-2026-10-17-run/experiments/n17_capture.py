"""Loss-identical N17 training with every model retained for deferred test AULC."""
import argparse
import json
from pathlib import Path
import traceback
import torch
import n17_common as c
from n17_train import Runner as BaseRunner

class Runner(BaseRunner):
    def __init__(self,a):
        super().__init__(a);self.capture_sha=c.sha(__file__)
        if a.resume:assert self.config['model_capture_source_sha256']==self.capture_sha
        else:
            self.config.update(model_capture_source_sha256=self.capture_sha,all_epoch_models_retained=True,test_prediction_deferred=True)
            c.dump(self.dir/'config.json',self.config)
    def save(self,epoch,name):
        if name=='last_resume.pt':
            folder=self.dir/'all_models';folder.mkdir(exist_ok=True);target=folder/f'epoch{epoch:03d}_model.pt';tmp=folder/'save.tmp.pt'
            torch.save(dict(epoch=epoch,model=c.base.cpu(self.model.state_dict())),tmp);tmp.replace(target)
        super().save(epoch,name)
    def fit(self):
        super().fit();assert c.sha(__file__)==self.capture_sha
        assert all((self.dir/'all_models'/f'epoch{epoch:03d}_model.pt').exists() for epoch in range(1,self.a.epochs+1))
        receipt=json.loads((self.dir/'receipt.json').read_text());receipt.update(all_epoch_models_retained=True,model_capture_source_sha256=self.capture_sha)
        c.dump(self.dir/'receipt.json',receipt)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=['SN','SB'],required=True)
    p.add_argument('--optimizer',choices=['adam','sgd'],required=True);p.add_argument('--lr',type=float,required=True)
    p.add_argument('--phase',default='N1_confirmation');p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=40960)
    p.add_argument('--device',default='cuda:0');p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:Runner(a).fit()
    except Exception:
        c.dump(a.run_dir/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
