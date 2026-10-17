"""Complete N0 calibrations from previously certified exact-prefix prediction references."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,log_loss
from n17_n0 import calibrated_fit
import n17_common as c

def main():
    out=c.OUT/'N0';manifest=json.loads((out/'readout_manifest.json').read_text());frame=pd.read_csv(out/'calibration_cell_metrics.csv');new=[]
    cells=pd.read_csv(c.HISTORY/'analysis/test_baseline7_summary/seed_cell_test_readouts.csv');recoveries=[]
    for item in manifest['prediction_sources']:
        if item['status']!='MISSING_PREDICTIONS':continue
        vp=Path(item['validation']);tp=Path(item['test']);run=tp.parent
        provenance=run/'test_prediction_reuse_provenance.json'
        if not provenance.exists():continue
        pr=json.loads(provenance.read_text());epoch=int(tp.stem[-3:]);ref=pr['prediction_references'][str(epoch)];tp=Path(ref['path'])
        if not tp.exists() or not vp.exists():continue
        assert c.sha(tp)==ref['sha256']
        dataset='movielens' if item['group']=='MovieLensSGD_03' else 'kuairand';b=c.bundle(dataset,item['seed']);sp=c.split(dataset,item['seed']);v=np.load(vp)['predictions'].astype(float);t=np.load(tp)['predictions'].astype(float)
        for d in range(3):
            trainix=sp['selection'][b.val_x[sp['selection'],-1]==d]
            for task in range(2):
                fit,a,bb=calibrated_fit(b.val_y[trainix,task],logit(np.clip(v[trainix,task],1e-12,1-1e-12)))
                for name,pred,y,domains,ix in [('selection',v,b.val_y,b.val_x[:,-1],sp['selection']),('diagnostic',v,b.val_y,b.val_x[:,-1],sp['diagnostic']),('test',t,b.test_y,b.test_x[:,-1],np.arange(len(t)))]:
                    rows=ix[domains[ix]==d];yy=y[rows,task];pp=pred[rows,task];z=logit(np.clip(pp,1e-12,1-1e-12));score=a*z+bb;prob=expit(score);auc=roc_auc_score(yy,pp)
                    old=cells[(cells.group==item['group'])&(cells.seed==item['seed'])&(cells.readout==item['readout'])&(cells.domain==d)&(cells.task==task)&(cells.metric=='auc')]
                    residual=None if name!='test' or len(old)!=1 else float(auc-old[item['arm']].iloc[0])
                    new.append(dict(group=item['group'],seed=item['seed'],arm=item['arm'],readout=item['readout'],epoch=epoch,split=name,domain=d,task=task,n=len(rows),a=float(a),b=float(bb),fit_success=bool(fit.success),fit_message=str(fit.message),auc_before=float(auc),auc_after_score=float(roc_auc_score(yy,score)),auc_after_probability=float(roc_auc_score(yy,prob)),auc_delta_score=float(roc_auc_score(yy,score)-auc),logloss_before=float(log_loss(yy,np.clip(pp,1e-12,1-1e-12),labels=[0,1])),logloss_after=float(np.mean(np.logaddexp(0,score)-yy*score)),brier_before=float(np.mean((pp-yy)**2)),brier_after=float(np.mean((prob-yy)**2)),clipped=int(np.sum((pp<1e-12)|(pp>1-1e-12))),unique_before=int(len(np.unique(pp))),unique_after=int(len(np.unique(prob))),stored_auc_residual=residual))
        item.update(status='RECOVERED_CERTIFIED_PREFIX',test=str(tp),test_sha=ref['sha256'],epoch=epoch,validation_sha=c.sha(vp),prefix_provenance=str(provenance),prefix_provenance_sha=c.sha(provenance));recoveries.append(dict(group=item['group'],seed=item['seed'],arm=item['arm'],epoch=epoch))
        print('recovered',item['group'],item['seed'],item['arm'],epoch,flush=True)
    frame=pd.concat([frame,pd.DataFrame(new)],ignore_index=True);assert not frame.duplicated(['group','seed','arm','readout','split','domain','task']).any()
    frame.to_csv(out/'calibration_cell_metrics.csv',index=False);c.dump(out/'readout_manifest.json',manifest)
    status='COMPLETE' if len(frame)==2016 else 'PARTIAL'
    c.dump(out/'calibration_diagnostics.json',dict(status=status,rows=len(frame),fits=int(sum(frame.split=='selection')),failed_fits=int(sum((frame.split=='selection')&~frame.fit_success)),max_auc_score_change=float(frame.auc_delta_score.abs().max()),max_stored_auc_residual=float(frame.stored_auc_residual.abs().max()),fit_population='selection only',method='strict positive logit affine; frozen L2/bounds',user_id_available=False,certified_prefix_recoveries=recoveries))
    c.dump(out/'recovery_receipt.json',dict(status=status,source_sha=c.sha(__file__),new_calibration_rows=len(new),exact_prefix_records=recoveries,remaining_missing=[x for x in manifest['prediction_sources'] if x['status']=='MISSING_PREDICTIONS']))

if __name__=='__main__':main()
