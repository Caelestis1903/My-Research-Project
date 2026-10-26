"""Recompute historical evaluation estimands; calibrate only on selection rows."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,log_loss
import n17_common as c

def calibrated_fit(y,z):
    def fun(v):
        a=np.exp(v[0]);s=a*z+v[1];res=expit(s)-y
        return float(np.mean(np.logaddexp(0,s)-y*s)+5e-5*np.dot(v,v)),np.array([np.mean(res*a*z),np.mean(res)])+1e-4*v
    fit=minimize(fun,[0.,0.],jac=True,bounds=[(-8,8),(-30,30)],method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-8,'maxiter':500})
    return fit,np.exp(fit.x[0]),fit.x[1]

def main():
    out=c.OUT/'N0';out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((c.HISTORY/'test_baseline7_summary_manifest.json').read_text())
    src=c.HISTORY/'analysis/test_baseline7_summary/seed_cell_test_readouts.csv'
    frame=pd.read_csv(src);ledger=[];weights={};manifests=[];populations=[];calibration=[];pred_sources=[]
    for group,pairs in meta['groups'].items():
        for pair in pairs:
            receipt=json.loads((Path(pair)/'receipt.json').read_text());seed=receipt['seed'];runs=list(map(Path,receipt['source_runs']))
            conf=json.loads((runs[0]/'config.json').read_text());dataset=conf['dataset'];b=c.bundle(dataset,seed);sp=c.split(dataset,seed)
            p=np.asarray(conf['p']);dtest=b.test_x[:,-1].astype(int);r=np.bincount(dtest,minlength=3)/len(dtest)
            weights[(group,seed)]=(p,r)
            budget=max(int(x) for x in pd.read_csv(runs[0]/'metrics_test.csv').epoch)
            budget=min(budget,max(int(x) for x in pd.read_csv(runs[1]/'metrics_test.csv').epoch))
            manifests.append(dict(group=group,seed=seed,budget=budget,source_runs=list(map(str,runs)),selected_epochs=receipt['selected_epochs'],selection_population='fixed validation selection half',evaluation_population='historically exposed test',train_weights=p.tolist(),evaluation_weights=r.tolist()))
            for split_name,ds,yy,ix in [('train',b.train_x[:,-1],b.train_y,np.arange(len(b.train_y))),('selection',b.val_x[:,-1],b.val_y,sp['selection']),('diagnostic',b.val_x[:,-1],b.val_y,sp['diagnostic']),('test',dtest,b.test_y,np.arange(len(b.test_y)))]:
                for d in range(3):
                    rows=ix[ds[ix]==d];y=yy[rows].astype(int)
                    populations.append(dict(group=group,seed=seed,split=split_name,domain=d,n=len(rows),t0_pos=int(y[:,0].sum()),t1_pos=int(y[:,1].sum()),t0_neg=int(len(rows)-y[:,0].sum()),t1_neg=int(len(rows)-y[:,1].sum()),joint00=int(np.sum((y[:,0]==0)&(y[:,1]==0))),joint01=int(np.sum((y[:,0]==0)&(y[:,1]==1))),joint10=int(np.sum((y[:,0]==1)&(y[:,1]==0))),joint11=int(np.sum((y[:,0]==1)&(y[:,1]==1))),user_clusters=None))
            for arm,run in zip(('SN','SB'),runs):
                selected=receipt['selected_epochs']['macro_auc'][arm]
                for label,epoch in [('selected_macro_auc',selected),('common_epoch50',50)]:
                    vp=run/f'validation_epoch{epoch:03d}.npz';tp=run/f'test_epoch{epoch:03d}.npz'
                    if not vp.exists() or not tp.exists():
                        pred_sources.append(dict(group=group,seed=seed,arm=arm,readout=label,status='MISSING_PREDICTIONS',validation=str(vp),test=str(tp)));continue
                    v=np.load(vp)['predictions'].astype(float);t=np.load(tp)['predictions'].astype(float)
                    pred_sources.append(dict(group=group,seed=seed,arm=arm,readout=label,epoch=int(epoch),validation=str(vp),validation_sha=c.sha(vp),test=str(tp),test_sha=c.sha(tp),status='AVAILABLE'))
                    for d in range(3):
                        trainix=sp['selection'][b.val_x[sp['selection'],-1]==d]
                        for task in range(2):
                            z=logit(np.clip(v[trainix,task],1e-12,1-1e-12));fit,a,bb=calibrated_fit(b.val_y[trainix,task],z)
                            for name,pred,y,domains,ix in [('selection',v,b.val_y,b.val_x[:,-1],sp['selection']),('diagnostic',v,b.val_y,b.val_x[:,-1],sp['diagnostic']),('test',t,b.test_y,dtest,np.arange(len(t)))]:
                                rows=ix[domains[ix]==d];yy=y[rows,task];pp=pred[rows,task];zz=logit(np.clip(pp,1e-12,1-1e-12));score=a*zz+bb;new=expit(score)
                                auc=roc_auc_score(yy,pp);auc_score=roc_auc_score(yy,score);auc_prob=roc_auc_score(yy,new)
                                historical=frame[(frame.group==group)&(frame.seed==seed)&(frame.readout==label)&(frame.domain==d)&(frame.task==task)&(frame.metric=='auc')]
                                delta_stored=None if name!='test' or len(historical)!=1 else float(auc-historical[arm].iloc[0])
                                calibration.append(dict(group=group,seed=seed,arm=arm,readout=label,epoch=int(epoch),split=name,domain=d,task=task,n=len(rows),a=float(a),b=float(bb),fit_success=bool(fit.success),fit_message=str(fit.message),auc_before=float(auc),auc_after_score=float(auc_score),auc_after_probability=float(auc_prob),auc_delta_score=float(auc_score-auc),logloss_before=float(log_loss(yy,np.clip(pp,1e-12,1-1e-12),labels=[0,1])),logloss_after=float(np.mean(np.logaddexp(0,score)-yy*score)),brier_before=float(np.mean((pp-yy)**2)),brier_after=float(np.mean((new-yy)**2)),clipped=int(np.sum((pp<1e-12)|(pp>1-1e-12))),unique_before=int(len(np.unique(pp))),unique_after=int(len(np.unique(new))),stored_auc_residual=delta_stored))
            print('N0',group,seed,'calibration rows',len(calibration),flush=True)
    for (group,seed,readout,metric),part in frame.groupby(['group','seed','readout','metric']):
        assert len(part)==6
        p,r=weights[(group,seed)]
        for name,w in [('uniform',np.ones(3)/3),('train_p',p),('evaluation_p',r)]:
            cellw=w[part.domain.to_numpy(dtype=int)]/2
            sn=float(np.dot(part.SN,cellw));sb=float(np.dot(part.SB,cellw))
            ledger.append(dict(group=group,seed=int(seed),readout=readout,metric=metric,weights=name,SN=sn,SB=sb,difference=sb-sn,SN_epoch=None if part.SN_epoch.isna().all() else int(part.SN_epoch.iloc[0]),SB_epoch=None if part.SB_epoch.isna().all() else int(part.SB_epoch.iloc[0]),evaluation_population='test',selection_population='selection half of validation',actual_seed_unit=True))
    tab=pd.DataFrame(ledger);tab.to_csv(out/'readout_ledger.csv',index=False)
    cell=frame[(frame.group=='calAdam_0003')&(frame.readout=='common_epoch50')&(frame.metric=='auc')].copy()
    for name in ('uniform','train_p','evaluation_p'):
        cell[name+'_contribution']=[row.difference*({'uniform':np.ones(3)/3,'train_p':weights[(row.group,row.seed)][0],'evaluation_p':weights[(row.group,row.seed)][1]}[name][int(row.domain)])/2 for row in cell.itertuples()]
    cell.to_csv(out/'calAdam_E50_cell_contributions.csv',index=False)
    means=cell.groupby(['domain','task'])[['difference','uniform_contribution','train_p_contribution','evaluation_p_contribution']].mean().reset_index()
    c.dump(out/'cell_contributions.json',dict(cells=means.to_dict('records'),macro_delta=float(means.uniform_contribution.sum()),train_p_delta=float(means.train_p_contribution.sum()),evaluation_p_delta=float(means.evaluation_p_contribution.sum())))
    interp=[]
    for seed,part in cell.groupby('seed'):
        p,r=weights[('calAdam_0003',seed)]
        for alpha in np.linspace(0,1,101):interp.append(dict(seed=int(seed),alpha=float(alpha),difference=float(np.dot(part.difference,((1-alpha)/3+alpha*r)[part.domain.to_numpy(dtype=int)]/2))))
    pd.DataFrame(interp).to_csv(out/'evaluation_weight_curve.csv',index=False)
    pd.DataFrame(populations).to_csv(out/'split_populations.csv',index=False);pd.DataFrame(calibration).to_csv(out/'calibration_cell_metrics.csv',index=False)
    c.dump(out/'readout_manifest.json',dict(source_table=str(src),source_sha=c.sha(src),runs=manifests,prediction_sources=pred_sources,scope='historical recomputation; no new model test read'))
    residuals=[abs(x['stored_auc_residual']) for x in calibration if x['stored_auc_residual'] is not None]
    c.dump(out/'calibration_diagnostics.json',dict(status='COMPLETE' if len(calibration)==7*4*2*2*6*3 else 'PARTIAL',rows=len(calibration),fits=sum(x['split']=='selection' for x in calibration),failed_fits=sum(x['split']=='selection' and not x['fit_success'] for x in calibration),max_auc_score_change=max(abs(x['auc_delta_score']) for x in calibration),max_stored_auc_residual=max(residuals,default=None),fit_population='selection only',method='strict positive logit affine; frozen L2/bounds',user_id_available=False))
    summary=tab.groupby(['group','readout','metric','weights']).agg(n=('seed','nunique'),difference=('difference','mean')).reset_index();summary.to_csv(out/'seed_means.csv',index=False)
    c.dump(out/'receipt.json',dict(status='COMPLETE',source_sha=c.sha(__file__),ledger_rows=len(tab),historical_cells=len(frame),calibration_rows=len(calibration),new_training_runs=0))

if __name__=='__main__':main()
