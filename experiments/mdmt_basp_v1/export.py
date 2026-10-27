"""Export measured risks, predictions, mechanism data and explicit missing states."""
from pathlib import Path
import csv
import json
import numpy as np
import torch
from .identity import write_json,plain
from .metrics import evaluate,prototype_metrics


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows=[plain(row) for row in rows]
    if not rows:
        path.write_text('status\nNOT_TRIGGERED\n')
        return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value
                             for key,value in row.items()})


def export_prediction(root,role,state,view,y,mask,logits,extras,identity):
    root=Path(root)
    metrics=evaluate(y,logits,view['domain'],mask)
    write_json(root/f'{role}_{state}_metrics.json',dict(**metrics,identity=identity))
    arrays=dict(row_id=view['ids'],domain=view['domain'],labels=y,mask=mask,logits=logits)
    arrays.update({k:v for k,v in extras.items() if isinstance(v,np.ndarray)})
    np.savez_compressed(root/f'{role}_{state}_predictions.npz',**arrays)
    write_csv(root/f'{role}_{state}_CELL_METRICS.csv',metrics['cells'])
    if extras.get('need_mass') is not None:
        write_csv(root/f'{role}_{state}_PROTOTYPE_METRICS.csv',
                  prototype_metrics(y,logits,view['domain'],extras['need_mass']))
    return metrics


def calibration(target,prediction,constant=None):
    y,p=np.asarray(target,np.float64),np.asarray(prediction,np.float64)
    if y.shape!=p.shape:
        raise ValueError('Gain target and prediction differ')
    if not y.size:
        return dict(status='UNMEASURED')
    base=np.mean(y,axis=0,keepdims=True) if constant is None else np.asarray(constant)
    sse=float(np.mean((p-y)**2));ssec=float(np.mean((base-y)**2))
    eligible=np.abs(y)>=1e-4
    sign=[]
    for positive in (False,True):
        chosen=eligible&((y>0)==positive)
        if chosen.any():
            sign.append(float(((p[chosen]>0)==positive).mean()))
    off=np.zeros((*y.shape[:-1],1))
    ya=np.concatenate([off,y],-1);pa=np.concatenate([off,p],-1)
    chosen=np.take_along_axis(ya,pa.argmax(-1)[...,None],-1)[...,0]
    pp=p.ravel();yy=y.ravel();center=pp-pp.mean();variance=float(center@center)
    slope=float(center@(yy-yy.mean())/variance) if variance>0 else None
    return dict(mse=sse,constant_mse=ssec,skill=1-sse/ssec if ssec>0 else None,
                sign_balanced_accuracy=float(np.mean(sign)) if sign else None,
                near_zero_fraction=float((~eligible).mean()),
                best_action_regret=float(np.mean(ya.max(-1)-chosen)),
                calibration_slope=slope,target_rms=float(np.sqrt(np.mean(y*y))),
                prediction_rms=float(np.sqrt(np.mean(p*p))))


def fold_ridge_skill(features,target,folds,ridge=.01):
    x,y=np.asarray(features,np.float64),np.asarray(target,np.float64)
    shape=y.shape;y=y.reshape(len(y),-1)
    prediction=np.zeros_like(y);constant=np.zeros_like(y)
    for fold in (0,1):
        train,test=np.asarray(folds)!=fold,np.asarray(folds)==fold
        if not train.any() or not test.any():
            return dict(status='INSUFFICIENT_SUPPORT'),None
        mean=x[train].mean(0);std=np.maximum(x[train].std(0),1e-6)
        xx=(x[train]-mean)/std;xt=(x[test]-mean)/std
        xx=np.column_stack([xx,np.ones(len(xx))]);xt=np.column_stack([xt,np.ones(len(xt))])
        penalty=np.eye(xx.shape[1])*ridge;penalty[-1,-1]=0
        coef=np.linalg.solve(xx.T@xx+penalty,xx.T@y[train])
        prediction[test]=xt@coef;constant[test]=y[train].mean(0)
    result=calibration(y.reshape(shape),prediction.reshape(shape),constant.reshape(shape))
    result.update(predictor='two_fold_ridge',full_pipeline_oof=False)
    return result,prediction.reshape(shape)


def state_manifest(identity,epoch,step,state):
    return dict(**identity,epoch=int(epoch),main_step=int(step),state=state,storage='RAM_ONLY')


