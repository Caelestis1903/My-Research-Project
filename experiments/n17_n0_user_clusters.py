"""Recover native ML user IDs and paired cluster uncertainty of historical fixed models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import n17_common as c

def auc_plan(y,p,users):
    order=np.argsort(p,kind='stable');ss=p[order];starts=np.r_[0,np.flatnonzero(ss[1:]!=ss[:-1])+1]
    return y[order],users[order],starts

def weighted_auc(plan,counts):
    y,u,starts=plan;w=counts[u];pos=np.add.reduceat(w*y,starts);neg=np.add.reduceat(w*(1-y),starts);pn=pos.sum()*neg.sum()
    if pn==0:return np.nan
    return float(np.dot(pos,np.cumsum(neg)-neg*.5)/pn)

def main():
    out=c.OUT/'N0';manifest=json.loads((out/'readout_manifest.json').read_text());raw=Path('/home/xzf/data/ml-1m/ml-1m.csv')
    assert c.sha(raw)=='f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4'
    data=pd.read_csv(raw,usecols=['user_id','rating']);uids=data.user_id.to_numpy();population=pd.read_csv(out/'split_populations.csv');results=[];means={};user_counts=[]
    # Raw row IDs are the original CSV's zero-based integer positions, not encoded OOV IDs.
    for seed in range(2022,2026):
        b=c.bundle('movielens',seed);sp=c.split('movielens',seed)
        for split,ids,domains,ix in [('train',b.train_row_ids,b.train_x[:,-1],np.arange(len(b.train_x))),('selection',b.val_row_ids,b.val_x[:,-1],sp['selection']),('diagnostic',b.val_row_ids,b.val_x[:,-1],sp['diagnostic']),('test',b.test_row_ids,b.test_x[:,-1],np.arange(len(b.test_x)))]:
            for d in range(3):
                rows=ix[domains[ix]==d];n=int(len(np.unique(uids[ids[rows]])));mask=(population.group=='MovieLensSGD_03')&(population.seed==seed)&(population['split']==split)&(population.domain==d);population.loc[mask,'user_clusters']=n
                user_counts.append(dict(seed=seed,split=split,domain=d,unique_users=n))
        test_ids=b.test_row_ids;yy=(data.rating.to_numpy()[test_ids,None]>np.array([3,4])).astype(np.float32);assert np.array_equal(yy,b.test_y)
        _,users=np.unique(uids[test_ids],return_inverse=True);nusers=int(users.max()+1);domains=b.test_x[:,-1].astype(int)
        rng=np.random.default_rng(20261017);counts=rng.multinomial(nusers,np.ones(nusers)/nusers,size=200)
        for readout in ('selected_macro_auc','common_epoch50'):
            arms={}
            for arm in ('SN','SB'):
                record=next(x for x in manifest['prediction_sources'] if x['group']=='MovieLensSGD_03' and x['seed']==seed and x['arm']==arm and x['readout']==readout)
                pred=np.load(record['test'])['predictions'].astype(float);plans=[];ll=[];cellusers=[]
                for d in range(3):
                    ix=np.flatnonzero(domains==d)
                    for task in range(2):
                        plan=auc_plan(b.test_y[ix,task],pred[ix,task],users[ix]);assert abs(weighted_auc(plan,np.ones(nusers))-roc_auc_score(b.test_y[ix,task],pred[ix,task]))<1e-12
                        plans.append(plan);pp=np.clip(pred[ix,task],1e-12,1-1e-12);y=b.test_y[ix,task];ll.append(-(y*np.log(pp)+(1-y)*np.log1p(-pp)));cellusers.append(users[ix])
                auc=np.array([[weighted_auc(plan,w) for plan in plans] for w in counts]);loss=np.array([[float(np.dot(w[u],l)/w[u].sum()) for l,u in zip(ll,cellusers)] for w in counts]);arms[arm]=(auc,loss)
            for metric,j in [('auc',0),('logloss',1)]:
                differences=arms['SB'][j]-arms['SN'][j];valid=np.isfinite(differences).all(1);macro=differences[valid].mean(1);means.setdefault((readout,metric),[]).append(differences)
                results.append(dict(seed=seed,readout=readout,metric=metric,endpoint='macro',users=nusers,valid_replicates=int(sum(valid)),replicates=200,ci95_low=float(np.quantile(macro,.025)),ci95_high=float(np.quantile(macro,.975)),scope='test-user cluster uncertainty conditional on fitted models and frozen selection; not training-seed CI'))
                for cell in range(6):
                    values=differences[valid,cell];results.append(dict(seed=seed,readout=readout,metric=metric,endpoint=f'd{cell//2}t{cell%2}',users=nusers,valid_replicates=int(sum(valid)),replicates=200,ci95_low=float(np.quantile(values,.025)),ci95_high=float(np.quantile(values,.975)),scope='test-user cluster conditional interval'))
        print('ML user clusters',seed,nusers,flush=True)
    for (readout,metric),arrays in means.items():
        a=np.mean(arrays,axis=0);valid=np.isfinite(a).all(1);values=a[valid].mean(1)
        results.append(dict(seed='four_fitted_seed_mean',readout=readout,metric=metric,endpoint='macro',users=nusers,valid_replicates=int(sum(valid)),replicates=200,ci95_low=float(np.quantile(values,.025)),ci95_high=float(np.quantile(values,.975)),scope='same user resample across four fitted seeds; conditional test-user uncertainty only'))
    population.to_csv(out/'split_populations.csv',index=False);pd.DataFrame(user_counts).to_csv(out/'ML_user_cluster_counts.csv',index=False);pd.DataFrame(results).to_csv(out/'ML_paired_user_cluster_intervals.csv',index=False)
    c.dump(out/'user_cluster_receipt.json',dict(status='COMPLETE',raw_file=str(raw),raw_sha=c.sha(raw),user_link='native raw user_id via original row IDs; labels matched exactly',replicates=200,shared_across_tasks_domains_arms_and_fitted_seeds=True,KR_user_ids_available=False,ML_user_ids_available=True,training_seed_inflation=False,source_sha=c.sha(__file__)))

if __name__=='__main__':main()
