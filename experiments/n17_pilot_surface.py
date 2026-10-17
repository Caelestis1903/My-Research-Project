"""Primary-only frozen N1 tuning response surface; not confirmation evidence."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import n17_common as c


def main():
    manifest=json.loads((c.OUT/'N1_pilot_manifest.json').read_text());status=json.loads((c.OUT/'N1_pilot_manifest.status.json').read_text())
    freeze=json.loads((c.OUT/'N1_winners_frozen.json').read_text());assert freeze['status']=='FROZEN'
    assert all(x['status'] in ('COMPLETE','FAILED') for x in status['tasks'].values())
    out=c.OUT/'N1_pilot_surface';out.mkdir(exist_ok=False);rows=[]
    for task in manifest['tasks']:
        if status['tasks'][task['name']]['status']!='COMPLETE':continue
        run=c.OUT/'runs'/task['name'];conf=json.loads((run/'config.json').read_text())
        trajectory=pd.read_csv(run/'trajectory.csv').set_index('epoch');metric=pd.read_csv(run/'metrics_selection.csv')
        assert list(trajectory.index)==list(range(1,101));ll=metric.groupby('epoch').logloss.mean()
        for horizon in (10,20,50,100):
            selected=int(trajectory.loc[1:horizon].selection_macro_auc.idxmax())
            row=dict(name=task['name'],optimizer=conf['optimizer'],arm=conf['arm'],seed=conf['seed'],lr=conf['lr'],horizon=horizon,
                selected_epoch=selected,best_selection_auc=float(trajectory.loc[selected,'selection_macro_auc']),
                selection_LL_at_selected=float(ll.loc[selected]),fixed_selection_auc=float(trajectory.loc[horizon,'selection_macro_auc']),
                fixed_selection_LL=float(ll.loc[horizon]),fixed_train_J=float(trajectory.loc[horizon,'train_empirical_objective']),
                AULC_auc_1_H=float(trajectory.loc[1:horizon].selection_macro_auc.mean()),steps=int(trajectory.loc[horizon,'steps']),examples=int(trajectory.loc[horizon,'examples']))
            rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(out/'raw_seed_response.csv',index=False)
    own=table.groupby(['optimizer','arm','lr','horizon']).agg(n=('seed','nunique'),mean=('best_selection_auc','mean'),minimum=('best_selection_auc','min'),maximum=('best_selection_auc','max')).reset_index()
    common=table.groupby(['optimizer','lr','horizon']).agg(n=('seed','count'),mean=('best_selection_auc','mean'),minimum=('best_selection_auc','min'),maximum=('best_selection_auc','max')).reset_index()
    own.to_csv(out/'q_response.csv',index=False);common.to_csv(out/'common_eta_response.csv',index=False);choices=[]
    for optimizer in ('adam','sgd'):
        for horizon in (10,20,50,100):
            candidate=common[(common.optimizer==optimizer)&(common.horizon==horizon)&(common.n==4)].sort_values(['mean','lr'],ascending=[False,True])
            assert len(candidate);winner=float(candidate.iloc[0].lr)
            if horizon==100:assert winner==freeze['winners'][optimizer]['common']
            choices.append(dict(optimizer=optimizer,horizon=horizon,arm='COMMON',lr=winner,n=4))
            for arm in ('SN','SB'):
                candidate=own[(own.optimizer==optimizer)&(own.arm==arm)&(own.horizon==horizon)&(own.n==2)].sort_values(['mean','lr'],ascending=[False,True]);assert len(candidate)
                winner=float(candidate.iloc[0].lr)
                if horizon==100:assert winner==freeze['winners'][optimizer]['own'][arm]
                choices.append(dict(optimizer=optimizer,horizon=horizon,arm=arm,lr=winner,n=2))
    pd.DataFrame(choices).to_csv(out/'horizon_winner_context.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharey=True)
    for i,optimizer in enumerate(('adam','sgd')):
        for j,arm in enumerate(('SN','SB','COMMON')):
            source=common[common.optimizer==optimizer] if arm=='COMMON' else own[(own.optimizer==optimizer)&(own.arm==arm)]
            for horizon,color in zip((10,20,50,100),('tab:blue','tab:orange','tab:green','tab:red')):
                part=source[source.horizon==horizon].sort_values('lr');ax=axes[i,j]
                ax.plot(part.lr,part['mean'],marker='o',color=color,label=f'H{horizon}')
                ax.fill_between(part.lr,part.minimum,part.maximum,color=color,alpha=.08)
            ax.set_xscale('log');ax.set_title(f'{optimizer} {arm}');ax.set_xlabel('learning rate');ax.grid(alpha=.2)
        axes[i,0].set_ylabel('best selection macro AUC')
    axes[0,2].legend(fontsize=8);fig.suptitle('Tuning only: means and raw range (q n=2; common n=4 runs)')
    fig.tight_layout();fig.savefig(out/'selection_response.pdf');fig.savefig(out/'selection_response.png',dpi=160);plt.close(fig)
    c.dump(out/'receipt.json',dict(status='COMPLETE_TUNING_READOUT',source_sha=c.sha(__file__),rows=len(table),winners_sha=c.sha(c.OUT/'N1_winners_frozen.json'),
        scope='selection-only descriptive tuning response; raw range is not a confidence interval; H100 winners already frozen; horizons do not create extra independent seeds; no test predictions'))
    print('pilot surface rows',len(table),'winners independently match frozen selection')


if __name__=='__main__':main()
