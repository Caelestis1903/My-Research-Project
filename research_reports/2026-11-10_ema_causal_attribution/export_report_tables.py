"""Export saved scalar evidence and plots; no model, guardian, or fitting imports."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEEDS={2022,2023,2024,2025,2026}
T4=2.7764451052
DIMS=['dataset','objective','anchor','population','descriptor_id','reference_id','contrast_id',
      'contrast','cut','group','background','h','scope','variant','readout','component','space',
      'compared_state','identity_kind','metric','effect']
TERMS=['independent_first','independent_second','deletion_first','deletion_second','interaction','total']
STAGES={'E5':'early','E10':'early','E20':'middle','E35':'late','E50':'late','tF':'tF','Test':'Test'}


def load(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:
        return json.load(stream)


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(row,metric,effect,value):
    return dict(row,metric=metric,effect=effect,value=value)


def cells_from_metrics(rows):
    for row in rows:
        for population,cells in row.get('metrics',{}).items():
            for cell,values in enumerate(cells):
                yield dict({k:v for k,v in row.items() if k!='metrics'},population=population,
                           domain=cell//2,task=cell%2,**values)


def baseline_key(row,descriptor):
    return tuple(row.get(k) for k in ('dataset','objective','seed','anchor','population','domain','task'))+(descriptor,)


def metric_rows(rows,baselines):
    for row in rows:
        for metric in ('auc','logloss'):
            value=row.get(metric)
            yield emit(row,metric,'value',value)
            for reference in ('O','F'):
                baseline=baselines.get(baseline_key(row,reference),{})
                ref=baseline.get(metric)
                if value is not None and ref is not None:
                    derived=dict(row,source_reference_id=row.get('reference_id'),
                        source_reference_state_id=row.get('reference_state_id'),reference_id=reference,
                        reference_state_id=baseline.get('state_id'),reference_epoch=baseline.get('epoch'))
                    yield emit(derived,metric,'delta_'+reference,value-ref)


def summarize(frame,keys):
    if frame.duplicated(keys+['seed']).any():raise ValueError('Repeated seed in summary: '+str(keys))
    grouped=frame.groupby(keys,dropna=False,sort=False)
    result=grouped['value'].agg(n='count',mean='mean',sd='std').reset_index()
    values=grouped.agg(seeds=('seed',list),values=('value',list)).reset_index()
    values['seed_values']=[json.dumps({str(int(s)):None if pd.isna(v) else float(v) for s,v in zip(seeds,vals)})
                           for seeds,vals in zip(values.seeds,values['values'])]
    values=values.drop(columns=['seeds','values'])
    result=result.merge(values,on=keys,validate='one_to_one')
    result['status']=np.where(result.n==5,'COMPLETE','INCOMPLETE_SEEDS')
    result['ci95_low']=np.where(result.n==5,result['mean']-T4*result.sd/np.sqrt(5),np.nan)
    result['ci95_high']=np.where(result.n==5,result['mean']+T4*result.sd/np.sqrt(5),np.nan)
    return result


def export_family(records,directory,name):
    rows=list(records)
    if not rows:return pd.DataFrame()
    frame=pd.DataFrame(rows)
    keys=[k for k in DIMS if k in frame.columns]
    # Preserve all original IDs, descriptors, counts, pair denominators and provenance in cells.
    for col in frame:
        if frame[col].map(lambda v:isinstance(v,(dict,list))).any():
            frame[col]=frame[col].map(lambda v:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v)
    frame.to_csv(directory/f'{name}_cells.csv.gz',index=False)
    seed=frame[keys+['seed','domain','task','value']].copy()
    if not set(seed.seed.unique()).issubset(SEEDS):raise ValueError('Unexpected seed')
    if seed.duplicated(keys+['seed','domain','task']).any():raise ValueError('Duplicate scientific cell in '+name)
    seed['aggregation']='cell'
    groups=seed.groupby(keys+['seed'],dropna=False,sort=False)
    macro=groups.value.agg(value='mean',valid_cells='count',cells='size').reset_index()
    macro.loc[(macro.valid_cells!=6)|(macro.cells!=6),'value']=np.nan
    macro=macro.drop(columns=['valid_cells','cells']).assign(domain=-1,task=-1,aggregation='six_cell_equal')
    seed=pd.concat([seed,macro],ignore_index=True)
    seed.to_csv(directory/f'{name}_per_seed.csv.gz',index=False)
    groupkeys=keys+['domain','task','aggregation']
    summary=summarize(seed,groupkeys)
    summary.to_csv(directory/f'{name}_summary.csv.gz',index=False)
    seed['stage']=seed.anchor.map(STAGES)
    if seed.stage.isna().any():raise ValueError('Unknown anchor in '+name)
    stagekeys=[k for k in groupkeys if k!='anchor']+['stage','seed']
    stage=seed.groupby(stagekeys,dropna=False,sort=False).agg(value=('value','mean'),
        anchors=('anchor',lambda v:','.join(sorted(set(v)))),anchor_count=('anchor','nunique'),
        finite_anchors=('value','count')).reset_index()
    stage.loc[stage.finite_anchors!=stage.anchor_count,'value']=np.nan
    stage.to_csv(directory/f'{name}_stage_per_seed.csv.gz',index=False)
    stage_summary=summarize(stage,[k for k in stagekeys if k!='seed'])
    stage_summary.to_csv(directory/f'{name}_stage_summary.csv.gz',index=False)
    return summary


def plot_family(summary,path,title,filter_rows,label_columns,metrics=None):
    if summary.empty:return False
    view=summary[(summary.aggregation=='six_cell_equal')&filter_rows(summary)].copy()
    if view.empty:return False
    if metrics is not None:view=view[view.metric.isin(metrics)]
    if view.empty:return False
    facets=list(view.groupby(['population','metric'],dropna=False,sort=False))
    fig,axes=plt.subplots(len(facets),1,figsize=(12,max(3,3*len(facets))),squeeze=False)
    order={'E5':0,'E10':1,'E20':2,'E35':3,'E50':4,'tF':5,'Test':6}
    for ax,((population,metric),part) in zip(axes[:,0],facets):
        cols=[c for c in label_columns if c in part]
        part['label']=part[cols].astype(str).agg(' / '.join,axis=1)
        for label,line in part.groupby('label',sort=False):
            line=line.sort_values('anchor',key=lambda v:v.map(order))
            if line.anchor.duplicated().any():raise ValueError('Plot would merge distinct contrasts: '+label)
            ax.errorbar(line.anchor.map(order),line['mean'],yerr=(line.ci95_high-line.ci95_low)/2,
                        marker='o',capsize=2,label=label)
        ticks=[a for a in order if a in set(part.anchor)]
        ax.set_xticks([order[a] for a in ticks],ticks)
        if not (metric in ('auc','logloss') and (part.effect=='value').all()):ax.axhline(0,color='gray',linewidth=.5)
        ax.set_title(f'{population} | {metric} | six-cell mean within seed; 95% t4 CI only for n=5')
        ax.legend(fontsize=6,loc='best',ncol=2);ax.grid(alpha=.2)
    fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
    return True


def plot_group_time(summary,path,title):
    view=summary[(summary.aggregation=='six_cell_equal')&(summary.population=='V')&
                 summary.effect.isin(['independent_first','deletion_first','interaction'])].copy()
    if view.empty:return False
    view['label']=view[['contrast_id','group','background','h','effect']].astype(str).agg(' / '.join,axis=1)
    fig,axes=plt.subplots(2,1,figsize=(13,max(8,.5*view.label.nunique())),squeeze=False)
    for ax,metric in zip(axes[:,0],['auc','logloss']):
        grid=view[view.metric==metric].pivot(index='label',columns='anchor',values='mean')
        grid=grid.reindex(columns=[a for a in STAGES if a in grid.columns])
        limit=max(float(np.nanmax(np.abs(grid.to_numpy()))),1e-15)
        cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#dddddd')
        im=ax.imshow(grid.to_numpy(),aspect='auto',cmap=cmap,vmin=-limit,vmax=limit)
        ax.set_yticks(range(len(grid)),grid.index,fontsize=7);ax.set_xticks(range(len(grid.columns)),grid.columns)
        ax.set_title(metric+' | V | five-seed mean; missing anchor grey; CI in tables')
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
    return True


def export_block(block,directory,manifest):
    index=load(block/'ARTIFACT_INDEX.json');directory.mkdir(parents=True,exist_ok=True)
    def table(name):
        if name not in index:
            manifest['skipped'].append(str(block.name)+'/'+name+': missing table');return []
        source=block/Path(index[name]['path']).name
        if digest(source)!=index[name]['sha256']:raise ValueError('Source checksum mismatch: '+str(source))
        data=load(source)
        manifest['sources'][f'{block.name}/{name}']=dict(path=str(source),sha256=index[name]['sha256'])
        if not isinstance(data,list):return []
        return [dict(r,source_table=name,source_row_id=f'{block.name}/{name}:{i}') for i,r in enumerate(data)]
    interventions=table('INTERVENTION_METRICS')
    baselines={baseline_key(r,r['descriptor_id']):r for r in interventions if r['descriptor_id'] in ('O','F')}
    plots={}
    summary=export_family(metric_rows(interventions,baselines),directory,'interventions')
    plots['stage_O_F']=plot_family(summary,directory/'stage_O_F.png',block.name+' O/F and fixed interventions',
        lambda s:(s.population=='V')&(s.effect=='value')&s.descriptor_id.isin(['O','F','B/S','B/N','C_PRIMARY/recent','C_PRIMARY/old']),['descriptor_id'],['auc','logloss'])
    for name in ('FACTORIAL_EFFECTS','TEMPORAL_EFFECTS','FUNCTIONAL_EFFECTS'):
        rows=table(name)
        summary=export_family((emit(r,r['metric'],term,r.get(term)) for r in rows for term in TERMS),directory,name.lower())
        if name=='TEMPORAL_EFFECTS':
            plots['group_time']=plot_group_time(summary,directory/'group_time.png',block.name+' group × time factorial effects')
    rows=table('PAIR_TRANSITIONS')
    for r in rows:
        if r.get('pairs'):
            c=np.array(r['counts']);delta=(np.arange(3)[None,:]-np.arange(3)[:,None])/2
            assert np.isclose((c*np.maximum(delta,0)).sum()/r['pairs'],r['corrected'])
            assert np.isclose((c*np.maximum(-delta,0)).sum()/r['pairs'],r['broken'])
    summary=export_family((emit(r,'auc_contribution',field,r.get(field)) for r in rows
        for field in ('corrected','broken','delta_auc')),directory,'exact_pair')
    plots['corrected_broken_net']=plot_family(summary,directory/'corrected_broken_net.png',block.name+' exact pair contributions (already normalized)',
        lambda s:(s.population=='V')&(s.descriptor_id=='F')&(s.reference_id=='O'),['effect'])
    for name,fields in [('FUNCTIONAL_MARGIN',['representation_mean','downstream_mean','interaction_mean','interaction_rms','net',
                                           'representation_net','downstream_net','lost_correction_without_representation','lost_correction_without_downstream']),
                        ('FACTORIAL_SCORE_MARGIN',['score_interaction_mean','score_interaction_rms','margin_first_mean','margin_second_mean','margin_interaction_mean','margin_interaction_rms']),
                        ('SOURCE_OVERLAP',['recovered','lost_recent_benefit','correction_not_prior_damage','damage_not_prior_benefit']),
                        ('SOURCE_ASSAY_OVERLAP',['recovered','lost_recent_benefit','correction_not_prior_damage','damage_not_prior_benefit']),
                        ('PAIR_OVERLAP',['corrected_overlap','broken_overlap','first_unique_correction','second_unique_correction'])]:
        summary=export_family((emit(r,field,'value',r.get(field)) for r in table(name) for field in fields),directory,name.lower())
        if name=='FUNCTIONAL_MARGIN':
            plots['h_fused_functional']=plot_family(summary,directory/'h_fused_functional.png',block.name+' O→F h/fused margins and rank changes',
                lambda s:(s.population=='V_mech')&(s.contrast=='O_F')&(s.space=='logit'),['cut','space'],['interaction_mean','net','representation_net','downstream_net'])
        if name=='PAIR_OVERLAP':
            plots['ensemble_overlap']=plot_family(summary,directory/'ensemble_overlap.png',block.name+' ensemble vs F pair overlap',
                lambda s:(s.population=='V_B')&s.descriptor_id.isin(['q_probability','q_logit']),['descriptor_id'],['corrected_overlap','broken_overlap','first_unique_correction','second_unique_correction'])
    for name in ('FUSION_EFFECTS','SOURCE_ASSAY','ENSEMBLE_METRICS'):
        rows=table(name)
        if name!='ENSEMBLE_METRICS':rows=list(cells_from_metrics(rows))
        summary=export_family(metric_rows(rows,baselines),directory,name.lower())
        if name=='FUSION_EFFECTS':
            plots['fusion_deletion']=plot_family(summary,directory/'fusion_deletion.png',block.name+' fusion deletion: modified state minus F',
                lambda s:(s.population=='V')&(s.effect=='delta_F'),['contrast','component'],['auc','logloss'])
    for name in ('BRIDGE_VALUE','BRIDGE_SUMMARY','BRIDGE_FOLDS','FEATURE_MANIFEST'):
        rows=table(name)
        if rows:pd.DataFrame(rows).to_csv(directory/(name+'.csv.gz'),index=False)
        if name=='BRIDGE_SUMMARY' and rows:
            values=pd.DataFrame(rows);view=values[(values.population=='V_B')&values.metric.isin(['utility','delta_static','macro_logloss_delta'])]
            if not view.empty:
                fig,axes=plt.subplots(1,3,figsize=(16,5))
                for ax,metric in zip(axes,['utility','delta_static','macro_logloss_delta']):
                    part=view[view.metric==metric];labels=part.menu+' / '+part.selector
                    ax.barh(labels,part['mean']);ax.set_title(metric+' | V_B');ax.axvline(0,color='gray')
                fig.suptitle(block.name+' saved bridge summaries; no refit');fig.tight_layout()
                fig.savefig(directory/'bridge_action_value.png',dpi=150);plt.close(fig);plots['bridge_action_value']=True
    for name in ('GROUP_SELECTION','CONTRAST_SELECTION','H2_TRIGGER','SOURCE_ASSAY_DECISION','STAR_CROSS_DECISION','ANCHOR_ALIASES'):
        table(name)
    manifest['plots'][block.name]=plots
    for name in ('stage_O_F','corrected_broken_net','group_time','h_fused_functional','fusion_deletion','ensemble_overlap','bridge_action_value'):
        if not plots.get(name):manifest['skipped'].append(block.name+'/'+name+': no matching saved rows')


def endpoints(out,dest,manifest):
    path=out/'ENDPOINTS.csv'
    if not path.exists():manifest['skipped'].append('Test five endpoints: ENDPOINTS.csv not present');return
    raw=pd.read_csv(path);raw.to_csv(dest/'ENDPOINTS_source.csv.gz',index=False)
    manifest['sources']['ENDPOINTS']=dict(path=str(path),sha256=digest(path))
    rows=[]
    for r in raw.to_dict('records'):
        for metric in ('auc','logloss'):
            # Source metrics writes d{domain}t{task}_{metric} scalar columns.
            for d in range(3):
                for t in range(2):
                    key=f'd{d}t{t}_{metric}'
                    if key not in r:raise ValueError('Endpoint cell column missing: '+key)
                    rows.append(dict(r,anchor='Test',population='Test',domain=d,task=t,metric=metric,
                        selected_epoch=r['epoch'],identity_kind='endpoint_value',value=r[key],effect='value'))
    lookup={(r['dataset'],r['objective'],r['seed'],r['domain'],r['task'],r['metric'],r['readout']):r for r in rows}
    export_family(rows,dest,'test_endpoints_values')
    for family,names,references in [('independent_selected',('O','F'),('O','F')),
                                     ('same_time_atF',('F','OatF','S_atF','N_atF'),('OatF','F'))]:
        effects=[]
        for r in rows:
            if r['readout'] not in names:continue
            for ref in references:
                key=tuple(r[k] for k in ('dataset','objective','seed','domain','task','metric'))+(ref,)
                if key not in lookup:continue
                baseline=lookup[key]
                if family=='same_time_atF' and r['epoch']!=baseline['epoch']:raise ValueError('Endpoint time mismatch')
                effects.append(dict(r,identity_kind=family,reference_id=ref,reference_epoch=baseline['epoch'],
                    effect='delta_'+ref,value=r['value']-baseline['value']))
        export_family(effects,dest,'test_endpoints_'+family)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--dest',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();dest=args.dest.resolve()
    if dest==out or out in dest.parents:raise ValueError('Report destination must be outside frozen attempt')
    dest.mkdir(parents=True,exist_ok=True)
    manifest=dict(source_attempt=str(out),exporter_sha256=digest(Path(__file__)),sources={},plots={},skipped=[],
        statistics=dict(seeds=sorted(SEEDS),n_unit='seed',t4=T4,macro='equal mean of six domain-task cells within seed',
            stage='mean available anchors within seed; anchor coverage recorded; tF separate',
            stages=STAGES,ci='only n=5; no cell/epoch replication'),
        semantics={'C_PRIMARY/recent':'F minus recent; deletion, not recent-only',
                   'C_PRIMARY/old':'F minus old; deletion, not old-only','C/{h}/recent':'recent-only on O',
                   'logloss':'raw differences; negative means lower loss',
                   'PAIR_TRANSITIONS.corrected/broken':'already normalized AUC contributions; preserved without second division',
                   'delta_F':'same dataset/objective/seed/anchor/population/domain/task minus F'})
    for dataset in ('movielens','kuairand'):
        for objective in ('clean','gaussian_d0'):
            name=dataset+'__'+objective;block=out/'blocks'/name
            if not (block/'BLOCK_COMPLETE.json').exists():manifest['skipped'].append(name+': BLOCK_COMPLETE.json not present');continue
            export_block(block,dest/name,manifest)
    endpoints(out,dest,manifest)
    dump(dest/'EXPORT_MANIFEST.json',manifest)
    (dest/'SKIPPED.txt').write_text('\n'.join(manifest['skipped'])+'\n')
    print(json.dumps(dict(status='EXPORTED',dest=str(dest),blocks=list(manifest['plots']),skipped=manifest['skipped'])))


if __name__=='__main__':main()
