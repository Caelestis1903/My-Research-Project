from pathlib import Path
import json,pandas as pd
P=Path(__file__).resolve().parent
S=json.loads((P/'FINAL_performance/seed_summary.json').read_text());Q=json.loads((P/'N2_quota_performance/seed_summary.json').read_text());H=json.loads((P/'FINAL_performance/primary_common_eta_Holm.json').read_text());G=json.loads((P/'N3_confirmation_summary/confirmation_gate.json').read_text());D=json.loads((P/'FINAL_domain_context/domain_tradeoff_pattern_counts.json').read_text());C=pd.read_csv(P/'FINAL_domain_context/domain_task_baseline_population_context.csv');groups=['N1_common_adam','N1_common_sgd','N2_quota_adam','N5_common_adam','N5_own_adam']
def get(group,metric='macro_auc',readout='selected'):return next(x for x in S if x['group']==group and x['metric']==metric and x['readout']==readout)
def qs(effect,metric='macro_auc',readout='selected'):return next(x for x in Q if x['effect']==effect and x['metric']==metric and x['readout']==readout)
def fmt(x):return f'{x:+.6f}'
def ci(x):return '['+', '.join(fmt(v) for v in x['ci95'])+']'
def md(df):
 df=df.copy();return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'
def selected_table(metric='macro_auc'):
 return md(pd.DataFrame([dict(group=g,SN=f"{get(g,metric)['SN_mean']:.6f}",SB=f"{get(g,metric)['SB_mean']:.6f}",delta=fmt(get(g,metric)['mean']),t95=ci(get(g,metric)),pos_neg=f"{get(g,metric)['positive']}/{get(g,metric)['negative']}") for g in groups]))
def stages_table():
 return md(pd.DataFrame([dict(group=g,**{r:fmt(get(g,'macro_auc',r)['mean']) for r in ['E10','E20','E50','E100','selected','AULC1_20','AULC1_100']}) for g in groups]))
def quota_table(metric='macro_auc'):
 return md(pd.DataFrame([dict(effect=e,mean=fmt(qs(e,metric)['mean']),t95=ci(qs(e,metric)),pos_neg=f"{qs(e,metric)['positive']}/{qs(e,metric)['negative']}") for e in ['IID_SB_minus_SN','quota_SB_minus_SN','quota_interaction','SN_quota_minus_IID','SB_quota_minus_IID']]))
selected=[x for x in S if x['readout']=='selected'];pd.DataFrame(selected).to_csv(P/'FINAL_SELECTED_ENDPOINTS.csv',index=False);pd.DataFrame([x for x in S if x['metric']=='macro_auc']).to_csv(P/'FINAL_STAGE_MACRO_AUC.csv',index=False);pd.DataFrame([x for x in selected if x['metric'].startswith('d') and 't' in x['metric'] and x['metric'].endswith(('_auc','_logloss'))]).to_csv(P/'FINAL_SELECTED_CELL_METRICS.csv',index=False)
weighted=md(pd.DataFrame([dict(group=g,macro=fmt(get(g)['mean']),train_p_cell=fmt(get(g,'train_p_cell_auc')['mean']),test_p_cell=fmt(get(g,'evaluation_p_cell_auc')['mean'])) for g in groups]))
base=C[(C.readout=='selected')&(C.metric=='auc')&(C.group.isin(['N1_common_adam','N5_common_adam']))].groupby(['group','domain','task'],as_index=False)[['train_domain_n','test_cell_n','test_positive_rate','SN_reference','SB_candidate','difference']].mean();base['LL_delta']=[get(r.group,f'd{r.domain}t{r.task}_logloss')['mean'] for r in base.itertuples()]
for col in ['train_domain_n','test_cell_n']:base[col]=base[col].astype(int)
for col in ['test_positive_rate','SN_reference','SB_candidate','difference','LL_delta']:base[col]=base[col].map(lambda x:f'{x:.6f}')
base_table=md(base)
trade=md(pd.DataFrame([dict(group=x['group'],strict_both_small_gain_large_loss=x['both_smaller_gain_larger_loss_seeds'],either_small_gain_large_loss=x['either_smaller_gain_larger_loss_seeds'],reverse_both_small_loss_large_gain=x['both_smaller_loss_larger_gain_seeds'],practical_strict=x['practical_both_smaller_gain_larger_loss_seeds']) for x in D['results'] if x['readout']=='selected' and x['group'] in groups]))
num=dict(status='COMPLETE',primary=H,selected=selected,macro_AUC_all_stages=[x for x in S if x['metric']=='macro_auc'],quota_selected=[x for x in Q if x['readout']=='selected'],local_confirmation_gate=G,method_guards=json.loads((P/'FINAL_method_guards/numerical_benefit_guard_summary.json').read_text()),quota_absolute_method_guards=json.loads((P/'N2_absolute_quota_method_guards/numerical_benefit_guard_summary.json').read_text()),domain_patterns=D,scope='All statistics from frozen actual raw outputs. Primary Holm family2; other rows secondary pointwise. Eight training seeds per group, aliases not independent repeats.')
(P/'FINAL_NUMERIC_FINDINGS.json').write_text(json.dumps(num,ensure_ascii=False,indent=2)+'\n')
