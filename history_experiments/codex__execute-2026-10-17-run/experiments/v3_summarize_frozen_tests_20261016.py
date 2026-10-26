from pathlib import Path
import json,time
import pandas as pd
O=Path('/home/xzf/Code/phase0-3-20261016-v3-git/research_plans/2026-10-16_v3/execution_20261016');load=lambda p:json.loads(p.read_text());assert load(O/'analysis/test_all_cohorts_primary_verification/receipt.json')['status']=='PASS';B=load(O/'analysis/test_baseline7_summary/test_seed_summary.json');G=load(O/'test_baseline7_summary_manifest.json');I=load(O/'analysis/test_all_cohorts_primary_verification/test_contrast_summary.json');records=[]
for g in G['groups']:
 pre=next(x['readout'] for x in B if x['group']==g and x['readout'].startswith('common_pre_AULC'))
 for ro in [pre,'common_epoch50','final_epoch','selected_macro_auc','selected_macro_logloss']:
  for metric in ['macro_auc','macro_logloss']:
   r=next(x for x in B if (x['group'],x['readout'],x['metric'])==(g,ro,metric));records.append(r)
 print(g)
 for ro in [pre,'common_epoch50','final_epoch','selected_macro_auc']:
  a=next(x for x in records if (x['group'],x['readout'],x['metric'])==(g,ro,'macro_auc'));ll=next(x for x in records if (x['group'],x['readout'],x['metric'])==(g,ro,'macro_logloss'));print(ro,'SN/SB',round(a['SN_mean'],7),round(a['SB_mean'],7),'gap',round(a['mean'],7),'CI',round(a['ci_low'],7),round(a['ci_high'],7),'signs',a['positive'],a['negative'],'LLgap',round(ll['mean'],7),'seeds',[round(v,7) for v in a['seed_values']])
selected=[]
for x in I:
 if x['cohort'] in ['V5_SGD_final','V8_final','V11_final','V10_final','V4_optimizer_final'] and x['variable']=='macro_auc' and (x['readout'] in ['common_epoch50','selected_macro_auc','final_epoch'] or x['readout'].startswith('common_pre_AULC')):
  selected.append(x);print('INT',x['cohort'],x['condition'],x['readout'],'n',x['n'],'inter',round(x['mean'],7),'CI',round(x['ci_low'],7),round(x['ci_high'],7),'signs',x['positive'],x['negative'],'SNabs',round(x['SN_condition_minus_baseline_mean'],7),'SBabs',round(x['SB_condition_minus_baseline_mean'],7))
f=O/'final_test_scientific_readout_tables.json';assert not f.exists();f.write_text(json.dumps(dict(status='ACTUAL_PRIMARY_REVIEW_TABLES_NOT_AUTOMATIC_DECISION',time=time.time(),baseline_macro_records=records,selected_predeclared_macro_interactions=selected,scope='Onlyfrozenalreadydeclaredendpoints. Allvariables/cells/allseeds remaininoriginaltables; decisionbyprimaryafterreview.'),indent=2)+'\n')
