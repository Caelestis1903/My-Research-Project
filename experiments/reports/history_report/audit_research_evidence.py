from pathlib import Path
import json, re, hashlib, itertools
import numpy as np
import pandas as pd
import math
_nodes,_weights=np.polynomial.legendre.leggauss(192)
def t_cdf(v,df):
    x=abs(v)*(_nodes+1)/2
    c=math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2))/math.sqrt(df*math.pi)
    z=abs(v)/2*float(np.sum(_weights*c*(1+x*x/df)**(-(df+1)/2)))
    return .5+math.copysign(z,v)
def t_quantile(p,df):
    lo,hi=0.,100.
    for _ in range(70):
        mid=(lo+hi)/2
        if t_cdf(mid,df)<p: lo=mid
        else: hi=mid
    return (lo+hi)/2
assert abs(t_quantile(.975,3)-3.182446305)<1e-7

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
OUT = ROOT / 'research_diagnosis_20260909'
OUT.mkdir(exist_ok=True)

# Read all supplied reports, including every table; retain source fingerprints.
manifest=[]
for p in sorted(ROOT.glob('*.md')):
    s=p.read_text(encoding='utf-8-sig')
    manifest.append(dict(file=p.name,bytes=p.stat().st_size,lines=len(s.splitlines()),
        table_rows=sum(x.startswith('|') for x in s.splitlines()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

def summary(x):
    x=np.asarray(x,dtype=float); n=len(x); se=x.std(ddof=1)/np.sqrt(n)
    q=t_quantile(.975,n-1); ci=[x.mean()-q*se,x.mean()+q*se]
    return dict(n=n,mean=float(x.mean()),std=float(x.std(ddof=1)),ci95=list(map(float,ci)),
                p_t=max(0.,2*(1-t_cdf(abs(x.mean()/se),n-1))),positive=int((x>0).sum()),
                median=float(np.median(x)),min=float(x.min()),max=float(x.max()))

cells=pd.read_csv(EXP/'P0_5_comp1_x_weighting/P0_5_cells.csv')
run=cells.groupby(['setting','seed','weighting','comp1']).joint_auc.mean().unstack(['weighting','comp1'])
diffs=pd.DataFrame({'gain_equal':run['equal','on']-run['equal','off'],
 'gain_sample':run['sample','on']-run['sample','off'],
 'weighting_gain':run['sample','off']-run['equal','off']})
diffs['interaction']=diffs.gain_sample-diffs.gain_equal
result={'P0_5':{}}
for k in diffs:
    result['P0_5'][k]={'run12_descriptive':summary(diffs[k]),'seed4_block':summary(diffs[k].groupby('seed').mean()),
                       'settings':diffs[k].groupby('setting').mean().to_dict()}
result['P0_5']['negative_cells']=cells.groupby(['weighting','comp1']).delta_transfer.apply(lambda x:int((x<0).sum())).to_dict()
result['P0_5']['negative_cells']={str(k):v for k,v in result['P0_5']['negative_cells'].items()}
result['P0_5']['L_details']=[]
for (w,c),g in cells[cells.role=='L'].groupby(['weighting','comp1']):
    byseed=g.groupby('seed').delta_transfer.mean()
    means=g.groupby(['setting','domain','task']).delta_transfer.mean()
    result['P0_5']['L_details'].append({'weighting':w,'comp1':c,'seed4_block':summary(byseed),
      'negative_seed_cells':int((g.delta_transfer<0).sum()),'negative_mean_cells':int((means<0).sum()),
      'cell_means':{str(k):v for k,v in means.items()}})
diffs.to_csv(OUT/'paired_run_differences.csv',encoding='utf-8-sig')

c4=pd.read_csv(EXP/'P0_4_loss_weighting/P0_4_cells.csv')
result['P0_4_censored']=c4[c4.censored==1].drop_duplicates(['weighting','selection','setting','seed'])[['weighting','selection','setting','seed']].to_dict('records')

pairs=pd.read_csv(EXP/'P1_2_curriculum/P1_2_paired.csv')
pairs=pairs[pd.to_numeric(pairs.seed,errors='coerce').notna()].copy()
result['P1_2']={}
for pair,g in pairs.groupby('pair'):
    result['P1_2'][pair]={}
    for metric in ['delta_overall','delta_worst']:
        x=pd.to_numeric(g[metric]); byseed=g.assign(v=x).groupby('seed').v.mean()
        result['P1_2'][pair][metric]={'run12_descriptive':summary(x),'seed4_block':summary(byseed)}

# Complete per-run evidence from saved configs and labeled metric columns.
long=[]
for dirname in ['P0_1_role_reversal','P0_2_ratio_sweep','P0_3_mmoe_transfer',
                'P0_4_loss_weighting','P0_5_comp1_x_weighting','P1_2_curriculum']:
    for p in sorted((EXP/dirname).rglob('metrics_summary.csv')):
        cfgp=p.with_name('config.json')
        cfg=json.loads(cfgp.read_text(encoding='utf-8-sig')) if cfgp.exists() else {}
        m=pd.read_csv(p).iloc[0].to_dict()
        sizes={int(k):int(v) for k,v in re.findall(r'(\d+)=(\d+)',str(m.get('domain_train_samples','')))}
        oracle=m.get('method')=='oracle'
        for col in m:
            match=re.fullmatch(r'auc_d(\d+)t(\d+)',col)
            if not match or pd.isna(m[col]): continue
            d,t=map(int,match.groups())
            if oracle and str(d)!=str(m['domain']): continue
            weight=cfg.get('loss_weighting',cfg.get('loss_weighting_arg','equal'))
            comp=cfg.get('momentum_mode',cfg.get('comp1','none'))
            cur=cfg.get('curriculum_mode','none'); ratio=cfg.get('ratio',20 if dirname.startswith('P0_1') else '未报告')
            long.append(dict(Experiment_ID=dirname,Dataset=m.get('dataset','ml-1m'),Setting=m['setting'],Domain=d,
                Task=t,Domain_size=sizes.get(d,'未报告'),Domain_ratio=ratio,
                Domain_share=(sizes[d]/sum(sizes.values()) if d in sizes and not oracle else '未报告'),
                Model=m.get('model','未报告'),Baseline='同setting/seed单域参考' if oracle else '同模型、setting和协议的无改动臂',
                Modification=m.get('method','未报告'),Curriculum=cur,Kappa=cfg.get('kappa0',0 if cur=='none' else '未报告'),
                Momentum=comp,Gamma0=cfg.get('gamma0','未报告'),Gamma_max=cfg.get('gamma_max','未报告'),
                Learning_rate=cfg.get('learning_rate','未报告'),Optimizer='Adam（源码核实；comp1共享β1=0）',
                Weighting=weight,Seed=int(m['seed']),Evaluation_metric='domain-task test AUC',Value=float(m[col]),
                Logloss=m.get(f'log_d{d}t{t}','未报告'),Best_epoch=m.get('best_epoch','未报告'),
                Stopped_epoch=m.get('stopped_epoch','未报告'),Arm=cfg.get('arm',m.get('method','未报告')),
                Source=str(p.relative_to(EXP))))
df=pd.DataFrame(long)
df.to_csv(OUT/'evidence_run_intermediate.csv',index=False,encoding='utf-8-sig')
groupcols=[x for x in df.columns if x not in ['Seed','Value','Logloss','Best_epoch','Stopped_epoch','Source']]
agg=[]
for key,g in df.groupby(groupcols,dropna=False,sort=False):
    row=dict(zip(groupcols,key)); row.update(Seed=','.join(map(str,sorted(g.Seed))),Mean=g.Value.mean(),Std=g.Value.std(ddof=1),
        N=len(g),Delta_vs_baseline='未报告',Stable='未判断',Previous_consistency='见诊断正文',Source='; '.join(g.Source))
    agg.append(row)
a=pd.DataFrame(agg)
# Baseline pairing is within stage/model/setting/weighting; curriculum compares to same momentum none-cur arm.
for i,row in a.iterrows():
    if row.Modification=='oracle':
        a.at[i,'Previous_consistency']='新对齐单域参照；与旧全量oracle不可直接比较'
        continue
    base=a[(a.Experiment_ID==row.Experiment_ID)&(a.Model==row.Model)&(a.Setting==row.Setting)&
           (a.Domain==row.Domain)&(a.Task==row.Task)&(a.Domain_ratio==row.Domain_ratio)&
           (a.Weighting==row.Weighting)&(a.Curriculum=='none')&(a.Modification!='oracle')]
    if row.Curriculum!='none': base=base[base.Momentum==row.Momentum]
    else: base=base[base.Momentum=='none']
    if row.Experiment_ID=='P0_4_loss_weighting':
        base=a[(a.Experiment_ID==row.Experiment_ID)&(a.Setting==row.Setting)&(a.Domain==row.Domain)&(a.Task==row.Task)&(a.Weighting=='equal')]
    if row.Experiment_ID=='P0_5_comp1_x_weighting':
        base=a[(a.Experiment_ID=='P0_4_loss_weighting')&(a.Setting==row.Setting)&(a.Domain==row.Domain)&(a.Task==row.Task)&(a.Weighting=='sample')]
    if len(base)==1:
        br=base.iloc[0]; a.at[i,'Delta_vs_baseline']=f'{row.Mean-br.Mean:.6f}'
        a.at[i,'Baseline']=f'{br.Experiment_ID}/{br.Arm}/{br.Weighting}/{br.Momentum}'
        ga=df[(df.Experiment_ID==row.Experiment_ID)&(df.Setting==row.Setting)&(df.Domain==row.Domain)&(df.Task==row.Task)&(df.Domain_ratio==row.Domain_ratio)&(df.Arm==row.Arm)&(df.Weighting==row.Weighting)]
        gb=df[(df.Experiment_ID==br.Experiment_ID)&(df.Setting==br.Setting)&(df.Domain==br.Domain)&(df.Task==br.Task)&(df.Domain_ratio==br.Domain_ratio)&(df.Arm==br.Arm)&(df.Weighting==br.Weighting)]
        z=ga[['Seed','Value']].merge(gb[['Seed','Value']],on='Seed',suffixes=('_a','_b'))
        dv=z.Value_a-z.Value_b
        a.at[i,'Stable']=f'正/负/零={int((dv>0).sum())}/{int((dv<0).sum())}/{int((dv==0).sum())}；非显著性判据'
        if row.Curriculum!='none': consistency='旧课程未见稳定贡献；本行仅单cell，跨期稳定性未验证'
        elif row.Momentum!='none': consistency='与旧总体正向一致，非同配置复现' if row.Mean>br.Mean else '反例：未重复旧总体正向增益'
        else: consistency='新目标/比例/角色配置；旧实验未报告同配置'
        a.at[i,'Previous_consistency']=consistency
a.to_csv(OUT/'evidence_summary_intermediate.csv',index=False,encoding='utf-8-sig')
cols=['Experiment_ID','Dataset','Setting','Domain','Task','Domain_size','Domain_share','Domain_ratio','Model','Arm','Baseline','Modification','Weighting','Curriculum','Kappa','Momentum','Gamma0','Gamma_max','Learning_rate','Optimizer','Seed','Evaluation_metric','Mean','Std','Delta_vs_baseline','Stable','Previous_consistency']
with (OUT/'EXPERIMENT_EVIDENCE_TABLE.md').open('w',encoding='utf-8') as f:
    f.write('# Experiment Evidence Table\n\n全部新阶段的逐domain×task×配置证据。Dataset=MovieLens-1M；Evaluation metric=domain-task test AUC。Mean、Std由保存的带列名指标复算，Std为4 seeds样本标准差ddof=1；独立训练重复只有4个seed。\n\n')
    f.write('Domain为原身份，Setting为大小角色轮换。ratio为joint场景L:S；oracle行的size为自身训练数。Task 0=rating>3，Task 1=rating>4（预处理代码核实）。\n\n')
    f.write('Baseline：无课程动量臂与同weighting无动量比较；课程臂与同weighting、同momentum的无课程臂比较；oracle不作方法收益差。P0-5仅新增sample+comp1，其baseline见P0-4；复用证据不可计为独立重复。参数未保存者标未报告，不用当前默认值回填历史配置。\n\n')
    f.write('所有数值来自每run的config.json和metrics_summary.csv；来源逐项见下方分组。稳定列仅为配对符号，不等于统计显著。\n\n')
    for stage,g in a.groupby('Experiment_ID',sort=False):
        f.write('## '+stage+'\n\n')
        f.write('| '+' | '.join(cols[1:])+' |\n| '+' | '.join(['---']*(len(cols)-1))+' |\n')
        for _,r in g.iterrows():
            vals=[]
            for c in cols[1:]:
                v=r[c]
                vals.append(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v).replace('|','/'))
            f.write('| '+' | '.join(vals)+' |\n')
        f.write('\n数据源：`experiments/'+stage+'/**/config.json` 和 `metrics_summary.csv`。\n\n')
result['evidence_counts']={'run_metric_rows':len(df),'summary_rows':len(a),'source_reports':len(manifest),'run_counts':df.groupby('Experiment_ID').Source.nunique().to_dict()}
result['epoch_summary_mismatch']=[]
result['epoch_schema_missing_test_auc']=[]
for source,g in df.groupby('Source'):
    ep=pd.read_csv((EXP/source).with_name('metrics_epoch.csv'))
    if 'test_auc' not in ep:
        result['epoch_schema_missing_test_auc'].append(source)
        continue
    for _,r in g.iterrows():
        z=ep[(ep.epoch==r.Best_epoch)&(ep.domain==r.Domain)&(ep.task==r.Task)]
        if len(z)==1 and abs(float(z.iloc[0].test_auc)-r.Value)>1.1e-6:
            result['epoch_summary_mismatch'].append(dict(source=source,d=int(r.Domain),t=int(r.Task),diff=r.Value-float(z.iloc[0].test_auc)))
p02=pd.read_csv(EXP/'P0_2_ratio_sweep/P0_2_cells.csv')
result['P0_1_gain_signs']={'positive_seed_cells':int((p02[p02.ratio==20].comp1_gain>0).sum()),'total_seed_cells':int((p02.ratio==20).sum()),'positive_mean_cells':int((p02[p02.ratio==20].groupby(['setting','domain','task']).comp1_gain.mean()>0).sum())}
result['P0_2_baseline_exceptions']=p02[((p02.role=='L')&(p02.ratio==50)&(p02.delta_transfer>=0))|((p02.role=='S')&(p02.ratio==50)&(p02.delta_transfer<0))|((p02.ratio==1)&(p02.delta_transfer<0))].to_dict('records')
result['P1_2_arm_means']={}
for arm,g in df[df.Experiment_ID=='P1_2_curriculum'].groupby('Arm'):
    x=g.groupby(['Setting','Seed']).Value.mean()
    result['P1_2_arm_means'][arm]={'mean':float(x.mean()),'setting_seed_std':float(x.std(ddof=1)),
       'seed_block_std':float(x.groupby('Seed').mean().std(ddof=1)),
       'logloss_mean':float(pd.to_numeric(g.Logloss).mean())}
(OUT/'audit_statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
