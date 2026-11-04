"""Audit and expose the existing MovieLens d0 Gaussian five-seed results."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HISTORY = Path('/home/xzf/Code/phase0-3')
AB = ROOT/'research_reports/2026-11-03_overnight_AB'
AIGM = HISTORY/'artifacts/gradient_aug_aig_m_20261101/attempt_20261101_1720'
AB_RUN = HISTORY/'artifacts/mdmt_overnight_ab_20261103/attempt1'
SEEDS = tuple(range(2022, 2027))
CELLS = tuple(f'd{d}t{t}' for d in range(3) for t in range(2))
ALIASES = {('GAUSSIAN_A1','fixed_ema'):'G-D0', ('A0','online'):'M3OE',
           ('HYPERGATE','online'):'HYPERGATE', ('AUTOIFS','online'):'AUTOIFS'}
METRICS = tuple(c+'_'+m for c in CELLS for m in ('auc','logloss')) + (
    'macro_auc','macro_logloss','sample_weighted_cell_auc','sample_weighted_cell_logloss',
    'pooled_task_mean_auc','pooled_logloss')
HYPERPARAMETERS = {
    'movielens':dict(target_domains=[0],epochs=50,batch_size=40960,lr=.01,weight_decay=1e-4,
                     kappa=.05,lambda_aug=.5,rho=.9828205985452511),
    'kuairand':dict(target_domains=[0],epochs=50,batch_size=40960,lr=.01,weight_decay=1e-5,
                    kappa=.1,lambda_aug=.5,rho=.995),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (HERE/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def write_json(name, value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def stats(values):
    assert len(values)==5
    mean,sd=statistics.fmean(values),statistics.stdev(values)
    margin=2.7764451051977987*sd/math.sqrt(5)
    return dict(n=5,mean=mean,sd=sd,ci95_low=mean-margin,ci95_high=mean+margin,
                positive_seeds=sum(v>0 for v in values),negative_seeds=sum(v<0 for v in values))


def score(row):
    return math.fsum(float(row[c+'_auc']) for c in CELLS)/6


def metric_check(row):
    assert math.isclose(score(row),float(row['macro_auc']),abs_tol=1e-15,rel_tol=0)
    assert math.isclose(statistics.fmean(float(row[c+'_logloss']) for c in CELLS),
                        float(row['macro_logloss']),abs_tol=1e-15,rel_tol=0)
    denominator=sum(float(row[c+'_count']) for c in CELLS)
    for metric in ('auc','logloss'):
        traffic=sum(float(row[c+'_count'])*float(row[c+'_'+metric]) for c in CELLS)/denominator
        assert math.isclose(traffic,float(row['sample_weighted_cell_'+metric]),abs_tol=1e-15,rel_tol=0)
    assert math.isclose((float(row['task0_pooled_auc'])+float(row['task1_pooled_auc']))/2,
                        float(row['pooled_task_mean_auc']),abs_tol=1e-15,rel_tol=0)


def tier(delta):
    return ('STRONG_ALGORITHM' if delta>=.003 else 'CANDIDATE_ALGORITHM' if delta>=.002
            else 'POSITIVE_WIN' if delta>0 else 'NOT_ABOVE_BEST_PUBLISHED')


def decision(candidate, reference):
    assert {int(r['seed']) for r in candidate}=={int(r['seed']) for r in reference}==set(SEEDS)
    assert len(candidate)==len(reference)==5
    a={int(r['seed']):r for r in candidate}
    b={int(r['seed']):r for r in reference}
    delta=statistics.fmean(float(a[s]['macro_auc'])-float(b[s]['macro_auc']) for s in SEEDS)
    ll=statistics.fmean(float(a[s]['macro_logloss'])-float(b[s]['macro_logloss']) for s in SEEDS)
    cells={c:statistics.fmean(float(a[s][c+'_auc'])-float(b[s][c+'_auc']) for s in SEEDS) for c in CELLS}
    return dict(n=5,macro_auc_delta=delta,macro_logloss_delta=ll,auc_tier=tier(delta),
                cell_protection_pass=all(v>=-.001 for v in cells.values()),cell_auc_deltas=cells,
                logloss_improved=ll<0)


def main():
    assert tier(.003)=='STRONG_ALGORITHM' and tier(.002)=='CANDIDATE_ALGORITHM'
    assert tier(.001)=='POSITIVE_WIN' and tier(0)=='NOT_ABOVE_BEST_PUBLISHED'
    old_endpoints_sha=sha(HERE/'ENDPOINTS.csv')
    assert old_endpoints_sha==read_json(HERE/'RUN_MANIFEST.json')['endpoint_sha256']
    assert len(read_csv(HERE/'ENDPOINTS.csv'))==270
    sources={}
    def evidence(path):
        path=Path(path)
        sources[str(path)]=sha(path)
        return read_json(path) if path.suffix=='.json' else None
    resolved=evidence(AIGM/'resolved_config.json')
    frozen=evidence(AIGM/'formal_configuration_frozen.json')
    source_manifest=evidence(AIGM/'source_manifest.json')
    raw_data=evidence(AIGM/'data_manifest.json')['movielens']
    raw_selection=evidence(AIGM/'selection_frozen.json')
    evidence(AIGM/'test_metrics.csv')
    evidence(AIGM/'validation_epoch_metrics.csv')
    data=evidence(AB_RUN/'DATA_MANIFEST.json')['movielens']
    evidence(AB_RUN/'CONFIGURATION_FROZEN.json')
    assert source_manifest['hash']==frozen['code_hash']==raw_selection['code_hash']
    ml=resolved['datasets']['movielens']
    conf=frozen['configurations']['movielens']['A1']
    assert ml['augmented_domain']==0 and conf['kind']=='H2'
    assert conf['config']['use_augmentation'] and conf['config']['use_shadow_ema']
    assert not conf['config']['use_sad'] and not conf['config']['use_aig_m']
    for key,value in (('lr',.01),('weight_decay',1e-4),('noise_scale',.05),('lambda_aug',.5)):
        assert conf['config'][key]==value
    assert conf['config']['shadow_rates']['H2']==.9828205985452511
    assert resolved['training']['epochs']==50 and resolved['training']['batch_size']==40960
    assert not resolved['training']['amp'] and not resolved['training']['tf32']
    assert resolved['model']['fusion']==dict(exp_d=1,exp_t=1,bal_d=1,bal_t=1)
    assert resolved['model']['fcn_dims']==[512,256,256,64] and resolved['model']['expert_num']==4
    assert not resolved['model']['learned_fusion'] and not resolved['model']['darts']
    assert resolved['rng']['loader_offsets']['train']==11000
    for field in ('raw_sha256','array_hashes','split_row_fingerprints','split_row_id_hashes'):
        assert ml[field]==raw_data[field]==data[field]
    assert raw_data['features']['encoder_hash']==data['encoder_hash']==ml['encoder_hash']
    assert raw_data['features']['scaler_hash']==data['scaler_hash']==ml['scaler_hash']
    assert data['preprocessing_fit']=='train_only' and ml['loader_setting']=='natural'
    assert data['train_domain_counts']==[155949,318599,325619]
    assert data['split_rows']==dict(train=800167,validation=100021,test=100021)
    assert raw_data['preprocessing_metadata']['split_rule']=='global_row_order_80_10_10'
    sources[data['raw_path']]=sha(data['raw_path'])
    assert sources[data['raw_path']]==data['raw_sha256']
    primitive_root=ROOT/'experiments/mdmt_history_information_20261103/vendor/v3/vendor/ema02/vendor/gradient_aug_20261031_v2'
    primitives=[]
    for name in ('objectives.py','views.py','protocol.py','vendor_ga31/model.py'):
        original='experiments/gradient_aug_20261031_v2/'+name
        expected=next(row['sha256'] for row in source_manifest['files'] if row['path']==original)
        actual=sha(primitive_root/name)
        assert actual==expected
        sources[str(primitive_root/name)]=actual
        primitives.append(dict(original_path=original,available_path=str(primitive_root/name),sha256=actual))
    evidence(AB/'ENDPOINTS.csv')
    evidence(AB/'VALIDATION_CURVES.csv')
    rows=read_csv(AB/'ENDPOINTS.csv')
    curves=read_csv(AB/'VALIDATION_CURVES.csv')
    raw_tests=read_csv(AIGM/'test_metrics.csv')
    imported=[]
    selection_audit=[]
    for arm,estimator in ALIASES:
        selected=[(i+2,r) for i,r in enumerate(rows) if r['dataset']=='movielens' and r['arm']==arm
                  and r['estimator']==estimator and r['checkpoint']=='selected']
        assert len(selected)==10 and {int(r['seed']) for _,r in selected}==set(SEEDS)
        curve_arm='Gaussian' if arm=='GAUSSIAN_A1' else arm
        for seed in SEEDS:
            peers=[r for _,r in selected if int(r['seed'])==seed]
            assert {r['split'] for r in peers}=={'validation','test'} and len(peers)==2
            validation=next(r for r in peers if r['split']=='validation')
            test=next(r for r in peers if r['split']=='test')
            assert (validation['epoch'],validation['model_hash'])==(test['epoch'],test['model_hash'])
            assert validation['protocol_id']==test['protocol_id']=='AUC_SELECTED_V1'
            assert validation['data_protocol']==test['data_protocol']=='FULLTRAIN_NATURAL80_10_10'
            current=[r for r in curves if r['dataset']=='movielens' and r['arm']==curve_arm
                and int(r['seed'])==seed and (r.get('estimator') or r['model_kind'])==estimator
                and (arm not in ('HYPERGATE','AUTOIFS') or r['stage']==('reuse' if arm=='AUTOIFS' else 'train'))]
            assert len(current)==50 and {int(r['epoch']) for r in current}==set(range(1,51))
            winner=max(sorted(current,key=lambda r:int(r['epoch'])),key=score)
            assert int(winner['epoch'])==int(validation['epoch'])
            assert all(float(winner[c+'_auc'])==float(validation[c+'_auc']) for c in CELLS)
            path=Path(test['source_path'])
            evidence(path)
            assert sources[str(path)]==test['source_sha256']
            evidence(validation['source_path'])
            assert sources[validation['source_path']]==validation['source_sha256']
            final=next(r for r in rows if r['dataset']=='movielens' and r['arm']==arm
                and r['estimator']==estimator and r['checkpoint']=='E50' and r['split']=='test' and int(r['seed'])==seed)
            assert int(final['epoch'])==50 and final['trajectory_id']==test['trajectory_id']
            if arm in ('GAUSSIAN_A1','A0'):
                raw=next(r for r in raw_tests if r['dataset']=='movielens' and int(r['seed'])==seed
                         and r['checkpoint']=='selected' and r['model_hash']==test['model_hash'])
                assert raw['run_hash']==test['trajectory_id'] and raw['config_hash']==test['config_hash']
                assert all(float(raw[m])==float(test[m]) for m in METRICS)
                seal=next(r for r in raw_selection['checkpoints'] if r['dataset']=='movielens'
                          and r['seed']==seed and r['checkpoint']=='selected' and r['model_hash']==test['model_hash'])
                assert seal['global_step']==int(test['epoch'])*20
                if arm=='GAUSSIAN_A1':
                    assert seal['model_kind']=='H2' and test['source_hash']==frozen['code_hash']
                    raw_run=AIGM/f"runs/{raw['stage']}/movielens/{raw['trial']}/seed{seed}/attempt1"
                    if (raw_run/'runtime_stats.json').exists():
                        runtime=evidence(raw_run/'runtime_stats.json')
                        per_run=evidence(raw_run/'resolved_config.json')
                        assert runtime['epochs']==50 and runtime['global_step']==1000
                        assert runtime['run_hash']==test['trajectory_id']
                        assert per_run['rng']['view_key']=='GA31-R3|dataset|seed|train|global_step|gaussian|1'
            else:
                seal=read_json(path)
                assert seal['configuration_frozen_sha256']==sources[str(AB_RUN/'CONFIGURATION_FROZEN.json')]
                assert seal['endpoints']['online']['selected']['model_hash']==test['model_hash']
                native=evidence(path.parent/'endpoints.json')
                native_test=next(r for r in native if r['split']=='test' and r['checkpoint']=='selected')
                assert native_test['model_hash']==test['model_hash']
                assert all(float(native_test[m])==float(test[m]) for m in METRICS)
            selection_audit.append(dict(arm=ALIASES[arm,estimator],source_arm=arm,seed=seed,
                epoch=int(test['epoch']),validation_curve_epochs=50,model_hash=test['model_hash'],
                trajectory_id=test['trajectory_id'],selection_id=test['selection_id'],
                config_hash=test['config_hash'],source_hash=test['source_hash'],
                source_path=test['source_path'],source_sha256=test['source_sha256'],
                validation_source_path=validation['source_path'],validation_source_sha256=validation['source_sha256']))
        for line,row in selected:
            metric_check(row)
            imported.append(dict(row,arm=ALIASES[arm,estimator],source_arm=arm,
                source_table=str(AB/'ENDPOINTS.csv'),source_table_sha256=sources[str(AB/'ENDPOINTS.csv')],
                source_table_line=line,supplement_action='REUSE_EXISTING'))
    imported.sort(key=lambda r:(r['arm'],int(r['seed']),r['split']))
    write_csv('MOVIELENS_ENDPOINTS.csv',imported)
    tests=[r for r in imported if r['split']=='test']
    absolute=[]
    for arm in ('G-D0','M3OE','HYPERGATE','AUTOIFS'):
        current=[r for r in tests if r['arm']==arm]
        assert len(current)==5
        for metric in METRICS:
            absolute.append(dict(dataset='movielens',arm=arm,metric=metric,**stats([float(r[metric]) for r in current])))
    write_csv('MOVIELENS_ABSOLUTE_SUMMARY.csv',absolute)
    candidate=[r for r in tests if r['arm']=='G-D0']
    paired,paired_summary,decisions=[],[],[]
    for arm in ('M3OE','HYPERGATE','AUTOIFS'):
        reference=[r for r in tests if r['arm']==arm]
        current=[]
        for seed in SEEDS:
            a=next(r for r in candidate if int(r['seed'])==seed)
            b=next(r for r in reference if int(r['seed'])==seed)
            out=dict(dataset='movielens',comparison_id='G-D0_vs_'+arm,seed=seed,
                candidate_epoch=int(a['epoch']),reference_epoch=int(b['epoch']),
                candidate_model_hash=a['model_hash'],reference_model_hash=b['model_hash'])
            for metric in METRICS:
                out.update({f'candidate_{metric}':float(a[metric]),f'reference_{metric}':float(b[metric]),
                            f'delta_{metric}':float(a[metric])-float(b[metric])})
            paired.append(out)
            current.append(out)
        for metric in METRICS:
            paired_summary.append(dict(comparison_id='G-D0_vs_'+arm,metric=metric,
                candidate_mean=statistics.fmean(r['candidate_'+metric] for r in current),
                reference_mean=statistics.fmean(r['reference_'+metric] for r in current),
                **stats([r['delta_'+metric] for r in current])))
        decisions.append(dict(reference=arm,**decision(candidate,reference)))
    write_csv('MOVIELENS_PAIRED_SEED.csv',paired)
    write_csv('MOVIELENS_PAIRED_SUMMARY.csv',paired_summary)
    published_means={arm:statistics.fmean(float(r['macro_auc']) for r in tests if r['arm']==arm)
                     for arm in ('M3OE','HYPERGATE','AUTOIFS')}
    best=max(published_means,key=published_means.get)
    write_json('MOVIELENS_DECISIONS.json',dict(best_published=best,published_macro_auc=published_means,
        classification=next(r for r in decisions if r['reference']==best),comparisons=decisions,
        rules=dict(strong_auc_gain=.003,candidate_auc_gain=.002,minimum_cell_auc_delta=-.001,
                   ci_is_gate=False,require_all_seeds_positive=False,logloss_independent=True)))
    kr=read_csv(HERE/'ENDPOINTS.csv')
    two=[]
    for dataset,records in (('movielens',tests),('kuairand',[r for r in kr if r['checkpoint']=='selected'
                 and r['split']=='test' and r['estimator']==('fixed_ema' if r['arm']=='G-D0' else 'online')])):
        current=[r for r in records if r['arm']=='G-D0']
        assert len(current)==5 and {int(r['seed']) for r in current}==set(SEEDS)
        means={arm:statistics.fmean(float(r['macro_auc']) for r in records if r['arm']==arm)
               for arm in ('M3OE','HYPERGATE','AUTOIFS')}
        best_arm=max(means,key=means.get)
        verdict=decision(current,[r for r in records if r['arm']==best_arm])
        out=dict(dataset=dataset,arm='G-D0',estimator='fixed_ema',n=5,
            protocol_id='AUC_SELECTED_V1',data_protocol='FULLTRAIN_NATURAL80_10_10',
            best_published=best_arm,reference_macro_auc=means[best_arm],**HYPERPARAMETERS[dataset],
            **{k:v for k,v in verdict.items() if k not in ('n','cell_auc_deltas')})
        for metric in METRICS:
            values=[float(r[metric]) for r in current]
            out.update({metric+'_mean':statistics.fmean(values),metric+'_sd':statistics.stdev(values)})
        two.append(out)
    write_csv('G_D0_TWO_DATASETS.csv',two)
    headers=['数据集','n']+[c+' AUC / LL' for c in CELLS]+['Macro AUC ± SD','Macro LL ± SD']
    lines=['# G-D0双数据集对比','',
        '每行n=5，seed2022–2026；固定EMA按validation六格Macro AUC最高、并列最早选点后的test结果。',
        '每格为五seed均值AUC / LogLoss；逐格SD及Traffic/Pooled见CSV。MovieLens5条Gaussian轨迹复用，新增训练0；KuaiRand沿用原报告。','',
        '| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']
    for r in two:
        values=[r['dataset'],5]+[f"{r[c+'_auc_mean']:.6f} / {r[c+'_logloss_mean']:.6f}" for c in CELLS]
        values += [f"{r[m+'_mean']:.6f} ± {r[m+'_sd']:.6f}" for m in ('macro_auc','macro_logloss')]
        lines.append('| '+' | '.join(map(str,values))+' |')
    lines += ['', '| 数据集 | 增强域 | batch | epoch | Adam lr | coupled wd | Gaussian κ | λ | EMA ρ |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for dataset,c in HYPERPARAMETERS.items():
        lines.append('| '+' | '.join(map(str,[dataset,'d0',c['batch_size'],c['epochs'],c['lr'],c['weight_decay'],c['kappa'],c['lambda_aug'],c['rho']]))+' |')
    lines += ['', '两数据集均为fixed1 M3oE、FP32、原始自然人口、Sample目标、train-only预处理。'
              'MovieLens领域按固定年龄划分，任务为rating>3/rating>4；KuaiRand域为tab0/1/4，任务为is_click/long_view。',
              '原KuaiRand270端点和10条新增训练的统计保持原归档；本补充复用MovieLens G-D0五条及发表模型十五条历史轨迹。',
              '', '[逐seed端点](MOVIELENS_ENDPOINTS.csv) · [来源审计](MOVIELENS_SOURCE_AUDIT.json) · [中文补充分析](MOVIELENS_SUPPLEMENT.md)']
    (HERE/'G_D0_TWO_DATASETS.md').write_text('\n'.join(lines)+'\n')
    audit=dict(status='COMPLETE_REUSE',task='MovieLens existing G-D0 five-seed supplement',
        source_branch_base='f6ec1bf65745b891deb96f1e74ab5c86cfdb7224',
        counts=dict(movielens_gaussian_reused_trajectories=5,movielens_published_reused_trajectories=15,
                    new_training_runs=0,new_seeds=0,movielens_selected_validation_test_endpoints=40,
                    kuairand_prior_comparison_trajectories=20,original_kuairand_endpoint_rows=270),
        hyperparameters=HYPERPARAMETERS,data_identity=data,
        gaussian_configuration=conf,source_code_hash=frozen['code_hash'],
        original_gaussian_primitives=primitives,selection_audit=selection_audit,
        imported_rows=imported,upstream_files=sources,supplement_script_sha256=sha(__file__),
        original_kuairand_endpoints_sha256=old_endpoints_sha)
    write_json('MOVIELENS_SOURCE_AUDIT.json',audit)
    assert sha(HERE/'ENDPOINTS.csv')==old_endpoints_sha
    assert len(tests)==20 and len(paired)==15 and len(two)==2
    for row in absolute:
        source=[r for r in tests if r['arm']==row['arm']]
        assert row['mean']==statistics.fmean(float(r[row['metric']]) for r in source)
    print(json.dumps(dict(status=audit['status'],counts=audit['counts'],two_datasets=two,
                          movielens_comparisons=decisions),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
