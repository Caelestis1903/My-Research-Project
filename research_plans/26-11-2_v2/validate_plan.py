"""Static validation of the planning package; does not train a research model."""
from __future__ import annotations
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    '00_README.md', '01_证据基础与研究问题.md', '02_多方向候选比较.md',
    '03_主候选算法与实现合同.md', '04_实验矩阵与有限超参数搜索.md',
    '05_指标与研究决策.md', '06_Codex完整执行提示词.md',
    '07_相关工作与来源.md', 'experiment_queue.yaml',
]

def main() -> None:
    for name in REQUIRED:
        path = ROOT / name
        assert path.is_file() and path.stat().st_size > 0, name
    q = yaml.safe_load((ROOT / 'experiment_queue.yaml').read_text(encoding='utf-8'))
    s, methods = q['stages'], set(q['methods'])
    assert q['status'] == 'planning_only_no_gpu_experiments'
    assert q['protocol']['loss_normalizer'] == 'original_batch_size_times_2'
    assert q['protocol']['epochs'] == 50 and q['protocol']['tasks'] == 2
    assert q['protocol']['perturbation_full_detach']
    assert q['protocol']['training_gate_input_detach']
    assert not q['protocol']['probe_gate_input_detach']
    assert not q['resources']['disk_tensor_checkpoints_allowed']
    assert set(q['protocol']['seed_dev']).isdisjoint(q['protocol']['seed_confirm'])
    for item in s.values():
        assert set(item.get('arms', [])).issubset(methods)
    for arms in s['S4']['arms_by_winner'].values():
        assert set(arms).issubset(methods)
        assert len(arms) == len(set(arms))
        assert {'A0', 'A1'}.issubset(arms)
    extra = s['S4']['additional_control_by_winner']
    for winner, spec in extra.items():
        assert winner in s['S4']['arms_by_winner']
        assert set(spec['pool']).issubset(methods)
        assert spec['count'] == 1 and spec['tie_prefer'] in spec['pool']
        assert not set(spec['pool']) & set(s['S4']['arms_by_winner'][winner])
    max_confirm_arms = max(len(arms)+extra.get(winner,{}).get('count',0)
        for winner,arms in s['S4']['arms_by_winner'].items())
    counts = {
        'S1': len(s['S1']['datasets'])*len(s['S1']['seeds'])*len(s['S1']['arms']),
        'S1R': s['S1R']['max_datasets']*len(s['S1R']['seeds'])*len(s['S1R']['arms']),
        'S2M': s['S2M']['max_datasets']*len(s['S2M']['seeds'])*len(s['S2M']['new_kappa_multipliers'])*max(map(len,s['S2M']['family_arms'].values())),
        'S3M': s['S3M']['max_datasets']*len(s['S3M']['seeds']),
        'S2B': len(s['S2B']['datasets'])*len(s['S2B']['seeds'])*len(s['S2B']['arms']),
        'S4': s['S4']['max_datasets']*len(s['S4']['seeds'])*max_confirm_arms,
        'S5': s['S5']['max_datasets']*len(s['S5']['seeds'])*len(s['S5']['arm_roles']),
    }
    for name, count in counts.items():
        assert count == s[name]['max_trajectories'], (name,count)
    totals = {name: sum(counts[x] for x in path) for name,path in q['paths'].items()}
    assert totals['main'] == q['budgets']['main_max_unique'] == 86
    assert totals['backup'] == q['budgets']['backup_max_unique'] == 82
    assert max(totals.values()) == q['budgets']['global_max_unique']
    assert not set(q['paths']['backup']) & {'S2M','S3M'}
    assert 'S2B' not in q['paths']['main']
    assert q['path_choice_immutable_after_first_exclusive_job']
    for data in q['datasets'].values():
        assert sum(data['train_counts']) == data['train_total']
        assert data['kappa_half'] == data['kappa']/2
    report = dict(status='PASS', checks='static_files_yaml_counts_protocol_mutual_exclusion',
                  stage_max_unique=counts, path_max_unique=totals,
                  gpu_experiments_run=False, algorithm_implementation_validated=False,
                  measured_performance=False)
    (ROOT/'PLAN_VALIDATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
