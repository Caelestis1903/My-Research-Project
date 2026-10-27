#!/usr/bin/env python3
"""Validate the design package, not a trainer or a claim of model correctness.

Usage: python -B research_plans/2026-10-27/08_VALIDATE_PLAN.py
Read-only; prints JSON. Requires PyYAML, no torch/GPU/data access.
"""
from __future__ import annotations
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from urllib.parse import unquote

sys.dont_write_bytecode = True
try:
    import yaml
except ImportError:
    print(json.dumps({'status': 'BLOCKED_DEPENDENCY', 'missing': 'PyYAML'}))
    raise SystemExit(3)

ROOT = Path(__file__).resolve().parent
FILES = [
    '00_README.md', '01_研究问题与方案修订.md', '02_核心算法数学与机制规范.md',
    '03_Codex实现技术规范.md', '04_实验矩阵与研究决策.md', '05_CODEX_EXECUTION.md',
    '06_实现验收与风险清单.md', '07_EXECUTION_CONFIG.yaml', '08_VALIDATE_PLAN.py',
    '09_SOURCE_AUDIT.json', '10_REVIEW_RECORD.md',
]
checks: list[str] = []


def require(condition: bool, name: str) -> None:
    if not condition:
        raise ValueError(name)
    checks.append(name)


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f'duplicate YAML key: {key}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def main() -> dict:
    hashes = {}
    links = 0
    for name in FILES:
        path = ROOT / name
        require(path.is_file(), f'exists:{name}')
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        require(len(raw) > 100 and text.endswith('\n'), f'complete_utf8:{name}')
        hashes[name] = {
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest(),
        }
        if name.endswith('.md'):
            require(not re.search(r'\b(?:TODO|TBD|PLACEHOLDER)\b|待补充', text), f'no_placeholders:{name}')
            stack = None
            for line in text.splitlines():
                if line.startswith(('```', '~~~')):
                    token = line[:3]
                    if stack is None:
                        stack = token
                    else:
                        require(token == stack, f'fence_type:{name}')
                        stack = None
            require(stack is None, f'fences_closed:{name}')
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if '://' in target or target.startswith('#'):
                    continue
                dest = ROOT / unquote(target.split('#')[0])
                require(dest.is_file(), f'local_link:{name}:{target}')
                links += 1
    ast.parse((ROOT / '08_VALIDATE_PLAN.py').read_text())
    checks.append('python_syntax')
    cfg = yaml.load((ROOT / '07_EXECUTION_CONFIG.yaml').read_text(), Loader=UniqueLoader)
    audit = json.loads((ROOT / '09_SOURCE_AUDIT.json').read_text())
    require(cfg['status'] == audit['status'] == 'DESIGN_ONLY', 'design_not_training')
    require(cfg['source_main'] == audit['source_main'], 'source_commit_consistency')
    require(cfg['protocol'] == 'MDMT-AP-v1', 'protocol')
    require(cfg['model']['xi_dim'] == 256 + 2 + 3 + 2, 'request_dim')
    require(cfg['model']['phi_dim'] == 2 * cfg['prefix']['pilot']['dims'][1], 'source_dim')
    require(cfg['data']['domains'] == 3 and cfg['data']['tasks'] == 2, 'MDMT_dimensions')
    require(cfg['data']['mask'] == 'all_observed', 'no_fake_click_mask')
    require(cfg['data']['task_weights'] == [1.0, 1.0], 'task_weights')
    require(not cfg['data']['domain_resampling'], 'natural_distribution')
    totals = {}
    for dataset in ('movielens', 'kuairand'):
        d = cfg['data'][dataset]
        require(bool(re.fullmatch('[0-9a-f]{64}', d['raw_sha256'])), f'raw_hash:{dataset}')
        totals[dataset] = {}
        for role, cells in d['counts'].items():
            require(len(cells) == 3 and all(isinstance(n, int) and n > 0 for n in cells), f'counts:{dataset}:{role}')
            totals[dataset][role] = sum(cells)
        require(sum(totals[dataset].values()) == d['raw_rows'], f'row_partition_total:{dataset}')
    require(totals['movielens']['F'] == 48010, 'ML_formation_rows')
    require(totals['kuairand']['F'] == 120367, 'KR_formation_rows')
    require(cfg['data']['exposed_roles'] == ['E', 'legacy'], 'historical_test_boundary')
    require(cfg['prefix']['native']['batch'] == cfg['prefix']['native']['physical_batch'] == 40960, 'native_physical_batch')
    require(not cfg['prefix']['native']['learned_fusion'] and not cfg['prefix']['native']['automl'], 'fixed_native')
    require(cfg['prefix']['native']['gate_input_detach'], 'native_training_graph')
    require(cfg['numerics']['source_clip'] is None, 'teacher_no_clip')
    require(cfg['numerics']['ll'] == 'float64_logit_softplus_minus_yz', 'logit_LL')
    require(cfg['numerics']['auc_score'] == 'float64_logits', 'logit_AUC')
    require(isinstance(cfg['numerics']['adam_eps'], float), 'yaml_numeric_eps')
    common = (263 * 32 + 32) + (32 * 4 + 4) + 12 * 64 + 6
    parameters = {
        'F0': common + 512 * 6 + 33 * 3,
        'F1_initial': common + 512 * 3,
        'F1_max': common + 512 * 8 + 33 * 5,
        'WIDE': common + 512 * 6,
        'READ': 268 * 43 + 778,
        'TARGET_ONLY': 263 * 47 + 47 + 47 + 1,
        'DIRECT': 6 * (258 * 40 + 40 + 40 + 1),
    }
    for arm, n in parameters.items():
        require(cfg['model']['expected_trainable_parameters'][arm] == n, f'parameters:{arm}')
    require(cfg['model']['reference_parameters'] == 6 * (258 * 32 + 32 + 32 + 1), 'B_parameters')
    require(cfg['model']['source_hidden_deployment_parameters'] == 6 * (256 * 32 + 32), 'pilot_hidden_parameters')
    require(cfg['model']['physical_growth'] and not cfg['model']['preallocated_Kmax'], 'physical_not_mask_growth')
    require(not any(cfg['model'][key] for key in ('merge', 'prune', 'epoch_rollback', 'explicit_gain_network', 'forced_diversity')), 'bounded_first_version')
    require(cfg['training']['lineage_alpha'] == 'conserved_on_split', 'anchor_mass')
    require(cfg['training']['gamma_LL_interval'] == [0.0, 1.0], 'independent_source_projection')
    require(cfg['structure']['event_passes'] == [6, 8, 10, 12, 14], 'structure_windows')
    require(len(cfg['structure']['event_passes']) == cfg['data']['structure_slices'], 'one_slice_per_window')
    require(cfg['structure']['commit_rejected'] == 'trained_KEEP', 'reject_commits_trained_KEEP')
    require(cfg['structure']['whole_lightweight_fresh_adam'], 'matched_restart')
    require(cfg['structure']['demand']['ridge_target'] == 'raw_sketch', 'no_target_scale_leak_into_fold_fit')
    require(cfg['structure']['reactivate']['maximum_per_source'] == 1, 'bounded_reactivation')
    require(not cfg['structure']['reactivate']['increments_K'], 'reactivation_not_growth')
    require(cfg['structure']['acceptance']['evaluate_normal_gate_only'], 'deployment_gate_acceptance')
    require(cfg['selection']['primary'] == 'V_macro_LL_min', 'selection_metric')
    require(cfg['selection']['stage2_passes'] == list(range(15, 21)), 'post_structure_selection')
    require(cfg['seeds']['development'] == [3101, 3102], 'dev_seeds')
    require(cfg['seeds']['formal'] == [6101, 6102, 6103, 6104], 'formal_seeds')
    require(not set(cfg['seeds']['development']) & set(cfg['seeds']['formal']), 'seeds_disjoint')
    require(len(cfg['experiments']['stage1']) == len(set(cfg['experiments']['stage1'])) == 6, 'stage1_arms')
    require(len(cfg['experiments']['stage2']) == len(set(cfg['experiments']['stage2'])) == 6, 'stage2_arms')
    require(len(cfg['experiments']['formal_core']) == len(cfg['experiments']['formal_dynamic']) == 7, 'formal_methods')
    dev_max = 6 * 2 * 2 + 6 * 2 * 2 + 4 + 4
    formal_max = 7 * 2 * 4
    require(cfg['experiments']['max_development_main_traces'] == dev_max, 'development_cap')
    require(cfg['experiments']['max_formal_main_traces'] == formal_max, 'formal_cap')
    require(cfg['bridge']['campaign_limit'] == 1 and cfg['bridge']['max_main_traces'] == 4, 'single_bounded_bridge')
    require(cfg['attribution']['T0_source_bce_and_risk_labels'] == 'joint_permuted', 'PERM_no_true_source_labels')
    require(cfg['attribution']['S_source_audit_labels'] == 'original_real', 'PERM_S_safety_scope')
    require(not cfg['experiments']['external_SOTA_baselines_in_campaign'], 'no_unrun_SOTA_claim')
    require(cfg['cost']['charge_rejected_and_failed'], 'failure_cost')
    require(cfg['functional_analysis']['max_refits_per_candidate'] == 4, 'bounded_functional_analysis')
    # Tiny algebra checks only. These do not instantiate or train the research model.
    rng = random.Random(20261027)
    vec = lambda n: [rng.uniform(-1, 1) for _ in range(n)]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    r, mu, a = vec(64), vec(64), vec(64)
    std = [abs(x) + 0.001 for x in vec(64)]
    bias, native = 0.13, -0.27
    phi = [(x - m) / sd for x, m, sd in zip(r, mu, std)]
    v = [sd * x for sd, x in zip(std, a)]
    teacher_before = native + bias + dot(a, r)
    teacher_after = native + bias + dot(a, mu) + dot(v, phi)
    require(abs(teacher_before - teacher_after) < 1e-10, 'numeric_teacher_affine_invariance')
    parent, q, alpha = vec(64), 0.37, 0.25
    combined = [q * x + (1 - q) * x for x in parent]
    require(max(abs(x - y) for x, y in zip(parent, combined)) < 1e-12, 'numeric_split_forward_invariance')
    require(abs(alpha / 2 + alpha / 2 - alpha) < 1e-12, 'numeric_split_anchor_mass')
    gelu = lambda x: x * 0.5 * (1 + math.erf(x / math.sqrt(2)))
    V0, V1 = [vec(64) for _ in range(4)], [vec(64) for _ in range(4)]
    U0, U1 = [vec(4) for _ in range(64)], [vec(4) for _ in range(64)]
    b0, b1 = [gelu(dot(row, phi)) for row in V0], [gelu(dot(row, phi)) for row in V1]
    small = [q * dot(u0, b0) + (1 - q) * dot(u1, b1) for u0, u1 in zip(U0, U1)]
    wide = [dot([q * z for z in u0] + [(1 - q) * z for z in u1], b0 + b1) for u0, u1 in zip(U0, U1)]
    require(max(abs(x - y) for x, y in zip(small, wide)) < 1e-9, 'numeric_constant_q_rank8_equivalence')
    pred, center0, center1 = vec(16), vec(16), vec(16)
    direct = sum((x - y) ** 2 for x, y in zip(pred, center1)) - sum((x - y) ** 2 for x, y in zip(pred, center0))
    affine = 2 * dot(pred, [x - y for x, y in zip(center0, center1)]) + dot(center1, center1) - dot(center0, center0)
    require(abs(direct - affine) < 1e-12, 'numeric_distance_difference_affine')
    local_steps = {d: min(128, math.ceil(counts['F'] / cfg['training']['batch'])) for d, counts in totals.items()}
    require(local_steps == {'movielens': 24, 'kuairand': 59}, 'local_steps')
    return {'status': 'PASS', 'scope': 'plan_static_and_tiny_algebra_only_no_training',
            'checks': len(checks), 'files': len(FILES), 'relative_links': links,
            'parameters': parameters, 'role_totals': totals, 'local_steps': local_steps,
            'maximum_main_traces': {'development': dev_max, 'formal': formal_max},
            'files_manifest': hashes}


if __name__ == '__main__':
    try:
        print(json.dumps(main(), ensure_ascii=False, indent=2))
    except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as exc:
        print(json.dumps({'status': 'FAIL', 'error': str(exc), 'checks_completed': len(checks)}, ensure_ascii=False))
        raise SystemExit(2)
