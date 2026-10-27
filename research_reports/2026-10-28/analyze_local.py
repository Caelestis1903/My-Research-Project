"""Apply the registered hard-group policy to common-row candidate/control predictions."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def local_risk(y, a, c, mask, rows, tasks):
    chosen = rows[:, None] & np.isin(np.arange(y.shape[1]), tasks)[None] & mask
    delta = (np.logaddexp(0., c)-y*c)-(np.logaddexp(0., a)-y*a)
    n = int(chosen.sum())
    return dict(valid_group_labels=n, valid_total_labels=int(mask.sum()),
                group_gain=float(delta[chosen].mean()) if n else None,
                natural_contribution=float(delta[chosen].sum()/mask.sum()) if n else None)


def register(root):
    definitions = []
    for path in sorted((root/'runs').glob('*/*/seed202[23]/F_selected_GROUP_REGISTRATION.json')):
        dataset, method, seed_text = path.parts[-4:-1]
        registration = json.loads(path.read_text())
        raw = np.load(path.with_name('F_selected_GAIN_TABLES.npz'))
        q, d, b, folds = (raw[k] for k in ('need_mass', 'domain', 'benefits', 'folds'))
        hard = q.argmax(1)
        groups = {}
        domain_skills = {}
        for dom in range(3):
            actions = registration['actions'][dom]
            take = d == dom
            if 'fold_prediction' in raw:
                target = b[take, :, :len(actions)]
                prediction = raw['fold_prediction'][take, :, :len(actions)]
                constant = np.zeros_like(target)
                for fold in (0, 1):
                    constant[folds[take] == fold] = target[folds[take] != fold].mean(0)
                sse = float(np.mean((prediction-target)**2))
                base = float(np.mean((constant-target)**2))
                domain_skills[dom] = 1-sse/base if base > 0 else None
            else:
                domain_skills[dom] = None
            for leaf in np.unique(hard[d == dom]):
                rows = (d == dom)&(hard == leaf)
                if rows.sum() < 100:
                    continue
                profile = b[rows, :, :len(actions)].mean(0)
                best = np.column_stack([np.zeros(2), profile]).argmax(1)-1
                preferences = ['OFF' if a < 0 else 'source'+str(actions[a]['source']) for a in best]
                stable = []
                for task, preference in enumerate(preferences):
                    panel_preferences = []
                    for fold in (0, 1):
                        panel = rows&(folds == fold)
                        if not panel.any():
                            break
                        mean = b[panel, task, :len(actions)].mean(0)
                        a = int(np.r_[0., mean].argmax())-1
                        panel_preferences.append('OFF' if a < 0 else 'source'+str(actions[a]['source']))
                    stable.append(panel_preferences == [preference, preference])
                for tasks in ((0,), (1,), (0, 1)):
                    if not all(stable[t] for t in tasks):
                        continue
                    key = (dom, tasks, tuple(preferences[t] for t in tasks))
                    group = groups.setdefault(key, dict(leaves=[], f_hard_rows=0, profiles=[]))
                    group['leaves'].append(int(leaf)); group['f_hard_rows'] += int(rows.sum())
                    group['profiles'].append(profile.tolist())
        for (dom, tasks, preferences), group in groups.items():
            definitions.append(dict(dataset=dataset, method=method, seed=int(seed_text[4:]),
                domain=dom, tasks=list(tasks), preferences=list(preferences), **group,
                state_identity=registration['identity'],
                F_predictor_skill=domain_skills[dom],
                references=['TARGET_MATCH', 'STATIC_LOCAL', 'FIXED_READ']))
    return definitions


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=True)
    definitions = register(root)
    # Only F arrays determine membership, preferences and unions; Q is opened below.
    (output/'F_LOCAL_CONDITION_REGISTRY.json').write_text(json.dumps(definitions, indent=2)+'\n')
    rows = []
    for definition in definitions:
        base = root/'runs'/definition['dataset']/definition['method']/f'seed{definition["seed"]}'
        a = np.load(base/'Q_selected_predictions.npz')
        members = (a['domain'] == definition['domain'])&np.isin(a['need_mass'].argmax(1), definition['leaves'])
        metric_a = json.loads((base/'Q_selected_metrics.json').read_text())
        for reference in definition['references']:
            if reference == definition['method']:
                continue
            control = base.parents[1]/reference/base.name
            if not (control/'Q_selected_predictions.npz').is_file():
                continue
            c = np.load(control/'Q_selected_predictions.npz')
            assert np.array_equal(a['row_id'], c['row_id']) and np.array_equal(a['labels'], c['labels'])
            metric_c = json.loads((control/'Q_selected_metrics.json').read_text())
            cell_delta = [aa['logloss']-cc['logloss'] for aa, cc in zip(metric_a['cells'], metric_c['cells'])]
            row = dict(**{k:definition[k] for k in ('dataset','method','seed','domain','F_predictor_skill')},
                tasks=','.join(map(str, definition['tasks'])), preferences=','.join(definition['preferences']),
                reference=reference, hard_Q_rows=int(members.sum()),
                macro_ll_delta=metric_a['macro_logloss']-metric_c['macro_logloss'],
                macro_auc_delta=metric_a['macro_auc']-metric_c['macro_auc'], cell_ll_deltas=cell_delta,
                **local_risk(a['labels'], a['logits'], c['logits'], a['mask'], members, definition['tasks']))
            rows.append(row)
    serial = [{**r, 'cell_ll_deltas':json.dumps(r['cell_ll_deltas'])} for r in rows]
    pd.DataFrame(serial).to_csv(output/'LOCAL_GROUP_RAW.csv', index=False)
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ('dataset','method','domain','tasks','preferences','reference'))
        groups.setdefault(key, []).append(row)
    results = []
    for key, values in groups.items():
        complete = {v['seed'] for v in values} == {2022, 2023} and len(values) == 2
        predictable = complete and all(v['F_predictor_skill'] is not None and v['F_predictor_skill'] > 0 for v in values)
        safe = complete and all(v['macro_ll_delta'] <= .0002 for v in values) and \
            np.mean([v['macro_auc_delta'] for v in values]) >= -.0005 and \
            np.max(np.mean([v['cell_ll_deltas'] for v in values], axis=0)) <= .003
        local = complete and all(v['group_gain'] is not None and v['group_gain'] >= .001 and
                                 v['natural_contribution'] >= .00005 for v in values)
        results.append(dict(zip(('dataset','method','domain','tasks','preferences','reference'),key),
            paired_seeds=[v['seed'] for v in values], assessable=complete, F_predictable=bool(predictable),
            safety=bool(safe), local_risk=bool(local), pass_local=bool(predictable and safe and local)))
    (output/'LOCAL_GATE_RESULTS.json').write_text(json.dumps(results, indent=2)+'\n')
    return dict(conditions=len(definitions), comparisons=len(rows),
                passing=[r for r in results if r['pass_local']])


if __name__ == '__main__':
    test = local_risk(np.ones((2,2)), np.ones((2,2)), np.zeros((2,2)),
                      np.ones((2,2), bool), np.array([True,False]), [0])
    assert np.isclose(test['natural_contribution'], test['group_gain']/4)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.output), ensure_ascii=False))
