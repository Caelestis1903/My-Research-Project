"""Summarize exported formation, intervention and cost measurements."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
from scipy.special import expit


def extract(root, output):
    output.mkdir(parents=True, exist_ok=True)
    mechanism, routing, interventions, costs, parameters, sketches, calibration = [], [], [], [], [], [], []
    for run in sorted((root/'runs').glob('*/*/seed*')):
        dataset, method, seed_text = run.parts[-3:]
        common = dict(dataset=dataset, method=method, seed=int(seed_text[4:]))
        caps_path = run/'GRADIENT_PERMISSIONS.csv'
        caps = pd.read_csv(caps_path, usecols=['epoch','aux_gamma']) if caps_path.is_file() else None
        source_path = run/'SOURCE_CONSTRAINTS.csv'
        source = pd.read_csv(source_path) if source_path.is_file() else None
        extra = {}
        if caps is not None:
            active = caps[caps.epoch >= 3].aux_gamma
            extra.update(aux_gamma_mean=float(active.mean()), aux_cap_fraction=float((active < 1-1e-9).mean()))
        if source is not None and 'gamma' in source:
            extra.update(source_gamma_min=float(source.gamma.min()),
                         source_projection_fraction=float((source.gamma < 1-1e-9).mean()))
        cost_path = run/'COST_LEDGER.json'
        if cost_path.is_file():
            costs.append(dict(**common, **json.loads(cost_path.read_text())))
        count_path = run/'PARAMETER_AND_MAC_LEDGER.json'
        if count_path.is_file():
            count = json.loads(count_path.read_text())
            parameters.append(dict(**common, **{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in count.items()}))
        for state in ('selected','final'):
            p = run/f'F_{state}_GAIN_TABLES.npz'
            f_benefits, f_domains, gain_scale = None, None, None
            if p.is_file():
                with np.load(p) as a:
                    q, domain = a['need_mass'], a['domain']
                    f_benefits, f_domains = a['benefits'].astype(float), domain
                    if 'prediction' in a:
                        stored = json.loads((run/f'F_{state}_GAIN_CALIBRATION.json').read_text())['current_student']
                        raw_rms = np.sqrt(np.mean(a['prediction'].astype(float)**2))
                        gain_scale = np.float32(stored['prediction_rms']/raw_rms) if raw_rms > 0 else np.float32(0.)
                    for d in np.unique(domain):
                        z = q[domain == d].astype(float)
                        routing.append(dict(**common, state=state, domain=int(d),
                            max_uniform_deviation=float(np.max(np.abs(z-1/z.shape[1]))),
                            mean_uniform_deviation=float(np.mean(np.abs(z-1/z.shape[1]))),
                            entropy=float(np.mean(-np.sum(z*np.log(np.maximum(z,1e-30)),axis=1))),
                            mean_mass=json.dumps(z.mean(0).tolist()),
                            hard_counts=json.dumps(np.bincount(z.argmax(1),minlength=z.shape[1]).tolist())))
            for role in ('F','Q','E'):
                c = run/f'{role}_{state}_GAIN_CALIBRATION.json'
                if c.is_file():
                    values = json.loads(c.read_text())
                    for kind, record in values.items():
                        if not isinstance(record, dict) or not set(record).intersection({'mse','skill','status'}):
                            continue
                        mechanism.append(dict(**common,state=state,role=role,kind=kind,**record,**extra))
            for role in ('Q','E'):
                gain_path = run/f'{role}_{state}_GAIN_TABLES.npz'
                if gain_scale is not None and gain_path.is_file():
                    with np.load(gain_path) as a:
                        target = a['benefits'].astype(float)
                        prediction = (a['prediction']*gain_scale).astype(float)
                        constant = np.empty_like(target)
                        for d in np.unique(a['domain']):
                            constant[a['domain']==d] = f_benefits[f_domains==d].mean(0)
                        mse = float(np.mean((target-prediction)**2))
                        constant_mse = float(np.mean((target-constant)**2))
                        recorded = json.loads((run/f'{role}_{state}_GAIN_CALIBRATION.json').read_text())['student']['mse']
                        assert np.isclose(mse,recorded,rtol=1e-6,atol=1e-12), 'Recovered gain units differ from measured MSE'
                        risks = np.concatenate([np.zeros((*target.shape[:-1],1)),target],axis=-1)
                        def policy_gain(profile):
                            actions = np.concatenate([np.zeros((*profile.shape[:-1],1)),profile],axis=-1).argmax(-1)
                            return float(np.take_along_axis(risks,actions[...,None],axis=-1).mean())
                        mechanism.append(dict(**common,state=state,role=role,kind='F_trained_domain_constant',
                            mse=mse,constant_mse=constant_mse,skill=1-mse/constant_mse if constant_mse>0 else None,
                            student_single_action_gain=policy_gain(prediction),constant_single_action_gain=policy_gain(constant),
                            recovered_gain_scale=float(gain_scale),scale_recovery='nearest FP32 from stored prediction RMS; validated against measured MSE',**extra))
                p = run/f'{role}_{state}_predictions.npz'
                if p.is_file():
                    with np.load(p) as a:
                        y, z, valid = a['labels'], a['logits'].astype(float), a['mask']
                        probability = expit(z)
                        for domain in np.unique(a['domain']):
                            for task in range(y.shape[1]):
                                chosen = (a['domain'] == domain)&valid[:,task]
                                if chosen.any():
                                    observed = y[chosen,task]
                                    predicted = probability[chosen,task]
                                    calibration.append(dict(**common,role=role,state=state,domain=int(domain),
                                        task=task,n=int(chosen.sum()),positive_rate=float(observed.mean()),
                                        mean_probability=float(predicted.mean()),
                                        calibration_bias=float(predicted.mean()-observed.mean()),
                                        brier=float(np.mean((predicted-observed)**2))))
                        if 'null_mass' in a and 'need_mass' in a:
                            null = np.sum(a['null_mass']*a['need_mass'][:,None,:],axis=-1)
                            mechanism.append(dict(**common,state=state,role=role,kind='deployed_NULL',
                                                  null_mean=float(null.mean()),**extra))
            p = run/'interventions'/state/'FUNCTION_INTERVENTIONS.csv'
            if p.is_file():
                a = pd.read_csv(p)
                interventions.extend(a[(a.role == 'Q')&(a.level == 'GLOBAL')].to_dict('records'))
        p = run/'GRADIENT_SKETCH_DIAGNOSTICS.csv'
        if p.is_file():
            frame = pd.read_csv(p)
            if 'nodes' in frame:
                for row in frame.to_dict('records'):
                    for node in json.loads(row['nodes']):
                        sketches.append(dict(**common,**node))
    for name, rows in [('MECHANISM_SUMMARY',mechanism),('ROUTING_SUMMARY',routing),
                       ('FUNCTION_INTERVENTION_SUMMARY',interventions),('COST_LEDGER',costs),
                       ('PARAMETER_AND_MAC_LEDGER',parameters),('GRADIENT_SKETCH_NODES',sketches),
                       ('CALIBRATION_DIAGNOSTICS',calibration)]:
        pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
    if calibration:
        c = pd.DataFrame(calibration)
        paired = c[c.method=='CLUSTER_RELATION'].merge(c[c.method=='TARGET_MATCH'],
            on=['dataset','seed','role','state','domain','task'],suffixes=('_a','_c'),validate='one_to_one')
        paired['brier_delta'] = paired.brier_a-paired.brier_c
        paired['abs_bias_delta'] = abs(paired.calibration_bias_a)-abs(paired.calibration_bias_c)
        paired.to_csv(output/'PAIRED_CALIBRATION.csv',index=False)
    from collections import defaultdict
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from experiments.mdmt_basp_v1.config import load_config, arms, P1
    from experiments.mdmt_basp_v1.model import make_model
    cfg, scopes = load_config(), []
    for name in P1:
        model = make_model(cfg,arms()[name],seed=2022)
        categories, counts = model.parameter_categories(), defaultdict(int)
        for parameter_name, parameter in model.named_parameters():
            counts[categories[parameter_name]] += parameter.numel()
        foreign = bool(getattr(model,'foreign',False))
        excluded = {'owner'} if foreign else {'owner','source','reader'}
        if not getattr(model,'flags',{}).get('GAIN_CLUSTER',False):
            excluded.add('center')
        scopes.append(dict(method=name,trainable_adapter_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            allocated_adapter_parameters=sum(counts.values()),parameter_categories=dict(counts),
            deployed_adapter_parameters=sum(v for k,v in counts.items() if k not in excluded),
            excluded_training_or_inactive_categories=sorted(excluded),
            frozen_source_hidden_parameters_if_foreign=49344 if foreign else 0,
            scope='existing forward graph; fixed projection/centers, G and normalization buffers are additional state'))
    assert scopes[0]['trainable_adapter_parameters']==63966 and scopes[-1]['trainable_adapter_parameters']==63680
    (output/'DEPLOYMENT_PARAMETER_SCOPE.json').write_text(json.dumps(scopes,indent=2)+'\n')
    return dict(mechanism_records=len(mechanism), routing_records=len(routing),
                intervention_records=len(interventions), sketch_nodes=len(sketches))


if __name__ == '__main__':
    q = np.array([[.5,.5],[.5,.5]])
    assert np.allclose(np.sum(np.ones((2,2,2))*.5*q[:,None,:],axis=-1), .5)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.root,args.output)))
