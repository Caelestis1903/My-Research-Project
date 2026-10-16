"""Fixed-theta, finite-pool optimizer replay. Never calls a training runner."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
OLD = ROOT / 'research_plans/2026-10-15/execution_20261015/C2_numeric_oracle_20261016'
REPLAYS, STEPS = 4096, 64


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def extract(dataset, seed, target):
    import torch
    import torch.nn.functional as F
    from c2_numeric_oracle_20261016 import CONFIGS, flatten_gradient
    from stage_S1_data import load_stage_s1_bundle
    from stage_S1_models import build_model, initial_state_hash
    torch.set_num_threads(2)
    old_path = OLD / dataset / f'seed{seed}/manifest.json'
    old = json.loads(old_path.read_text())
    setting, directory, filename, expected = CONFIGS[dataset]
    assert sha(Path(directory) / filename) == expected
    bundle = load_stage_s1_bundle(dataset, directory, setting, seed)
    model = build_model(bundle.features, 3, 2, 'm3oe', expert_num=4,
                        fcn_dims=(512,256,256,64), seed=seed).cpu().eval()
    before = initial_state_hash(model)
    assert before == old['actual_initial_parameter_hash']
    position = {int(row): i for i, row in enumerate(bundle.train_row_ids)}
    ids = np.array(old['pool_row_ids'])
    positions = np.array([position[int(row)] for row in ids])
    coords = np.array([c['index'] for c in old['selected_coordinates']])
    groups = np.array([c['partition'] for c in old['selected_coordinates']])
    params = [p for _, p in model.named_parameters()]
    theta = np.concatenate([p.detach().numpy().ravel() for p in params])[coords].astype(float)
    attrs = [(m, m.tau) for m in model.modules() if hasattr(m, 'softmax_type')]
    gradients, losses = [], []
    for i in positions:
        x = torch.tensor(bundle.train_x[i:i+1], dtype=torch.float32)
        y = torch.tensor(bundle.train_y[i:i+1], dtype=torch.float32)
        pred = model(x.T)
        for m, tau in attrs: m.tau = tau
        terms = F.binary_cross_entropy(pred, y, reduction='none')[0]
        losses.append(terms.detach().numpy())
        # Match prior C2's float32 task aggregation before float64 projection.
        g0 = flatten_gradient(terms[0], params, retain=True)
        g1 = flatten_gradient(terms[1], params)
        gradients.append(((g0 + g1)*.5)[coords].astype(float))
    assert initial_state_hash(model) == before and all(p.grad is None for p in params)
    assert all(m.tau == tau for m,tau in attrs)
    assert sha(Path(directory) / filename) == expected
    domains = bundle.train_x[positions,-1].astype(int)
    assert np.array_equal(domains, old['pool_domain_labels'])
    np.savez_compressed(target / 'gradient_pool.npz', gradients=np.array(gradients),
         losses=np.array(losses), theta=theta, groups=groups, row_ids=ids,
         domains=domains, p=np.array(old['full_train_domain_probabilities']),
         labels=bundle.train_y[positions], coordinates=coords)
    dump(target / 'pool_receipt.json', {'dataset': dataset, 'seed': seed,
         'scope': 'FINITE_POOL_INITIAL_THETA_PROJECTED_COORDINATES', 'model_unchanged': True,
         'initial_hash': before, 'old_manifest_sha256': sha(old_path),
         'dataset_sha256': expected, 'data_manifest': bundle.manifest_fields(),
         'gradient_pool_sha256': sha(target / 'gradient_pool.npz'),
         'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
             ROOT/'experiments/c2_numeric_oracle_20261016.py', ROOT/'experiments/stage_S1_models.py',
             ROOT/'experiments/stage_S1_data.py', ROOT/'m3oe/MDMTRec/models/multi_domain/model.py']}})


def propagate(grad, theta, lr, optimizer, decay=1e-5):
    n, steps, k = grad.shape
    m = np.zeros((n,k)); v = m.copy(); cumulative = m.copy()
    records, final = [], None
    for t in range(1, steps+1):
        g = grad[:,t-1] + (decay*theta if optimizer != 'AdamW' else 0)
        if optimizer == 'SGD':
            delta = -lr*g
            mh = g; vh = g*g
        else:
            m = .9*m + .1*g; v = .999*v + .001*g*g
            mh = m/(1-.9**t); vh = v/(1-.999**t)
            delta = -lr*mh/(np.sqrt(vh)+1e-8)
            if optimizer == 'AdamW': delta -= lr*decay*theta
        cumulative += delta
        if t in [1,2,8,16,32,64] or t == steps:
            records.append((t, m.mean(0).copy(), v.mean(0).copy(), mh.mean(0).copy(),
                 vh.mean(0).copy(), delta.mean(0).copy(), cumulative.mean(0).copy()))
        final = dict(m=m, v=v, mhat=mh, vhat=vh, delta=delta, cumulative=cumulative)
    return records, final


def torch_check():
    import torch
    rng = np.random.default_rng(721)
    grad = rng.normal(size=(1,64,5))
    theta = rng.normal(size=5); lr = np.full(5,.01)
    largest = 0.
    for optimizer in ['SGD','Adam','AdamW']:
        for wd in [0.,1e-5]:
            records, expected = propagate(grad, theta, lr, optimizer, decay=wd)
            p = torch.nn.Parameter(torch.tensor(theta, dtype=torch.float64))
            cls = getattr(torch.optim, optimizer)
            kwargs = dict(lr=.01, weight_decay=wd, foreach=False)
            if optimizer != 'SGD': kwargs.update(betas=(.9,.999), eps=1e-8)
            opt = cls([p], **kwargs)
            cum = np.zeros(5)
            for t in range(64):
                # Fixed-theta replay: state advances, physical theta is reset.
                with torch.no_grad(): p.copy_(torch.tensor(theta))
                p.grad = torch.tensor(grad[0,t], dtype=torch.float64)
                opt.step()
                delta = p.detach().numpy()-theta; cum += delta
                if optimizer != 'SGD':
                    state = opt.state[p]
                    partial = propagate(grad[:,:t+1],theta,lr,optimizer,decay=wd)[1]
                    for a,b in [('exp_avg','m'),('exp_avg_sq','v')]:
                        largest=max(largest,float(np.max(np.abs(state[a].numpy()-partial[b][0]))))
            largest=max(largest,float(np.max(np.abs(cum-expected['cumulative'][0]))),
                        float(np.max(np.abs(delta-expected['delta'][0]))))
    assert largest < 1e-12
    return largest


def replay(dataset, seed, target):
    with np.load(target/'gradient_pool.npz',allow_pickle=False) as z:
        pool={k:z[k] for k in z.files}
    g=pool['gradients']; loss=pool['losses'].mean(1); domain=pool['domains']
    p=pool['p']; theta=pool['theta']; groups=pool['groups']; lr=np.where(groups=='A_arch',.001,.01)
    statistics=[]; state_rows=[]; contrasts=[]; coordinates=[]; covariance={}
    for objective, alpha in [('Equal',np.ones(3)/3),('Sample',p)]:
        mean=(alpha[domain]/16) @ g
        for batch in [16,40960]:
            streams={}; finals={}
            for sampler,q in [('Natural',p),('Balanced',np.ones(3)/3)]:
                prob=q[domain]/16; weight=alpha[domain]/q[domain]
                exact=(prob*weight) @ g
                residual=float(np.linalg.norm(exact-mean)/max(np.linalg.norm(mean),1e-12))
                assert residual <= 1e-10
                cov=((prob*weight**2)[:,None]*g).T@g-np.outer(mean,mean)
                cov/=batch
                cov=(cov+cov.T)*.5
                covariance[f'{objective}_{sampler}_B{batch}']=cov
                rng=np.random.default_rng(20261016+seed+(dataset=='movielens')*10000)
                counts=rng.multinomial(batch,prob,size=REPLAYS*STEPS)
                weighted=counts*weight/batch
                estimates=(weighted@g).reshape(REPLAYS,STEPS,-1)
                streams[sampler]=estimates
                # All steps are independent draws at fixed theta.
                z=np.abs(estimates.mean((0,1))-mean)/(np.sqrt(np.maximum(np.diag(cov),0)/(REPLAYS*STEPS))+1e-12)
                assert float(z.max()) <= 8
                dc=counts.reshape(-1,3,16).sum(2)
                sumw=counts@weight; sumw2=counts@(weight**2)
                ess=sumw**2/sumw2
                statistics.append(dict(dataset=dataset,seed=seed,objective=objective,sampler=sampler,
                    batch_size=batch,replays=REPLAYS,steps=STEPS,exact_gradient_relative_residual=residual,
                    mc_gradient_max_z=float(z.max()),target_loss=float((alpha[domain]/16)@loss),
                    mc_loss_mean=float((weighted@loss).mean()),projected_cov_trace=float(np.trace(cov)),
                    gradient_mean_norm=float(np.linalg.norm(mean)),
                    mc_mean_gradient_norm=float(np.linalg.norm(estimates,axis=2).mean()),
                    mean_ess=float(ess.mean()),mean_ess_fraction=float(ess.mean()/batch),
                    unique_rows_mean=float((counts>0).sum(1).mean()),
                    importance_min=float(weight.min()),importance_max=float(weight.max()),
                    **{f'alpha_d{d}':float(alpha[d]) for d in range(3)},
                    **{f'q_d{d}':float(q[d]) for d in range(3)},
                    **{f'w_d{d}':float(alpha[d]/q[d]) for d in range(3)},
                    **{f'exposure_d{d}':int(dc[:,d].sum()) for d in range(3)},
                    **{f'absent_d{d}_observed':float((dc[:,d]==0).mean()) for d in range(3)},
                    **{f'absent_d{d}_expected':float((1-q[d])**batch) for d in range(3)}))
                for optimizer in ['SGD','Adam','AdamW']:
                    records, final=propagate(estimates,theta,lr,optimizer)
                    finals[(sampler,optimizer)]=final
                    for step,m,v,mh,vh,delta,cum in records:
                        for group in ['C','P','A_arch']:
                            mask=groups==group
                            state_rows.append(dict(dataset=dataset,seed=seed,objective=objective,sampler=sampler,
                                batch_size=batch,optimizer=optimizer,step=step,partition=group,
                                mean_m_norm=float(np.linalg.norm(m[mask])),mean_v_norm=float(np.linalg.norm(v[mask])),
                                mean_mhat_norm=float(np.linalg.norm(mh[mask])),mean_vhat_norm=float(np.linalg.norm(vh[mask])),
                                mean_delta_norm=float(np.linalg.norm(delta[mask])),mean_cumulative_delta_norm=float(np.linalg.norm(cum[mask])),
                                bias_correction1=1-.9**step,bias_correction2=1-.999**step,weight_decay=1e-5,clipping=False))
                del counts,weighted,dc
            for optimizer in ['SGD','Adam','AdamW']:
                a=finals[('Natural',optimizer)]; b=finals[('Balanced',optimizer)]
                for variable in ['m','v','mhat','vhat','delta','cumulative']:
                    difference=b[variable]-a[variable]
                    for group in ['C','P','A_arch']:
                        mask=groups==group; d=difference[:,mask]
                        means=d.mean(0); se=d.std(0,ddof=1)/np.sqrt(REPLAYS)
                        contrasts.append(dict(dataset=dataset,seed=seed,objective=objective,batch_size=batch,
                            optimizer=optimizer,variable=variable,partition=group,
                            mean_difference_norm=float(np.linalg.norm(means)),
                            mc_se_vector_norm=float(np.linalg.norm(se)),
                            max_coordinate_z=float(np.max(np.abs(means)/(se+1e-14))),
                            pathwise_rms_difference=float(np.sqrt(np.mean(np.sum(d*d,axis=1)))),
                            natural_mean_norm=float(np.linalg.norm(a[variable][:,mask].mean(0))),
                            balanced_mean_norm=float(np.linalg.norm(b[variable][:,mask].mean(0)))))
                    # Coordinate-level mean/SE at final step: no norm-only claims.
                    for k in range(g.shape[1]):
                        coordinates.append(dict(dataset=dataset,seed=seed,objective=objective,batch_size=batch,
                            optimizer=optimizer,variable=variable,coordinate=int(pool['coordinates'][k]),
                            partition=str(groups[k]),natural_mean=float(a[variable][:,k].mean()),
                            balanced_mean=float(b[variable][:,k].mean()),
                            difference=float(difference[:,k].mean()),
                            mc_se=float(difference[:,k].std(ddof=1)/np.sqrt(REPLAYS))))
            # Exact SGD mean and expected Adam vhat under fixed-theta IID replay.
            for sampler in ['Natural','Balanced']:
                final=finals[(sampler,'SGD')]['cumulative']
                expected=-STEPS*lr*(mean+1e-5*theta)
                z=np.abs(final.mean(0)-expected)/(final.std(0,ddof=1)/np.sqrt(REPLAYS)+1e-12)
                assert z.max()<8
                vv=finals[(sampler,'Adam')]['vhat']
                expected_v=(mean+1e-5*theta)**2+np.diag(covariance[f'{objective}_{sampler}_B{batch}'])
                z=np.abs(vv.mean(0)-expected_v)/(vv.std(0,ddof=1)/np.sqrt(REPLAYS)+1e-12)
                assert z.max()<8
            del streams,finals
    for name,data in [('estimator_statistics',statistics),('state_trace',state_rows),
                      ('state_contrasts',contrasts),('coordinate_contrasts',coordinates)]:
        pd.DataFrame(data).to_csv(target/f'{name}.csv',index=False)
    np.savez_compressed(target/'coordinate_covariance.npz',**covariance)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    parser.add_argument('--dataset',choices=['kuairand','movielens']);parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.check:
        residual=torch_check()
        dump(OUT/'optimizer_formula_check.json',{'status':'PASS_FIXED_THETA_FORMULA','maximum_absolute_residual':residual,
             'steps':64,'optimizers':['SGD','Adam','AdamW'],'decays':[0,1e-5],'independent_review':False})
        print(residual);return
    assert args.seed in [2022,2023,2024,2025]
    target=OUT/'numeric_oracle'/args.dataset/f'seed{args.seed}'
    target.mkdir(parents=True,exist_ok=False)
    started=time.time(); source=sha(__file__)
    extract(args.dataset,args.seed,target)
    replay(args.dataset,args.seed,target)
    assert sha(__file__)==source
    dump(target/'receipt.json',{'status':'PASS_NUMERIC_ONLY','training_runs':0,
        'dataset':args.dataset,'seed':args.seed,'elapsed_seconds':time.time()-started,
        'source_sha256':source,'scope':'INITIAL_THETA_FINITE_POOL_PROJECTED_COORDINATES_FIXED_THETA_REPLAY',
        'replays':REPLAYS,'steps':STEPS,'independent_review':False,
        'artifact_hashes':{p.name:sha(p) for p in target.iterdir() if p.is_file()}})
    print(json.dumps({'dataset':args.dataset,'seed':args.seed,'status':'PASS_NUMERIC_ONLY'}),flush=True)


if __name__=='__main__': main()
