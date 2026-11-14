# Experiment Matrix

## Fixed platform

- Datasets: MovieLens and KuaiRand.
- Model: M3oE fixed1, exact current source snapshot.
- Tasks: MovieLens rating>3 / rating>4; KuaiRand click / long_view.
- Domains: repository's fixed three-domain mapping.
- Training: preserve natural sample proportions, batch size, split, preprocessing, epoch budget, optimizer, learning rate, weight decay, and seed mapping from accepted FULLTRAIN_NATURAL80_10_10 / AUC_SELECTED_V1 protocol.
- Seeds: 2022–2026.
- Checkpoint: earliest strict maximum of validation six-cell Macro AUC.
- EMA: fixed shadow as internal control; same rho and implementation as source baseline.

## Stage 0: semantic and correctness checks

| Test | Expected result |
|---|---|
| Pair sampler | Positive/negative target labels only; same domain; no other-task-label matching |
| Projection | Damped projection reduces Jq residual and does not increase target-gradient norm |
| Nullspace edge case | Near-zero q causes skip, no NaN/Inf |
| Actual forward | Non-target logit drift is within frozen tolerance or bounded backtracking succeeds |
| Label identity | Positive remains target label 1, negative remains target label 0 |
| Clean path | L_clean and optimizer update match fixed1 baseline exactly when lambda_aug=0 |
| Randomness | Independent deterministic RNG stream; clean data order unchanged |
| Resume | Model, optimizer, EMA, RNG, pair sampler and epoch/step restore exactly |

## Stage 1: five-seed pilot

Run both datasets, all five seeds:
- B0: clean online + EMA.
- B1: TGA, fixed pair budget.
- B2: DT-PGNA, same pair budget and nominal perturbation radius.
- B3: random-direction matched-norm augmentation.
- B4: isotropic Gaussian perturbation matched to DT-PGNA per-sample norm.

Use pilot to establish numerical efficacy, cell-wise validation trajectories, and cost. Do not use pilot test metrics to select.

## Stage 2: lightweight epsilon search

After Stage 1 passes:
- epsilon ratio alpha in {0.01, 0.03, 0.05};
- fixed K=128 pairs per domain-task cell per batch, subject to available positives/negatives;
- lambda_aug fixed at 0.25 for first search;
- same five seeds, both datasets, TGA and DT-PGNA.

Select a single common alpha using validation-only five-seed paired Macro AUC with six-cell protection. If no alpha passes, do not search a larger grid. A per-dataset alpha is allowed only if common alpha has opposite dataset effects and the per-dataset rule was predeclared in the manifest.

## Stage 3: confirmation

Freeze alpha and all implementation semantics. Run:
- clean online / fixed EMA;
- TGA online / fixed EMA;
- DT-PGNA online / fixed EMA;
- matched Gaussian and random controls if not already identical to frozen pilot identities.

No further hyperparameter changes after confirmation begins. Evaluate published models using same accepted protocol; do not retrain published baselines unless identities or source configurations are invalid.

## Stage 4: targeted ablation

Only if DT-PGNA passes:
- projection vs no projection at matched perturbation norm;
- target-only BCE vs clean BCE only;
- actual non-target drift vs first-order predicted drift;
- K pair budget sensitivity (64 vs 128 vs 256) only if cost or skip rate changes the decision.

No all-factor factorial grid. No domain-size stratification. Do not infer difficulty from sample counts.
