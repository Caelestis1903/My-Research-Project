# Hyperparameters, Metrics and Statistical Protocol

## 1. Immutable training protocol

| Setting | Value |
|---|---|
| Model | M3oE fixed1, 3 domains × 2 tasks |
| Datasets | MovieLens, KuaiRand |
| Seeds | 2022, 2023, 2024, 2025, 2026 only |
| Epochs | 50 |
| Batch | 40960 |
| Optimizer | inherited Adam, lr=.01, betas=(.9,.999), eps=1e-8 |
| Weight decay | coupled WD: MovieLens 1e-4, KuaiRand 1e-5 |
| EMA | inherited fixed EMA; ML rho=.9828205985452511, KR rho=.995 |
| Sampling | natural row population; no artificial domain balancing in clean objective |
| Precision | FP32 training and evaluation |
| Checkpoint | earliest strict maximum of validation six-cell Macro AUC; online/EMA independently |
| Test | sealed until frozen evaluation |

If current main source differs, Stage 0 must report the discrepancy and resolve it without modifying historical code or silently changing the protocol.

## 2. Hyperparameter policy

Stage 1 adds no performance hyperparameter. Zero means exact delta=0.

Stage 2 initially uses one global mixture lambda, with the first value fixed at .25 only when the convex mixture formula is correctly normalized. This is not the old extra-loss coefficient. If a search is triggered, use lambda in {0.1, 0.25, 0.5}, all five seeds, both datasets, 50 epochs per configuration.

Radius alpha is held at .03 initially to isolate objective design. Search alpha in {.01,.03,.05} only if the objective passes the Stage 2 validation gate and radius sensitivity is the remaining decision-relevant uncertainty. Each hyperparameter value is evaluated using all five seeds. Do not use one or two seeds for screening.

## 3. Metrics

For every dataset/configuration/seed/readout:
- six cell AUC and unscaled LogLoss;
- Macro AUC and Macro LogLoss;
- Traffic AUC/LogLoss;
- Pooled AUC/LogLoss;
- selected epoch and validation curves;
- online and fixed-EMA results;
- training wall time, examples/sec, GPU peak allocated/reserved memory;
- perturbation norm distribution, zero-delta/fallback fraction, skip rate, per-task non-target drift;
- objective contribution ledger by domain, task and label.

## 4. Pairing and intervals

For a candidate-control contrast, compute delta_s = metric(candidate,s)-metric(control,s) for each of the five fixed seeds. Report:
- mean paired delta;
- sample SD across the five seeds;
- 95% paired t interval: mean ± 2.776445 * SD / sqrt(5);
- number of positive seeds.

Seed is the independent unit. Epochs, examples, domain-task cells, batches and online/EMA readouts are not independent replicates. Do not pool these as n.

## 5. Selection and decision rules

Validation is used to select checkpoint and route. Use the earliest strict maximum of six-cell Macro AUC. Do not select by LogLoss.

Validation mechanism gate:
- A positive perturbation increment means the five-seed paired mean is >0 against its matching zero control on at least one dataset, with the other dataset and six-cell trade-offs reported.
- A positive validation increment is a reason to consider the next stage, not a formal algorithm pass.
- Do not impose all-seed positivity or require the CI to exclude zero as an extra hard gate.

Formal Test gate, only after frozen evaluation:
- mean paired Test Macro AUC gain >= +0.003 vs strongest applicable published model;
- every cell's five-seed mean AUC delta >= -0.001;
- no all-seed-positive requirement;
- report mean, SD, 95% CI and positive seed count.
LogLoss is independent: mean delta <0 indicates lower LogLoss. Do not trade LogLoss improvement for AUC acceptance.

## 6. Fixed EMA accounting

Report three comparisons separately:
1. Candidate fixed EMA vs clean online: total system gain.
2. Candidate fixed EMA vs clean fixed EMA: incremental gain of the new component.
3. Candidate online vs clean online: training-method gain without EMA readout.

Never call the first comparison the independent gain of the proposed augmentation.
