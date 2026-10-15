# Stage S0 Main Analysis

## Decision

**STOP Momentum — uniform smoothing is sufficient.**

Stage S0 does reproduce the fresh comp1 residual against the sample-weighted Adam baseline, but it falsifies the stronger claim that the residual is specific to domain-conditioned per-domain momentum. The validation-only uniform EMA family is better than comp1 in every seed after averaging the three fixed role rotations.

This decision closes the Momentum method line. No further gamma, EMA, controller, or momentum-schedule method design is justified by Stage S0.

## Evidence base and protocol validity

- 84/84 formal runs are valid: 7 arms x 3 role rotations x 4 seeds.
- All 12 rotation-seed blocks passed pairing audits for initialization, data/split, batches, optimizer/architecture steps, examples processed, and protocol fingerprints.
- Seeds 2022/2023/2024/2025 are the repeat unit; rotations are fixed crossed conditions within each seed.
- Model/backbone: the current `MDMTRec` implementation of M3OE on MovieLens under the strong 150000/34500/3000 imbalance setting.
- All formal comparisons use sample-proportional weighting and validation-only checkpoint/control selection.
- Historical A/B artifacts could not be admitted to the formal audit because the old files did not contain exactly six finite selected-cell metrics. The historical `+0.0054` is therefore compared descriptively with the fresh effect, not pooled with it.

## 1. Did the +0.0054 residual reproduce?

**Yes — STRONG evidence.**

At validation-selected checkpoints, comp1 minus Sample Adam macro-AUC is:

- mean `+0.005307`;
- seed-level SD `0.000523`;
- range `+0.004641` to `+0.005921`;
- positive in `4/4` seeds.

All three rotation means are positive: `d0_large +0.003205`, `d1_large +0.007843`, and `d2_large +0.004873`. The effect persists at all common epochs: `+0.007555`, `+0.010935`, `+0.009985`, `+0.009175`, and `+0.008194` at epochs 9/19/29/39/49. It is therefore not a selected-checkpoint-only artifact relative to Sample Adam.

Macro LogLoss also improves by `-0.023781` (lower is better) in all `4/4` seeds. Worst-cell AUC improves by `+0.006645` on average.

The fresh estimate is numerically very close to the historical `+0.0054`, but exact historical paired reproduction cannot be claimed because the historical audit inputs were invalid/missing under the new six-cell requirement.

## 2. Is the residual specific to domain-conditioned momentum?

**No — STRONG evidence.**

Against the validation-only selected uniform EMA family (`C_val`):

- comp1 macro-AUC difference: `-0.003033`;
- seed-level SD: `0.000815`;
- range: `-0.003778` to `-0.001931`;
- positive seeds: `0/4`.

The uniform selector chose `uniform_eq_tau` in 9/12 rotation-seed blocks, `uniform_p_tau` in 2/12, and `uniform_090` in 1/12. Thus this is not a result of test-set tuning or one arbitrarily privileged scalar match.

The conclusion is stable across training stage. Comp1 minus `C_val` mean macro-AUC is negative at every common epoch: `-0.002295`, `-0.001219`, `-0.001932`, `-0.003099`, and `-0.001020` at epochs 9/19/29/39/49. At fixed-final it is `-0.001020`, positive in only 1/4 seeds.

Against the single preregistered sample-weighted memory match C1, comp1 is approximately tied at selected checkpoints (`-0.000180`, only 1/4 seeds positive), although it is better later in training. The stronger family comparison removes any basis for attributing the residual uniquely to domain-conditioned gamma.

## 3. Can ordinary smoothing or ordinary Adam memory explain it?

### Uniform smoothing

**Yes — STRONG evidence.**

The best validation-selected uniform EMA family exceeds comp1 by `0.003033` macro-AUC on average and does so in every seed. `uniform_eq_tau` also has the highest unconditional selected mean macro-AUC among the seven arms (`0.739251`, versus comp1 `0.736151`). Ordinary smoothing is therefore sufficient to explain, and improve upon, the residual gain.

There is a metric trade-off: comp1 has slightly lower macro LogLoss than `C_val` by `0.002817` on average, with the lower-loss direction in 3/4 seeds, while losing AUC. This does not rescue the domain-conditioned mechanism because the preregistered KEEP gate requires a clear AUC advantage over the uniform family.

### Ordinary Adam memory

**Mixed — MEDIUM evidence.**

Comp1 beats the validation-selected Adam family at selected checkpoints by `+0.002243` macro-AUC, positive in 4/4 seeds, and improves LogLoss by `-0.012435`. It also beats the preregistered D1 control by `+0.003400`, positive in 4/4 seeds.

However, the advantage is stage-dependent: versus `D_val`, common-epoch differences decline from `+0.003843` at epoch 9 to `-0.002918` at epoch 49, and fixed-final is `-0.002918` with only 1/4 seeds positive. Ordinary Adam memory alone is not as complete an explanation as uniform EMA, but checkpoint/convergence behavior remains a competing explanation for the selected-checkpoint advantage.

## 4. Per-role, per-task, per-domain, loss, and negative-transfer evidence

### Size role

Comp1 minus Sample Adam selected AUC gains are:

- Large: `+0.002270` (4/4 seed means positive);
- Medium: `+0.001143` (2/4 positive);
- Small: `+0.012507` (4/4 positive).

No role has a seed-aggregated loss of `-0.003` or worse. Corresponding LogLoss differences are `-0.002158`, `-0.006067`, and `-0.063118`. Thus there is **no small-domain improvement / large-domain decline trade-off** relative to Sample Adam. Evidence: **STRONG** for absence of a large-role decline, **MEDIUM** for a positive medium-role AUC effect.

Relative to `C_val`, comp1 is worse for Large (`-0.003363`) and Medium (`-0.005206`) and approximately tied but still lower for Small (`-0.000529`). This is not a small-domain-win/large-domain-loss trade-off either; it is broad AUC underperformance, weakest on the Small role.

### Tasks and physical domains

Both tasks improve over Sample Adam: task 0 `+0.004683` and task 1 `+0.005931`, each positive in 4/4 seed means. All physical domains also improve: domain 0 `+0.005569`, domain 1 `+0.005331`, and domain 2 `+0.005020`, each positive in 4/4 seeds. The fresh residual is therefore not driven by one task or one physical domain. Evidence: **STRONG**.

### Domain size versus difficulty

Sample-Adam baseline AUC falls by assigned size role (Large `0.749412`, Medium `0.733488`, Small `0.709632`) within the crossed design. Physical domain 2 is also intrinsically harder: its mean baseline AUC is `0.709001`, versus `0.738618` and `0.744913` for domains 0 and 1, and it remains lower at every size role.

The comp1-minus-baseline gain changes much more by size role (`+0.002270/+0.001143/+0.012507`) than by physical-domain identity (roughly `+0.0050` to `+0.0056`). Domain size is therefore a plausible main modifier of the apparent comp1 gain, while physical-domain difficulty independently explains baseline level. Evidence: **MEDIUM**, because this stage manipulates role assignment but was not designed as a full causal size/difficulty decomposition.

### Negative transfer and worst-cell behavior

At selected checkpoints, mean practical AUC negative-transfer rate is `0.111111` for Sample Adam, `0.041667` for comp1, and `0.000000` for `uniform_eq_tau`. Concordant practical NT is `0.041667`, `0.027778`, and `0.000000`, respectively. Worst-cell AUC means are `0.697487`, `0.704132`, and `0.707800`.

These observations reinforce the main result: comp1 improves the baseline, but the uniform control does at least as well on residual NT and worst-cell performance. Evidence: **MEDIUM**, since rates are coarse over six cells per run and Stage 0 was not the dedicated residual-NT reproduction stage.

## 5. Stable phenomena, contradictions, and competing explanations

Stable repeated phenomena:

1. **STRONG:** comp1 improves Sample Adam by about `+0.0053`, with 4/4 seed consistency and persistence across common epochs.
2. **STRONG:** a validation-selected uniform EMA family outperforms comp1 by about `+0.0030`, with 4/4 seed consistency.
3. **STRONG:** gains over Sample Adam occur across both tasks and all physical domains, and do not require sacrificing the Large role.
4. **MEDIUM:** the largest comp1-minus-baseline improvement is concentrated in the Small role.
5. **MEDIUM:** all methods select relatively early checkpoints (comp1 mean epoch `8.17`; arm ranges are approximately epochs 4-17), and late-training behavior changes the Adam-family comparison.

Contradictory/qualifying evidence:

- Comp1 loses AUC to `C_val` but has slightly better LogLoss. This is a metric trade-off, not proof of unique momentum value.
- Comp1 beats the Adam family at selected checkpoints but loses at fixed-final. Checkpoint/convergence remains a plausible explanation for that comparison.
- Historical fresh-versus-old pairing is unavailable, so only the fresh residual magnitude—not exact run-level historical reproducibility—is established.

Competing explanations ranked by current support:

1. **Ordinary temporal smoothing:** strongly supported and sufficient.
2. **Size-dependent regularization benefit:** moderately supported as an explanation of where the baseline gain appears.
3. **Checkpoint/convergence effects:** moderately supported, especially versus Adam controls.
4. **Domain-conditioned momentum as a unique mechanism:** falsified by the uniform family comparison.
5. **Random fluctuation or one-task/one-domain artifact:** strongly disfavored.

## 6. Formal Stage S0 answers

1. **Did `+0.0054` residual reproduce?** Yes: fresh `+0.005307`, 4/4 seeds positive.
2. **Is it unique to domain-conditioned momentum?** No.
3. **Can ordinary smoothing/effective momentum explain it?** Ordinary uniform smoothing is sufficient and better; Adam memory is a partial, stage-dependent explanation.
4. **Does Momentum leave the paper's method main line?** Yes. Formally **STOP Momentum**.
5. **Is there a publishable negative result?** Yes: after objective correction, a seemingly domain-specific momentum residual reproduces against the baseline but disappears—and reverses—under stronger matched uniform-smoothing controls. This demonstrates why objective correction and mechanism-matched controls are both necessary before attributing MDMT gains to a domain-conditioned optimizer.

## 7. Next-stage decision

**Stage 1 GO: Residual Negative Transfer Reproduction.**

The most discriminating next experiment is the already specified Stage 1 program: with sample-proportional weighting fixed and Momentum removed from the method line, independently reproduce cell-level residual negative transfer across the prescribed backbones/rotations/seeds and compare against matching single-domain/STL references. This tests whether the remaining scientific problem is real rather than continuing to tune a mechanism already falsified by Stage S0.

Per the repository workflow, Stage 1 must begin with a new designer call that designs only Stage 1; this report does not design Stage 1 or any later stage.
