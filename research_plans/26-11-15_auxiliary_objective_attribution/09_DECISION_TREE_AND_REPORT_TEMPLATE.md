# Decision Tree and Final Report Template

## 1. Stage 1 decision tree

For each dataset, report the four principal contrasts:
- Z-PAIR - CLEAN;
- Z-EXAMPLE - CLEAN;
- B - Z-PAIR;
- C - Z-EXAMPLE.

Also report Z-PAIR - Z-EXAMPLE. Do not reduce all findings to a single pooled mean.

### Route decisions

1. **Both zero controls are negative; both perturbation increments are non-positive.** Stop current auxiliary-objective and adversarial-view family. Do not launch Stage 2 or broaden HPO.
2. **Z-PAIR negative, Z-EXAMPLE neutral/positive.** Remove pair-balanced auxiliary objective. If C-Z-EXAMPLE is non-negative, proceed with the natural-row objective (NDP-A or NDP-B depending on cost).
3. **Z controls neutral/positive; perturbation increments negative.** The auxiliary target is not the main problem; current adversarial directions/caps are. Do not keep the current A/B/C recipe. A simpler task-specific view may be considered only if it changes the objective rather than merely retuning the same direction.
4. **One perturbation increment is positive but the full arm remains below CLEAN.** Preserve it as a conditional signal; redesign the objective to preserve natural row-task weights. Stage 2 is justified if the gain is stable across seeds and not driven by one cell.
5. **A zero control itself is positive and the perturbation increment is near zero.** Prefer the zero-perturbation auxiliary objective if it is simpler and yields the same ranking outcome; do not claim input perturbation as the contribution.
6. **All effects are small/uncertain and do not alter route choice.** Stop; do not add experiments for theoretical completeness.

The numerical values above are interpreted with the five-seed paired mean, SD, CI and per-cell effects. CI crossing zero is descriptive, not an automatic veto. No requirement that every seed be positive.

## 2. Stage 2 decision

Advance NDP-A/NDP-B only if Stage 1 identifies either a viable perturbation increment or a clear objective-weighting defect that the natural mixture corrects.

If NDP-A improves validation Macro AUC relative to CLEAN fixed EMA and its zero-delta counterpart, keep it for bounded HPO. If only NDP-B is feasible, compare it with CLEAN and its matching zero-delta subset control. If the improvement is only LogLoss, retain it as a probability-loss finding, not an AUC algorithm success.

If neither candidate provides a positive validation AUC signal, stop this task-sensitive input augmentation branch.

## 3. Formal acceptance

After freezing the selected candidate, all hyperparameters, code, data and checkpoint protocol, evaluate Test for all five seeds. Compare against the strongest applicable published model under the same population/split and selection protocol. A formal AUC pass requires mean paired Test Macro AUC gain >= +0.003 and every six-cell mean delta >= -0.001. Do not require every seed positive. LogLoss is evaluated separately; a mean decrease is an LL improvement.

If an applicable published model lacks paired seed endpoints, do not fabricate paired confidence intervals. Report the limitation or run a matched published baseline only if its implementation and protocol can be verified.

## 4. Report structure

1. Executive decision and exact stage status.
2. Repository/source/data/protocol identities.
3. Research evidence table: historical line, intervention, result, accepted conclusion, design constraint.
4. Stage 1 objective and support definitions.
5. Effective-weight ledger and per-cell label/domain/task contributions.
6. Five-seed validation results: six cells, Macro, Traffic, Pooled; online and fixed EMA.
7. Paired contrasts with mean, sample SD, 95% CI and positive-seed count.
8. Training-stage results and selected epochs.
9. Cost, memory and implementation checks.
10. Competitive explanations and what is/ is not identified.
11. Stage 2 trigger decision and exact next action.
12. Explicit NOT_TRIGGERED / NOT_RUN / ENGINEERING_BLOCKED labels.
13. Test status and confirmation that no Test data was accessed before freeze.

## 5. Required machine-readable decision fields

decision.json must include:
- status, stage, dataset, seed list and run count;
- source/config/data/selection hashes;
- all paired contrast values and confidence intervals;
- six-cell mean deltas and per-seed deltas;
- effective objective weights and support overlap;
- online and fixed-EMA endpoints;
- Test accessed boolean;
- stage advancement decision and reason;
- untriggered experiments explicitly listed;
- independent review verdict and blockers.

## 6. Final scientific questions

The report must directly answer:
1. What did the 2026-11-15 pilot establish?
2. How much of the observed loss is reproduced by Z-PAIR and Z-EXAMPLE?
3. Does perturbation add independent value conditional on the auxiliary objective?
4. Are pair balancing and example sampling meaningfully different?
5. Which exact loss weights are applied to each domain/task/label in practice?
6. Is natural-distribution-preserving augmentation worth continuing?
7. Is strict cross-task projection needed, or is a simpler unprotected method preferable?
8. What is the next single experiment?
9. What result stops the current branch?
