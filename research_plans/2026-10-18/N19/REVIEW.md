# N19 initial primary review

Authority: revised start.md principle11; primary self-review, not independent.
Inputs: DESIGN_SPECIFICATION, implementation and IMPLEMENTATION_REPORT.

Verdict: **PASS for engineering runs; formal runs require these to finish cleanly.**

- Relevant forward change checked against actual historical adapter (max5.96e−8).
  Baseline fixed fusion, single Adam, objective, train-meta exclusion and full
  population are explicit. No hidden revival of learned architecture weights.
- Original reference is an explicit separate protocol port; source includes
  `~bool` optimizer overlap and extra training-data architecture step. It cannot
  isolate only a fusion effect. original_sample is the fair same-objective arm.
- New P3 factors replace dominant slot matrices, rather than extending linear
  heads. Same maximum-rank warmup and exact48 active-rank budget are compared.
  Deployed counts exclude inactive factors; masked supernet storage is not claimed
  physically compressed until active-factor export is recorded.
- P4 double centering applies over destination domain/task axes, preserving a
  separate unconstrained-capacity control. P5 scores finite same-tower functions;
  regret targets are detached, while the supervised network trains end to end.
- Noise training uses only noisy outcomes and declared clean audits; nested
  joint likelihood includes both labels and uses known nested support. No outside
  audit clean labels enter the objective. Natural business noise is not claimed.
- P6 mask marginals .55 and allowed joint endpoints .1/.55 are valid. Exact
  correction applies to observed loss. EMA control variate uses predictions only.
  Copula statistics do not query extra labels. Physical forward computation is
  still dense: label/estimator budget is separated from measured runtime.
- All seeds and full budgets are mandatory. Selection uses selection half only;
  improvements use development only. Test is deferred to a frozen decision stage.
- Minimal engineering cases cover actual full-batch GPU training, original
  two-optimizer path, joint likelihood, task CF, copula+CV, regret auxiliary, and
  the ten-epoch rank-pruning transition. No unrelated boundary-test matrix.

Any blocking implementation failure is returned to primary implementation. No
formal scientific conclusion is licensed by these engineering receipts.

## Formal-training gate

Seven engineering training cases completed, exit0 and COMPLETE_ENGINEERING,
including original two-Adam, noisy joint likelihood, task counterfactual,
copula/control-variate, full-batch KR regret and ten-epoch rank transition.
ADDENDUM_01_SPECTRAL_RANK strengthens the paired rank baseline before formal
results. Both arms share canonicalization and moment reset. No unresolved blocker.
Verdict: **PASS WITH FIXES, applied; proceed to full formal budgets.**

## Final readout admission and optimizer-transfer reporting

Primary self-review (not independent): PASS for n19_freeze.py and the expanded
P3 figure. The admission script reads no test performance, requires all declared
formal budgets and four seeds per group, checks selected/final states and the
post-pruning selection rule, fingerprints the primary decision/source/receipts,
then adds only zero-optimization evaluation commands to the existing queue. It
must be invoked only after the primary writes the final pre-test development
disposition; file existence alone is not a scientific stopping rule. Historical
test use remains disclosed. Freeze cannot be recreated over an existing freeze.
Figure comparisons stay within optimizer tags and include the declared ML AdamW
rank transfer; no undeclared ML AdamW dictionaries are implied. These changes do
not alter model training, selection, population membership or any saved result.
Both edited scripts pass syntax parsing; no additional training/test grid.
