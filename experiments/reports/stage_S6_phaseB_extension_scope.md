# S6 Phase B execution extension — 2026-10-07

## Authorization and scope

The user explicitly requested completion of the full F/T/HT/DH optimizer
history × current matrix for S6, directly through luna_runner, because the
design and implementation stages and part of execution were already complete.
The user authorized autonomous execution without further approval and required
two concurrent Python training processes for each multi-seed batch.

This request overrides the original Phase A STOP → no Phase B execution gate.
It does not change the original Phase A STOP decisions or thresholds. Phase B
is an explicitly requested diagnostic extension after inspecting Phase A,
not a GO-triggered confirmatory experiment.

Complete only missing HT/DH runs for KuaiRand/natural and MovieLens/d2_large,
seeds 2022, 2023, 2024, 2025. Reuse canonical F/T runs after source identity and
cross-phase pairing checks. Reuse existing S6/S6ML training and report code,
four-arm definitions, Sample objective, initialization, data exposure,
optimizer configuration, 50 epochs, and checkpoint selection. Preserve
Phase A outputs and analysis before any merged collection overwrites outputs.

## Role boundary

luna_runner executes existing protocols, validates invariants and pairing,
records raw outcomes and failures/retries, and collects results. It does not
implement algorithms or interpret scientific findings. The primary agent
performs analysis after execution and raw validation finish. Any necessary
implementation repair must return to coder and independent reviewer before
affected experiments resume.

## Planned analysis of the complete matrix

Rows are persistent Adam history (FULL/TARGET); columns are current applied
gradient source (FULL/TARGET):

| History / current | FULL | TARGET |
|---|---|---|
| FULL | F | DH |
| TARGET | HT | T |

For each seed, dataset, checkpoint, domain and task, let Y be test AUC or
LogLoss. Analyze the original contrasts T−F, HT−F, DH−F, HT−T, DH−T, plus:

- History effect at FULL current: HT−F.
- History effect at TARGET current: T−DH.
- Current effect at FULL history: DH−F.
- Current effect at TARGET history: T−HT.
- Average history contrast: (HT + T − F − DH) / 2.
- Average current contrast: (DH + T − F − HT) / 2.
- Interaction: T − HT − DH + F.

These are contrasts between trained trajectories. Selected outcomes use each
arm's independently validation-selected checkpoint and must be distinguished
from same-epoch contrasts at common9/19/29/39/49. Final equals common49 and
must not be counted as an independent replication.

Report absolute outcomes, per-seed contrasts, means and variation, positive
and negative seed counts, target/non-target cells, six-cell macro,
worst-domain and worst-cell, AUC and LogLoss. Retain practical AUC threshold
0.003 and strong threshold 0.005. Compare with Equal S5/S5ML anchors subject
to source/configuration comparability.

Report HR=mean(HT−F)/mean(T−F) and CR=mean(DH−F)/mean(T−F), but flag them as
unstable when T−F is near zero or sign-inconsistent; do not interpret large
ratios as recovered rescue in that circumstance. Assess absolute contrasts
and trajectory structure first.

The final report must independently assess stable phenomena, contradictions,
evidence strength (STRONG/MEDIUM/WEAK), competing explanations, size versus
difficulty, non-target trade-offs, and the experiment best suited to falsify
competing explanations. Both tested targets are the largest domains; domain
size alone cannot be isolated from identity, task difficulty, or dataset
structure in this extension.
