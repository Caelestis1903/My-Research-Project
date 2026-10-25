# LPS coder implementation report

Stage IMPLEMENT, following the completed DESIGN SPECIFICATION. No research outcome analysis or formal training was performed by this coder. Independent review remains required.

## Files and interfaces

- `experiments/mdmt_next_v3_20261025/lps.py`: `LpsTrainer(cfg,campaign,out=None,device='cuda:0').train(resume=None,until=None)`, six core arms and internal `WARM_LPS_READ`.
- `experiments/mdmt_next_v3_20261025/semantics_lps.py`: `validate_semantics(campaign,device)`, compact capacity, permission, Adam, hypergradient and actual-path/resume examples.

Current file SHA256: `lps.py` = `b8ad3a0c78a75fd76af5cd0ccaaaba0dafe675dea3a63fe6f3295b969f8cb5c3`; `semantics_lps.py` = `bcbc3ee200af3d1d586d118071af7f29c7873dfea7a9817941850c228ab2dd00`. The orchestrator must freeze the complete imported-source receipt after all worker/reviewer fixes.

## Implemented specification

The native adapter uses historical `risk=False` training semantics. Detached expert-wise normalized h256 feeds 3×4 separate no-bias rank4 blocks; owner sums every block. Reader matrices have independent target/task/source parameter identities. Current a32/u34 contain fixed Gaussian projection and detached native logits, without foreign geometry.

Main Adam owns native, bank and reader only. Owner activation routing multiplies bank gradients once by detached w; native/reader are unweighted, foreign bank paths are detached, inactive source/reader operations are omitted to preserve None/Adam-step semantics. L3 weights are precomputed across the entire logical batch and source/task before microbatch accumulation; the default is the full 40960 logical graph.

L3/L4/L5 formation uses each source fit512 and target P_form512, natural source mass, current actual bank moments/per-parameter steps and coupled functional Adam. Virtual support uses differentiable w and `create_graph=True`. Outer target owns current real z0 and only foreign content is virtual, with separately computed .75 deployed/.25 .5 backgrounds. Only allocator Adam commits; uniform reference and real-bank/optimizer no-submit checks are logged.

All six core arms are implemented. L0 is independent all-OFF and fits MONO/FEATURE on all P_label, with independent validation-AUC selected checkpoints. L1 is coarse two-source gating; L2 uniform block formation; L3 scalar meta; L4 block meta; L5 FP64 bounded L-BFGS-B constant reading with old/two optimized whole-policy candidates. Read heads use independent head optimizers/shuffles and complete-policy P_cal selection, including the unchanged policy/optimizer candidate.

Shared read-family warmup saves `epoch10_pre_refresh` before specialization; forks restore only the common core/real optimizer/state/RNG, then initialize each own formation/read refresh once. Warmup validation prefix is referenced/replayed into logical run metrics. L0 cannot use the read-family fork. Epoch10 initial refresh and every-even subsequent formation/read refresh precede validation, odd epochs also validate. Full atomic last and own selected/AUC/LL/terminal snapshots include policy/allocator optimizers and trackers.

Outputs include per-run effective config/data manifest, long validation/Q_explore metrics, exact selected/terminal predictions, checkpoint identity hashes, labelled `test_metrics.csv`, model/optimizer state links, component costs, information access and cached repeated label-use counts. Refresh journals include weights/support, two outer backgrounds, uniform reference, gradient cosine/norm and virtual displacement differences, exact row IDs, candidate scores and solver states. Fixed selected/terminal diagnostics retain foreign/owner block and source deletions, label-free gate quartile groups, and L4 same-bank strong constant-versus-conditional reading. Native/bank/reader are not changed by those diagnostics. Inference timing includes complete local MONO/FEATURE pipelines and measured warmup work.

Conditional CAL_PROB_CLIP, OWNER_META, GEOM and PAIR are deliberately not silently enabled: core dispatch rejects unreviewed variants. If an explicit P2 trigger activates them, their bounded specification must be implemented and reviewed before that stage; no alternate outcome-dependent defaults exist.

## Necessary checks run

Python syntax compilation passed. Compact admission checks ran under `CUDA_VISIBLE_DEVICES=0` with the verified Python3.13/torch2.9 environment; no GPU1 context or formal run was opened. Receipt: `artifacts/mdmt_next_v3/v3_0/semantics_lps.json`.

| Check | Actual observation |
|---|---|
| FP64 block sum vs rank16 | forward max error 1.39e-17; gradient max 1.67e-16 |
| Real owner route vs explicit one-w bank gradient | max error 0; forward/native/reader errors 0 |
| Foreign target-to-bank direct gradient / inactive Adam | None; no inactive states or parameter changes |
| Scalar full-source/task normalization | mean error 0; intentionally split normalization differs by .01294 |
| Whole-batch coverage | one selected batch in each of ten complete ten-batch blocks |
| Initialized functional coupled Adam vs real Adam | max absolute error 0, None skips |
| FP64 hypergradient finite differences | analytic 9.834512e-7; relative errors 1.31e-7 to 8.84e-7 across frozen step sweep |
| Production small real-batch + full512 support/query meta path | finite hypergradient norm 5.57e-5; allocator changed; complete real model/main optimizer unchanged |
| Full production save/restore roundtrip | core, main/meta optimizers, allocator, policy and training trackers equal |

The production acceptance batch has 57 real fit rows solely to exercise semantic paths, and uses isolated temporary output, excluded from research trajectories. A separate small synthetic cached packet exercised actual direct read fitting and FP64 constant solver completion, without native/model training. These are implementation checks, not evidence of method efficacy.

## Defects corrected before admission

Reviewer found a real duplicate `z0` keyword when the production meta support packet reentered `from_features`. It now copies feature fields then overwrites computed outputs; the actual production meta example exercises that path. Self-review also removed unused foreign computation on OFF paths, added explicit whole-policy optimizer restoration/access accounting, same-bank comparison, cross-filesystem state links and exact resume identity checking. No research trajectory was affected because none was started.

No unresolved core algorithm blocker is known to this coder. The independent reviewer must decide admission; triggered conditional implementations and final cost-package freezing remain later-stage work, not claimed complete here.
