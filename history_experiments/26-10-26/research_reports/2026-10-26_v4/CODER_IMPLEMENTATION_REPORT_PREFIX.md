# CODER IMPLEMENTATION REPORT — prefix / P0 / P1

Native/pilot production core is implemented in `experiments/mdmt_dfd_20261026/`. This coder has not run formal trajectories or analyzed research results. Initial syntax compilation passes; independent reviewer owns admission checks. The native/pilot core and completed P1 OOF path have separate source scopes for staged independent admission.

## Production prefix scope ready for independent review

`data.py`, `native_adapter.py`, `state.py`, `plans.py`, `metrics.py`, `costs.py`, `prefix.py`, `functions.py`, and vendored dependencies define the prefix source hash. `demand.py`, `readers.py` and later orchestration are intentionally outside this core hash because they do not participate in native/pilot training. This is a dependency boundary, not permission to change the reviewed prefix code while running. The source receipt enumerates exact files and hashes.

`prefix.run_prefix(RoleBundle(dataset), seed, artifact_root, device='cuda:0', resume=True, stop_after='native1')` executes the required scratch native first epoch, retains a complete continuation boundary, and returns. Rerunning with `stop_after=None` continues the same trajectory, then caches selected native outputs and trains owner-only pilots. `stop_after='pilot1'` similarly retains the first required pilot pass. Native is 24 epochs, eligible validation LL 12–24/earliest tie; pilot is 12 passes, no selection. No shortened quality trial is introduced.

Data wrapper checks complete saved array/manifest/encoder/role hashes and reconstructs all role integer identities without labels. `RoleBundle.verify(output_dir,full=True)` additionally verifies raw hashes/domains/labels, exact prepared encodings and vocabulary/scaler fit-only scope; it records task support. ML raw user/item IDs are preserved; KR missing original identity is declared.

Native mechanically mirrors pinned fixed pre-sigmoid/shared outputs, preserving gate detach. Original model is initialized first under dedicated SeedSequence native stream; learned fusion tensors are dropped and canonical backbone copied to fixed. Native sigmoid parity and actual expert-wise LN h256 have separate reviewer receipts. No AutoML/DARTS is restored.

Training uses coupled Adam(.001,.9,.999,1e-8,wd1e-5), one step per full logical and physical batch40960, loss sums divided by 2*actual logical B, FP32/no AMP/TF32 and short final batches. Pilot loss creates only actual source heads; absent source grad=None. Raw source order/ratios remain unchanged.

Complete checkpoints preserve current stage/pass/next logical step, model/optimizer state, gradients/None, modes/requires_grad, RNG, row-plan hash, selected history and immutable source/data/config/plan identity. Cache uses original global row indexing and records selected native state/extractor/encoder/x hashes; no labels are cached. `load_prefix` checks pilot->cache and selected native->cache dependencies.

Storage: metadata under `/home/xzf/Code/phase0-3/artifacts/mdmt_dfd_v4/v4_0/prefix/<dataset>/seed<seed>/`; temporary full states/cache under this same campaign root in `_temporary/prefix/...` via explicit internal aliases. No new write targets `/home_new`. Existing artifacts are not modified. Cost leaf and label-access journals are separate from rollback state.

## Fixes before any formal execution

A duplicate `dependencies` keyword in generic payload construction was corrected to construct then update before training. Resume dependency checks were strengthened before admission. Neither issue affected an experimental trajectory, since none has been launched by this coder.

## Scope not yet claimed

This coder report does not itself admit execution; independent review controls admission. P2/P3 implementation has its own owner/report, and conditional P4 requires exact author adaptations and separate review. Seven pinned dependencies have independent remote Git-blob identity PASS receipts.

## Superseding pre-run revisions — current implementation

The following pre-run corrections define the current implementation:

1. **Physical batch is full40960 for both datasets and pilots.** The optional micro4096 implementation gave a small FP32 GEMM accumulation difference in ML (maximum gradient difference about4.02e-6;19 elements exceeded the locked atol1e-6/rtol1e-4), while KR passed. The denominator was correct and both full40960 schema forward/backward paths fit GPU0. Production therefore uses the original permitted full logical batch physically; no tolerance was relaxed, no learning rate, loss or Adam-step semantics changed. The failed optional optimization remains in the reviewer diagnostic receipt and is not admitted for production. `semantics.verify` now checks the actual full physical production path against its direct full reference.
2. **User storage revision controls all new writes.** No new write targets `/home_new`. Prefix model/Adam/checkpoint/h/z/pilot-residual files are temporary under `artifact_root/_temporary/prefix/<dataset>/seed<seed>/`. Metadata aliases inside this campaign point only to that internal location. A few rolling last/selected/epoch24 files support exact continuation and final prediction/reconstruction; the finalizer removes all model files and large intermediate caches after required downstream work and locked evaluation. Logs, row predictions, metrics, config/source/data/selection receipts remain. The user explicitly overrides the original specification's permanent checkpoint retention requirement.
3. Seven vendored pinned dependencies were independently fetched from exact remote `c5618e28` paths and match their Git blob hashes. `vendor/pinned_fixed/SOURCE.json` records PASS; complete receipt is `research_reports/2026-10-26_v4/PINNED_SOURCE_IDENTITY.json`.

Prefix core source is frozen for independent admission at this current version. No formal trajectory has been run by the coder.

## P1 implementation completion and minimal smoke receipt

`demand.py`, `diagnostics.py`, `semantics.py` and `cli.py` are now implemented. P1 ETA_FULL, LOGIT and DIRECT each use exactly the heldout-fold complement for20 passes, batch512/Adam.001wd0, target fit normalizers, dedicated immutable component streams and no heldout selection. Component0/1 name the **heldout** fold, so their training rows are the opposite fold; exact train/heldout row hashes and intersection receipts are stored. Both tasks share the model-seed-independent row fold. CONST independently solves the three-candidate train-fold simplex with the production solver and feasibility/FW-gap acceptance.

DEMAND/LOGIT actions use argmax of OFF/analytic μ only, with OFF/source-ID tie order, never full-pool m or τ. Risk conversion uses FP64 clipped diagnostic logits; DIRECT training separately computes exact un-clipped half-mixture logits by stable log-probability/log-complement logsumexp, so its training risk does not inherit the diagnostic clip. OOF outputs save candidates, η/μ, actions, four strategy p/loss, true train-fold constant d prediction, fold/rows/y/domain and all actual trained head/optimizer hashes. Full-pool refit remains a separate API invoked only after dataset-level P1 passing.

`refit_eta` returns six heads/normalizers and actual Adam state; `make_responsibilities` uses full-pool observed m, equal-target weighted SD around the weighted pooled mean and frozen positive two-slot mass. It checks complete cache/pilot/η/normalizer/OOF/fit-row identity before reusing a responsibility cache. Source domain is unchanged; source inference features contain no target label. Source/target clipping, μ, ESS/mass, joint responsibility correlation and label×local-confidence-decile summaries are retained.

Minimal actual-schema smoke used64 real MovieLens P_form rows with a real pinned native packet and frozen pilot. ETA and LOGIT output shape64, DIRECT shape64×3; all finite, four Adam parameter states per head. Exact training half-mixture logits reconstructed candidate probabilities with maximum error0. The semantic smoke used one batch/pass solely to validate API/backward/state serialization, not a scientific trajectory; it is excluded from research results and its temporary model files were deleted. Machine receipt: `artifacts/mdmt_dfd_v4/v4_0/CODER_P1_SMOKE_RECEIPT.json`. Syntax compilation passes for the entire package. No algorithm efficacy analysis or formal training was performed by the coder.

Executable entry: `python -m experiments.mdmt_dfd_20261026.cli` with inspect/build-plan/verify-semantics/run-stage --stage p1 --dataset movielens --seed5101 --resume, decide --stage p1, freeze-evaluation, evaluate-exposed --locked-only, summarize. Standard argparse requires `--seed 5101` with a space. P2/P3 commands invoke the separately owned `pipeline.run_pipeline/decide_stage`, and return honest NO_START after failed predecessor gates. A triggered P4 explicitly waits for its locked exact external implementation/review, rather than fabricating a simplified baseline.

Final exposed evaluation and cleanup are orchestration steps after all triggered outcomes are known. Any scientific negative remains a result, not a reason to change seed/model/threshold or rerun. The parent orchestrator owns analysis and final report writing.

OOF export recovery accepts a completed bundle only when its sidecar identity and NPZ hash match. An interrupted export is rebuilt from the same complete head checkpoints without repeating completed training. Dataset-level P1 arithmetic additionally validates the complete sidecar, NPZ SHA, matching checkpoint identities and current demand source hash; missing/corrupt output remains INCOMPLETE.

A final P1 wrapper fix renamed cost-journal metadata from `component` to `rng_component`, avoiding a Python duplicate argument in `CostLedger.measure/access`. A minimal real64-row check exercised ETA/LOGIT/DIRECT through the actual cost ledger:3 leaf events and3 label-access events, all completed; no formal trajectory or quality analysis. `CODER_P1_LEDGER_SMOKE_RECEIPT.json` records this check and its temporary models were deleted. Current demand/readers/diagnostics/CLI bytes are frozen for the independent P1 source-matched admission.
