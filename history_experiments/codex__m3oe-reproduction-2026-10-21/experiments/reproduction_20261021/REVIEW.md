# REVIEW REPORT

## 1. Verdict

**PASS WITH FIXES.** All identified blocking implementation issues have been corrected and independently rechecked. No CRITICAL issue remains. The reviewed implementation may enter the RUN stage: luna-runner must first complete the actual full-batch, one-epoch profiles, then launch formal jobs only after those profiles succeed. A profile is not a formal result.

Reviewed frozen design SHA256: `3a1270cc7b434802c2a68f6d0dbcf281470cf63195f266f070fde816341ae65f`.

This is an implementation review, not an effectiveness assessment. The reviewer ran small forward/backward/update checks only, with no full epoch, formal training, hyperparameter selection, or research-result analysis.

## 2. Requirement-by-requirement verification

| Requirement | Verification and result |
| --- | --- |
| Complete packages inside both requested folders | PASS. Both directories contain basic/features/initializers/layers, actual MDMTRec models, trainers and utils plus executable run.py. `diff -qr` finds the two MDMTRec packages identical, excluding bytecode. Trainer classes explicitly adapt the frozen runner; historical positional API compatibility is not claimed. |
| Actual original M3oE backbone | PASS. Independently diffed original.py against `m3oe/MDMTRec/models/multi_domain/model.py`: changes are only keeping routing masks and output/result allocations on the input device. STAR, shared/domain/task experts, detached-input gates, six towers and fusion operations remain original. D/G instantiate supplied MDMTRec, not the unused PLE class. |
| O-original-ported | PASS. Four learned sigmoid Weights modules retain raw initialization 1. Main Adam includes all parameters, reproducing the historical truthy `~bool` path; auxiliary Adam includes the same four fusion tensors. Source trainer confirms persistent **training** iterator, pooled BCE and extra epoch update. Independent check confirmed auxiliary step changes fusion parameters only. The isolated auxiliary loader and unified selection/encoding are explicit controlled-port differences. |
| B-fixed | PASS. Diff against original.py changes only four learned modules/calls to scalar coefficients. All coefficients are 1, with one Adam. Original ordinary backbone tensors are preserved. |
| D/G supplied augmentation and weighting | PASS. Supplied model.py is byte-identical to both package model.py files; layers preserve original EmbeddingLayer and add supplied DBO classes only. `batch_loss` chooses smallest real training domain, copies its rows/labels once, appends a Boolean mask, and adds std .1 Gaussian noise to the complete flattened appended embedding before STAR/auxiliary calculations. Routing IDs and categorical inputs do not change. Real weights are `.1+.5*specificity`, with gradient retained; copies use detached real mean. Complexity is observed but absent from the loss. |
| Loss formula and tensor ordering | PASS. Input is features×batch, predictions/labels are batch×2, cells ordered t0d0,t0d1,t0d2,t1d0,t1d1,t1d2. Each cell uses mean(weight×BCE); the six cell means are summed without division by six. Copy rows participate in the scarce cell denominator. Empty groups use graph-safe zero. |
| G all-parameter balancing | PASS. Six autograd gradients use all model parameters, Euclidean norms and detached scale `.5+.5*max(norm)/norm`; zero norms use 1. Final gradients sum the scaled cell gradients. Every parameter receives an explicit zero gradient when disconnected, preserving supplied behavior. Independent numeric scale check and empty-domain backward passed. |
| Gradient paths and tau0 audit | PASS. Independent checks confirm specificity receives nonzero gradient, D complexity gradients are None and G complexity gradients are zero. D and G with balancing disabled are bitwise equal after one matched Adam step. Independent tau0 check confirms None-vs-zero occurs for the two complexity matrices. Full coder receipts also retain small prediction-parameter floating accumulation/update differences. Consequently G−D is the supplied balancing-implementation increment; it is not a strict pure-amplitude contrast. |
| Canonical paired initialization | PASS. `make_model` constructs original seed-specific backbone once per invocation and copies matching tensors into B/D/G, excluding learned fusion and DBO-only tensors. Independent checks on both datasets confirm shared canonical/copied hashes and identical complete D/G initial state. Independent loader generators isolate main/auxiliary/evaluation order; noise has its own deterministic seeded fork. |
| Data identity, natural domains, train-only fitting | PASS. All 21 source snapshots independently match both manifest SHA and current originals. Raw hashes match design. Audited loader diff removes only MovieLens role downsampling and its assertions. Categorical vocabularies/OOV0 and MovieLens age scaler fit real global training prefix, routing IDs stay 0/1/2, feature order is explicit. KuaiRand excludes tab input and adds no user/video ID. |
| Split and counts | PASS. Independently verified cached encoded-array hashes and exact row-ID ranges floor(.8N)/floor(.9N), without shuffle, sorting or downsampling. MovieLens split sizes 800167/100021/100021; train domain counts 155949/318599/325619. KuaiRand sizes 2006132/250767/250767; train domain counts 172782/1743898/89452. Row-order provenance is honestly qualified; upload_dt is not used as interaction time. |
| Training/selection/baselines | PASS by direct code inspection and basic update checks. Adam .01/.001, decay1e-5, batch40960, dims[512,256,256,64], expert4, embedding16, no AMP/TF32/clipping/scheduler. Epoch limit50, strict validation six-cell macro AUC, earliest tie, patience10. Best model and matching optimizer/RNG/loader states saved on improvement; best model restored after either early stop or epoch limit; last-epoch metrics retained. Test occurs once after restoration. O and B remain separate mandatory controls. |
| Metrics | PASS. `metrics` computes independent binary BCE directly; no multi-output multiclass log_loss normalization. Per-cell AUC/LogLoss/counts/positives, six-cell macro, domain/task macro, per-task pooled AUC, sample-weighted cell AUC and pooled BCE are distinct. Independent asymmetric two-output fixture matches direct Bernoulli calculations. Undefined cell AUC is NaN with counts, not 0.5. |
| Mechanism probes | PASS. Shared subset includes embeddings/shared STAR/skip/shared experts and excludes private/auxiliary modules. Cell norm/cosine/conflict, real/generated cosine, weights/complexity/specificity and actual Adam delta/m/v diagnostics are present. Probe uses existing graph and extra autograd.grad only. Independent all-arm checks on both datasets confirm unchanged model state, CPU/CUDA RNG and existing parameter gradients; no extra forward/noise or optimizer step is performed by probe. |
| Artifacts/provenance | PASS. Per-attempt configs, source SHA paths, raw/encoded/split manifests, row IDs, epoch/last/test/mechanism metrics, logs, completion, best model+optimizer states are implemented. Root supplied sources remain unchanged. Historical grad companion is explicitly paired-from-supplied-model/ported; no VUG/adaptive-mask algorithm is invented. |
| Queue/run protocol | PASS WITH FIX described below. Mandatory scheduler matrix is four arms×four seeds×two datasets; README orders MovieLens2022, remaining MovieLens, frozen KuaiRand. Profiles are full actual one-epoch pipelines and required before formal scheduling. Queue caps five owned processes/GPU, budgets external and outstanding reserved memory below90%, retains failed attempts, halts new launches on failure and rejects duplicate valid completed identities. |
| Orchestrator analysis script | Limited ancillary review PASS after correction below. Seed is aggregation unit, sample SD and paired t intervals use matched seeds, LogLoss signed deltas remain raw, and common-epoch support is recorded. No research interpretation was reviewed or inferred. |

## 3. Bugs/deviations found

1. **MAJOR — blocking, now fixed:** original queue budget `max(used,reserved)+new_estimate` omitted simultaneous external allocation plus not-yet-realized owned-process reservations. This could admit excess concurrency and cause OOM/interrupted experiments. The profile path also lacked a busy-GPU preflight.
2. **MINOR — ancillary continuation logic, now fixed:** initial analyze.py screening combined an absolute mean threshold with three same-sign seeds, rather than requiring practical per-seed deltas of at least .001 in three seeds. This could trigger unwarranted later ablations; it did not affect the first-round training algorithm.
3. **No CRITICAL implementation issue found.** The retained G zero-fill semantics and tau0 numerical differences are explicit design requirements/limitations, not silently corrected defects.

Blocking issues remaining: **none in the reviewed implementation**. The required full-epoch pipeline/memory validation remains a RUN-stage acceptance gate before formal runs.

## 4. Corrections made

The reviewer requested narrowly scoped changes; coder corrected queue accounting and profile preflight. Current projection is:

`current GPU used + sum(max(owned estimate - observed owned PID memory, 0)) + new estimate`.

Missing/inconsistent PID accounting falls back to full reservation. Synthetic used increments were removed. Unknown first-profile footprint is explicitly recorded, runs one process at a time and requires at least2GiB available below the90% threshold; this is an admission preflight, not proof of peak footprint. Runner must measure and inspect the actual profile.

The orchestrator corrected analysis screening to count individual seed deltas ≥.001 or ≤−.001, with at least three qualifying seeds in a direction. The reviewer re-read both fixes and independently executed external-plus-pending, partially/fully realized, multiple pending and inconsistent-sample scheduler cases. No training algorithm was changed during review.

Reviewer created `REVIEW_CHECKS.json` with independent two-dataset verification receipts. Syntax compilation of the shared code and both complete packages passed. Coder `smoke_*.json`, `scheduler_checks.json`, source snapshots/manifest and compatibility.diff were inspected alongside actual code; their claims were not treated as substitutes for direct inspection.

## 5. Remaining risks and limits

- Full-batch one-epoch profiles have not yet completed at review time. Unknown first-profile memory, end-to-end checkpoint I/O and full validation/test evaluation must pass in RUN. Reduce concurrency first if capacity is tight; do not change batch/scientific settings without a reviewed amendment.
- Queue estimates are based on measured allocated peaks plus margin; unrelated future GPU allocations and separate nvidia-smi sampling cannot be guaranteed away. Runner must record actual footprint and preserve failures. No unrelated process may be interrupted.
- O is a controlled original-model port, not exact historical training: train-only encoders, canonical initialization, isolated auxiliary shuffle and always restoring best are disclosed protocol changes. O−B also changes the optimizer path, so it is not fusion-only attribution.
- G−D includes historical zero-vs-None handling and floating accumulation order. A separately specified/reviewed matched tau0 control is required before a pure amplitude causal claim. The first-round matrix itself is valid for comparing the supplied implementations.
- Full historical VUG/masking source and original launch identity remain unestablished despite archived bounded-search receipts. Historical maxima cannot replace current controls or define numerical reproduction success.
- Natural domain sizes confound identity, labels and difficulty. This implementation cannot establish causal size effects; four seeds limit uncertainty claims.

## 6. Whether experiments are safe to run

**YES for the prescribed RUN stage, with the mandatory profile-first gate.** All blocking code issues identified by this review are corrected. Start luna-runner with the full-batch MovieLens one-epoch profiles; only successful, finite, complete profile artifacts and acceptable measured GPU memory permit corresponding formal jobs. Follow the documented ordered32-run matrix, preserve raw receipts, and return any scientific configuration change for design/review. No formal experiment was launched by the reviewer.
