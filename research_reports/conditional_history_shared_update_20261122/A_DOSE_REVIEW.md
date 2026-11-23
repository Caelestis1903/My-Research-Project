# REVIEW REPORT — A_DOSE

Date: 2026-11-23. Independent review of the actual repository implementation; no scientific training or Test evaluation performed. No implementation files changed.

## 1. Verdict: PASS

The registered A40_HISTORY_L025/A41_ADAM_L025 matrix is safe to run under the current-conversation foreground runner, subject to the existing runtime stop, source, live RAM-generation and resource checks. Blocking issues: none. This receipt approves 20 registered trajectories and their Validation tables; it does not authorize A_SCOPE, additional configurations, or Test.

Reviewed source hash: `b5ee444c6e723a195aaae385f88b0c0beb9d5b893d5931ecf88894437614fb37`.

Engineering check SHA256: `f8014a7d8f9a6d1b88635c2ee9d26cb8416583dcd1c16cf21e4c31442ba5a450`.

## 2. Requirement-by-requirement verification

| Requirement | Independent verification | Result |
|---|---|---|
| Original scientific specification | Read original 03_DIRECTION_A_DESIGN, 06_IMPLEMENTATION_SPEC and 09_DECISION_RULES, this round's DESIGN_SPECIFICATION and CODER_IMPLEMENTATION_REPORT_DOSE. Compared all three original plan files byte-for-byte by SHA256 with commit 4a3e4f1b21f1aebb6daa2df6635c3d7ca547a983. | PASS |
| Authorized A_DOSE gate | dose_stage.gate checks COMPLETE A_HISTORY, five prescribed seeds, no missing primary rows, Validation/fixed_ema/selected identity, previous gate, statistics-script and evidence-file hashes. Actual recorded target/form/scope are movielens/A21/S_DENSE. Recorded A21 target increments are positive versus A00 and A10; the frozen A21 has current KuaiRand cell cost, which supplies the registered dose motivation. No new Test values enter this gate. | PASS |
| Matched finite matrix | Recomputed the manifest and all job keys: exactly 20 unique jobs, two datasets, seeds 2022–2026 and both arms. A40 inherits A21 and A41 inherits P00; their only changed parent configuration field is lambda_pair .05 to .025. Both are static, 50 epochs, batch 40960. The accepted .05 references retain their own source/job identities. | PASS |
| Coefficient and λ=0 | dose_runner.dose_config validates the frozen .05 parent before changing the coefficient in the new adapter. The old v3 registry is unchanged. Both new arms at λ=0 return source A0 with ordinary .9 Adam throughout. Independently asserted both factories and optimizer groups. | PASS |
| Objective, gradient graph, shapes and sampling | Both workers call the frozen v3 objectives.train_step. BCE remains a mean over B×2; PAIR uses the real forward's tower logits and softplus margins. Raw r is requested only for S_dense history, while full mixed backward reaches all originally reachable parameters. Existing gate embedding.detach is preserved. No new normalization, routing or population change. Sampler keys exclude arm and dose; q, w/q and 2048 replacement pairs retain source semantics. | PASS |
| A40 formula and one update per parameter | Actual HistoryOptimizer delegates live collapsed updates to source PairOptimizer: Q=.99Q+.01(w/q)r, pair clock advances only on a present pair; mR uses .9 on real data+WD; v uses .999 on full mixed data+WD; private bias corrections and λQ injection match the design. Ordinary parameters receive native .9 Adam once; full-model EMA advances once afterward. Independently checked explicit .025 equations on five synthetic steps including absent-pair steps. | PASS |
| A41 Adam, WD and None semantics | Actual runner builds source ordinary PairOptimizer with full mixed backward. Independently compared native .9 Adam updates, ordinary non-history parameters, explicit-zero steps and unused None parameters; all matched. No extra WD or private-step advancement. | PASS |
| Same-trajectory diagnostic states | A40 directly uses history_runner.worker, including E5/E20/E35/E50 last-batch pre-step anchors. step_anchor captures model/buffers/modes, complete optimizer and histories/shadows, EMA, RNG, loader permutation/cursor, physical row IDs, pair/raw/mixed gradients, selection metadata and actual post-step hashes. The callback runs before any shadow/live step. M9 contains the dose-scaled ordinary auxiliary first moment; six Q shadows contain unscaled collapsed residuals. Independently checked M9+mR reconstruction and Q reconstruction. Anchor key/schema are accepted by A_TARGET's A40 parent lookup. No code from the unreviewed downstream execution path is licensed by this receipt. | PASS |
| State integrity and RAM_ONLY | Capture/restore includes source/job/data/group identity, RNG, loader, selection and optimizer states. Training tensors travel only through guardian IPC; commit performs actual get/hash verification. Inspected dose_checks: A40 saved tail anchors are retrieved and stepped on clones, comparing model, EMA and complete optimizer identity, then fixture keys are released. Independently confirmed current generation equals audit/check generation. | PASS |
| Source regression fixtures | Inspected actual dose_checks and final A_DOSE_CHECK.json. Both arms/datasets compare source objective/optimizer with the adapted coefficient on natural first/tail batches, including model, EMA, native Adam, history, live moments, selected clocks, demand, loss and pair hashes. Recomputed the check-file SHA256; cross-checked both arms' physical-row/pair hashes and their pairs against accepted .05 A21 fixtures. A40 tail model/EMA hashes match restored-anchor actual-post receipts. These recorded GPU checks were inspected, not rerun. | PASS |
| Budget, selection and endpoint identity | Both worker loops run E1–50 without early stopping and retain natural tails; accepted result checks require 1000/2450 steps. Online and fixed_ema select independently with strict >, retaining the earliest tied maximum and that point's complete metrics/LL. Independently checked a tied synthetic selection fixture. E10/E50 are separate readouts. A41's dedicated-process endpoint adapter correctly preserves NEW_TRAIN provenance while its physical cost class remains NEW_SCIENTIFIC_TRAIN. | PASS |
| Tables and downstream gate | analyze requires current review authority and COMPLETE run, verifies job/data/step/result identities, combines only included A00/A10/A21/A40/A41 accepted Validation rows, and keeps source identities distinct. Preliminary tables use A_DOSE_PRELIM; official A_DOSE requires COMPLETE A_TARGET_RUN and the exact localization file hash. results.py retains five-seed pairing, t4 intervals, protected-cell preference and matched A40−A41 comparison. Three existing results tests passed. | PASS |
| Supervision, resources and stop controls | run and each mechanical child call dose_stage.reviewed; the child verifies its direct foreground supervisor PID. Both workspace stop controls are checked through history_stage.stopping. dose_scheduler uses current per-card memory<85%, current power<85%, host RAM<90%, plus measured full-process physical fit within CUDA-usable capacity. It resamples after each startup, permits allocation-pending protection and has no fixed worker cap. Nonzero or unexpected incomplete worker exits pause the stage. | PASS |
| Baselines and Test seal | dose_source recomputes the reviewed closure and revalidates all frozen v3 source-manifest entries. No old source or .05 configuration was modified. Neither dose worker nor analyze reads Test; their evaluation paths pass only Validation arrays. New configurations or Test remain subject to the original distinct gates. | PASS |

## 3. Bugs/deviations found

CRITICAL: none. MAJOR: none. MINOR: none found in the reviewed A_DOSE scope.

The source configuration arm remaining A21/P00 while the registered job arm is A40/A41 is intentional: objective dispatch uses the frozen numerical kernel, and external trajectory/endpoints retain the registered treatment identity. A41's temporary endpoint binding is confined to its dedicated mechanical process and restored in finally.

## 4. Corrections made

None. Only this report and the independent A_DOSE_REVIEW_RECEIPT.json were written.

## 5. Remaining risks and validation limits

- The GPU engineering fixtures cover two natural batches per arm/dataset and retained-anchor replay; I inspected their code and hash-bound final receipt instead of rerunning GPU fixtures or 50-epoch trajectories. Independent reviewer execution covered CPU synthetic arithmetic/clock/None/zero/selection checks and the three results-table tests.
- RAM retention and measured GPU/host headroom can change. Current generation matched `f2a6427b61b812f225f910e6991ede53507fd72383c1a03eaac50db9ea01d353`; runtime must continue enforcing generation, stop controls and current telemetry. Check fit profiles are ML 4515168256 bytes and KR 5540675584 bytes; CUDA-usable total is 25307578368 bytes on each device. No state-loss or resource failure may be reported as a scientific negative.
- Official dose decisions remain conditional on complete, independently reviewed A_TARGET localization execution; this receipt does not certify that downstream implementation or its future measurements.

## 6. Whether experiments are safe to run

Yes: the 20 source-bound A_DOSE trajectories may run through the foreground supervised dose_stage entry point after this receipt validates. No scientific run was started during review. Test remains sealed.
