# S0 independent implementation review

Current verdict after corrected-code re-review: **PASS for scoped S0 only; no blocking issues remain.** The original two findings below are retained as revision history and have been corrected. The matching REVIEW_AUTHORIZATION_S0.json binds the reviewed source/configuration; subsequent covered-file changes invalidate it until re-reviewed.

Scope: the shared S0 native/source-pilot/local-pilot prefix only. This review does not authorize S1 readers, S2 formation, S3 controls, SCALE, or campaign/outer evaluation. The reviewer read the complete DESIGN_SPECIFICATION.md including its API appendix, protocol.py, state.py, prefix.py, verify_prefix.py, the frozen vendor source manifest and relevant native/PilotBank implementation, CODER_IMPLEMENTATION_REPORT_PREFIX.md, and VERIFY_PREFIX_NATIVE.json. No formal experiment was run.

## Blocking findings

1. **Strict identity and replay are not implemented at the prefix boundary.** In prefix.py lines 105–117, the RAM key contains dataset/seed/config/device but omits actual source identity, data identity, and output root. Existing PREFIX_COMPLETE.json is accepted without comparing its saved identity; config.json is then overwritten. A changed source/config can therefore append a new trajectory into an existing run directory or return an old in-memory prefix while presenting a different actual implementation or destination. This can mix initializations, logs, and paired results and invalidate the claimed same-identity replay. Before any mutation, construct and freeze a scoped S0 source manifest/hash, include data/config/source/root in the run/RAM identity, and compare existing start/config/complete identities. Refuse mismatches. Record a true same-identity process-loss replay separately from continuation. The source scope should include protocol.py, state.py, prefix.py and the imported frozen dependency tree plus SOURCE_MANIFEST.json; exclude still-evolving baselines/readers/formation/campaign from this S0 hash. Verification-code hashes may be recorded separately.

2. **The epoch24 complete training state is discarded.** At prefix.py lines 139–140, `terminal_state=cpu_state(model)` retains model tensors only; the terminal optimizer and RNG/position state are not attached to the returned bundle. Only a selected-epoch snapshot remains in diagnostics. This fails the design's complete selected/terminal state contract and prevents a genuine epoch24 next-step restoration if needed. Capture the model, optimizer, RNG, and terminal positions into a RAM-only terminal snapshot before restoring selected weights, retain it in diagnostics, and preserve the existing native_terminal model-only field for API compatibility. Do not add disk checkpoints.

Both are narrow pre-launch state/identity fixes. They do not require altering the algorithm, adding experimental arms, or rerunning any formal trajectory, since none has started.

## Semantics inspected and supported

- Native uses the exact restored Original → backbone excluding `_weight_*` → fixed-model construction and strict copy. The NativeAdapter preserves the native operation order and detached gate inputs; H is four separately normalized 64-dimensional shared expert outputs concatenated to 256.
- Native optimization is the two-task natural Sample BCE sum divided by `2*actual_batch_rows`, batch40960 including the short final batch, coupled Adam .001/.9/.999/1e-8/wd1e-5, 24 complete epochs, validation Macro LL selection restricted to epochs12–24 with earliest ties. AMP is unused and TF32 is disabled.
- Source and two independent local pilot banks receive only their owners' fit labels for 12 natural passes. Inputs H/Z detach; unavailable owner heads are skipped before their forward call and remain grad=None without Adam state.
- Group cuts use only target-fit native logit medians with equality assigned to group0. Matrix normalizers use source-fit hidden activations, preserving raw hidden parameters for later delta copying.
- New training code writes data/log/identity files and keeps model/optimizer snapshots in memory. No new model checkpoint helper or disk weight write was found in the scoped implementation. The frozen vendor still contains historical helpers, but S0 does not call them.
- RNG registry is explicit and follows SeedSequence coordinates with `91000+stage`; it does not use process order or Python hash.

## Independently executed minimum checks

Executed with `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`:

- Recomputed Git blob SHA1 and SHA256 for all **26** restored files against SOURCE_MANIFEST.json: all exact.
- Reran `verify_native_required(device='cpu')` on 64 real MovieLens fit rows: canonical copy true; native probability forward maximum error **0**; h shape `[64,256]`; explicit Sample mean error **0**; two continuous Adam steps and one step + deep-copy RAM restore + next step produced identical model/optimizer state hashes.
- Independently ran two pilot Adam steps on 32 real source-domain0 MovieLens fit rows: all domain1/2 head gradients remained None and their Adam states remained absent; native parameters received no gradient; active owner parameters received nonzero gradients.

The TF32 deprecation warning and verification scalar-conversion warning do not change these checks or the current training graph and are nonblocking. No AUC/performance conclusion, GPU training timing, full data audit, or four-prefix completion is claimed.

## Conditions for scoped S0 release

After the two fixes are inspected, run the actual full raw/prepared/role audit and preserve its receipts before formal training. Start S0 only through a reviewed resident-memory controller/worker that keeps the returned PrefixBundle available for downstream use. GPU concurrency must respect each card's memory and power rule, leave unrelated processes intact, and preserve the physical batch40960. Later stages require their own integrated review and actual numerical/permission acceptance.

The API can materialize outer features and labels for audited export, but S0 does not calculate outer performance or use outer labels for selection. Global test/legacy evaluation remains locked until the campaign freeze. Full downstream raw-X-to-final-prediction consistency is a later export acceptance item; the current prefix only checks native cached features on a bounded sample.

Original review status: fixes requested to coder and parent; no token was issued for that original version.

## Corrected-code re-review and scoped release

Both blocking fixes were inspected in the actual corrected implementation. `prefix.source_manifest()` now binds protocol/state/prefix and the frozen imported v4/vendor Python tree. `build_prefix()` constructs data/config/source/resolved-root identity, compares existing config/complete identities before writes, and includes that identity and device in its resident key. A full terminal model/Adam/RNG/epoch-next-position RAM snapshot is captured before restoring selected weights and is retained as `diagnostics['native_terminal_complete']` while the native_terminal API stays model-only.

The reviewer independently exercised source/config/data/output-root identity mismatches and confirmed rejection with unchanged receipt bytes, and reran the 64-real-row native forward/Sample/complete-memory-resume check: all exact as in the original check. No formal trajectory was executed.

The resident S0 path was inspected in campaign.py and cli.py: prefix workers import build_prefix without importing S1/S2 algorithms, retain the returned bundle in process memory, and return small summaries over a pipe. The local authenticated Unix socket carries commands and summaries; training model state is not written. `Campaign.prefix()` requires the S0 reviewer token and checks registered data/seeds; `require_review('s0')` validates configuration, verdict, absence of blockers, reviewed file SHA256 values, and scope. Per-card dispatch requires both memory and power below85% and does not stop unrelated processes or change physical batch size. The worker remains resident after completing S0. This is a scoped infrastructure inspection, not authorization of its later-stage code paths.

The reviewer actually ran the full CLI `audit` using the project interpreter. Exit code0. Both datasets passed raw SHA256, prepared-array identity, exhaustive/disjoint roles, raw labels/domain mapping, fit-only categorical encoding and age scaler checks. Exact fit and atomic-role counts match the design. Receipts are `artifacts/mdmt_rcf_20261026/v1/DATA_ROLE_MANIFEST.json` and `audit/audit/{dataset}_DATA_ROLE_MANIFEST.json`; the top manifest SHA256 is `1ff7f0fe42d0373c9e1e0b867e3a76ebc36c56d6541a7b09739937f9f712a917`. Prepared identity ML `fde41e1bb0e92eec66073e8e63162296f2a02a6c059d425eb5a59c3b9730237c`; KR `b43efcb1ec4c0947c3696e809889950100fa1b3282c6d5df0a7f153da90282c5`.

**Scoped S0 PASS** permits only ML/KR×5201/5202 common native/pilot prefixes under the registered configuration, with their memory kept for later stages. Launch the resident service after the final reviewed token/source version so loaded modules match reviewed bytes. S1/S2/S3/S4/SCALE and complete downstream predictor/export acceptance remain outside this authorization. The original full-audit precondition is now satisfied. Any later changes to covered infrastructure require a small re-review and token refresh; those code-only changes do not themselves require repeating unchanged data audit.

Infrastructure delta re-review: inspected the subsequent campaign.py/artifacts.py changes, verify_campaign.py and appended coder report. The removed fake-PASS fallback now records NOT_RUN when an actual complete predictor callback is absent; NATIVE/LOCAL/B_REFIT carry their real callbacks. S3 initializes its contribution record before the incomplete-control branch. These changes do not alter S0 prefix training or resident dispatch. Only campaign.py and artifacts.py differed from the previously authorized34-file map; their SHA256 values were refreshed after inspection. The unchanged full-data audit receipt was reverified, and the actual `Campaign.require_review('s0')` accepted the refreshed token. No formal trajectory was executed for this re-review.
