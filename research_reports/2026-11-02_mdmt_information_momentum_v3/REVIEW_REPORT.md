# Independent implementation review

Verdict: **PASS**. No blocking implementation defects were found. The reviewed source hash is `5adc90554898ce9eefe707de2365ef9b13865402d77b360efaee2f255e91db0a` (52 Python/JSON files). Real-data S0 remains the next required stage before formal training.

The review compared DESIGN_SPECIFICATION, CODER_IMPLEMENTATION_REPORT and the actual new package, including the runner and RAM guardian. All 40 vendored files match the pinned EMA02V2 Git blobs; the implementation source hash independently matches its recorded manifest. No algorithm files were changed during review.

## Scientific and execution semantics

- Boundaries use observed joint labels and within-domain positive/negative rows. Target weights retain missing-cell mass, probabilities normalize on active conditions, and dynamic demand is read before its post-step update. Pair streams exclude arm/lambda and do not consume the loader stream.
- Tower hooks retain the training graph and leave the source gate detach intact. BCE uses actual batch/task mean; pair loss is additive. Lambda zero and A0/A1 call the original numerical path. A1 retains the original EMA02V2 context, and initialization/loader seed and order follow the source.
- Shared history membership follows outer module names, including shared experts whose inner layer names contain domain_specific. Other parameters update through ordinary Adam. Each parameter is updated once.
- ECCM applies frequency-corrected selected history refresh, active-condition clocks, source-channel first moment and one coupled decay contribution to the current second moment. CV reads old tables before correction and writes the selected table after Adam. PCOLL has one independently named collapsed table and pair clock. Empty conditions suppress history injection; ordinary None/explicit-zero semantics are preserved.
- Parameters, EMA, optimizer/history/demand, RNG, loader and selection state remain in RAM. Epoch snapshots retain E10, each estimator's earliest strictly best validation state and E50. The source model has no dropout or batch-normalization state that would be changed by the hook check.
- The runner requires a matching independent-review source hash and S0 PASS before dispatch. S1 contains 24 unique trajectories. Later stages require explicit orchestrator dispatch, use only permitted seeds and enforce the registered development/confirmation caps. Scientific promotion remains the orchestrator's responsibility.
- Test evaluation is reached only after final decision and selection freezing, checks each model hash, evaluates all six selected/E10/E50 online/EMA endpoints, and releases RAM after completed scalar output. Source evaluation retains six-cell, macro, pooled, traffic, task/domain and loss metrics.

## Independent checks

`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/conditional_information_momentum_20261102_v3/tests` completed with **14 passed in 2.47 s**. Coverage includes finite-system expectations, observed support, source lambda-zero equivalence, eight-arm RAM serialization/next-step equivalence, hand-calculated coupled decay/history updates, active versus selected clocks, empty conditions, None/zero and CV old-table ordering. The reviewer also recomputed source hashes, checked all 40 vendor blobs and constructed the initial 24-job dispatch.

Real CUDA/full-batch data identity, lambda-zero parity and P11/CV/PCOLL smoke checks are implemented in S0 and must pass before the formal queue. This review did not run GPU training or inspect research outcomes. It did not add unrelated validation or change the experimental matrix.
