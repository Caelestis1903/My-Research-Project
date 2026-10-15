# F0-TRAIN-ML-v1 implementation report

## Result and support boundary

The reviewed implementation target is the required F0-B1-ML first block:
natural MovieLens, the M3OE/MMoE/CGC-1 backbones, MDMT and three physical-domain
SDMT views, Sample/Q, the search/formal queues, and the required state and
diagnostic artifacts. No MovieLens trajectory, CUDA training, or research
result analysis was run. CPU unit tests include small synthetic optimizer and
resume fixtures only.

| Requirement | Implementation | Verification and limit |
|---|---|---|
| Natural MovieLens prefix, train-only encoders, rating>3/rating>4 labels; global schema with MDMT/SDMT views | `experiments/stage_F0_data.py` | Synthetic CSV split, label, and SDMT schema test. Real file/provenance audit is still a run gate. |
| Frozen Q widths; M3OE, MMoE, terminal CGC-1 | `experiments/stage_F0_protocol.py`, `experiments/stage_F0_models.py` | Model counts checked against all six MDMT/SDMT widths. Production builder test constructs every physical domain × all three backbones. M3OE unmatched `slot_bias.N` tensors are explicitly zero-initialized. |
| Same-seed/backbone/setting MDMT source reused across scopes; local SDMT embeddings audited | `experiments/stage_F0_models.py` | Builder and tensor-hash tests compare source modules, shared tensors, source state hashes, and scope-local physical-domain embedding rows/table hashes. |
| Sample objective and weighted batch-mean aggregation; Equal remains diagnostic | `experiments/stage_F0_objectives.py` | Synthetic full-pool/batch recombination and fixed-share formula tests. |
| First-four paired validation trigger with df=3 | `experiments/stage_F0_protocol.py` | Uses t critical value 3.182446305284263, requires exactly four finite paired deltas for each of all five comparisons, and validates all five vectors before trigger evaluation. Tests include a triggered first comparison followed by malformed/nonfinite input. Trigger input is validation-only. |
| 12-unit search coverage (288 rows), frozen selection, 48-row formal queue | `experiments/stage_F0_queue.py` | Canonical run IDs and queue validation enforce full coverage, identity/hash consistency, no test access, and deterministic numeric tie-breaking. |
| Add-on seed trigger | `experiments/stage_F0_queue.py`, `experiments/stage_F0_protocol.py` | A positive registered first-four trigger mechanically expands the full 12-unit block for seeds 2026–2029 (48 additional rows); otherwise it adds none. Test checks all 48 or zero, never a partial block. |
| Complete checkpoint transaction and recovery | `experiments/stage_F0_run.py` | Resume, selected, fixed, and final snapshots have explicit roles. Recovery derives selected/fixed snapshots from the authoritative epoch commit. Fault injection covers all seven projection/commit points and compares recovered selected/fixed snapshots with uninterrupted execution. |
| Formal selected/final train/validation routing probes and no-op audit | `experiments/stage_F0_run.py` | Formal completion probes selected and final snapshots over train and validation, records deterministic per-cell routing summaries, repeats each probe, and checks model, optimizers, RNG, gradients, `requires_grad`, module modes, and next-batch identity before/after and after restoration. Test covers all three backbones and MDMT physical-domain routing. |
| Selected-test release bound to frozen selected checkpoint | `experiments/stage_F0_run.py` | API accepts a selected checkpoint path and frozen identity only; it checks role, seed role, run/config/epoch identity and file/model/optimizer hashes, then loads that checkpoint internally. Arbitrary model injection is rejected by API shape. Add-on test release also requires full add-on completion. |
| Resource preflight and epoch logger | `experiments/stage_F0_queue.py`, `experiments/stage_F0_run.py` | Preflight queue asserts 3 backbones × 2 batch sizes. Schema includes CPU RAM, GPU inventory/peak allocated and reserved memory, disk free space, checkpoint bytes, epoch time, throughput, and explicit pending measurements before the first real search epoch. Epoch logs record observed throughput/RAM/GPU/storage/checkpoint fields. Synthetic tests assert schema and bytes. |
| Reproducibility hash inventory | `experiments/stage_F0_queue.py` | Canonical queue hashes cover all F0 modules/tests plus reused S1 data/model/protocol/metrics, P7 model, and M3OE source/package dependencies; missing inventory files fail closed. Test checks the reused dependency set. |
| Multiworker recovery and process cap | `experiments/stage_F0_queue.py`, `experiments/stage_F0_run.py`, `experiments/tests/conftest.py` | Canonical-run flock claims prevent duplicate claims; OS-released locks support interruption recovery. When `run()` rewrites canonical-lock ownership, it preserves the same GPU index/UUID mapping written to the GPU lock; a metadata-only test exercises this without a CUDA context or epoch. Shared five-slot process leases are acquired by queue/run workers and each pytest process. Training additionally takes one GPU lock per worker and is capped at min(5, preflight-usable GPU count). The cap was exercised with a synthetic process-slot test; no live worker was launched. |
| Whole matrix accounting | `experiments/stage_F0_protocol.py` | Required ledger totals remain 122 units / 2,928 search / 488 formal / 3,416 trajectories. B1 is implemented but approval-pending; all other required blocks remain explicitly `BLOCKED_NOT_IMPLEMENTED` (KuaiRand also needs provenance). `not_in_matrix` is limited to Ali-CCP, optional full-structure P, gradient/mechanism interventions, and inference-cost microbenchmarks. |

## Frozen width table

| Backbone | MDMT c/b/a | MDMT non-embedding | SDMT c/b/a | SDMT non-embedding |
|---|---:|---:|---:|---:|
| M3OE | 64 / 256 / 512 | 576,482 | 79 / 316 / 632 | 573,474 |
| MMoE | 113 / 452 / 0 | 576,104 | 130 / 520 / 0 | 579,290 |
| CGC-1 | 68 / 272 / 0 | 581,300 | 101 / 404 / 0 | 573,490 |

## Verification

All six F0 test modules passed: 38 tests total. The five independent pytest
groups ran concurrently with no more than five project Python processes:

```text
conda run --no-capture-output -n phase0-3-env python -m pytest -q experiments/tests/test_stage_F0_protocol.py experiments/tests/test_stage_F0_queue.py
  12 passed
conda run --no-capture-output -n phase0-3-env python -m pytest -q experiments/tests/test_stage_F0_data.py
  1 passed
conda run --no-capture-output -n phase0-3-env python -m pytest -q experiments/tests/test_stage_F0_models.py
  5 passed
conda run --no-capture-output -n phase0-3-env python -m pytest -q experiments/tests/test_stage_F0_objectives.py
  2 passed
conda run --no-capture-output -n phase0-3-env python -m pytest -q experiments/tests/test_stage_F0_run.py
  18 passed (one PyTorch TF32 deprecation warning)

conda run --no-capture-output -n phase0-3-env python -m py_compile \
  experiments/stage_F0_data.py experiments/stage_F0_models.py \
  experiments/stage_F0_objectives.py experiments/stage_F0_protocol.py \
  experiments/stage_F0_queue.py experiments/stage_F0_run.py \
  experiments/tests/conftest.py experiments/tests/test_stage_F0_data.py \
  experiments/tests/test_stage_F0_models.py \
  experiments/tests/test_stage_F0_objectives.py \
  experiments/tests/test_stage_F0_protocol.py \
  experiments/tests/test_stage_F0_queue.py experiments/tests/test_stage_F0_run.py
  passed
```

No experiment is approved by this implementation report. Real-data audit,
independent review, frozen data/queue/gate hashes, and successful GPU/storage
preflight remain required before any research trajectory starts. Resource
measurements that require an actual search epoch are present in the schema and
will remain explicitly pending until an authorized run.

The final implementation inventory contains 28 files and its aggregate
SHA-256 is `b656a1bfc931259c4a0ec81f150fce518f973aa178363536f676bed1672e3f19`.
