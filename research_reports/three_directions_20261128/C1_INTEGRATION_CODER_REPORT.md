# CODER IMPLEMENTATION REPORT | C1 shared integration

## Scope and changes

Integrated C1 into the foreground runner after B1 completed 30 of 30 jobs and released the supervision lock. The runner dispatches the initial twenty C2 and C3 jobs through the existing scheduler and RAM journal. It delegates roster and binding decisions to the reviewed C1 protocol and validates the exact frozen per-seed input record supplied by the adapter at E10 and on recovery. The initial roster keeps C4 as `NOT_TRIGGERED`. [Approved design](C1_EXECUTION_DESIGN.md) · [Core review](C1_CORE_FINAL_REVIEW.md) · [Worker](../../experiments/three_directions_20261128/runner.py)

The shared update accepts optional pair selection and reduction hooks after the stock clean BCE forward. It preserves the original A pair expression, forward ordering, single gradient-bearing backward, diagnostic VJP, Adam update and EMA update. A C1 event performs one pair forward over the ordered 4,096 endpoints and records the actual work, source and applied pair identity. With no event the transaction performs the clean BCE update, with zero pair work and null pair identity and weights. C1 warmup traces also record the sign counts of the stock current margins. [Shared transaction](../../experiments/three_directions_20261128/a_step.py)

Initialization and prefix receipts are checked before memory training. The worker finishes the E10 bank installation after the final update and before Validation or a RAM commit, and writes the E10 boundary to the attempt and epoch records. Recovery validates the adapter against the enclosing epoch before restoring training state. A diagnostic attempt follows the same E10 installation path and measures the first memory forward with pair work. Measurement records include full process residency at the transition and memory forward. Diagnostic attempts publish attempt and measurement receipts without committing epochs. The E10 bank metadata carries the compact array size and completion records carry the full serialized E10 RAM packet size. [Worker](../../experiments/three_directions_20261128/runner.py) · [C adapter](../../experiments/three_directions_20261128/c_step.py)

On completion the worker reads each epoch's committed writer, checks its binding and complete step sequence, and replays exposure from E1 through E50. This excludes diagnostic attempts and discarded work after rollback. The replay verifies source, selected condition, weights, pair work, uniqueness, bank hash and draw identity before aggregating. Raw completion is written only after replay and selected-model verification. [Worker](../../experiments/three_directions_20261128/runner.py) · [Replay](../../experiments/three_directions_20261128/memory_bank.py)

## Stage output schema

| C1 output | Rows and identities |
| --- | --- |
| `ENDPOINTS.json` and `SELECTED.json` | Mixed new and linked rows. Use `logical_arm` for C0, C1, C2 and C3. `linked_control` is true on the exact original A1.B0/P0 rows. Original `stage`, `arm`, `job_key`, source, config, data, algorithm, scope and model identities remain on those rows. |
| `SIX_CELL.json` | Selected readout cells carry `logical_arm`, `linked_control`, `stage` and `job_key` in addition to the standard cell metrics. |
| `COST.json` and `VERIFIED_MODEL_READBACKS.json` | Physical and linked records carry `logical_arm` and `linked_control`. Cost rows use the physical original job key with `job`, and readback receipts retain their original RAM identity. |
| `LINKED_CONTROLS.json` | Frozen source, journal, completion, endpoint and RAM receipts keyed by original A1 job key. |
| `EXPOSURE.json` | Verified replay rows keyed by completed physical C1 job, with its journal binding. |
| `E10_BOUNDARIES.json` | Completed physical jobs grouped by dataset and seed, with each arm's E10 prefix, bank and packet size. `prefix_matches` is the equality of the C2/C3 prefix evidence, or null if only one arm has completed. |
| `STAGE_STATUS.json` | `jobs` and `completed` count physical C2/C3 jobs only. |

## Implementation checks

- The CPU suite passed 74 tests across the C1 integration and core, C0 audit, A transaction, B runtime, RAM, and freshness modules. CUDA was hidden for this check and 14 CUDA cases were skipped. All four direct CPU integration cases passed again on the final source. The reduced worker used the real transaction and named RAM journal with synthetic Train/Validation inputs. It tested the exact seed-record callback, E10 commit, E10 and E9 diagnostic recovery without a new logical commit, rollback of corrupted E10 packet bytes, deterministic E10 replay and E11 draw, and exclusion of discarded and diagnostic rows from the published exposure. Continuous and recovered runs matched online, EMA, optimizer, loader and RNG state and Validation endpoints. Direct bank, fallback and no-event tests verified endpoint ordering, actual forward and autograd calls, and exact stock or clean BCE updates where applicable. [Integration tests](../../experiments/three_directions_20261128/tests/test_c1_integration.py)
- All 14 CUDA engineering cases passed. The C1 comparisons exercised seed 2022 on both datasets at the first 40,960-row batch and at the exact natural tail of 21,927 MovieLens rows and 40,052 KuaiRand rows. Archived A1.P0 and C1 warmup matched the committed A1 batch receipts and each other for ordered BCE and pair forwards, pair identity, losses, exact gradients, Adam, fixed EMA, loader, RNG and predictions. The CUDA set also passed the twelve existing A engineering cases. The card used for this set had current memory 20,071,841,792 of 25,769,803,776 bytes and current power 210.86 of 350 W, with host RAM 85.34% used before launch. PyTorch emitted the existing TF32 API deprecation warning. [Real-batch checks](../../experiments/three_directions_20261128/tests/test_c1_integration.py) · [Existing A checks](../../experiments/three_directions_20261128/tests/test_direction_a.py)
- Read-only source and input checks verified all 16 archive blobs, 20 exact historical controls, 40 selected model readbacks, ten P0 fixed-EMA E10 readbacks, and ten per-seed C0 input records for the twenty physical C1 jobs. All ten scoped stop controls were clear. The edited Python files compiled, and `git diff --check` passed. The original A1 and B1 roster checks rejected the shared C1 runtime bytes as expected. [Archive manifest](../../artifacts/three_directions_20261128/A_B_SOURCE_SNAPSHOT.json) · [B1 terminal status](../../artifacts/three_directions_20261128/B1/STAGE_STATUS.json)

## Archived source identity

Local archive commit `70455ed0ebc898ae80275d17f14e7be80f8cfd57` descends from `641b5b8c4242c1ba71f7e4d348a41ec7c630a7fc`. The original A1/B1 roster identities and live RAM references are preserved.

| Archived file | SHA-256 |
| --- | --- |
| A1 `ROSTER.json` | `cb8b0cd85f3b128233b5c60be9941efda68cee21595080b211e1d0498992eecb` |
| B1 `ROSTER.json` | `93c530a5c45ebd08e73944bb2079febec68f9bf37a5ce34a2a51ff46baaeac49` |
| `DATA_AUDIT.json` | `d0db007241b35b44a81f58e8a37bd99556b0fbbd2ff4a84ea8a22711b7b34d41` |
| `SOURCE_AUDIT.json` | `201e489d949b8654ed8daaa3717bded97aa5de7f9fc8b4ed1e65105d0d86d54a` |
| GA31 callback `objectives.py` | `df8845ab9076a1362ed35542dcda6c8632140d583c55e90832d8470b008a7afd` |
| `__init__.py` | `5297137b25f199868209f6d0bd58da79abbf9bf0357a41be34f38e8a97eae13e` |
| `a_step.py` | `09a27d2749acb9f854aa712c9d3d24444cb91829c2a36adf1b3867ae85baf8f5` |
| `freshness_optimizer.py` | `bc5ffe850e08a742215a7f1481103862288a19006db8c6236014007c50f3776e` |
| `macro_pair_sampler.py` | `b82d757b2b4e2a342fbfadcfb0801bad68cfc19a93ec9d024de645e47e776592` |
| `protocol.py` | `b84b2de97e2b3e165f64b6c6994818056c63d845934ecf8aea450e020b41410d` |
| `ram_state.py` | `f468a7b87f4b35f2da6c59d14ffabc15e6705dd7338dd57c6ed04bbdfae2e9c6` |
| `runner.py` | `0d422d92192d2c3e0af579a7800a52dc6cfdd5684fe2db3aec03bd7e9cd8ac33` |
| `scheduler.py` | `d927dcfd62f7f5914396fb721747afeb43a066a4d617660d963e5e734ea613ee` |
| v3 `IMPLEMENTATION_SOURCE_MANIFEST.json` | `7034a3bb59e6d08e3296183258e0e7b16ab74d0248ee0a3b75c526a680462cc0` |
| `A_EXECUTION_DESIGN.md` | `c6aeba9424d1f06c923215731231f64536cc4a8e88718013fd3daafc02eea850` |
| `B_EXECUTION_DESIGN.md` | `49491cb0144b8129e0904cc6f96b2263b89878bec7b8986b69d73056d75d71bf` |

## Review handoff

The independent reviewer can compare `runner.py`, `a_step.py` and `tests/test_c1_integration.py` with the execution design and core final review. The verified live C1 source digest is `660d554cc2d852ab5a7d61d8ded57a7d23b7c9e6c478633affc81f26b542552f` before roster freeze. Live C1 roster freeze and the end-to-end E10 teacher match on the real 50-epoch jobs remain for the approved RUN stage. This implementation check used zero formal C1 jobs and zero Test accesses.
