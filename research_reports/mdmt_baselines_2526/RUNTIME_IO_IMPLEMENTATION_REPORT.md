# Runtime I/O R2 implementation report

Status: **IMPLEMENTED; independent review pending.** This implementation does not establish that memmap access caused either native crash. No real training or E evaluation was run.

## Implemented scope

- `experiments/mdmt_sota_baselines_2526/data.py`: `Data.__init__` loads only `x.npy` as a normal resident ndarray and makes it read-only. `y.npy` remains a read-only memmap; role-specific label access and the constructor's no-E-label-read behavior remain in place.
- `experiments/mdmt_sota_baselines_2526/report.py`: default `build_freeze(run_root, output_path)` retains strict single-source behavior. Its explicit `runtime_io_amendment` argument accepts only the fixed R1/R2 pair after exact inventory and full-byte checks, confirmed current-package R2 bytes, independent review receipt, CPU feature evidence, and the required successful replay receipt. It classifies every complete matrix entry as R1 or R2 and stores both full snapshots, the amendment evidence, and a per-run `source_revision` beside the unchanged run/checkpoint identity. Config, method-config, source-commit, data, run, and selected/fixed-final checkpoint checks remain active.
- `experiments/mdmt_sota_baselines_2526/evaluation.py`: E clears prior authorization, binds the exact matrix run to its frozen R1/R2 snapshot, verifies the currently installed implementation against the approved R2 snapshot, then invokes the existing exact run/checkpoint/data authorization before label access. Evaluation arithmetic and batching are unchanged.
- `research_reports/mdmt_baselines_2526/reproduction_guide.md`: documents the explicit Python API call for a completed mixed-revision freeze. The default CLI remains single-source and rejects mixed snapshots.

## Source-pair receipt

`RUNTIME_IO_SOURCE_EQUIVALENCE.json` contains the complete R1 snapshot taken from `artifacts/mdmt_baselines_2526/PRE_FORMAL_FREEZE.json`, the current R2 direct-source snapshot, exact changed-path list, each changed file's full R1/R2 bytes and unified diff, configuration, source-commit and prepared-manifest provenance. The R1/R2 inventory has 18 files and is identical. Actual changed executable files are exactly `data.py`, `evaluation.py`, and `report.py`; all other source/config/reference files compare byte-for-byte equal. The record is explicitly **REVIEW_REPLAY_PENDING**. Its review and real replay receipts remain pending; it cannot authorize a final freeze or E.

The amendment freezer requires status `REVIEW_REPLAY_COMPLETE`, a review receipt naming this amendment and PASS, passing CPU receipts including both real prepared-data X comparisons, and a successful full HyperGate/KuaiRand/seed-2022 R2 run with exact first-five-epoch trajectory fields. The acceptance fixture uses temporary synthetic receipts only to exercise mixed-source binding; it does not alter the pending project record or represent review/replay evidence.

## CPU evidence

`RUNTIME_IO_CPU_EVIDENCE.json` records the synthetic checks and links the orchestrator's actual X-only comparison in `RUNTIME_IO_FEATURE_EQUIVALENCE.json`. That direct comparison reports full feature value and ordered-batch equality for MovieLens (1,000,209 × 8, float32) and KuaiRand (2,507,666 × 34, float32), unchanged Python/NumPy/Torch CPU RNG, and no y loading.

Using `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`:

- `python -m pytest experiments/mdmt_sota_baselines_2526/tests/test_pipeline.py experiments/mdmt_sota_baselines_2526/tests/test_runtime_io.py -q`: **16 passed**.
- Synthetic checks cover ordinary read-only x, lazy memmapped y and E/S/legacy permission, constructor RNG preservation, supported M3OE logits/loss/gradients/one Adam update/model and optimizer state/RNG exact equality, default mixed-source rejection, explicit mixed 30-entry freeze acceptance, unchanged selected/fixed-final checkpoint identities, pending/failed replay rejection, same-size third-source mutation rejection, unapproved training-source mutation rejection, unknown revision rejection before E labels, and current-package drift rejection before E labels.

No real train, replay, checkpoint evaluation, or E-label read occurred. No dependency was added and no checksum was calculated. The failed HyperGate/KuaiRand/2022 attempt and the 20 completed R1 runs were not edited or relabeled.

## Review and stop conditions

Independent review of the exact R1/R2 byte pair and narrow diff is pending. Do not launch the bounded real retry until review PASS. After PASS, run only the approved full-budget HyperGate/KuaiRand/2022 retry from initialization, preserve its full logs/checkpoints, compare its first five epochs exactly as specified, and stop if any scientific trajectory field differs or a native crash recurs. The source pair must remain unchanged after review.

## IOFIX1 — follow-up to independent runtime review

The initial independent runtime review remains **FAIL** in `RUNTIME_IO_REVIEW_REPORT.md` and its structured receipt. The root archived that candidate pair and failed receipt under `artifacts/mdmt_baselines_2526/runtime_io_review_history/candidate01/`. I did not edit that history or the 20 completed R1 runs/checkpoints/snapshots.

- **IO1:** the amended freeze now reads the independent reviewer receipt as JSON. It requires amendment ID and verdict/status PASS, then compares the complete reviewed R1/R2 snapshot objects, changed paths, original configuration, source commits, and prepared-data manifests with the candidate. It also compares the receipt's full CPU and feature evidence dictionaries with the currently referenced evidence files. Every formal run's configuration, source commits, and dataset manifest must equal those reviewed originals. A candidate R2 and current package changed together cannot reuse a receipt for the previous byte pair.
- **IO2:** the amended freeze no longer accepts summary flags as replay evidence. It opens the fixed preserved R1 failed run, its archived source snapshot, epoch log, and native-crash receipt; it then opens the actual retry run, R2 source snapshot, epoch log, and selected/fixed-final checkpoint paths. The retry must be the completed formal HyperGate/KuaiRand/2022 run with exact original config/source/data identity. It parses the first five ordered `train` records from both JSONL files and compares every field except `seconds`, including train loss, update/exposure counts, components, temperature, all six validation cells, and validation macro metrics.
- The synthetic replay acceptance test reads the real preserved failed-attempt metadata and first five epoch records, then compares them against temporary JSON/log/checkpoint fixtures. Negative fixtures reject nonexistent retry paths despite `status=PASS`, fewer than five records, a changed scientific field, and an R1 retry snapshot. This is validator evidence only; it is not the real R2 replay.

The pending `RUNTIME_IO_SOURCE_EQUIVALENCE.json` now contains the updated full R2 bytes and diff. Its review and replay fields remain PENDING. R1/R2 still have 18 identical source paths, with exactly `data.py`, `evaluation.py`, and `report.py` changed; all 15 other files remain byte-identical. The failed review record was not overwritten.

Using the configured project Python, `python -m pytest experiments/mdmt_sota_baselines_2526/tests/test_pipeline.py experiments/mdmt_sota_baselines_2526/tests/test_runtime_io.py -q` completed with **16 passed in 1.90s**. `RUNTIME_IO_CPU_EVIDENCE.json` records the structured-receipt, original-identity, actual-record replay-validation fixture results and links the previously completed actual-dataset X-only equivalence receipt.

No real retry, real training, E evaluation, or E-label access occurred. Do not launch the retry before a new independent review PASS. After the retry actually completes and its first five scientific fields compare exactly, the orchestrator may complete the replay evidence and later use the amended final freeze; until then, final freeze and E remain blocked.
