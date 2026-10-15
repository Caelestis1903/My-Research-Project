# Stage E4 CODER IMPLEMENTATION REPORT

## Scope

Implemented Prompt 4 / Stage E4 from the approved `stage_E4_design.md` and
`stage_E4_dataset_audit.md`.  This is an implementation report only.  No
formal E4 matrix was launched and no research conclusion was calculated.

The implementation is isolated from E1/E3 artifacts and does not modify the
core MDMTRec model, trainer, preprocessing code, processed CSV, or split.

## Design-to-file mapping

| Design requirement | Implementation |
|---|---|
| Equal/Sqrt/Sample coefficient law from train shares | `experiments/stage_E4_weighting.py` (`weighting_coefficients`) |
| Processed CSV, global row-order 80/10/10 split and audit counts | `stage_E4_weighting.py` (`load_processed_dataset`, `audit_dataset_bundle`) |
| Exact tab mapping 0/1/4 → d0/d1/d2 and task mapping | `stage_E4_weighting.py` |
| Explicit six masked-mean BCE dot product | `stage_E4_run.py` + `stage_E4_weighting.py` |
| Shared-only observer, raw/weighted contribution, variance/SNR, reconstruction and update alignment | `stage_E4_weighting.py`; invoked by `stage_E4_run.py` |
| Strict main/architecture parameter grouping | `stage_E4_run.py`, `stage_E4_oracle_run.py` using `build_parameter_partition(..., strict=True)` |
| One joint `(weighting,seed)` run | `experiments/stage_E4_run.py` |
| Split-first/filter-second matched oracle | `experiments/stage_E4_oracle_run.py` and `filter_split_first_domain` |
| Exact 12 joint + 12 oracle queue, resume/integrity checks | `experiments/stage_E4_orchestrator.py` |
| Fixed-path structural summary and run report | `experiments/stage_E4_report.py` |
| Synthetic coefficient/gradient/data/oracle tests | `experiments/tests/test_stage_E4_weighting.py`, `test_stage_E4_data_oracle.py` |

## Implemented protocol

- Joint queue: `equal`, `sqrt`, `sample` × seeds `2022, 2023, 2024, 2025` = 12.
- Oracle queue: physical domains `d0,d1,d2` × the same four seeds = 12.
- Batch size 40,960; maximum 50 epochs; patience 10; main Adam `1e-2`,
  architecture Adam `1e-3`, weight decay `1e-5`; no scheduler.
- Natural `DataLoader(shuffle=True, num_workers=0, drop_last=False)` sampling.
- Shared probes at `step % 10 == 0`; no observer optimizer step or second
  forward pass for a probe.
- Validation checkpoint requires all six joint cells (or both oracle tasks) to
  have finite AUC and LogLoss before selection.
- Test evaluation is wrapped by CPU/all-CUDA RNG preservation and records
  before/after evidence.
- All raw runs record ordered split fingerprints, counts/positive rates, batch
  fingerprints, parameter hashes, optimizer partition, metrics, contribution
  tables, non-finite inventory and formal validity.

The following mechanisms are explicitly absent from both runners and are
listed in every manifest: EMA, Momentum, Comp1, Curriculum, adaptive
controller, temporal schedule, balanced sampling and importance correction.

## Verification performed by coder

1. Python syntax compilation passed for all five E4 implementation modules.
2. Focused tests passed: `5 passed` for the two E4 test files.
3. Coefficient table smoke matched the design values within `1e-8`; domain sum
   was 3, six-cell sum was 6, and task-major coefficients repeated identically.
4. Read-only real processed-CSV loader smoke passed the audited 2,507,666 row
   count, all train/validation/test domain counts and all 12 domain/task
   positive counts.  It produced stable ordered split fingerprints.
5. Real-data CUDA two-batch joint smoke passed for the explicit weighted loss;
   shared reconstruction relative error was `2.55e-7` and weighted norm
   identity error was `0.0` in the observed probe.
6. Real-data CUDA two-batch d0 oracle smoke confirmed filtered counts,
   original-to-local mapping and train execution.  No formal oracle/joint
   matrix was started.

## REVIEW #1 blocking corrections

- `stage_E4_orchestrator.py` now treats a missing or incomplete weighting arm
  as `PENDING`; `_finalize_joint` keeps that arm `OK` and continues the queue.
  Only a complete Equal/Sqrt/Sample triad is allowed to become `VALID` or
  `INVALID_BLOCK`, so the weighting-first queue can finish all 12 joint arms
  from an empty raw-results directory.
- `block_status`, `block_valid`, and `block_validity` are propagated into each
  joint manifest and `metrics_summary.csv`.  The structural report combines
  per-arm validation with same-seed block validity; a pending or invalid block
  cannot be labelled `FORMAL_VALID`.
- `metrics_summary.csv` now records selected-checkpoint presence and, after
  triad completion, actual common epoch count, common trace count, and whether
  the selected epoch lies in common support.  Unknown support remains blank
  while the triad is pending; it is never represented by a constant zero.
- Added focused queue lifecycle, invalid-block report, and common-support
  fixtures in `experiments/tests/test_stage_E4_queue_report.py`.

The review correction tests pass together with the original focused tests:
`8 passed`.  No formal run or scientific analysis was performed.

## REVIEW #1 follow-up correction

The nested `_set_block_status.update_summary` helper now uses a local
`support_map` derived from the outer selected-support mapping, avoiding a
read-before-assignment scope collision when a real non-empty
`metrics_summary.csv` is present.  The queue fixture now emits a real summary
row and asserts that the first and second arms receive `PENDING` block
metadata, while the third arm updates all three summaries to `VALID` with
non-empty common support.  The focused suite passes (`8 passed`).

The implementation is frozen for the third and final review; no formal
matrix or analysis was started.

The first loader smoke exposed the expected `domain_indicator` contract issue
(LabelEncoder reserves index 0 for ordinary sparse features but MDMTRec needs
local domain ids 0/1/2); the isolated loader now explicitly performs the same
post-encoding remap as `run_kuaishou.py`.  The focused and real-data smokes
were rerun after this correction.

## Formal execution command

After DESIGN and REVIEW approval, the runner stage may be started with:

```text
python experiments/stage_E4_orchestrator.py --device cuda:0
```

The orchestrator audits all 12 matched oracle references first, runs missing
or invalid oracle references, and launches the 12 joint runs only after oracle
coverage is complete.  `--dry_run` prints the queue without launching it.

## Limitations and handoff

- This report does not certify the formal matrix; `stage_E4_review.md` must be
  produced by the independent reviewer before RUN.
- The orchestrator/report intentionally perform structural validation only;
  cross-seed, transfer, macro-vs-pooled, task/domain stratification and
  falsification analysis belong to the runner output and main analysis stage.
- The legacy historical Comp1 artifacts are not used as E4 oracle data.
- A short smoke has `formal_valid=false` and cannot enter the resume `done_set`.
