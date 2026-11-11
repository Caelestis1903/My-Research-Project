# Scientific implementation report

Implementation follows `DESIGN_SPECIFICATION.md` and frozen plan 04/12–14. No experimental results were produced by the coder.

## Files and source reuse

- `optimizer_direction.py`: captures unmodified data gradients and pre-step EMBED coordinates, applies one FP64 radius-matched direction after native Adam, and splits A_RAW parameter groups without resetting v/step. Alpha zero directly preserves native values.
- `history_readout.py`: actual FP32 F snapshots at sparse registered origins, reference consumption, FP64 stable two-centroid construction, S/common-N representation and shared-only alternative.
- `conditional_readout.py`: canonical BLAKE2b pair identities, finite PA risk fit including the documented midpoint tie-break, absolute one-epoch-lagged coefficients and real parameter-readout support.
- `evaluation.py`: source raw-logit accessor and source full-population metrics, exact task-column PARAM assembly, common FP32 mixing arithmetic for streamed selection and callable endpoint inference, strict earliest selection including student e0, frozen-Test check.
- `trajectory.py`: source session/model/loss/Adam/F adapter, complete simultaneous registered readouts, B/C diagnostics, E20/E35 RAM anchors, immutable best descriptors, prefix retention, source-compatible restore and conditional continuation.
- `distillation.py`: teacher freeze, original-train-row soft targets cached through guardian, exact same-epoch parent-F initialization, fresh Adam/F, paired data order and source-scale hard/soft BCE.
- `tests/test_science.py`: ten focused correctness tests.

Existing scientific source files are unchanged. Model construction, parameter layout/support, source training loss, fixed EMA recursion, source panels and source metrics are imported from the pinned experiment packages. New parameters, optimizer states, F history, anchors and selected packets remain ordinary RAM objects.

## Integration contract

`create_parent(job, prepared, device)` and `create_student(job, prepared, device, teacher_packet, objects)` return live sessions. `advance_epoch(session)` returns `(record, new_objects)`; `export_session` returns mutable live state plus object references. `reduce_session` runs only after a successful guardian commit.

Immutable predictor descriptors reference tensor bases using `base_ref` and optional exact `parent_F_ref`. Their `refs` expose dependencies to the guardian. E20 B prefixes omit the teacher-parent reference but retain the same tensor base. `materialize_packet` resolves descriptors for evaluation. Current epoch objects that are not reachable from the updated live snapshot are excluded from the commit payload.

`fork_at_epoch` accepts an anchor or live snapshot, filters selected prefix/history to the frozen branch menu and resets the branch job while preserving source numerical state. A E26 continuation preserves its actual short-branch state. Student e0 metrics are returned in `initial_record` before any optimizer update. KD target cache commands are `teacher_get/teacher_put`; target data is keyed by teacher state and checked against train-row hash.

Each epoch record contains full `readouts` metrics/state/config identity, natural training loss/order hash, B reconstruction/pair diagnostics, C applied and pending policy plus PB future differences, direction summary and separate training/observation cost.

## Checks

Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/ema_next_20261111/tests/test_science.py
```

Result: **10 passed**, 12.80 seconds. Source emits its existing PyTorch TF32 API deprecation warning.

Coverage: native/alpha-zero source theta/Adam/F/loader identity over two real-data batches; alpha-one source oracle and second-gradient feedback; ML/KR sparse cache peaks and KR277; S/common-N and barycenter residuals; deterministic unique C pairs, midpoint tie and lag; C stays on S while both B scopes run; actual task-index PARAM output; bitwise identical streamed/callable FP32 mixture scores; KD target-row alignment and teacher=y equality; e0/strict ties/Test guard; independent restore storage and exact next source update.

Review corrections already applied: C initially matched H without scope and could be overwritten by all-scope B; it now matches both H and scope, with a regression test. Streamed and callable mixture arithmetic now share the same function. Optimizer restore clones input state before loading, protecting an immutable CPU anchor.

GPU smoke, complete-epoch resource calibration, scheduler integration and formal five-seed runs remain with the independent review/runner stages. No extra HPO, model dependencies, old attribution reruns or model-weight exports were added.
