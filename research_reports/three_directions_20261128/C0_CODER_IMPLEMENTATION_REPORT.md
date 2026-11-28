# CODER IMPLEMENTATION REPORT | C0 offline audit

C0 is implemented for independent review against the approved design. [Design specification](C0_EXECUTION_DESIGN.md)

- The foreground CPU entry uses the existing A1 roster, job binding, committed epoch records and named RAM journal. It reads the three Train arrays through immutable mappings, checks their Train projection against pinned and audited identities, and verifies E05, E10 and selected fixed EMA packets with their expected epochs, steps and hashes. Selection is independently checked against the recorded earliest exact Validation Macro AUC maximum. [Implementation](../../experiments/three_directions_20261128/c0_audit.py)
- Candidates and the A sampler replay are shared by dataset and seed. Replay verifies the first and last logical steps of eligible windows against the committed writers and checks matching records from earlier attempts. The outputs retain strict signs, weighted conditional rates, observed and uniform-support exposure, E10 confidence bins, E5 formation, Validation joins and seed-level paired summaries.
- C0 writes numeric receipts under `C0/` and checks the shared stop controls before inspection and at job boundaries. Its per-job JSON references the atomically replaced margin archive by SHA-256. The final status links the manifest by SHA-256 and accounts for all 20 trajectories. Missing roster or completion inputs produce `INPUT_PENDING`. Known compact or identity failures produce `TECHNICAL_INPUT_FAILURE` with a job disposition. `INPUT_PENDING` is a normal CLI exit, and the status and complete-accounting flag indicate readiness.
- `COST.json` records combined RAM read, integrity verification and decode latency because `Journal.read_model` exposes one synchronous call. Wall seconds per scored candidate use each common candidate key once per scored trajectory. Memory exposure remains conditional on an assigned bank size.

## Checks

```bash
PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/bin/conda run \
  --no-capture-output -n phase0-3-env python -m pytest -q \
  experiments/three_directions_20261128/tests/test_c0_audit.py
```

All 16 focused checks passed. They cover candidate enumeration and sampling, cell mass, independent weighted arithmetic, FP32 task and domain routing, transition timing, earlier exact Validation ties, same-epoch identity and shared logits, retry trace checks, exact repeat exposure, missing inputs, input corruption, and stop preservation. The two-control integration fixture uses actual Journal readback on synthetic named RAM packets and checks that A1 fixture files retain their original hashes. Both Python files compiled and the CLI help worked from the specified working directory. PyTorch 2.9 emitted one deprecation warning from the pinned v3 TF32 backend setup. [Focused checks](../../experiments/three_directions_20261128/tests/test_c0_audit.py)

## Pending

Independent implementation review and supervised study execution remain pending. The 20 live trajectory dispositions, real Train cache readback, retained A1 model identities and study execution costs require that run. Verification here used synthetic inputs and performed zero Test accesses.
