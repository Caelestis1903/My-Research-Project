# Runtime Crash 3 implementation report

Status: **IMPLEMENTED; ready for independent review.** This report records tooling only. No real CUDA diagnostic, formal retry, training, or E evaluation was run.

## Deliverable

`runtime_crash3_tool.py` is an external report-directory script. It does not change the frozen experiment package, default config, R1/R2 source pair, or prior run artifacts.

- `diagnose --profile D4|D1` runs in the process that invokes it. Invoke each profile in a separate fresh project-Python process. D4 sets OMP/MKL to4 and records inherited OpenBLAS/NumExpr and effective PyTorch counts; D1 sets all four environment limits and PyTorch intra/inter-op counts to1 before workload.
- Each diagnostic opens only KuaiRand `x.npy` and `rows_validation.npy`; it does not open `y.npy` or labels. It preserves sorted V order, saves the exact row IDs for direct D4/D1 comparison, and performs two complete 512-row sweeps including the final short batch on `cuda:1`.
- The flushed stage log separates V slicing, X advanced indexing, NumPy copy, CPU/CUDA `as_tensor`, CUDA synchronization, and exact CPU/CUDA value validation. The JSON record includes storage descriptors, environment/runtime information, both tiny temporary-array lifetime checks, and direct Python/NumPy/Torch CPU/CUDA RNG-state equality.
- `launch --sidecar PATH -- <existing CLI argv>` sets the process-local D1 limits, writes a per-process sidecar with PID, exact CLI argv, effective environment/thread counts, config, source-pair identity, run/data identity when available, then uses `runpy` on the existing package entrypoint. The original CLI performs its own argument and E authorization checks. The same launcher can carry later formal runs and approved E evaluation; queue sequencing and launch authorization remain with the orchestrator.

## Commands for the runner

Run D4 then D1 sequentially as separate processes. The output directories must be new; D1 compares the exact saved D4 row-ID array:

```bash
PYTHONFAULTHANDLER=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python \
  research_reports/mdmt_baselines_2526/runtime_crash3_tool.py diagnose \
  --profile D4 --output-dir research_reports/mdmt_baselines_2526/runtime_crash3/D4

PYTHONFAULTHANDLER=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python \
  research_reports/mdmt_baselines_2526/runtime_crash3_tool.py diagnose \
  --profile D1 --output-dir research_reports/mdmt_baselines_2526/runtime_crash3/D1 \
  --compare-d4 research_reports/mdmt_baselines_2526/runtime_crash3/D4/v_row_ids.npy
```

The one conditional formal replacement invocation, after independent review authorizes it, uses the original CLI arguments and canonical output:

```bash
PYTHONFAULTHANDLER=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python \
  research_reports/mdmt_baselines_2526/runtime_crash3_tool.py launch \
  --sidecar research_reports/mdmt_baselines_2526/runtime_crash3/HYPERGATE_kuairand_seed2025.json -- \
  --config experiments/mdmt_sota_baselines_2526/default_config.json train \
  --method HYPERGATE --dataset kuairand --seed 2025 --phase formal --device cuda:1 \
  --output artifacts/mdmt_baselines_2526/runs/formal/kuairand/HYPERGATE/seed2025
```

Later D1 invocations use the same `launch` entry with their own sidecar path and the exact already-approved existing CLI argv. The script itself does not schedule the matrix or start any command on its own.

## Verification performed

Using `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`:

- `ast.parse` of the new script: **PASS**.
- `runtime_crash3_tool.py selfcheck`: **PASS_CPU_SYNTHETIC_ONLY**; CPU conversion and temporary-array lifetime values matched exactly, Python/NumPy/Torch CPU RNG states were unchanged, and D1 PyTorch thread counts were1/1.
- In-process launch-context check with the specified HG/KuaiRand/2025 argv: **PASS**; original argv, config, canonical run path and prepared-data manifest were captured. The CLI was not entered.

The D4/D1 CUDA diagnostics, gdb collection, formal replacement, remaining formal entries, and E were not run. No package/config/dependency changes or hash computations were made. Existing 23 complete runs and the archived Crash 3 attempt were not modified. CUDA behavior and real runtime thread counts remain unmeasured until the runner executes the approved diagnostics.
