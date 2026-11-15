# Resource, Scheduler and Recovery Plan

## 1. Cost envelope

Stage 1 planned new work: 20 trajectories × 50 epochs = 1000 epochs, plus smoke. At the 2026-11-15 report's observed worker costs, the zero-delta controls should be substantially cheaper than the perturbed arms because no gradient-based delta is applied after support validity is computed; nevertheless, profile actual cost rather than promising a fixed wall time.

Stage 2 NDP-A may be materially more expensive than Stage 1 because it computes task sensitivities for the full natural batch and forwards task-specific augmented views. Do not dispatch Stage 2 until a full-batch smoke has measured time and memory. If infeasible, use NDP-B with a predeclared inclusion probability and unbiased correction.

Do not reduce batch size, epoch count, seed count or dataset population to fit resources.

## 2. GPU and host admission

- Measure full-batch peak allocated and reserved memory on each dataset and arm.
- Admit a worker only when that GPU's memory and utilization/power policy is below the project threshold (use the inherited 85% GPU and 90% host RAM guard unless current main changes it).
- Keep at most one allocating worker per GPU until first-batch peak is known.
- Release probe graphs, hooks and temporary tensors before training backward.
- Never interrupt unrelated jobs.
- No silent batch-size fallback. If the fixed full batch cannot run, record ENGINEERING_BLOCKED and report the actual resource blocker.

## 3. Runtime and logging

Log per worker:
- wall time, epoch time, examples/sec;
- peak allocated/reserved GPU memory;
- CPU RAM and data-loader time;
- sensitivity/perturbation time separately from clean and augmented forward/backward;
- number of candidate/accepted rows, zero-delta rows and skipped pairs;
- optimizer and EMA update counters.

The 2026-11-15 pilot observed 5674.29 seconds wall time for 60 trajectories and 20.79 aggregate worker-hours under concurrent load. These are reference observations, not guaranteed future runtime.

## 4. Resume and retry

- Snapshot model, optimizer, EMA, loader generator, Python/NumPy/Torch/CUDA RNG states, epoch, global step, selected model states and current best validation scores in the RAM guardian.
- On restart, verify run identity and restore every state before continuing.
- Keyed sampler streams must be independent of global RNG restoration and reproducible from the batch context.
- A failed attempt is immutable; retry as attemptN+1 with the failure reason and source/config identity recorded.
- Do not mix partial trajectories across seeds or configurations.
- Do not treat a resumed run with a changed code hash as the same trajectory.
- Do not write full model or optimizer checkpoints into historical artifact folders.

## 5. Failure labels

- ENGINEERING_BLOCKED: numerical correctness, memory, runtime or source integration prevents execution.
- DATA_IDENTITY_MISMATCH: raw/processed data or row-order hash differs.
- PROTOCOL_MISMATCH: optimizer, EMA, batch, split, checkpoint or metric protocol differs.
- INCOMPLETE: fewer than 50 epochs or missing validation endpoint.
- NOT_TRIGGERED: a conditional stage was not authorized by its preceding result.
- SCIENTIFIC_NEGATIVE: only after the complete five-seed matrix and predeclared validation decision rule.
