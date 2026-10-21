# GPU usage policy

Latest user instruction (2026-10-21): **Do not run subsequent training on GPU device1.**

All training for the current 48-run matrix finished before this instruction. Subsequent jobs must use physical GPU0 only: pass `--gpus 0` to either scheduler and `--device cuda:0` to individual entrypoints. If CUDA_VISIBLE_DEVICES remaps ordinals, verify the physical device identity first. Do not start work on physical GPU1. No additional training is part of this report/publication step.

The frozen source snapshots and historical launch logs keep their actual two-GPU commands as evidence, not as the current execution policy.
