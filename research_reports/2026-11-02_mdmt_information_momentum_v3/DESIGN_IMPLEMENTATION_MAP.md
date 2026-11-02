# Design implementation map

| Design responsibility | Implementation |
|---|---|
| Pinned source and original bytes | `SOURCE_MANIFEST.json`, `vendor/ema02/` (source commit 38a53bc1) |
| Explicit data manifests, source fixed1 model and metrics | `protocol.py`, vendor data/evaluate/model; no implicit old-plan config |
| Same-domain observed joint labels, w/q, independent condition/pair RNG | `boundary_sampler.py` |
| True graph-connected raw tower logits, unchanged gate detach | `boundary_sampler.forward_logits`; source model remains byte-identical |
| Source A0/A1 and lambda0 bypass | `objectives.train_step`, source `objectives.train_step` |
| P00/P10/P01/P11/PCV/PCOLL and fixed EMA | `objectives.py`, `optimizers.PairOptimizer` |
| Shared outer-module parameter grouping; None versus zero | `optimizers.shared_name`, standard Adam dispatch, parameter manifest |
| ECCM active clock and selected counts, CV old-table update, PCOLL pair clock | `PairOptimizer.step` |
| All optimizer/history/D/clock state, config identity, RAM restore | `PairOptimizer.state_dict/load_state_dict`; `guardian.py` |
| Loader/source A1 RNG identity | `run.worker`, unchanged EMA02V2 context, loader seed+11000 |
| Epoch loss/diagnostics, complete validation and earliest selected/E10/E50 | `run.worker`, per-epoch JSON and batch JSONL |
| Explicit reviewed S1/S2/S2_EXTRA/S3 dispatch, finite seed/lambda/arm counts | `run.schedule`; scientific decisions remain orchestrator-owned dispatch files |
| Each-GPU 85% memory/power admission with measured peak | `run.choose_device/schedule`, `stage0.py` |
| Final selection hashes before any test call; six endpoints per trajectory | `run.freeze/final_evaluate` |
| CPU formulas/parity/resume; real full-batch S0 | `tests/test_semantics.py`, `stage0.py` |

Source adaptation notes: `guardian.py` adapts the source module entry/socket prefix. The runner retains source epoch RAM commit and loader semantics, adds E10, and replaces the old study's search/cost-stop rules with this study's explicit finite dispatch. No model weights are written to disk.
