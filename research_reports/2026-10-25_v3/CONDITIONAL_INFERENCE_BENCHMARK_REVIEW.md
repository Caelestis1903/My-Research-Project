# Independent static review — separate conditional inference benchmark

**Verdict: PASS WITH FIXES. All blocking issues corrected.**

Static source review only; no new quality records, arrays or research outcomes read, no GPU/benchmark execution, and no frozen experimental implementation modified. The existing base benchmark is unchanged.

## Requirement-by-requirement verification

| Requirement | Actual implementation | Result |
|---|---|---|
| Final admission | Requires complete52+10 base package, all registered P2/formal trajectories complete, final freeze membership, current composite source/design identity and conditional implementation PASS. Selects exactly six P2 CAL/OWNER/GEOM combinations and five formal seed4101 methods. No premature run while formal manifest/freeze is absent. | PASS |
| Selected identity | Only existing primary_best aliases, complete exact-config/source snapshots, legal selected epoch12..endpoint. Saved data_hash now equals digest(saved v3 manifest); full saved fit-ID array hash checked before slicing; x.npy SHA matches frozen feature manifest. Checkpoint SHA, epoch, input hashes and inventories recorded. No latency/quality selection or validation-score reads. | PASS AFTER FIX |
| Actual LPS/CAL forward | A minimal object of the actual ExtensionLpsTrainer class calls production predict_logits -> prediction_packet -> full current-real packet -> policy coefficients -> actual residual sum. CAL receives u,z0,residuals,domain and executes the production FP64 box minimum-L2 projection, including its production receipt-computation overhead. No substituted constant/foreign-only approximation. | PASS |
| OWNER/GEOM and formal deployment | OWNER/GEOM deploy the real block gate/native+owner+foreign-read model; learned allocator or centers influence training and are unnecessary in their deployment forward. Formal RHD/STGU use exact Fixed1Adapter.forward native graph, without training controller. | PASS |
| Feature-only batch | Loads saved role IDs and encoded x.npy only; first4096 sorted fit IDs, float32 shape4096xfeatures. No Data/Trainer constructor, no y.npy open, no label tensor or quality file used. | PASS |
| Training-state removal | CPU checkpoint extraction restores deployed model and policy state/attributes only. Adam moments, saved gradients, phi/allocator, geometry, controllers and RNG never move to GPU; CPU packet released before memory baseline. Deployed modules set eval and requires_gradFalse. | PASS |
| Common timing | GPU0 only, seed_runtime disables TF32; no AMP, torch.no_grad, identical4096 batch,5 warmups and30 synchronized measured reps as base benchmark. P50/P95 linear interpolation and throughput recorded. Isolation excludes other GPU0 compute processes before/after restore and every5 forwards. | PASS |
| Memory and parameter boundaries | Baseline allocated/reserved and total/incremental peaks separated. Inventory counts unique deployed parameters, saved training parameters separately; no assertion that all weights are active on every input. CPU extraction/model startup excluded from latency but included in measured section cost. | PASS |
| Separate cost ledger | New inference_conditional journal uses one exclusive whole-section leaf per method. Cost helper includes those as research-only operations; whole receipt wall is separately nonadditive with component seconds. No nested trainer context journals are created. Base inference journal remains separate. | PASS |

## Bugs, corrections and remaining risks

**MAJOR identity omission, corrected by helper owner:** initial minimal restore reported but did not validate checkpoint data_hash, and did not validate full fit-ID NPZ against manifest. Both checks are now enforced without labels. No research source changed.

No additional blocker found. Static review establishes source-path fidelity; runtime success, isolation and memory/latency numbers still require the final authorized measurement. Ordinary hardware scheduling jitter remains;35 forward timings are not training-seed replicates or energy measurements. Saved state is taken from preselected aliases, so this tool relies on the already reviewed selection lifecycle and does not reselect checkpoints.

## Safe execution

Safe after all final gates pass and GPU0 is idle. This receipt approves mechanical cost measurement only; it does not authorize formal training or quality-based method decisions.

## Reviewed source identities

- `benchmark_conditional_inference.py`: `4d9675314c762ab33210fc3adaff6fd9617afc55821acde5ef2de66a677d693e`
- `aggregate_cost_information.py`: `7a494f71d671c0db095e225b31b5e538c64d8e3e789e616cb8258bd633cdf7ca`
- `benchmark_inference.py`: `f50a14226efbd2d0f679c2c5bcaa7d17af257171682057444797dcf2368cf57b`
