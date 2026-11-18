# PRIMARY IMPLEMENTATION REVIEW

Reviewer identity: primary agent self-review under start.md §11; not an independent agent review. Reviewed DESIGN_SPECIFICATION, actual cm19 model/data/training/manifest/queue implementations, extracted author HAMUR reference, primary literature and CODER IMPLEMENTATION REPORT.

Verdict: PASS WITH FIXES for engineering execution; all identified blocking fixes applied before execution.

- Data: N19 natural cache/population/meta/selection split retained; domain/task mapping explicit; no test prediction used in training/selection.
- Optimizers: one main optimizer with disjoint parameter groups for uniform rules; two nonoverlapping optimizers for local hybrid. Original architecture optimizer intentionally overlaps the four learned fusion parameters. Source main-parameter filter uses bitwise ~ on bool and indeed includes them. Fixed fusion remains fixed; no restoration to the project baseline.
- Fixed original auxiliary update batch now cycles a separate training iterator as the source does, rather than N19's repeated first shuffled batch. Exact source data ordering/default hyperparameters not claimed.
- Corrected selected checkpoint diagnostics to avoid overwriting selected optimizer history with final history; snapshots retain full RNG/state. Gradient rows carry the step where gradients actually originated.
- Froze permanently unused intervention parameters and report both nominal and trainable counts; no claim frozen serialization equals physically pruned inference.
- Calibrated G2 bypass multiplier once on label-free train-meta activations; fixed thereafter.
- Matrices are explicit Cartesian products; parent G2 count typo corrected. WD0 and additive-capacity aliases are listed, never double-counted as new networks.
- Probe removal definitions differ appropriately by architecture; no common numerical threshold ranking. Author HAMUR uses batch-stat adapter normalization in evaluation; this is explicitly a public-code-style adaptation and a generalization limitation, not silently equated with paper DN.
- Queue uses independent per-GPU90% gate and at most5 processes, startup allocation gate and measured engineering memory. No automatic algorithm/hyperparameter retries.

Ready for engineering runner: YES. Run six representative KR full-batch/two-step+validation engineering checks. Formal release requires all receipts finite/exit0 and actual group/control checks, then append the formal release below. No scientific interpretation by execution worker.

## Formal release

All six KR full-batch engineering checks completed exit0, two optimizer steps plus validation, finite losses/metrics and receipts. Peak reserved memory1250–2916MiB; batch40960 retained. The engineering cohort is not formal evidence. The no-sharing normalization issue is addressed before formal release by G0MC32 additional independent joint controls; G4 matched sampling now preserves original per-domain exposure as well as update count. Total856 independent networks+72 aliases=928 nominal configurations. Updated source inspected; syntax passes. Remaining fidelity limits above are explicit interpretation limits, not undisclosed algorithm substitution. No formal results reviewed.

Formal verdict: PASS WITH FIXES, all identified blocking fixes corrected.
Ready for runner: YES.

Primary self-review only, not independent review. Full normal budget, all four seeds and every manifest task required. Any subsequent scientific design change must have an explicit addendum before affected runs; tests do not guide selection or tuning.
