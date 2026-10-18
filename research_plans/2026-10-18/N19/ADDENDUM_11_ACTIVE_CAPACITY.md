# P3 active-capacity optimizer falsification

Design recorded before implementing/running this extension. Development-only
finite slot removal found exactly zero float32 prediction changes for every
completed KuaiRand uniform/protected checkpoint. A targeted seed2022 inspection
found the full baseline shared-weight norm~1.2e-22, low-rank~1.6e-15, and float32
slot norms underflowing to0. MovieLens retains shared norms~6.3 and measurable
slot sensitivity. Trainable parameters do not imply active effective capacity.

This is a scientific limitation of interpreting the current optimizer regime,
not evidence that rank allocation works or that the implementation is broken.
Coupled Adam L2 at2350 KR updates versus500 ML updates is a plausible explanation;
no causal attribution is made from these checkpoint observations alone.

One declared alternative: decoupled AdamW, SAME lr.003,betas(.9,.999),eps1e-8,
weight_decay1e-5,batch40960,50epochs,population/meta/selection rules/seeds.
KuaiRand only,5methods x4seeds=20full networks: fixed,rank_uniform,rank_protected,
domain_codebook,cell_codebook. Full-budget all-parameter from-scratch training.
Both allocation and dictionary controls are necessary to distinguish an active
parameterization from allocation benefit. Existing Adam runs continue unchanged.
No multiple decay strengths, rates or optimizer grid. Comparisons remain WITHIN
optimizer tag; the full dense AdamW model is the same-optimizer strong baseline.
MMOE/PLE Adam runs remain descriptive context, not same-optimizer proof.

Implement optional --optimizer adam|adamw, default adam. Use automatic distinct
optimizer_adamw tag; actual config names optimizer. Update diagnostics to include
full/dictionary arms, both optimizer tags and double-precision parameter norms.
Use finite slot-zero perturbation on identical train-meta rows, selected/final,
without optimization. Compare effective branch sensitivity and per-cell risks;
optimizer changes also affect every other parameter, so restored sensitivity is
not an isolated causal estimate of slot regularization.

Acceptance: default Adam path unchanged; selected class matches flag; all model
parameters retained; weight-decay/rate/budget identical; config/source snapshots
distinguish arms; no test access, no seed cancellation. Primary self-review, not
independent. No benefit conclusion until all matched runs and test complete.

Implementation/review: optional constructor dispatch and run-ID suffix implement
the specification. AST parsing passes, AdamW class/default/parameter registration
checked without optimizer steps. Default is unchanged Adam. Full diagnostics
record double-precision norms and finite slot removal for both tags. Primary
self-review verdict: PASS. Raw basic check: active_capacity_basic_check.json.
