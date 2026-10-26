# P1 development-triggered architecture falsification

All28 primary P1 runs completed100epochs before this decision. Development
selected macro AUC: DRO .764532, known-rule mask .761993, consistency .707460,
ERM .579645. DRO-minus-mask +.002540 in4/4 seeds, descriptive95% t interval
[-.001972,.007051]; train-weighted +.000210 in2/4, task1 macro -.000528 in3/4.
Smallest-domain task0 has the largest mean gain but wide uncertainty; largest
domain task0 mean is slightly negative with mixed signs. This is not established
universal cell protection or a size-driven benefit. Final-checkpoint loss favors
DRO more strongly, but all methods overfit relative to selected checkpoints.

Competing explanations: task-compatible worst-case risk can regularize native
task0 beyond OOV masking; alternatively this small difference depends on STAR's
particular shared representation or optimization. The large ERM gap establishes
the controlled problem, not novelty of an ordinary DRO principle.

Extension: MMoE backbone on exactly the SAME controlled populations, main/meta,
selection/development, seeds and100epoch Adam/.001/batch40960 budget. Arms fixed
ERM, task_mask, task_dro;4seeds each=12 complete runs. All MMoE network parameters
train from scratch. Tag backbone_mmoe prevents mixing with primary M3oE results.
No extra shift grid, penalty search or training-seed selection. Test remains
deferred for both architectures until all methods/alternatives are frozen.

Implement an optional architecture override in the common Model constructor;
defaults preserve every existing run. P1 objectives operate on returned two-task
probabilities and need no M3oE-specific internals. Record architecture in configs
and propagate it into evaluation. Aggregation must keep extension tags separate.

Primary self-review acceptance: ordinary MMoE matches the existing MMoE path;
objectives unchanged, no frozen layers; config fusion flags reflect actual model;
unique run IDs and tagged paired comparisons. Verdict after minimal forward/
gradient and default-equivalence checks: PASS for full extension.
