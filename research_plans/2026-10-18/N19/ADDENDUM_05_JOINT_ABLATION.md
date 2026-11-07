# P2 likelihood isolation before noise results

Joint_axis versus nested_ind changes both channel sharing and likelihood.
Add nested_axis: same nested posterior architecture and SAME additive axis
channel as joint_axis, but ordinary marginal composite likelihood. Four full ML
seeds,100epochs. Then joint_axis minus nested_axis isolates joint co-observation
likelihood; nested_axis minus nested_ind isolates axis sharing under the same
posterior support. This is a necessary mechanistic ablation, not a new candidate.

Implementation consists solely of selecting already reviewed nested-prediction
and additive-channel branches together, without joint likelihood. Primary
self-review: PASS; all full network parameters train, same audit/data/optimizer,
no additional information or unsupported labeling of this as an external method.
