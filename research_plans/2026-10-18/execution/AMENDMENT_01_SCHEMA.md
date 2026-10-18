# Pre-run amendment 01 — raw ML rating timestamps

Raw-schema inspection (before any substantive run) found that ML raw CSV includes
`timestamp`, although the cached model features omit it. KR still has only
`upload_dt`, not exposure time. Do not report that both raw datasets lack time.
Join ML **training** rows by preserved raw row IDs, verify rating-derived labels,
and record within-domain rating-time quartile label frequencies. These are
legitimate descriptive time environments, but no known nuisance/shift mechanism
or future policy is supplied. This addition refines the P1 data gate and does not
authorize treating any correlated feature as spurious or start stable training.

Also clarify P3 storage: 5K=325 is coefficient dimension, not total artifact size.
Report basis and feature-normalization storage, shared trunk and actual generic
compression factor storage. No method may claim a deployment saving from degrees
of freedom alone. Analysis for P4 uses the same fixed source actions (mean A/B)
and predicts target C responses from full versus additive target estimates.
