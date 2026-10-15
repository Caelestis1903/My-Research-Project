# Evidence audit specification

Scope: A0/A1/A2 only; no research algorithm changes. User explicitly prohibits subagents; DESIGN/IMPLEMENT/REVIEW/RUN/ANALYZE records will be authored sequentially by the primary agent. Self-review does not satisfy the plan's independent-review condition, and will not be represented as independent PASS.

1. Inventory accessible source/artifact bytes with SHA256; inspect archived Git tree without modifying historical files. Record missing per-run artifacts, not just directory presence.
2. Independently recompute saved AO04 predictions using float64 AUC and binary cross entropy at epsilon 1e-12; retain epsilon 1e-7 diagnostics. Match cell masks only using saved physical domains plus matching row IDs/labels or verified source data. Do not infer domains from counts. Validation arrays without row IDs are insufficient for L3 identity.
3. Recompute selected epoch from CSV (earliest maximum) and compare manifest/config/source hashes, initial and main/arch batch hashes across paired arms.
4. Recompute S5 and S6-R2 paired CSV contrasts across all four seeds, both tasks/domains; distinguish exact selected and coarse grid.
5. Bootstrap only verified paired predictions, stratifying by cell/label with a fixed analysis seed; report seed and row variation separately. No reconstructed or imputed missing outcomes.
6. Conditional B1/B2/C1/C2/D1/E1 remain unexecuted until their scientific/provenance requirements actually hold. Historical F0 receipts remain unchanged; this audit does not assert a new forward grant.
7. GPU policy for any future eligible run: independently test each GPU memory <=90%, at most five experiment processes per GPU, persistent launcher, no requirement that both GPUs pass together.

Acceptance: machine-readable inventory, explicit missing items, independent metric recalculation residuals and selection checks, scoped reports; no claim of full experiment completion while prerequisites or remote publication remain missing.
