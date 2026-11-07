# P2 conventional audit-calibrated transition control

Before examining noise-suite results, strengthen the conventional baseline:
audit_fixed estimates each false-positive/false-negative rate from the SAME
audited noisy/clean pairs with Jeffreys smoothing (flips+.5)/(eligible+1), clips
to [1e-5,.35-1e-5], then freezes all transition parameters. Network training uses
the ordinary independent-cell forward likelihood and audit clean BCE, exactly
as ind_channel except no joint transition/network optimization. This tests whether
the cross-axis method's apparent advantage comes from avoiding poorly identified
joint transition fitting. It gets no extra audited rows or validation information.
Four ML seeds,100epochs, same complete network and main training population.

Review acceptance: estimates restricted to audit AND main rows; class-specific
denominators; no clean labels outside audits; frozen channel omitted from Adam;
raw counts/rates retained. This is a conventional loss-correction control, not a
new method. Primary self-review follows start.md; no independent review claimed.
