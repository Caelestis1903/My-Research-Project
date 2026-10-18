# Protocol limits to carry into every conclusion

- The two native dataset populations, preprocessing and historical test sets have
  been reused in prior exploration. Four training seeds randomize optimization,
  not user populations, domains, splits, nuisance generators or noise channels.
  The unit of seed uncertainty is therefore optimization on a fixed population.
- All methods reserve the same5% main-train exclusion for train-meta decisions.
  They have equal available supervision, but methods need not consume identical
  subsets of that supervision. Record which method actually uses it.
- P1 selection/development already contains the controlled reversal(.1 versus
  training .95/.75), while final test has .05. This evaluates training under a
  known available shifted validation regime and slightly stronger held-out shift;
  it does not show discovery of an unknown causal rule or unseen shift family.
  Task1 and Z are synthetic. Global_cf is a negative intervention control, not
  proof that SAF, all decorrelation or all shared representations necessarily fail.
- P2 uses clean validation outcomes for all methods' checkpoint selection. The
  2% audit restriction concerns TRAINING likelihoods; it does not mean the entire
  study possesses only2% clean labels. The true simulator channel is not given
  to methods. Audit calibration uses only audit/main rows. Clean-oracle is an
  explicitly privileged upper reference, not a same-information competitor.
- P3 reports active parameter storage with a physically compact exported state;
  training retains a rank32/domain supernet and can use more memory. STAR's
  Hadamard shared matrix prevents inferring FLOP savings from a factor count.
  The task codebook evaluates two STAR/expert paths, a real compute cost.
- P4 zero means concern effective routing logits. Actual mixture probabilities
  need not retain zero domain/task effects after softmax. Replicated slope signs
  are a heuristic and can be numerically weak; no calibrated uncertainty claim.
- P5 finite candidate functions use the same current tower; this is meaningful
  credit measurement but not the trained performance of a separately optimized
  single-expert network. The utility comparator is a specified gradient rule,
  not a complete reproduction of all contemporary allocation methods.
- P6 measures per-visit label availability and estimator behavior. Physical
  computation remains dense, and repeated labels could be cached in real native
  data. EMA prediction control variates add computation. Shared-representation
  gradient alignment approximates, rather than equals, full-parameter alignment.
- Cell metrics are correlated; do not treat six cells as six independent domain
  samples or four seeds as sixteen independent tests. Paired t intervals with
  four seeds are descriptive and assume a distributional approximation. Multiple
  candidate comparisons are exploratory; no significance hunting or selective
  seed deletion. Domain size and difficulty must be distinguished empirically.
- Wall time under ten concurrent GPU jobs is an operational measurement, not an
  isolated latency benchmark. Peak memory is PyTorch process-reserved memory,
  whereas admission uses device-wide memory. No unmeasured deployment speedup.
