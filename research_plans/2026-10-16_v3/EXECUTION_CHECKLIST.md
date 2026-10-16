# 2026-10-16 v3：执行检查清单

## Gate 0 — Freeze

- [ ] 读取 v2 execution ledger、mechanism analysis、question registry
- [ ] 确认 S5–S8 的 Equal/Adam 旧路线已经关闭
- [ ] 确认 v2 P1 的 Adam/SGD 数值事实
- [ ] 冻结 code commit / data fingerprint / split
- [ ] 冻结 selection validation
- [ ] 建立 diagnostic validation（若采用）

## Gate 1 — Objective

- [ ] 验证 SN/SB 权重身份
- [ ] 验证 domain exposure
- [ ] 计算 exact empirical gradient
- [ ] 512+ estimator probes
- [ ] bias / covariance / cosine
- [ ] C/P/R/E 分块
- [ ] objective audit PASS

若 FAIL：停止机制研究。

## Gate 2 — Trajectory

- [ ] Adam SN/SB 4 seeds
- [ ] SGD SN/SB 4 seeds
- [ ] epoch 1..50
- [ ] pre-overfit AULC
- [ ] overfit onset
- [ ] common epoch
- [ ] selected epoch
- [ ] selection bootstrap

结论必须回答：早期、晚期、selection 三者谁贡献了 effect。

## Gate 3 — Optimization

- [ ] SGD extended LR calibration
- [ ] Adam extended LR calibration
- [ ] finalist 4 seeds
- [ ] equal examples
- [ ] equal optimizer steps
- [ ] larger batch / accumulation
- [ ] extended training budget

结论必须回答：是不是仅仅 learning speed。

## Gate 4 — Sampling decomposition

- [ ] N-WR
- [ ] N-PERM
- [ ] B-WR
- [ ] B-PERM
- [ ] exact exposure
- [ ] repetition histogram
- [ ] cumulative coverage
- [ ] same-multiset batch-order replay

结论必须分别回答 exposure / repetition / coverage / order。

## Gate 5 — Geometry

- [ ] exact gradient reference
- [ ] covariance trace
- [ ] top eigenvalues
- [ ] effective rank
- [ ] anisotropy
- [ ] temporal correlation
- [ ] cross-domain/task cosine
- [ ] C/P/R/E geometry

不得把 projection covariance 写成 full-model covariance。

## Gate 6 — Shared/private

- [ ] one-step C intervention
- [ ] one-step R intervention
- [ ] control blocks
- [ ] full-training privateization only if triggered
- [ ] necessity vs association decision

## Gate 7 — Domain factors

- [ ] size duplication
- [ ] difficulty perturbation
- [ ] task/domain cell audit
- [ ] identifiability assessment

禁止“小域=困难域”。

## Gate 8 — Boundary

- [ ] second environment SN/SB
- [ ] primary optimizer
- [ ] 4 seeds
- [ ] replicate only decisive mediator
- [ ] classify repeat / boundary

## Gate 9 — Prior work

- [ ] D3 objective audit
- [ ] SAF objective audit
- [ ] AdaTask objective-preserving comparison
- [ ] domain-order literature
- [ ] optimizer-memory literature
- [ ] importance-sampling literature
- [ ] 2026-10-16 literature cutoff

## Gate 10 — Causal synthesis

- [ ] CF1 q intervention
- [ ] CF2 optimizer intervention
- [ ] CF3 repetition intervention
- [ ] CF4 order intervention
- [ ] CF5 parameter-block intervention
- [ ] CF6 large-batch/deterministic control
- [ ] alternative explanations table

## Gate 11 — Method gate

- [ ] pre-overfit benefit
- [ ] not selection-only
- [ ] not late rescue
- [ ] not only optimization speed
- [ ] mechanism/boundary known
- [ ] prior work not covering
- [ ] 4-seed stable
- [ ] second environment or explicit boundary
- [ ] GO-METHOD or NO-GO

## Gate 12 — Final archive

- [ ] Q1–Q27 statuses
- [ ] Q28+ all included
- [ ] no question-count cap
- [ ] all failed runs retained
- [ ] all contradictory seeds retained
- [ ] final report
- [ ] final decision JSON
- [ ] raw artifact hashes
