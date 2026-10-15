# Stage S7-B: Same-θ Early/Late Causal Continuation Run Report

Per Algorithm Specification -- Stage S7-B.  T-branch semantics = S6 DH-arm current manipulation (moments<-g_F, shared step overwritten by pre-step Δ_T).  F-branch semantics = S6 F arm (bit-for-bit on step 0).

**Geometry口径差异声明**：S7-B geometry 是**单 actual-batch**（每个 step = 一次 train batch 的真实更新），与 S7-A 的 4-batch 平均**不可直接跨阶段比较绝对值**。本报告所有 U_k / H 读出、cell 几何仅在 S7-B 内部比较。

**Validity contract (V0 / V1 / V2 / V3)**：
- **V0 (硬门, step 0 only)**: fingerprint 逐位相等（已）+ |ΔL_full| ≤ 1e-5 + shared `delta_observed_step_norm` 相对差 ≤ 1e-3；任一失败即 abort。step 0 认证重建正确性。
- **V1 (硬门, 每步, 双 branch)**: `controlled_continuation_step` 内既有断言 （shared `assert_vector_match(observed, Δ)`、installed cosine、private 残差）— V1 失败时 step 不会产生 CSV 行。
- **V2 (仅上报, step ≥ 1)**: F-branch 对 S6 原轨迹的 L_full diff 与 shared delta 相对差逐步记录（沿用 ``validity_L_full_diff`` / ``validity_delta_norm_rel_diff`` 列）。**不设 abort 阈值**。任一步 |L_full diff| > 0.3 在该 run 的 report 行打 `REVIEW` 旗标。
- **V3 (硬门, 仅 step 0, pre-step)**: 双 branch 间 private 参数逐 tensor `torch.equal`；失败意味着 branch 对称性被破坏 → run abort。step ≥ 1 的 private 分歧是 T-branch 在 step 0 覆写 shared（Δ_T）后双 branch 的 pre-step shared θ 分歧、进而 g_P 分歧的结构后果，**非 V3 违反**，仅以 `v3_per_step` 记录。step-0 跨 branch cos(Δ_F, Δ_T) 一致性断言保留。

**Known approximation**：private moments not in any S6/S6ML product; per Algorithm Spec §4, private parameters are installed with no state (fresh Adam, step=0).  At step 0 the two branches' private parameters are bit-identical (same anchor deepcopy); at step ≥ 1 the private trajectories diverge together with the shared θ divergence — a structural consequence of the T-branch semantics, so the F vs T causal contrast on shared parameters is unaffected.  Per V2 above, step ≥ 1 的 F-branch 对 S6 原轨迹分歧是已声明的 private fresh-Adam 近似的预期后果，**不构成 run 无效**。

## Run matrix coverage

| dataset/seed | anchors |
|---|---|
| kuairand/seed2022 | anchor39=OK, anchor9=OK |
| kuairand/seed2023 | anchor39=OK, anchor9=OK |
| kuairand/seed2024 | anchor39=OK, anchor9=OK |
| kuairand/seed2025 | anchor39=OK, anchor9=OK |
| movielens/seed2022 | anchor39=OK, anchor9=OK |
| movielens/seed2023 | anchor39=OK, anchor9=OK |
| movielens/seed2024 | anchor39=OK, anchor9=OK |
| movielens/seed2025 | anchor39=OK, anchor9=OK |

## Fingerprint replay audit (epoch anchor+1 step 0..4)

| dataset | seed | anchor | all_match | row_count |
|---|---|---|---|---|
| kuairand | 2022 | 9 | True | 5 |
| kuairand | 2022 | 39 | True | 5 |
| kuairand | 2023 | 9 | True | 5 |
| kuairand | 2023 | 39 | True | 5 |
| kuairand | 2024 | 9 | True | 5 |
| kuairand | 2024 | 39 | True | 5 |
| kuairand | 2025 | 9 | True | 5 |
| kuairand | 2025 | 39 | True | 5 |
| movielens | 2022 | 9 | True | 5 |
| movielens | 2022 | 39 | True | 5 |
| movielens | 2023 | 9 | True | 5 |
| movielens | 2023 | 39 | True | 5 |
| movielens | 2024 | 9 | True | 5 |
| movielens | 2024 | 39 | True | 5 |
| movielens | 2025 | 9 | True | 5 |
| movielens | 2025 | 39 | True | 5 |

## Validity (F branch vs S6 F source) -- V0 step 0 / V2 step >= 1

| dataset | seed | anchor | step | layer | l_full_abs_diff | delta_norm_rel_diff | v0_abort | v2_review |
|---|---|---|---|---|---|---|---|---|
| kuairand | 2022 | 9 | 0 | V0 | 0.000e+00 | 5.614e-11 | False | False |
| kuairand | 2022 | 9 | 1 | V2 | 6.165e-02 | 1.524e+00 | False | False |
| kuairand | 2022 | 9 | 2 | V2 | 1.302e-01 | 2.777e+00 | False | False |
| kuairand | 2022 | 9 | 3 | V2 | 1.515e-02 | 2.221e+00 | False | False |
| kuairand | 2022 | 9 | 4 | V2 | 3.973e-02 | 1.679e+00 | False | False |
| kuairand | 2022 | 39 | 0 | V0 | 0.000e+00 | 9.940e-11 | False | False |
| kuairand | 2022 | 39 | 1 | V2 | 3.288e-02 | 8.228e-01 | False | False |
| kuairand | 2022 | 39 | 2 | V2 | 1.150e-01 | 1.879e+00 | False | False |
| kuairand | 2022 | 39 | 3 | V2 | 1.201e-02 | 1.554e+00 | False | False |
| kuairand | 2022 | 39 | 4 | V2 | 2.098e-02 | 2.237e+00 | False | False |
| kuairand | 2023 | 9 | 0 | V0 | 0.000e+00 | 5.543e-11 | False | False |
| kuairand | 2023 | 9 | 1 | V2 | 5.277e-02 | 1.561e+00 | False | False |
| kuairand | 2023 | 9 | 2 | V2 | 1.301e-01 | 2.067e+00 | False | False |
| kuairand | 2023 | 9 | 3 | V2 | 1.752e-02 | 1.300e+00 | False | False |
| kuairand | 2023 | 9 | 4 | V2 | 3.369e-02 | 2.237e+00 | False | False |
| kuairand | 2023 | 39 | 0 | V0 | 0.000e+00 | 1.273e-11 | False | False |
| kuairand | 2023 | 39 | 1 | V2 | 2.137e-02 | 6.457e-01 | False | False |
| kuairand | 2023 | 39 | 2 | V2 | 8.289e-02 | 1.858e+00 | False | False |
| kuairand | 2023 | 39 | 3 | V2 | 1.693e-02 | 1.830e+00 | False | False |
| kuairand | 2023 | 39 | 4 | V2 | 2.316e-02 | 3.140e+00 | False | False |
| kuairand | 2024 | 9 | 0 | V0 | 0.000e+00 | 3.785e-11 | False | False |
| kuairand | 2024 | 9 | 1 | V2 | 6.977e-02 | 1.434e+00 | False | False |
| kuairand | 2024 | 9 | 2 | V2 | 1.028e-01 | 2.811e+00 | False | False |
| kuairand | 2024 | 9 | 3 | V2 | 2.866e-02 | 2.146e+00 | False | False |
| kuairand | 2024 | 9 | 4 | V2 | 3.729e-02 | 1.696e+00 | False | False |
| kuairand | 2024 | 39 | 0 | V0 | 0.000e+00 | 2.504e-11 | False | False |
| kuairand | 2024 | 39 | 1 | V2 | 3.149e-02 | 8.438e-01 | False | False |
| kuairand | 2024 | 39 | 2 | V2 | 8.672e-02 | 1.488e+00 | False | False |
| kuairand | 2024 | 39 | 3 | V2 | 6.598e-03 | 1.387e+00 | False | False |
| kuairand | 2024 | 39 | 4 | V2 | 2.765e-02 | 2.493e+00 | False | False |
| kuairand | 2025 | 9 | 0 | V0 | 0.000e+00 | 5.799e-11 | False | False |
| kuairand | 2025 | 9 | 1 | V2 | 4.704e-02 | 1.628e+00 | False | False |
| kuairand | 2025 | 9 | 2 | V2 | 8.216e-02 | 2.487e+00 | False | False |
| kuairand | 2025 | 9 | 3 | V2 | 8.785e-03 | 1.903e+00 | False | False |
| kuairand | 2025 | 9 | 4 | V2 | 2.631e-02 | 1.633e+00 | False | False |
| kuairand | 2025 | 39 | 0 | V0 | 0.000e+00 | 9.695e-11 | False | False |
| kuairand | 2025 | 39 | 1 | V2 | 2.614e-02 | 6.721e-01 | False | False |
| kuairand | 2025 | 39 | 2 | V2 | 8.189e-02 | 1.565e+00 | False | False |
| kuairand | 2025 | 39 | 3 | V2 | 6.596e-03 | 1.301e+00 | False | False |
| kuairand | 2025 | 39 | 4 | V2 | 1.327e-02 | 1.580e+00 | False | False |
| movielens | 2022 | 9 | 0 | V0 | 0.000e+00 | 1.187e-11 | False | False |
| movielens | 2022 | 9 | 1 | V2 | 2.882e-02 | 4.372e-01 | False | False |
| movielens | 2022 | 9 | 2 | V2 | 2.361e-02 | 1.601e-01 | False | False |
| movielens | 2022 | 9 | 3 | V2 | 1.749e-02 | 4.435e-01 | False | False |
| movielens | 2022 | 9 | 4 | V2 | 9.744e-03 | 3.402e-02 | False | False |
| movielens | 2022 | 39 | 0 | V0 | 0.000e+00 | 5.865e-11 | False | False |
| movielens | 2022 | 39 | 1 | V2 | 4.280e-02 | 5.334e-01 | False | False |
| movielens | 2022 | 39 | 2 | V2 | 1.694e-01 | 1.309e+00 | False | False |
| movielens | 2022 | 39 | 3 | V2 | 5.290e-02 | 1.119e+00 | False | False |
| movielens | 2022 | 39 | 4 | V2 | 6.599e-02 | 8.463e-01 | False | False |
| movielens | 2023 | 9 | 0 | V0 | 0.000e+00 | 4.523e-11 | False | False |
| movielens | 2023 | 9 | 1 | V2 | 6.931e-02 | 3.519e-01 | False | False |
| movielens | 2023 | 9 | 2 | V2 | 3.780e-03 | 1.422e-01 | False | False |
| movielens | 2023 | 9 | 3 | V2 | 4.394e-02 | 1.115e-01 | False | False |
| movielens | 2023 | 9 | 4 | V2 | 3.166e-02 | 4.151e-01 | False | False |
| movielens | 2023 | 39 | 0 | V0 | 0.000e+00 | 1.541e-11 | False | False |
| movielens | 2023 | 39 | 1 | V2 | 1.057e-01 | 1.072e+00 | False | False |
| movielens | 2023 | 39 | 2 | V2 | 1.256e-01 | 8.422e-01 | False | False |
| movielens | 2023 | 39 | 3 | V2 | 5.627e-02 | 9.710e-01 | False | False |
| movielens | 2023 | 39 | 4 | V2 | 3.711e-02 | 7.227e-01 | False | False |
| movielens | 2024 | 9 | 0 | V0 | 0.000e+00 | 4.284e-12 | False | False |
| movielens | 2024 | 9 | 1 | V2 | 4.881e-02 | 3.416e-01 | False | False |
| movielens | 2024 | 9 | 2 | V2 | 2.810e-02 | 2.894e-01 | False | False |
| movielens | 2024 | 9 | 3 | V2 | 2.334e-02 | 2.273e-01 | False | False |
| movielens | 2024 | 9 | 4 | V2 | 2.972e-02 | 2.649e-01 | False | False |
| movielens | 2024 | 39 | 0 | V0 | 0.000e+00 | 2.609e-11 | False | False |
| movielens | 2024 | 39 | 1 | V2 | 4.614e-02 | 5.366e-01 | False | False |
| movielens | 2024 | 39 | 2 | V2 | 4.341e-02 | 4.607e-01 | False | False |
| movielens | 2024 | 39 | 3 | V2 | 1.725e-02 | 5.036e-01 | False | False |
| movielens | 2024 | 39 | 4 | V2 | 2.800e-02 | 2.761e-01 | False | False |
| movielens | 2025 | 9 | 0 | V0 | 0.000e+00 | 2.820e-11 | False | False |
| movielens | 2025 | 9 | 1 | V2 | 8.851e-02 | 9.908e-01 | False | False |
| movielens | 2025 | 9 | 2 | V2 | 4.647e-02 | 2.789e-01 | False | False |
| movielens | 2025 | 9 | 3 | V2 | 4.738e-02 | 8.102e-01 | False | False |
| movielens | 2025 | 9 | 4 | V2 | 4.605e-03 | 6.618e-01 | False | False |
| movielens | 2025 | 39 | 0 | V0 | 0.000e+00 | 1.129e-11 | False | False |
| movielens | 2025 | 39 | 1 | V2 | 6.746e-02 | 8.230e-01 | False | False |
| movielens | 2025 | 39 | 2 | V2 | 1.514e-01 | 1.145e+00 | False | False |
| movielens | 2025 | 39 | 3 | V2 | 7.675e-02 | 8.741e-01 | False | False |
| movielens | 2025 | 39 | 4 | V2 | 6.333e-02 | 8.185e-01 | False | False |

## Failure log

- (none)

## Source files

- `experiments/stage_S7B_protocol.py`
- `experiments/stage_S7B_run.py`
- `experiments/stage_S7B_orchestrator.py`
- `experiments/stage_S7B_report.py`

