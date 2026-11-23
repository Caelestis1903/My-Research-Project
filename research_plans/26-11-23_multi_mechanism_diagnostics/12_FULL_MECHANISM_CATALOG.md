# 12 — 全候选机制目录与运行注册

本文件作为 VARIANT_REGISTER 的静态设计来源。Codex 必须将下列每个 ID 展开为 dataset × seed × anchor × arm 的任务记录。

## M1 — Private-gradient shaping
- M1-A COSINE_CONSISTENCY
- M1-B MAGNITUDE_AWARE_CONSISTENCY
- M1-C PAIRWISE_CONFLICT_ENERGY
- Controls: P_BCE_PRESTEP、COORDINATE_SHUFFLE、JOINT_TWO_STEP、CONSISTENCY_ONLY_READOUT

## M2 — Row-task conditional influence
- M2-A TASK_CELL_IMPACT_WEIGHTING
- M2-B IMPACT_RANKED_TASK_GRAD_ACCUMULATION
- M2-C NEGATIVE_IMPACT_SUPPRESSION
- M2-D CELL_RESPONSE_WEIGHTING
- Controls: NATURAL_UNIFORM、TASK_AGNOSTIC、WITHIN_DOMAIN_SHUFFLE、RANDOM_MASS_MATCHED

## M3 — Stage/group attribution
- M3-A STAGEWISE_ROW_TASK_ATTRIBUTION
- M3-B SCORE_DEFINED_GROUP_INTERACTION
- M3-C RANDOM_MATCHED_GROUP_INTERACTION
- M3-D STAGE_BY_GROUP_INTERACTION
- Arms for each group pair: A_ONLY、B_ONLY、A_B、A_A、B_B
M3 always runs; it is not conditional on M2 success.

## M4 — Conditional feature dependence
- M4-A DOMAIN_TASK_CONDITIONAL_DEPENDENCE_MAP
- M4-B CONDITIONAL_FEATURE_GROUP_PERMUTATION
- M4-C TASK_CONDITIONED_RESIDUALIZATION
- M4-D TASK_CONDITIONED_HSIC_PENALTY
- Controls: SHAM_PERMUTATION、SHAM_PENALTY、UNMODIFIED

## M5 — Multi-label augmentation
- M5-A SPARSE_FIELD_MASKING
- M5-B EMBEDDING_PERTURBATION
- M5-C WITHIN_STRATUM_CATEGORICAL_SUBSTITUTION
- M5-D DENSE_WITHIN_BIN_PERTURBATION
- M5-E MULTIVIEW_LABEL_CONSISTENCY
- Controls: ORIGINAL_VIEW、GAUSSIAN_EMA_REFERENCE、SHAM_TRANSFORM
Each transform has validity, label-pair, domain-preservation and prediction-consistency receipts. Invalid transform does not exclude other M5 variants.

## M6 — Output sensitivity control
- M6-A SENSITIVITY_SCALAR_SCALING
- M6-B WORST_CELL_TRUST_REGION
- M6-C PARAMETER_GROUP_RESPONSE_GATING
- Controls: GLOBAL_NORM_CLIP、RANDOM_SCALE、UNMODIFIED_UPDATE

## M7 — Distributional attribution
- M7-A FIVE_SEED_INFLUENCE_DISTRIBUTION
- M7-B MEAN_MINUS_LAMBDA_SD_ROBUST_SCORE
- M7-C LEAVE_ONE_SEED_OUT_TRANSFER
- Controls: MEAN_ONLY、SHUFFLED_SCORE、SEED_IDENTITY_PERMUTATION

## M8 — Task conflict controller
- M8-A PCGRAD_REFERENCE
- M8-B CONFLICT_AWARE_SIMPLEX_WEIGHTS
- M8-C COMMON_DESCENT_PROJECTION
- M8-D CONFLICT_AWARE_CELL_GATING
- Controls: CANONICAL_JOINT、UNIFORM_WEIGHTS、RANDOM_WEIGHTS

## Cross-family interactions
- X1 M1×M2: CONSISTENCY_PLUS_ROW_TASK_IMPACT
- X2 M2×M3: IMPACT_WEIGHTING_PLUS_GROUP_INTERACTION
- X3 M4×M5: DEPENDENCE_GUIDED_MULTIVIEW_AUGMENTATION
- X4 M6×M8: SENSITIVITY_CONSTRAINED_CONFLICT_CONTROL
- X5 M1×M8: PRIVATE_GRADIENT_SHAPING_PLUS_SHARED_CONFLICT_CONTROL

Component arms reuse existing immutable single-family receipts. Each X interaction adds a combined arm and a matched-budget control arm. This is why the cross-family allocation is 2 new arms × 5 interactions × 30 anchors = 300 fork slots, not four new arms per interaction.

## Dataset feature group registry

Feature groups must be generated from actual S0 DATA_MANIFEST and model registry, not hardcoded assumptions.

MovieLens schema observed in D2:
- entity: user_id, movie_id;
- demographics/context: gender, age, occupation, zip;
- content: cate_id;
- routing: domain_indicator.

KuaiRand schema observed in D2:
- user/context: user_active_degree, is_live_streamer, is_video_author, follow_user_num_range, fans_user_num_range, friend_user_num_range, register_days_range;
- sparse content/context: onehot_feat0–onehot_feat17, video_type, upload_dt, upload_type, server_width, server_height, music_type, tag, counts;
- routing: domain_indicator.

This grouping is an operational feature registry, not a claim that every field is semantically interchangeable. Feature transformation validity is separately audited.
