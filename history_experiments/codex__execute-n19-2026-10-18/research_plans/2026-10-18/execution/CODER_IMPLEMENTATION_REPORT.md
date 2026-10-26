# N18 implementation report — before substantive execution

Implemented only DESIGN_SPECIFICATION and pre-run AMENDMENT_01_SCHEMA.

| Source | Responsibility |
|---|---|
| `experiments/n18_common.py` | Read cached frozen population, strict-load four E10 models, validation split, gradient flattening |
| `experiments/n18_theory_data.py` | P1 support LPs and timestamp/schema readout; P2 controlled masking/audits/worlds; P3 certificates; P5 product-fusion check |
| `experiments/n18_probe.py` | No-update P3 features and P4/P6 full-coordinate measurements, row IDs and raw Gram/moment data |
| `experiments/n18_fit_heads.py` | Analytic logit distillation and raw cell AUC/logloss; no conclusions |
| `experiments/n18_readout.py` | Preregistered primary-analysis arithmetic, conventional proposals and equivalent-cost aggregation |

Basic checks: Python compilation succeeded. Existing KR E10 checkpoint loads
strictly into actual `experiments/stage_S1_models.py::StageS1M3OEAdapter`, inherited
from `m3oe/MDMTRec/models/multi_domain/model.py`. `forward(x,test_flag=False)`
returns [2,2] for a two-row check. Model has 1,614,254 parameters, no nonzero
dropout/BatchNorm, no trainable fusion. Effective fixed fusion sigmoid(1).
No optimizer is created for checkpoint probes or analytic fitting. DARTS, DBO,
gradient scaling, scarce-domain augmentation and curriculum are disabled.
Parent checkpoint training was the fixed-fusion single-Adam Sample protocol.
Validation selection/diagnostic pool counts: [1346,123116,921] /
[1346,123116,922], sufficient for declared without-replacement measurements.

P3 labels never enter fitting; teacher logits are fit with ridge 1e-6 on a
train-standardized 64-vector plus bias. Equivalent repair parameterizations use
one canonical orthonormal span, so a parameterization-specific ridge penalty
cannot create a false difference. Full and additive coefficients, chosen residual,
SVD factors, normalizer, raw predictions, row IDs and per-cell metrics are saved.
The candidate requires frozen DT-dependent input features, retained gates/experts
and the trunk; it is explicitly a head compression study.

P4 full-coordinate Gram matrices retain all A/B/C cross-products needed for
recomputing predictions and interaction statistics. P6 stores each sampled row's
norm, six projections and proxy plus domain mean Gram to calculate centered
covariances. It uses 288 individual-row gradients per seed. Direction and norm
proposals are fitted only on design rows; heldout rows are evaluated once.

No research results have been analyzed at this stage. Independent review is not
claimed: latest start.md mandates primary-only work outside mechanical training.
