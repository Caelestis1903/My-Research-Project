> 本次仅发布研究报告与设计文档（含报告统计图）；运行代码、CSV/JSON结果、原始数据、日志、逐行预测和checkpoint均保留在服务器。正文数值与研究结论未改。
> 原文SHA256：`1c36fd6fbe6e394eff99ab41e4d1a602f2b50d3c99532bbec28bef308186bf99`。

# Recent-method adaptation specification

Frozen before baseline pilots. Same fit-only roles, natural logical batch40960, FP32, Adam .001, coupled wd1e-5. No outer labels. Total baseline development allowance per dataset is 72 measured native-equivalent epochs; search and retraining count. Review prior code against pinned author sources and perform one real-batch semantic check before admission.

STEM_DOMAIN_TASK reuses the locally checked adapter of LiangcaiSu/STEM@769e2af0d0d1a0be9f58b475b95e05f502a20df5 (Apache2.0). Six heads are domain×task, only two observed labels per row. Official default widths and cross-task detached borrowed experts are preserved. New fixed24/50 and eligible12 replace previous early-stop protocols; old scores are not reused.

AutoIFS reuses ChaohuaYang/AutoIFS@65fc27cdfad1eb6d7a4a2cc69cd31f61cc8e3d7e core equations and search/rewind/reuse implementation. Prespecify search10, rewind to start of search epoch10, then reuse to total24/50; sparse penalty .001 per task, temperature1→100, selector Adam1e-5/wd0. Both stages and copied frozen selector embedding count. This is a protocol adaptation, not reproduction of original paper dataset scores. No hyperparameter search.

STGU_DOMAIN_ADAM implements paper Algorithm1 (arxiv2502.11986v2), checked against wooseong97/sel-update-mtl@e5bc4565a4a1d33a8e9d6d8a2f13a63a72e7e3d3 (MIT). Three domain tasks each average its two labels; weighting p_d preserves natural full-batch target. Start singletons; shuffle group order independently; recompute all three losses after each real Adam group step. Track relative loss changes including active private changes, smooth with beta=.001. Opposite-sign within-group effects use negative max absolute effect, as paper Algorithm1. Unselected domain-private slot/expert/gate/tower parameters retain grad=None.

Explicit grouping adaptation: for exactly three domains, choose the maximum mutual-positive-affinity partition among all five set partitions, tie singleton-first. The official visual-task implementation instead uses normalized AffinityPropagation and delayed loss bookkeeping; those implementation choices are not claimed reproduced. This is a paper-mechanism/MDMT/Adam adaptation, retaining actual grouping and sequential updates, not an EMA C8 selector. External quality comparison must name the adaptation and cannot establish original STGU benchmark reproduction.

References: https://arxiv.org/html/2512.13396v1 ; https://arxiv.org/html/2308.13537v2 ; https://arxiv.org/html/2502.11986v2 . PRP/M2R remain outside these runnable slots; do not claim unresolved core code reproduced.
