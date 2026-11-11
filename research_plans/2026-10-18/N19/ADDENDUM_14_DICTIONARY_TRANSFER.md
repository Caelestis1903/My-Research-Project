# P3 bounded active-dictionary stage-transfer falsifier

Design before launch; all20 KR AdamW runs have finished full50epochs. Cell-minus-
domain dictionary selected macro AUC+.005095 in3/4, descriptive95% interval
[-.018263,.028454], loss+.000568 mixed2/2. Final macro AUC+.020424 in4/4,
interval[-.001926,.042774], train-weighted AUC+.006361 in4/4; task0 AUC+.007697
in4/4, interval[.000618,.014776]. Final loss-.007136 is mixed2/2. Against
uniform-rank final AUC+.008108 in4/4 but loss+.026607 worse4/4. Thus a repeated
late ranking effect is large enough to discriminate, not a uniform endpoint win.

Run ML domain_codebook/cell_codebook with AdamW,4seeds each=8 full100epoch
networks,lr.001,batch40960,weight_decay1e-5,unchanged data/meta/selection. Reuse
the already-declared same-optimizer ML dense/uniform/protected references. The
original ML Adam dictionaries stay separate. This completes the bounded native
dataset x optimizer capacity-family comparison, without a decay/rank/rate grid.

The question is whether task-conditioned sharing has a transferable late ranking
effect when slots are active, or whether task/optimizer/stage and calibration
explain a KR-specific difference. All budgets finish regardless of early signs.
Report selected/final AUC/loss, both tasks, six cells and population weights. A
late-only ranking gain, loss conflict or optimizer-specific behavior limits the
claim; none proves novel shared-dictionary principles or finite-risk protection.
The planned comparison is sufficient to characterize these tested mechanisms;
other hyperparameter regimes belong to future studies, not an open-ended grid.

Implementation: existing reviewed --optimizer adamw and dictionary dispatch.
Primary self-review (not independent): PASS. Same matched storage definitions,
all backbone parameters train, no new numerical training path or engineering run.
No test performance evaluated. Formal total increases320->328 (P3 exclusive72).
