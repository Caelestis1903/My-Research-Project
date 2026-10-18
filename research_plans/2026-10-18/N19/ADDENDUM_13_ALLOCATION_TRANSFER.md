# P3 bounded optimizer-condition transfer

Design before launch. All4 KuaiRand AdamW uniform/protected runs completed50
epochs. Protected-minus-uniform selected macro AUC+.003168 in4/4,95% descriptive
interval[-.002388,.008724], loss-.005195 in4/4, interval[-.012335,.001945].
Weighted AUC-.000969 in3/4; final macro AUC-.007372 in4/4 and task1 AUC-.014192
in4/4. Active path sensitivity has been measured, unlike the original Adam
KuaiRand regime. This is a potentially useful selected macro trade-off, not
uniform cell protection or a stable final advantage.

The predeclared bounded optimizer extension rule is triggered by the repeated
selected same-budget effect and its contradictory weighted/stage behavior.
Run MovieLens fixed/rank_uniform/rank_protected with AdamW,4seeds each=12 full
100epoch networks,lr.001,batch40960,weight_decay1e-5,all original splits/selection/
pruning rules. No rates,decays,budgets or optimizer grid. The dense same-optimizer
reference is necessary to distinguish allocation from a practical compression
trade-off. Existing MovieLens Adam evidence remains and is not pooled.

This tests whether the condition-dependent allocation effect transfers to the
second native dataset, where domain sizes/tasks and step count differ. Failure
to transfer or loss/ranking contradictions limit the mechanism; a favorable
mean alone is not independent novelty beyond established rank allocation.
No test performance has been assessed, and all12budgets complete regardless.

Implementation is the already-reviewed --optimizer adamw dispatch. Automatic
optimizer_adamw suffix, config optimizer and per-run snapshots distinguish runs.
Primary self-review (not independent): PASS; exact command uses ML defaults,
same active48-rank budget and >=epoch10 selection, no new training code.
