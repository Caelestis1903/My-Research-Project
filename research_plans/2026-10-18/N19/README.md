# N19 — full end-to-end MDMT study

Status: **RUNNING**, all final scientific judgments and test readout pending.
This stage follows the revised user start.md; N18's screening completion does
not satisfy this stage. Do not cite partial development tables as final results.

- [Entry archived as UTF8](source_start_utf8.md)
- [Design and mandatory full budgets](DESIGN_SPECIFICATION.md)
- [Mathematical scope/counterclaims](MATHEMATICAL_SCOPE.md)
- [Primary-source novelty audit](LITERATURE_UPDATE.md)
- [Protocol limits](PROTOCOL_LIMITS.md)
- [Development decisions and bounded extensions](DEVELOPMENT_DECISION_RULES.md)
- [Implementation report](IMPLEMENTATION_REPORT.md),[primary self-review](REVIEW.md),[alternative review](ALTERNATIVE_REVIEW.md)
- [Live execution snapshot](EXECUTION_STATUS.json),[queue manifest](queue_manifest.json),[mechanical status](queue_manifest.status.json)
- [Completed-run development tables](development_readout/method_means.csv),[paired effects](development_readout/paired_comparisons.csv),[cell effects](development_readout/paired_cells.csv)
- [Reproduction and local large states](REPRODUCE.md),[actual environment](ENVIRONMENT.json)

Declared full matrix currently totals300 formal runs:180 initial,8 P2 controls,
8 P4 additive controls,12 P1 MMoE,16 P2 audit-profile,16 P2 non-additive stress,
8 P6 residual/static-CV,32 preplanned P3/P4/P5 alternatives,20 P3 matched AdamW.
Seeds2022/2023/2024/2025 throughout. Native datasets for P3/P4/P5; explicit ML
controlled populations for P1/P2/P6. Some alternatives still await their primary
development launch triggers; the declared matrix is NOT yet complete. Engineering
runs are separately tagged/excluded. Every extension has a design/self-review
record and scientific trigger; no open-ended optimizer/hyperparameter grid.

All backbone/network parameters train from scratch. KR50epochs,ML100epochs,
complete fixed cached populations minus common5% train-meta. No early stopping,
single-seed cancellation, frozen-head substitute or test-driven method changes.
Only mechanical experiment execution is delegated; design, implementation,
review and scientific interpretation are primary-only under revised entry11.

Code: `experiments/n19_{data,models,objectives,train,queue,summarize,dynamics,rank_diagnostics,channels,evaluate,figures,tables,inventory}.py`.
Large model/optimizer/random states are under `/home_new/xzf_n19_artifacts`;
per-run configs identify exact directories. Logs and per-epoch/cell raw numbers
remain in `logs/` and `runs/`. The final archive will retain key raw evidence,
including negative outcomes, and record any large local-only prediction/state
artifacts explicitly. No overall COMPLETE until all triggered work finishes.
