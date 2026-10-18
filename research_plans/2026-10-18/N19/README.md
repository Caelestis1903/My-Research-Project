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

Initial matrix:180 formal runs, plus8 necessary P2 controls/ablations, each with
four seeds2022/2023/2024/2025. Both native datasets for P3/P4/P5; explicit ML
controlled populations for P1/P2/P6. P3/P4/P5 preplanned alternatives add32 full
runs after primary development readings. Engineering runs are separately tagged
and excluded from scientific tables. Further discriminating extensions depend
on actual development evidence and get their own design/review records.

All backbone/network parameters train from scratch. KR50epochs,ML100epochs,
complete fixed cached populations minus common5% train-meta. No early stopping,
single-seed cancellation, frozen-head substitute or test-driven method changes.
Only mechanical experiment execution is delegated; design, implementation,
review and scientific interpretation are primary-only under revised entry11.

Code: `experiments/n19_{data,models,objectives,train,queue,summarize,dynamics,evaluate,figures}.py`.
Large model/optimizer/random states are under `/home_new/xzf_n19_artifacts`;
per-run configs identify exact directories. Logs and per-epoch/cell raw numbers
remain in `logs/` and `runs/`. The final archive will retain key raw evidence,
including negative outcomes, and record any large local-only prediction/state
artifacts explicitly. No overall COMPLETE until all triggered work finishes.
