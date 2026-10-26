# RCF-v1 registered experiment

Workspace: `/home/xzf/Code/phase0-3`. The `/root` names in orchestration logs are logical agent names, not file locations.

Read the complete registered protocol in `research_plans/2026-10-26-v2/`, starting with `06_CODEX_EXECUTION.md`. Mathematical semantics are in `05_核心算法与完整实验规范.md`; branching rules are in `04_判别实验与研究决策.md`. The user's later minimum-four-training-seed requirement supersedes the original two-development-seed budget. The original plan bytes remain unchanged.

Effective configuration: `config_four_seed.json`; development seeds 5201–5204; independent locked seeds 6201–6204 only if registered gates trigger them. SCALE uses its separately registered 2022–2025 seeds. A replay of the same seed is not another statistical replicate.

Use the repository interpreter explicitly:

```bash
/home/xzf/miniconda3/envs/phase0-3-env/bin/python -m experiments.mdmt_rcf_20261026.cli --help
```

`cli.py` implements audit, build-plan, required verification, RAM-only service, prefix, read-probe, form, decide, diagnose, run-locked, freeze, evaluate, finalize, and the independent SCALE branch. Normal operation is the sequence in the execution document. A process with lost RAM cannot be resumed from logs; model/optimizer/RNG checkpoints are deliberately never serialized, as required by `start.md`.

The completed/active campaign's identity and state are recorded under `artifacts/mdmt_rcf_20261026/v2_four_seed/`; historical recovery evidence and the independent SCALE data are retained under the same task's `v1/`. Do not launch a second service over an existing socket or overwrite a frozen campaign. A new replication needs a separately registered output root and refreshed identity/review, preserving the original data/model/objective/selection semantics and all four seeds. It is not a disk-checkpoint continuation.

Source snapshots in `vendor_snapshot/` retain original bytes, Git blob SHA and SHA256. New adapters and algorithms live outside those frozen snapshots. Current published source and earlier loaded training versions have separate recorded hashes. Post-training observation repairs do not rewrite the training freeze or selected predictions.

Models remain in worker memory until raw paired predictions, source/formed functions, frozen requests/groups/policies, full raw-X reconstruction, metrics, and required measured deployment costs are accepted. Cleanup is limited to manifest-owned temporary row-data mappings in this campaign. Final research exports remain. There are no model, optimizer, checkpoint, or equivalent weight archives to download.

Research findings, four-seed raw tables, revisions, actual completion/cleanup receipts, and retained-server artifact references are in `research_reports/2026-10-26_v2/`. Large input-derived NPZ/NPY research artifacts stay on the server and are indexed by hash; they are not bundled into Git.
