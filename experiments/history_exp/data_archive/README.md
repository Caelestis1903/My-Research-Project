# Historical experiment data archive

This archive centralizes experiment data and generated results. Source code, prompt/design files, and analysis reports remain in their original folders.

- experiments/: raw run directories, stage outputs, analysis tables/figures, old review validations, failed-attempt logs, and the S6 reanalysis tables/logs.
- experiments_root_files/: flat stage CSV/JSON tables, manifests, queue state snapshots, and audit exports formerly directly under experiments/.
- workspace_root/_review_stage_E1/: archived E1 reviewer-run artifacts formerly at the workspace root.
- workspace_root_files/: workspace-level runner snapshot.
- archive_manifest.csv: original path, archive path, and archive/link verification status.
- cleanup_manifest.csv: cache and superseded-review cleanup record.

Directory paths left in place are NTFS junctions into this archive. Flat files left in place are hard links to their archived entries so existing scripts and report references continue to resolve.
