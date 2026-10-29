# Published metadata parts

Large JSON records are transported as exact byte parts because the connector could not upload whole records. Restore them with the standard-library command in [reproduction_guide.md](reproduction_guide.md#restore-metadata-in-a-published-checkout). Source files, CSV tables, figures and FINAL_REPORT.md are ordinary files. The shared workspace retains complete originals.

| Complete logical file | Bytes | Parts |
|---|---:|---:|
| `research_reports/mdmt_baselines_2526/FORMAL_RAW_RESULTS.json` | 5,717,425 | 12 |
| `research_reports/mdmt_baselines_2526/RUNTIME_IO_REVIEW_RECEIPT.json` | 1,337,261 | 3 |
| `research_reports/mdmt_baselines_2526/RUNTIME_IO_SOURCE_EQUIVALENCE.json` | 1,520,709 | 4 |
| `research_reports/mdmt_baselines_2526/experiment_manifest.json` | 3,744,154 | 8 |
| `artifacts/mdmt_baselines_2526/PRE_FORMAL_FREEZE.json` | 674,317 | 2 |
| `artifacts/mdmt_baselines_2526/final_freeze.json` | 7,492,606 | 15 |

The machine-readable part paths and exact ordering are in [PUBLICATION_PARTS.json](PUBLICATION_PARTS.json). Reassembly does not change a JSON value, historical receipt or encoded source snapshot. All sets were checked by direct byte equality before publication.
