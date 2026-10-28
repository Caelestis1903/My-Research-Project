# Historical preprocessing references

These JSON files are read-only reference inputs copied from the local historical experiment directories. The independent source snapshot does not track these files.

| Dataset | Original local source | Files |
|---|---|---|
| MovieLens | `artifacts/mdmt_relation_basis_v2/data/movielens/` | `PREPROCESSOR.json`, `DATA_MANIFEST.json` |
| KuaiRand | `artifacts/mdmt_relation_basis_v2/revision1/data/kuairand/` | `PREPROCESSOR.json`, `DATA_MANIFEST.json` |

The manifest's recorded SHA strings are provenance only. The pipeline does not compute or revalidate SHA values. Preparation refits the vocabulary/scaler from T0, compares JSON values, and rebuilds row arrays in the separate owned output directory.
